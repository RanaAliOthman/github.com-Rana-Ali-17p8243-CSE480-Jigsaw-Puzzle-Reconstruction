"""Train the graph neural network on the puzzle's true adjacencies.

    python3 -m ml.train_gnn --epochs 500

The GNN answers the same question as the Siamese pair classifier — are these two
piece sides adjacent in the solved puzzle? — but it is a genuinely different
model, not a re-parameterised one. The Siamese judges a pair in isolation. Here
every piece side is a *node* and every candidate pair an *edge*, and three
rounds of message passing let a side's representation be shaped by all the other
partners competing for it before the edge head commits to an answer.

Same splits, by construction
----------------------------
The edges are built from the examples :func:`ml.dataset.build_pairs` returns —
the same held-out 2x4 block, the same 84 training positives, the same 672
sampled negatives the Siamese sees. Both models are then ranked by the single
protocol in :mod:`ml.ranking`, against the classical measure on identical
candidate sets.

Reported honestly
-----------------
Only the ``neighbor_logit`` head is trained. ``orientation_logits`` and
``compatibility`` stay at their random initialisation and must not be read as
meaningful. Rank-1 is quoted over the held-out block only and is not comparable
to the 46.6% MGC figure measured against all 140 sides of the puzzle.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.edge_compatibility import compatibility, complementary
from .dataset import EDGE_DIM, NODE_DIM, build_graph, build_pairs, load_solved_pieces
from .gnn import JigsawGNN
from .ranking import rank_report

REPORT = Path("results/evaluation_results/gnn_training.json")
CHECKPOINT = Path("results/gnn/gnn.pt")


def smoke_train(output="results/gnn/smoke_checkpoint.pt"):
    """One forward/backward pass on deterministic tensors; not dataset training."""
    torch.manual_seed(480)
    m = JigsawGNN()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    nodes = torch.rand(5, 32)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 2]])
    edge_attr = torch.rand(6, 16)
    y = torch.tensor([[1.], [0.], [1.], [0.], [0.], [1.]])
    out = m(nodes, edge_index, edge_attr)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["neighbor_logit"], y)
    loss.backward()
    opt.step()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": m.state_dict(), "optimizer": opt.state_dict(),
                "loss": loss.item(), "verified_smoke_only": True}, output)
    return loss.item()


def _to_torch(graph, device):
    return (torch.from_numpy(graph["nodes"]).to(device),
            torch.from_numpy(graph["edge_index"]).to(device),
            torch.from_numpy(graph["edge_attr"]).to(device),
            torch.from_numpy(graph["labels"]).to(device))


def _gated_candidate_keys(pieces, block_cells):
    """Every ordered pair inside the block the tab/blank gate admits.

    These are exactly the pairs :func:`ml.ranking.rank_report` enumerates, so the
    scoring graph must contain an edge for each of them.
    """
    sides = [(cell, j) for cell in sorted(block_cells) for j in range(4)]
    keys = []
    for a_cell, a_side in sides:
        for b_cell, b_side in sides:
            if a_cell == b_cell:
                continue
            if complementary(pieces[a_cell].sides[a_side].kind,
                             pieces[b_cell].sides[b_side].kind):
                keys.append((a_cell, a_side, b_cell, b_side))
    return keys


def train(epochs=500, batch_size=32, lr=1e-3, seed=480, hidden=64, layers=3,
          output=CHECKPOINT, report=REPORT):
    """Train on the real adjacencies and rank against the classical measure."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    started = time.perf_counter()
    pieces, rows, cols = load_solved_pieces()
    train_set, valid_set, info = build_pairs(pieces, rows, cols, seed=seed)
    block_cells = set(info["held_out_cells"])

    train_graph = build_graph(train_set, pieces)
    valid_graph = build_graph(valid_set, pieces)
    # A third graph carries every gated candidate inside the block, so the
    # ranking protocol can ask about pairs that were never training examples.
    score_graph = build_graph(valid_set, pieces,
                              extra_edges=_gated_candidate_keys(pieces, block_cells))
    info = dict(info, train_nodes=train_graph["node_count"],
                train_edges=int(train_graph["edge_index"].shape[1]),
                valid_nodes=valid_graph["node_count"],
                valid_edges=int(valid_graph["edge_index"].shape[1]),
                scoring_edges=int(score_graph["edge_index"].shape[1]),
                node_features=NODE_DIM, edge_features=EDGE_DIM)
    print("graph: %d train nodes / %d edges, %d valid nodes / %d edges"
          % (info["train_nodes"], info["train_edges"],
             info["valid_nodes"], info["valid_edges"]), flush=True)

    model = JigsawGNN(node_dim=NODE_DIM, edge_dim=EDGE_DIM,
                      hidden=hidden, layers=layers).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr)
    pos_weight = torch.tensor([info["train_negative"] / max(info["train_positive"], 1)],
                              device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_tensors = _to_torch(train_graph, device)
    valid_tensors = _to_torch(valid_graph, device)
    generator = torch.Generator().manual_seed(seed)

    history, best = [], None
    for epoch in range(1, epochs + 1):
        # Message passing always runs over the whole training graph -- that is
        # what makes this a graph model -- but the loss is taken on a minibatch
        # of its edges, so the GNN gets the same number of gradient steps per
        # epoch, at the same batch size, as the Siamese.
        model.train()
        nodes, edge_index, edge_attr, y = train_tensors
        order = torch.randperm(y.shape[0], generator=generator).to(device)
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            optimiser.zero_grad()
            out = model(nodes, edge_index, edge_attr)
            loss = criterion(out["neighbor_logit"][batch], y[batch])
            loss.backward()
            optimiser.step()
            total += float(loss.item()) * len(batch)
        train_loss = total / len(order)

        model.eval()
        with torch.no_grad():
            nodes, edge_index, edge_attr, y = valid_tensors
            out = model(nodes, edge_index, edge_attr)
            valid_loss = float(criterion(out["neighbor_logit"], y).item())
            predicted = (torch.sigmoid(out["neighbor_logit"]) > 0.5).float()
            accuracy = float((predicted == y).float().mean().item())
            positives = y.squeeze(1) == 1
            recall = float((predicted.squeeze(1)[positives] == 1).float().mean().item())

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "valid_loss": valid_loss, "valid_accuracy": accuracy,
                        "valid_recall_on_true_seams": recall})
        if best is None or valid_loss < best["valid_loss"]:
            best = {"epoch": epoch, "valid_loss": valid_loss,
                    "state": copy.deepcopy(model.state_dict())}
        if epoch % 25 == 0 or epoch == 1:
            print("epoch %d/%d  train_loss %.4f  valid_loss %.4f  valid_acc %.3f  recall %.3f"
                  % (epoch, epochs, train_loss, valid_loss, accuracy, recall), flush=True)

    # --- ranking, on the same protocol and the same candidate sets ---
    score_tensors = _to_torch(score_graph, device)
    index = score_graph["index"]

    def learned_scores():
        model.eval()
        with torch.no_grad():
            nodes, edge_index, edge_attr, _ = score_tensors
            logits = model(nodes, edge_index, edge_attr)["neighbor_logit"]
        values = logits.squeeze(1).cpu().numpy()
        # The head emits a neighbour logit: higher means more likely adjacent,
        # so negate it to get a dissimilarity the ranker can sort ascending.
        return lambda a, b, c, d: -float(values[index[(a, b, c, d)]])

    final_rank = rank_report(learned_scores(), pieces, block_cells, cols=cols, rows=rows)
    final_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best["state"])
    best_rank = rank_report(learned_scores(), pieces, block_cells, cols=cols, rows=rows)
    model.load_state_dict(final_state)

    classical_rank = rank_report(
        lambda a, b, c, d: compatibility(pieces[a].sides[b], pieces[c].sides[d]),
        pieces, block_cells, cols=cols, rows=rows)

    print("\nheld-out block, identical candidate sets:")
    for label, figures in (("GNN, final epoch  ", final_rank),
                           ("GNN, best valid   ", best_rank),
                           ("classical MGC     ", classical_rank)):
        print("  %s: rank-1 %.1f%%  top-3 %.1f%%  median rank %.0f"
              % (label, 100 * figures["rank1"], 100 * figures["top3"],
                 figures["median_rank"]))

    elapsed = time.perf_counter() - started
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": final_state, "best_model": best["state"],
                "best_epoch": best["epoch"], "optimizer": optimiser.state_dict(),
                "epochs": epochs, "history": history, "dataset": info,
                "trained_heads": ["neighbor_logit"],
                "untrained_heads": ["orientation_logits", "compatibility"]}, output)
    summary = {
        "model": "JigsawGNN",
        "epochs": epochs, "device": str(device), "runtime_seconds": elapsed,
        "hyperparameters": {"optimiser": "Adam", "learning_rate": lr,
                            "batch": ("%d edges per step, message passing over "
                                      "the full %d-edge graph"
                                      % (batch_size, info["train_edges"])),
                            "loss": "BCEWithLogits, pos_weight=%.1f"
                                    % float(pos_weight.item()),
                            "hidden": hidden, "message_passing_layers": layers,
                            "augmentation": None},
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "dataset": info, "history": history,
        "best_epoch": best["epoch"], "best_valid_loss": best["valid_loss"],
        "held_out_block_rank": {"learned_final_epoch": final_rank,
                                "learned_best_valid": best_rank,
                                "classical_mgc": classical_rank},
        "trained_heads": ["neighbor_logit"],
        "untrained_heads": ["orientation_logits", "compatibility"],
        "caveat": ("One puzzle, %d training positives, the same split the Siamese "
                   "uses. Rank figures are over the held-out block only and are not "
                   "comparable to the 46.6%% MGC figure measured against all 140 "
                   "sides." % info["train_positive"]),
        "checkpoint": str(output),
    }
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=480)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--smoke", action="store_true",
                        help="run the one-step wiring test instead of training")
    args = parser.parse_args()
    if args.smoke:
        print({"smoke_loss": smoke_train()})
        return
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          seed=args.seed, hidden=args.hidden, layers=args.layers)


if __name__ == "__main__":
    main()

"""Train the Siamese pair classifier on the puzzle's true adjacencies.

    python3 -m ml.train_siamese --epochs 3

The model is asked one question: given the colour strips beneath two piece
sides, are those two sides actually adjacent in the solved puzzle?  That is the
same question :mod:`src.edge_compatibility` answers analytically, so the two can
be compared directly -- and the comparison is the point, because the assembly
work established that the pairwise measure, not the search, is what caps
reconstruction accuracy.

Only the ``neighbor_logit`` head is trained.  The model also exposes
``orientation_logits`` and ``compatibility``; those stay at their random
initialisation and must not be read as meaningful.

Reported honestly
-----------------
Rank-1 is quoted over the held-out block only, and the classical MGC measure is
evaluated on exactly the same restricted candidate set, so the two numbers are
comparable to each other.  Neither is comparable to the 46.6% MGC figure quoted
elsewhere, which ranks against all 140 sides of the puzzle rather than the 32
sides of the held-out block.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.edge_compatibility import compatibility
from .dataset import build_pairs, load_solved_pieces, side_strip
from .ranking import rank_report
from .siamese import SiameseCNN

REPORT = Path("results/evaluation_results/siamese_training.json")


def smoke_train(output="results/siamese/smoke_checkpoint.pt"):
    """Verified optimizer/checkpoint test on deterministic tensors; not dataset training."""
    torch.manual_seed(480)
    m = SiameseCNN()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    a = torch.rand(4, 3, 32, 32)
    b = torch.rand(4, 3, 32, 32)
    y = torch.tensor([[1.], [0.], [1.], [0.]])
    out = m(a, b)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["neighbor_logit"], y)
    loss.backward()
    opt.step()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": m.state_dict(), "optimizer": opt.state_dict(),
                "loss": loss.item(), "verified_smoke_only": True}, output)
    return loss.item()


def _tensors(examples, device):
    a = torch.from_numpy(np.stack([e[0] for e in examples])).to(device)
    b = torch.from_numpy(np.stack([e[1] for e in examples])).to(device)
    y = torch.tensor([[e[2]] for e in examples], dtype=torch.float32, device=device)
    return a, b, y


def train(epochs=3, batch_size=32, lr=1e-3, seed=480, resume=False,
          output="results/siamese/siamese.pt", report=REPORT):
    """Train on the real adjacencies and report against the classical measure.

    With ``resume``, the model and optimiser state are loaded from ``output`` and
    ``epochs`` more are run on top, so the reported epoch count is the running
    total.  The split, the seed and the batch order generator are unchanged, so a
    resumed run sees the same data in the same order as a fresh one would.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    started = time.perf_counter()
    pieces, rows, cols = load_solved_pieces()
    train_set, valid_set, info = build_pairs(pieces, rows, cols, seed=seed)
    block_cells = set(info["held_out_cells"])
    print("data: %d train (%d positive), %d valid (%d positive), strips %s"
          % (len(train_set), info["train_positive"], len(valid_set),
             info["valid_positive"], info["strip_shape"]), flush=True)

    model = SiameseCNN().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr)

    history = []
    completed = 0
    if resume:
        checkpoint = torch.load(output, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimiser.load_state_dict(checkpoint["optimizer"])
        history = list(checkpoint.get("history", []))
        completed = int(checkpoint.get("epochs", len(history)))
        print("resumed from %s at epoch %d" % (output, completed), flush=True)
    # Negatives outnumber positives by the sampling ratio; weight the positives
    # back up so the model cannot win by answering "not adjacent" every time.
    pos_weight = torch.tensor([info["train_negative"] / max(info["train_positive"], 1)],
                              device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_a, train_b, train_y = _tensors(train_set, device)
    valid_a, valid_b, valid_y = _tensors(valid_set, device)
    generator = torch.Generator().manual_seed(seed)

    best = None
    for record in history:
        if best is None or record["valid_loss"] < best["valid_loss"]:
            best = {"epoch": record["epoch"], "valid_loss": record["valid_loss"],
                    "state": None}
    for epoch in range(completed + 1, completed + epochs + 1):
        model.train()
        order = torch.randperm(len(train_set), generator=generator).to(device)
        total = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            optimiser.zero_grad()
            out = model(train_a[index], train_b[index])
            loss = criterion(out["neighbor_logit"], train_y[index])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(index)
        train_loss = total / len(order)

        model.eval()
        with torch.no_grad():
            out = model(valid_a, valid_b)
            valid_loss = criterion(out["neighbor_logit"], valid_y).item()
            predicted = (torch.sigmoid(out["neighbor_logit"]) > 0.5).float()
            accuracy = (predicted == valid_y).float().mean().item()
            positives = valid_y.squeeze(1) == 1
            recall = (predicted.squeeze(1)[positives] == 1).float().mean().item()
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "valid_loss": valid_loss, "valid_accuracy": accuracy,
                        "valid_recall_on_true_seams": recall})
        if best is None or valid_loss < best["valid_loss"]:
            best = {"epoch": epoch, "valid_loss": valid_loss,
                    "state": copy.deepcopy(model.state_dict())}
        if epoch % 25 == 0 or epoch == completed + 1 or epochs <= 25:
            print("epoch %d/%d  train_loss %.4f  valid_loss %.4f  valid_acc %.3f  recall %.3f"
                  % (epoch, completed + epochs, train_loss, valid_loss,
                     accuracy, recall), flush=True)

    # --- the comparison that matters, on identical candidate sets ---
    strips = {(cell, j): side_strip(pieces[cell].sides[j])
              for cell in block_cells for j in range(4)}
    reversed_strips = {(cell, j): side_strip(pieces[cell].sides[j], reverse=True)
                       for cell in block_cells for j in range(4)}

    @torch.no_grad()
    def learned(a_cell, a_side, b_cell, b_side):
        model.eval()
        a = torch.from_numpy(strips[(a_cell, a_side)][None]).to(device)
        b = torch.from_numpy(reversed_strips[(b_cell, b_side)][None]).to(device)
        # The head returns a neighbour logit; higher means more likely adjacent,
        # so negate it to get a dissimilarity the ranker can sort ascending.
        return -float(model(a, b)["neighbor_logit"].item())

    def classical(a_cell, a_side, b_cell, b_side):
        return compatibility(pieces[a_cell].sides[a_side], pieces[b_cell].sides[b_side])

    total_epochs = completed + epochs
    final_rank = rank_report(learned, pieces, block_cells, cols=cols, rows=rows)
    final_state = copy.deepcopy(model.state_dict())
    if best is not None and best["state"] is not None:
        model.load_state_dict(best["state"])
        best_rank = rank_report(learned, pieces, block_cells, cols=cols, rows=rows)
        model.load_state_dict(final_state)
    else:                       # every epoch of this run was worse than a resumed one
        best_rank = None
    classical_rank = rank_report(classical, pieces, block_cells, cols=cols, rows=rows)

    print("\nheld-out block, identical candidate sets:")
    print("  learned, final epoch %d: rank-1 %.1f%%  top-3 %.1f%%  median rank %.0f"
          % (total_epochs, 100 * final_rank["rank1"], 100 * final_rank["top3"],
             final_rank["median_rank"]))
    if best_rank is not None:
        print("  learned, best valid %-3d: rank-1 %.1f%%  top-3 %.1f%%  median rank %.0f"
              % (best["epoch"], 100 * best_rank["rank1"], 100 * best_rank["top3"],
                 best_rank["median_rank"]))
    print("  classical MGC          : rank-1 %.1f%%  top-3 %.1f%%  median rank %.0f"
          % (100 * classical_rank["rank1"], 100 * classical_rank["top3"],
             classical_rank["median_rank"]))

    elapsed = time.perf_counter() - started
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": final_state, "optimizer": optimiser.state_dict(),
                "best_model": None if best is None else best["state"],
                "best_epoch": None if best is None else best["epoch"],
                "epochs": total_epochs, "history": history, "dataset": info,
                "trained_heads": ["neighbor_logit"],
                "untrained_heads": ["orientation_logits", "compatibility"]}, output)
    summary = {
        "model": "SiameseCNN",
        "epochs": total_epochs, "device": str(device), "runtime_seconds": elapsed,
        "hyperparameters": {"optimiser": "Adam", "learning_rate": lr,
                            "batch_size": batch_size,
                            "loss": "BCEWithLogits, pos_weight=%.1f"
                                    % float(pos_weight.item()),
                            "augmentation": None},
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "dataset": info, "history": history,
        "best_epoch": None if best is None else best["epoch"],
        "best_valid_loss": None if best is None else best["valid_loss"],
        "held_out_block_rank": {"learned_final_epoch": final_rank,
                                "learned_best_valid": best_rank,
                                "classical_mgc": classical_rank},
        "trained_heads": ["neighbor_logit"],
        "untrained_heads": ["orientation_logits", "compatibility"],
        "caveat": ("One puzzle, %d training positives. Rank figures are over the "
                   "held-out block only and are not comparable to the 46.6%% MGC "
                   "figure measured against all 140 sides."
                   % info["train_positive"]),
        "checkpoint": str(output),
    }
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=480)
    parser.add_argument("--resume", action="store_true",
                        help="continue from the existing checkpoint for --epochs more")
    parser.add_argument("--smoke", action="store_true",
                        help="run the one-step wiring test instead of training")
    args = parser.parse_args()
    if args.smoke:
        print({"smoke_loss": smoke_train()})
        return
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          seed=args.seed, resume=args.resume)


if __name__ == "__main__":
    main()

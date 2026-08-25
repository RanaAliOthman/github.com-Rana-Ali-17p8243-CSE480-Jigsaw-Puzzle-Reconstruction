"""Tests for the learning milestone's data plumbing and model wiring.

These do not assert that either model is *good* — that is measured by the
training runs and reported in ``report/milestone_2_report.md``. What they assert
is the property the comparison depends on: both models see the same split, the
graph is built from the same labelled pairs the Siamese consumes, and the single
ranking protocol behaves correctly on a scorer whose answer is known.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ml.dataset import (EDGE_DIM, MissingAdjacencyGroundTruth, NODE_DIM,
                        build_graph, build_pairs, load_adjacency_manifest,
                        load_solved_pieces)
from ml.ranking import block_seams, rank_report

torch_missing = importlib.util.find_spec("torch") is None


@pytest.fixture(scope="module")
def solved():
    """The solved puzzle and the split both models are trained on."""
    if not Path("data/ground_truth/layout.json").exists():
        pytest.skip("ground truth not built; run scripts.build_ground_truth")
    pieces, rows, cols = load_solved_pieces()
    train_set, valid_set, info = build_pairs(pieces, rows, cols)
    return pieces, rows, cols, train_set, valid_set, info


def test_manifest_refuses_missing_truth(tmp_path):
    with pytest.raises(MissingAdjacencyGroundTruth):
        load_adjacency_manifest(tmp_path / "missing.csv")


# ------------------------------------------------------------------- the split

def test_the_split_never_puts_one_piece_on_both_sides(solved):
    _, _, _, train_set, valid_set, info = solved
    held_out = set(info["held_out_cells"])
    for *_, key in train_set:
        assert key[0] not in held_out and key[2] not in held_out
    for *_, key in valid_set:
        assert key[0] in held_out and key[2] in held_out


def test_the_split_reports_what_it_dropped(solved):
    _, _, _, _, _, info = solved
    assert info["positives_dropped_at_split_boundary"] > 0, \
        'a contiguous block must straddle some seams, and dropping them is the point'
    assert info["train_positive"] == 84 and info["train_negative"] == 672
    assert info["valid_positive"] == 20 and info["valid_negative"] == 160


# ------------------------------------------------------------------- the graph

def test_the_graph_is_built_from_the_very_same_examples(solved):
    pieces, _, _, train_set, _, _ = solved
    graph = build_graph(train_set, pieces)
    assert graph["edge_index"].shape == (2, len(train_set))
    assert graph["labels"].shape == (len(train_set), 1)
    assert graph["nodes"].shape[1] == NODE_DIM
    assert graph["edge_attr"].shape == (len(train_set), EDGE_DIM)
    assert int((graph["labels"] == 1).sum()) == 84
    assert int((graph["labels"] == 0).sum()) == 672


def test_every_edge_can_be_found_again_by_its_key(solved):
    pieces, _, _, _, valid_set, _ = solved
    graph = build_graph(valid_set, pieces)
    for row, (*_, key) in enumerate(valid_set):
        assert graph["index"][key] == row


def test_extra_edges_are_scored_but_never_labelled(solved):
    pieces, _, _, _, valid_set, info = solved
    held_out = sorted(info["held_out_cells"])
    extra = [(held_out[0], 0, held_out[1], 2), (held_out[1], 1, held_out[2], 3)]
    plain = build_graph(valid_set, pieces)
    widened = build_graph(valid_set, pieces, extra_edges=extra)
    added = widened["edge_index"].shape[1] - plain["edge_index"].shape[1]
    assert added == len([k for k in extra if k not in plain["index"]])
    for key in extra:
        row = widened["index"][key]
        if row >= len(valid_set):
            assert np.isnan(widened["labels"][row, 0]), 'unlabelled, so never trained on'


def test_node_features_carry_no_classical_score(solved):
    """The learned measures must not be handed the number they are compared with."""
    pieces, _, _, train_set, _, _ = solved
    graph = build_graph(train_set, pieces)
    assert np.isfinite(graph["nodes"]).all()
    assert np.isfinite(graph["edge_attr"]).all()
    # MGC costs on this puzzle run to tens; every feature here is a colour
    # statistic in 0..1, a relief in body units, or a flag.
    assert np.abs(graph["nodes"]).max() < 5.0
    assert np.abs(graph["edge_attr"]).max() < 5.0


# ---------------------------------------------------------------- the protocol

def test_block_seams_finds_the_seams_inside_the_block(solved):
    _, _, _, _, _, info = solved
    seams = block_seams(set(info["held_out_cells"]))
    assert len(seams) == 10, 'a 2x4 block has 10 internal seams'
    for a_cell, a_side, b_cell, b_side in seams:
        assert (a_side, b_side) in ((1, 3), (2, 0))


def test_an_oracle_ranks_first_and_an_adversary_ranks_last(solved):
    pieces, _, _, _, _, info = solved
    held_out = set(info["held_out_cells"])
    truth = {(a, b, c, d) for a, b, c, d in block_seams(held_out)}
    truth |= {(c, d, a, b) for a, b, c, d in block_seams(held_out)}

    oracle = rank_report(lambda *key: 0.0 if key in truth else 1.0, pieces, held_out)
    adversary = rank_report(lambda *key: 1.0 if key in truth else 0.0, pieces, held_out)
    assert oracle["rank1"] == 1.0 and oracle["median_rank"] == 1
    assert adversary["rank1"] == 0.0
    assert oracle["pairs_scored"] == adversary["pairs_scored"] == 20


def test_the_classical_measure_is_scored_by_the_same_protocol(solved):
    from src.edge_compatibility import compatibility
    pieces, _, _, _, _, info = solved
    figures = rank_report(
        lambda a, b, c, d: compatibility(pieces[a].sides[b], pieces[c].sides[d]),
        pieces, set(info["held_out_cells"]))
    assert figures["pairs_scored"] == 20
    assert figures["rank1"] == pytest.approx(0.55), \
        'the MGC baseline both models are compared against'


# -------------------------------------------------------------- model wiring

@pytest.mark.skipif(torch_missing, reason="PyTorch not installed in this interpreter")
def test_models_forward_backprop_checkpoint(tmp_path):
    from ml.train_gnn import smoke_train as train_gnn
    from ml.train_siamese import smoke_train as train_siamese
    assert train_siamese(tmp_path / "s.pt") > 0
    assert train_gnn(tmp_path / "g.pt") > 0


@pytest.mark.skipif(torch_missing, reason="PyTorch not installed in this interpreter")
def test_the_gnn_scores_every_edge_of_a_real_graph(solved):
    import torch
    from ml.gnn import JigsawGNN

    pieces, _, _, _, valid_set, _ = solved
    graph = build_graph(valid_set, pieces)
    model = JigsawGNN(node_dim=NODE_DIM, edge_dim=EDGE_DIM)
    with torch.no_grad():
        out = model(torch.from_numpy(graph["nodes"]),
                    torch.from_numpy(graph["edge_index"]),
                    torch.from_numpy(graph["edge_attr"]))
    assert out["neighbor_logit"].shape == (len(valid_set), 1)
    assert out["orientation_logits"].shape == (len(valid_set), 4)
    assert out["node_embeddings"].shape[0] == graph["node_count"]
    assert torch.isfinite(out["neighbor_logit"]).all()

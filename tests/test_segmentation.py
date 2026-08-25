import numpy as np

from src.segmentation import (binary_close, binary_dilate, binary_erode,
                              binary_open, components_from_labels,
                              connected_components, fill_holes,
                              propagate_labels, separate_touching)


def test_connected_components_labels_and_statistics():
    mask = np.zeros((20, 20), np.uint8)
    mask[2:6, 2:6] = 255      # 4x4 = 16 px at (2, 2)
    mask[12:18, 12:18] = 255  # 6x6 = 36 px at (12, 12)
    labels, components = connected_components(mask)

    assert len(components) == 2
    assert sorted(c.area for c in components) == [16, 36]
    first, second = components
    assert first.bbox == (2, 2, 4, 4)
    assert second.bbox == (12, 12, 6, 6)
    assert first.centroid == (3.5, 3.5)
    assert labels.max() == 2
    # Labels are assigned in raster order of first appearance.
    assert labels[2, 2] == 1 and labels[12, 12] == 2


def test_connected_components_connectivity_rule():
    mask = np.zeros((6, 6), np.uint8)
    mask[1, 1] = mask[2, 2] = 255   # touching only at a corner
    assert len(connected_components(mask, connectivity=8)[1]) == 1
    assert len(connected_components(mask, connectivity=4)[1]) == 2


def test_connected_components_merges_u_shape():
    # A shape whose two arms only join on a later row exercises union-find.
    mask = np.zeros((8, 9), np.uint8)
    mask[1:6, 1:3] = 255
    mask[1:6, 6:8] = 255
    mask[5:6, 1:8] = 255
    labels, components = connected_components(mask)
    assert len(components) == 1
    assert components[0].area == int((mask > 0).sum())


def test_morphology_round_trip():
    mask = np.zeros((21, 21), np.uint8)
    mask[8:13, 8:13] = 255
    assert binary_erode(mask, 3).sum() < mask.sum()
    assert binary_dilate(mask, 3).sum() > mask.sum()

    speckled = mask.copy()
    speckled[1, 1] = 255                      # isolated noise pixel
    assert binary_open(speckled, 3)[1, 1] == 0
    assert binary_open(speckled, 3)[10, 10] == 255

    pierced = mask.copy()
    pierced[10, 10] = 0                       # single-pixel hole
    assert binary_close(pierced, 3)[10, 10] == 255


def test_fill_holes_keeps_indentations_but_fills_enclosures():
    mask = np.zeros((30, 30), np.uint8)
    mask[5:25, 5:25] = 255
    mask[12:18, 12:18] = 0     # enclosed hole -> filled
    mask[5:10, 5:8] = 0        # notch open to the border -> kept
    filled = fill_holes(mask)
    assert filled[15, 15] == 255
    assert filled[6, 6] == 0


def test_propagate_labels_leaves_a_ridge_between_seeds():
    domain = np.zeros((11, 21), bool)
    domain[:, :] = True
    seeds = np.zeros((11, 21), np.int32)
    seeds[5, 2] = 1
    seeds[5, 18] = 2
    grown = propagate_labels(seeds, domain)
    assert grown[5, 3] == 1 and grown[5, 17] == 2
    # Pixels equidistant from both seeds are contested and stay unassigned.
    assert (grown == 0).any()


def test_separate_touching_splits_two_overlapping_blobs():
    mask = np.zeros((60, 110), np.uint8)
    ys, xs = np.mgrid[0:60, 0:110]
    for centre in (35, 75):
        mask[((xs - centre) ** 2 + (ys - 30) ** 2) <= 22 ** 2] = 255
    # The two discs overlap, so plain labelling sees one component.
    assert len(connected_components(mask)[1]) == 1

    labels, components = separate_touching(mask, expected_area=np.pi * 22 ** 2)
    assert len(components) == 2
    assert min(c.area for c in components) > 0.5 * max(c.area for c in components)
    assert components_from_labels(labels)[0].area == components[0].area


def test_separate_touching_leaves_single_pieces_alone():
    mask = np.zeros((40, 40), np.uint8)
    mask[10:30, 10:30] = 255
    _, components = separate_touching(mask)
    assert len(components) == 1
    assert components[0].area == 400

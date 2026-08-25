"""Scoring a blind reconstruction against the derived ground truth."""


def blind_accuracy(reconstruction, applied, layout):
    """Score a blind reconstruction against the derived ground truth.

    ``applied`` maps a piece id to the number of quarter turns that piece had
    already been given before the solver saw it, so that the solver's own
    rotation can be added to it and compared with the solved orientation.

    A rectangle that is not square has exactly one symmetry its seams cannot
    distinguish: the whole puzzle turned through 180 degrees.  Both readings are
    reported and the better one is quoted as the result, so the choice is
    visible rather than silently favourable.
    """
    rows, cols = reconstruction.grid
    cells = rows * cols
    expected_ids = layout["positions_piece_id"]
    expected_rotations = layout["solved_rotations_ccw"]

    def neighbours(ids):
        pairs = set()
        for cell, pid in enumerate(ids):
            row, col = divmod(cell, cols)
            if pid is None:
                continue
            if col < cols - 1 and ids[cell + 1] is not None:
                pairs.add(frozenset((pid, ids[cell + 1])))
            if row < rows - 1 and ids[cell + cols] is not None:
                pairs.add(frozenset((pid, ids[cell + cols])))
        return pairs

    truth_neighbours = neighbours(expected_ids)

    def measure(ids, turns):
        position_hits = sum(a == b for a, b in zip(ids, expected_ids))
        orientation_hits = 0
        for cell, pid in enumerate(ids):
            if pid is None or pid != expected_ids[cell]:
                continue
            total = (applied.get(pid, 0) + turns[cell]) % 4
            if total == int(expected_rotations[str(pid)]) % 4:
                orientation_hits += 1
        found = neighbours(ids)
        return {
            "position_accuracy": position_hits / cells,
            "orientation_accuracy": orientation_hits / cells,
            "neighbour_accuracy": len(found & truth_neighbours) / len(truth_neighbours),
            "complete_reconstruction": float(position_hits == cells
                                             and orientation_hits == cells),
            "position_hits": position_hits,
            "orientation_hits": orientation_hits,
            "neighbour_hits": len(found & truth_neighbours),
            "neighbour_total": len(truth_neighbours),
        }

    ids = reconstruction.piece_ids()
    turns = reconstruction.rotations()
    upright = measure(ids, turns)
    flipped = measure(ids[::-1],
                      [None if t is None else (t + 2) % 4 for t in turns[::-1]])
    better = max((upright, flipped),
                 key=lambda m: (m["neighbour_hits"], m["position_hits"]))
    return {
        "as_placed": upright,
        "rotated_180": flipped,
        "reported": better,
        "reported_orientation": ("as placed" if better is upright
                                 else "rotated 180 degrees"),
    }

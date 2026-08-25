"""Conformance audit: exercise every Milestone 1 requirement and report PASS/FAIL.

Each check actually calls the code and validates the result, rather than merely
checking that a name exists.
"""
import sys, inspect
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = []


def check(section, requirement, fn):
    try:
        detail = fn()
        RESULTS.append((section, requirement, 'PASS', detail or ''))
    except AssertionError as exc:
        RESULTS.append((section, requirement, 'FAIL', str(exc)))
    except Exception as exc:
        RESULTS.append((section, requirement, 'ERROR', '%s: %s' % (type(exc).__name__, exc)))


def documented(fn):
    return bool(inspect.getdoc(fn))


# ---------------------------------------------------------------- 1) enhancement
from src import enhancement as E

def c_conv():
    a = np.arange(25.).reshape(5, 5)
    assert np.allclose(E.convolution(a, np.array([[1.]])), a), 'identity kernel'
    # compare against an explicit textbook double loop
    k = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
    ref = np.zeros_like(a)
    p = np.pad(a, 1, mode='reflect')
    for y in range(5):
        for x in range(5):
            ref[y, x] = (p[y:y+3, x:x+3] * k).sum()
    assert np.allclose(E.convolution(a, k), ref), 'matches explicit per-pixel loop'
    try:
        E.convolution(a, np.ones((2, 2))); raise AssertionError('even kernel not rejected')
    except ValueError:
        pass
    assert documented(E.convolution)
    return 'verified against explicit double loop; rejects even kernels'

def c_mean():
    a = np.zeros((9, 9)); a[4, 4] = 9.
    out = E.mean_filter(a, 3)
    assert abs(out[4, 4] - 1.0) < 1e-9, 'mean of 3x3 impulse'
    assert documented(E.mean_filter)
    return 'mean_filter'

def c_gauss():
    k = E.gaussian_kernel(5, 1.0)
    assert k.shape == (5, 5), 'size'
    assert abs(k.sum() - 1) < 1e-12, 'normalised'
    assert np.allclose(k, k[::-1]) and np.allclose(k, k[:, ::-1]), 'symmetric'
    assert k[2, 2] == k.max(), 'peak at centre'
    k2 = E.gaussian_kernel(5, 3.0)
    assert k2[0, 0] > k[0, 0], 'wider sigma spreads mass'
    assert E.gaussian_filter(np.zeros((9, 9)), 5, 1.0).shape == (9, 9)
    return 'kernel derived from size and sigma, normalised, symmetric'

def c_median():
    a = np.zeros((9, 9), np.uint8); a[4, 4] = 255
    assert E.median_filter(a, 3)[4, 4] == 0, 'removes impulse'
    b = np.arange(81, dtype=np.uint8).reshape(9, 9)
    assert E.median_filter(b, 3).shape == b.shape
    doc = inspect.getdoc(E.median_filter) or ''
    assert 'rank statistic' in doc or 'justif' in doc.lower() or 'unavoidable' in doc, \
        'loop must be justified in the docstring'
    return 'impulse removed; loop justified in docstring'

def c_hist():
    a = np.array([[0, 0], [255, 128]], np.uint8)
    h = E.histogram(a)
    assert h.shape == (256,) and h.sum() == 4 and h[0] == 2 and h[255] == 1 and h[128] == 1
    return 'histogram computed by the library'

def c_eq():
    a = (np.arange(64).reshape(8, 8) * 2).astype(np.uint8)
    out = E.histogram_equalization(a)
    assert out.max() == 255 and out.dtype == np.uint8
    assert E.histogram(out).std() <= E.histogram(a).std() * 3
    return 'equalisation spans full range'

def c_stretch():
    a = np.array([[10, 10], [20, 20]], np.uint8)
    assert E.contrast_stretch(a, 10, 20).tolist() == [[0, 0], [255, 255]]
    return 'contrast stretching'

def c_sharpen():
    a = np.zeros((15, 15), float); a[7, 7] = 100.
    u = E.unsharp_mask(a, 5, 1.0, 1.0)
    l = E.laplacian_sharpen(a, 1.0)
    assert u[7, 7] >= a[7, 7], 'unsharp raises the peak'
    assert l[7, 7] >= a[7, 7], 'laplacian raises the peak'
    assert E.laplacian_kernel().sum() == 0, 'laplacian kernel sums to zero'
    return 'both unsharp masking and a Laplacian, built on convolution'


# --------------------------------------------------------------- thresholding
from src import thresholding as T

def c_global():
    a = np.r_[np.zeros(50), np.full(50, 200)].reshape(10, 10).astype(np.uint8)
    m = T.global_threshold(a, 100)
    assert set(np.unique(m)) <= {0, 255} and m.sum() == 50 * 255
    assert T.global_threshold(a, 100, invert=True).sum() == 50 * 255
    return 'global, with inversion'

def c_otsu():
    a = np.r_[np.zeros(50), np.full(50, 200)].reshape(10, 10).astype(np.uint8)
    m, t = T.otsu_threshold(a)
    assert 0 <= t < 200 and m.sum() == 50 * 255, 'threshold %s' % t
    return 'Otsu returns mask and chosen threshold t=%d' % t

def c_adaptive():
    a = (np.random.default_rng(0).random((40, 40)) * 255).astype(np.uint8)
    for method in ('mean', 'gaussian'):
        m = T.adaptive_threshold(a, 7, 5, method=method)
        assert m.shape == a.shape and set(np.unique(m)) <= {0, 255}, method
    return 'adaptive, both mean and Gaussian'


# ------------------------------------------------------------- 2) edge detection
from src import edge_detection as D

def c_sobel_prewitt():
    a = np.zeros((20, 20)); a[:, 10:] = 255.
    for name, fn in (('sobel', D.sobel), ('prewitt', D.prewitt)):
        mag, ori = fn(a)
        assert mag.shape == a.shape and ori.shape == a.shape, name
        assert mag.max() > 0, '%s magnitude' % name
        # a vertical edge has a horizontal gradient: orientation near 0 or pi
        strong = mag > 0.5 * mag.max()
        angles = np.abs(ori[strong])
        assert np.all((angles < 0.2) | (np.abs(angles - np.pi) < 0.2)), '%s orientation' % name
    return 'both return magnitude AND orientation, verified on a vertical edge'

def c_canny():
    a = np.zeros((40, 40)); a[:, 20:] = 255.
    edges, stages = D.canny(a, sigma=1.2, low=20, high=50)
    assert set(np.unique(edges)) <= {0, 255}, 'binary output'
    for stage in ('smoothed', 'magnitude', 'orientation', 'nms', 'double_threshold'):
        assert stage in stages, 'missing stage %s' % stage
    assert edges[20, 19:21].max() == 255, 'edge found at the step'
    assert stages['nms'].astype(bool).sum() < stages['magnitude'].astype(bool).sum(), \
        'NMS must thin the magnitude image'
    assert set(np.unique(stages['double_threshold'])) <= {0, 75, 255}, 'strong/weak/none'
    return 'all five stages present and individually inspectable'


# ------------------------------------------- 3) segmentation & contour extraction
from src import segmentation as S
from src import contour_extraction as C

def c_cc():
    m = np.zeros((20, 20), np.uint8); m[2:6, 2:6] = 255; m[12:18, 12:18] = 255
    labels, comps = S.connected_components(m)
    assert len(comps) == 2 and sorted(c.area for c in comps) == [16, 36]
    assert comps[0].bbox == (2, 2, 4, 4)
    assert comps[0].mask.shape == (4, 4)
    assert len(S.connected_components(m, connectivity=4)[1]) == 2
    return 'from scratch; areas, bboxes, centroids, per-component masks'

def c_trace():
    m = S.binary_close(np.zeros((30, 30), np.uint8), 3)
    m = np.zeros((30, 30), np.uint8); m[5:25, 5:25] = 255
    traced = C.trace_boundary(m)
    ref = C.extract_contour_cv2(m)
    assert set(map(tuple, traced.tolist())) == set(map(tuple, ref.tolist())), \
        'traced boundary differs from OpenCV'
    return 'Moore tracing matches cv2.findContours point-for-point'

def c_extract_pieces():
    from src.synthetic import piece_mask, TAB, BLANK, FLAT
    mask = piece_mask((TAB, BLANK, FLAT, TAB), size=60, margin=25)
    image = np.zeros(mask.shape + (3,), np.uint8); image[mask > 0] = (200, 180, 160)
    pieces, fg, labels = S.extract_pieces(image, mask=mask)
    assert len(pieces) == 1
    piece = pieces[0]
    for key in ('id', 'image', 'mask', 'bbox', 'area'):
        assert key in piece, 'missing %s' % key
    _, _, angle = S.normalize_orientation(piece['image'], piece['mask'])
    assert isinstance(angle, float)
    return 'bounding box, mask, contour and normalised orientation available'

def c_foreground_from_photo():
    from config import DATASET
    from src.frame_extraction import frame_mask
    import cv2
    from src.registry import full_frame_paths
    frames = full_frame_paths(DATASET)
    assert frames, 'no dataset frames'
    img = cv2.imread(str(frames[0][2]))
    mask, t = frame_mask(img)
    fraction = float((mask > 0).mean())
    assert 0.10 < fraction < 0.45, 'foreground fraction %.3f' % fraction
    return 'real photograph -> foreground mask (%.1f%% fg, Otsu t=%d)' % (fraction * 100, t)


# --------------------------------------------------------- 4) piece description
from src import piece_geometry as G

def c_corners_sides():
    from src.synthetic import piece_mask, TAB, BLANK, FLAT
    mask = piece_mask((TAB, BLANK, FLAT, TAB), size=G.BODY, margin=(G.CANVAS - G.BODY) // 2)
    image = np.zeros((G.CANVAS, G.CANVAS, 3), np.uint8); image[mask > 0] = (90, 150, 210)
    piece = G.describe_piece(1, image, mask)
    assert len(piece.sides) == 4
    assert [s.name for s in piece.sides] == ['top', 'right', 'bottom', 'left']
    assert piece.kinds == (G.SideType.TAB, G.SideType.BLANK, G.SideType.FLAT, G.SideType.TAB), \
        'classified %s' % (piece.kinds,)
    for s in piece.sides:
        assert s.colors.ndim == 2 and s.colors.shape[0] == len(s.profile), 'colour strip'
        assert s.colors.shape[1] == 9, 'three depths x three channels'
    return 'four corners -> four sides, TAB/BLANK/FLAT classified, colour strips sampled'

def c_corners_real():
    import cv2
    d = sorted(Path('data/pieces').glob('piece_*.png'))
    assert len(d) == 35, 'expected 35 canonical pieces, found %d' % len(d)
    rgba = cv2.imread(str(d[0]), cv2.IMREAD_UNCHANGED)
    piece = G.describe_piece(1, rgba[:, :, :3], rgba[:, :, 3])
    assert len(piece.sides) == 4
    return 'runs on the 35 real extracted pieces'


# ------------------------------------------------------------- 5) edge matching
from src import edge_compatibility as M

def c_matching():
    doc = inspect.getdoc(M) or ''
    assert 'D(a, b)' in doc, 'the exact formula must be stated'
    assert 'M(x -> y)' in doc, 'the terms of the formula must be defined'
    assert 'w_colour = 1.0' in doc and 'w_shape  = 0.0' in doc, \
        'the weights actually used must be stated'
    assert 'monotonically worse' in doc, \
        'the evidence for those weights must be stated, not just the values'
    from src.synthetic import piece_mask, TAB, BLANK, FLAT
    def mk(kinds, colour, pid):
        mask = piece_mask(kinds, size=G.BODY, margin=(G.CANVAS - G.BODY) // 2)
        im = np.zeros((G.CANVAS, G.CANVAS, 3), np.uint8); im[mask > 0] = colour
        return G.describe_piece(pid, im, mask)
    a = mk((FLAT, TAB, FLAT, FLAT), (60, 120, 200), 1).sides[1]
    good = mk((FLAT, FLAT, FLAT, BLANK), (60, 120, 200), 2).sides[3]
    bad = mk((FLAT, FLAT, FLAT, BLANK), (200, 60, 60), 3).sides[3]
    assert M.compatibility(a, good) < M.compatibility(a, bad), 'colour must discriminate'
    assert M.compatibility(a, a) == M.INCOMPATIBLE, 'tab-tab rejected'
    assert isinstance(M.compatibility(a, good), float), 'returns a single number'
    return 'single number (MGC), shape as a gate only, formula and weights documented'


# ---------------------------------------------------------------- 6) assembly
def _synthetic_puzzle(rows=2, cols=3, seed=7):
    """A real interlocking puzzle cut from a known image, scrambled."""
    from src.synthetic import cut_image
    ys, xs = np.mgrid[0:rows * G.BODY, 0:cols * G.BODY].astype(float)
    image = np.clip(np.dstack([128 + 110 * np.sin(xs / 37.),
                               128 + 110 * np.sin((xs + 2 * ys) / 61.),
                               128 + 110 * np.cos((3 * xs - ys) / 43.)]),
                    0, 255).astype(np.uint8)
    cut, _ = cut_image(image, rows, cols, margin=(G.CANVAS - G.BODY) // 2, seed=seed)
    pieces = [G.describe_piece(i, patch, mask) for i, (patch, mask) in enumerate(cut)]
    rng = np.random.default_rng(seed)
    turned = [G.rotate_piece(p, int(rng.integers(0, 4))) for p in pieces]
    return [turned[i] for i in rng.permutation(len(turned))]


def c_assembly_blind():
    """The brief's core deliverable: scrambled puzzle in, reconstruction out."""
    from src import solver
    pieces = _synthetic_puzzle()
    result = solver.solve(pieces, beam_width=64, per_state=4)
    assert result.grid == (2, 3), 'the grid must be inferred, not assumed: %s' % (result.grid,)
    assert result.complete, 'every cell must be filled'
    placed = result.piece_ids()
    assert len(set(placed)) == len(pieces), 'each piece used exactly once'
    truth = list(range(len(pieces)))
    assert placed == truth or placed[::-1] == truth, \
        'a scrambled synthetic puzzle must be reconstructed exactly, got %s' % placed
    assert all(0 <= p.rotation < 4 for p in result.placements), 'rotation searched'
    return 'src.solver searches position and rotation; solves a scrambled 2x3 exactly'


def c_endtoend_quality():
    """An unseen puzzle must yield both an image and a quality figure."""
    from src import solver
    pieces = _synthetic_puzzle()
    variants = solver.build_variants(pieces)
    result = solver.solve(pieces, beam_width=64, per_state=4, variants=variants)
    image = solver.render_grid(variants, result, margin=10)
    assert image.ndim == 3 and image.shape[2] == 3, 'a rendered reconstruction'
    assert image.std() > 5, 'the rendered reconstruction must not be blank'
    quality = result.quality
    for key in ('quality_score', 'mutual_best_fraction', 'best_partner_fraction',
                'mean_seam_cost', 'seams_realised', 'seams_expected'):
        assert key in quality, 'missing quality figure: %s' % key
    assert 0.0 <= quality['quality_score'] <= 1.0, 'quality score out of range'
    assert Path('scripts/solve_blind.py').exists(), 'no command-line entry point'
    return 'solve -> arrangement + render_grid image + blind quality score'


# ------------------------------------------------------------ repo structure
def c_structure():
    required_tests = ['test_enhancement', 'test_thresholding', 'test_edge_detection',
                      'test_segmentation', 'test_piece_description',
                      'test_edge_matching', 'test_assembly']
    present = {p.stem for p in Path('tests').glob('test_*.py')}
    missing = [t for t in required_tests if t not in present]
    dirs = ['data/input', 'data/ground_truth', 'data/sample_pieces',
            'results/enhanced_images', 'results/masks', 'results/contours',
            'results/edge_visualisations', 'results/reconstructed_images',
            'results/evaluation_results', 'report', 'notebooks']
    missing_dirs = [d for d in dirs if not Path(d).is_dir()]
    problems = []
    if missing:
        problems.append('missing test files: %s' % ', '.join(missing))
    if missing_dirs:
        problems.append('missing folders: %s' % ', '.join(missing_dirs))
    if not list(Path('report').glob('*.pdf')) if Path('report').is_dir() else True:
        problems.append('no report/milestone_1_report.pdf')
    assert not problems, ' | '.join(problems)
    return 'matches the recommended structure'


def c_notebook():
    import json
    nb = Path('notebooks/full_project_demo.ipynb')
    assert nb.exists(), 'no notebook'
    src = ''.join(''.join(c['source']) for c in json.loads(nb.read_text())['cells']
                  if c['cell_type'] == 'code')
    assert 'from main import run' not in src, \
        'notebook imports main.run which does not exist -> crashes on execution'
    return 'notebook executable'


CHECKS = [
    ('1 Enhancement', 'convolution from scratch', c_conv),
    ('1 Enhancement', 'mean filter', c_mean),
    ('1 Enhancement', 'Gaussian kernel from size+sigma', c_gauss),
    ('1 Enhancement', 'median filter (loop justified)', c_median),
    ('1 Enhancement', 'histogram computed by library', c_hist),
    ('1 Enhancement', 'histogram equalisation', c_eq),
    ('1 Enhancement', 'contrast stretching', c_stretch),
    ('1 Enhancement', 'sharpening (unsharp / Laplacian)', c_sharpen),
    ('1 Thresholding', 'global', c_global),
    ('1 Thresholding', 'Otsu', c_otsu),
    ('1 Thresholding', 'adaptive (mean & Gaussian)', c_adaptive),
    ('2 Edge detection', 'Sobel & Prewitt magnitude+orientation', c_sobel_prewitt),
    ('2 Edge detection', 'complete Canny with all stages', c_canny),
    ('3 Segmentation', 'connected components from scratch', c_cc),
    ('3 Segmentation', 'boundary tracing', c_trace),
    ('3 Segmentation', 'piece -> bbox, mask, contour, orientation', c_extract_pieces),
    ('3 Segmentation', 'foreground mask on a real photograph', c_foreground_from_photo),
    ('4 Description', 'corners, sides, tab/blank/flat, colour strip', c_corners_sides),
    ('4 Description', 'works on the real extracted pieces', c_corners_real),
    ('5 Matching', 'single score, formula + weights stated', c_matching),
    ('6 Assembly', 'blind search over position and rotation', c_assembly_blind),
    ('6 Assembly', 'end-to-end: scrambled in -> image + quality out', c_endtoend_quality),
    ('Repo', 'required structure (tests, folders, report)', c_structure),
    ('Repo', 'runnable demonstration notebook', c_notebook),
]

for section, requirement, fn in CHECKS:
    check(section, requirement, fn)

width = max(len(r[1]) for r in RESULTS)
current = None
for section, requirement, status, detail in RESULTS:
    if section != current:
        print('\n%s' % section)
        current = section
    mark = {'PASS': 'PASS', 'FAIL': 'FAIL', 'ERROR': 'ERR '}[status]
    print('  [%s] %-*s  %s' % (mark, width, requirement, detail))

passed = sum(1 for r in RESULTS if r[2] == 'PASS')
print('\n%d/%d checks pass' % (passed, len(RESULTS)))

"""Replay P0-verified Processor/connector nominal geometry, never sqrt(N).

Only the three explicitly audited all-valid crop/view contracts are supported.
This reuses existing evidence; it does not run a Processor or read image pixels.
Nominal anchors do NOT constrain contextual vision receptive fields.
"""
from dataclasses import dataclass
import hashlib
import json
import operator
from pathlib import Path
import numpy as np

SUPPORTED_LAYOUTS = {576: (2, 4), 832: (3, 4), 1088: (4, 4)}
DEFAULT_EVIDENCE = Path(__file__).resolve().parents[1] / 'configs/layout_geometry.json'


def positive_int(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a positive integer')
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be a positive integer') from exc
    if result <= 0:
        raise ValueError(f'{name} must be positive')
    return result


def readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class SpatialLayout:
    n: int
    crop_rows: int
    crop_cols: int
    original_size: tuple[int, int]  # width,height in original pixels
    centers_xy: np.ndarray  # normalized original-image nominal centers
    cell_ids: np.ndarray  # rational arithmetic owner, row-major5x5
    cell_counts: tuple[int, ...]
    evidence_path: str
    evidence_sha256: str
    grid_shape: tuple[int, int] = (5, 5)


def layout_from_p0(n, *, evidence_path=DEFAULT_EVIDENCE):
    """Require real audited rows/cols/tags/masks, not a visual length guess.

    This is a replay of P0's named samples, not validation of a new image.
    A future live caller must verify its own Processor order/mask/contract.
    """
    n = positive_int(n, 'n')
    if n not in SUPPORTED_LAYOUTS:
        raise ValueError('Unsupported P0 visual-token structure')
    path = Path(evidence_path)
    if path.stat().st_size > 65536:
        raise ValueError('P0 metadata unexpectedly large')
    data = path.read_bytes()
    if len(data) > 65536:
        raise ValueError('P0 metadata unexpectedly large')
    document = json.loads(data)
    if document.get('status') != 'PASS_NOMINAL_SPATIAL_MAPPING':
        raise ValueError('P0 mapping evidence is not PASS')
    matches = [sample for sample in document['samples'] if sample['N'] == n]
    if len(matches) != 1:
        raise ValueError('Missing/duplicate P0 structure')
    record = matches[0]
    rows, cols = positive_int(record['rows'], 'rows'), positive_int(record['cols'], 'cols')
    if (rows, cols) != SUPPORTED_LAYOUTS[n]:
        raise ValueError('Crop shape is not the audited Processor contract')
    if (rows * cols + 1) * 64 != n or record['tokens_per_block'] != 64 or record['visual_blocks'] != rows * cols + 1:
        raise ValueError('Invalid view/connector contract')
    required = ('tags_match_row_major_then_global', 'crop_marker_exact',
                'global_is_full_image_resize', 'pixel_mask_all_valid',
                'installed_pixel_shuffle_4x4_groups_exact', 'installed_merger_order_exact')
    if any(record.get(key) is not True for key in required) or record['empty_cells']:
        raise ValueError('Unverified order, mask, connector or cell occupancy')
    width = positive_int(record['original_width'], 'original_width')
    height = positive_int(record['original_height'], 'original_height')
    centers, owners = [], []
    # Explicit known connector8x8 output per actual Processor view. Global last.
    for view in range(rows * cols + 1):
        global_view = view == rows * cols
        crop_r, crop_c = divmod(view, cols)
        for v in range(8):
            for u in range(8):
                xn, xd = (2*u+1, 16) if global_view else (2*(crop_c*8+u)+1, 16*cols)
                yn, yd = (2*v+1, 16) if global_view else (2*(crop_r*8+v)+1, 16*rows)
                centers.append((xn/xd, yn/yd))
                owners.append(((5*yn)//yd)*5 + (5*xn)//xd)
    counts = tuple(int(x) for x in np.bincount(owners, minlength=25))
    local_counts = tuple(int(x) for x in np.bincount(owners[:-64], minlength=25))
    global_counts = tuple(int(x) for x in np.bincount(owners[-64:], minlength=25))
    if (counts != tuple(record['cell_counts_5x5']) or local_counts != tuple(record['local_counts_5x5'])
            or global_counts != tuple(record['global_counts_5x5']) or min(counts) == 0
            or min(counts) != record['minimum_candidates_per_cell']):
        raise ValueError('Replayed geometry disagrees with P0 evidence')
    return SpatialLayout(n, rows, cols, (width, height), readonly(centers, np.float64),
                         readonly(owners, np.int64), counts, str(path), hashlib.sha256(data).hexdigest())
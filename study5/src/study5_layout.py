"""Study4 Processor validation and transposed crop mapping, copied unchanged."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from coverpruner_attention_probe import prepare_processor_input
from coverpruner_spatial_layout import layout_from_p0, SUPPORTED_LAYOUTS, readonly

def processor_case(processor, path, prompt):
    with Image.open(Path(path)) as source:
        image = source.convert("RGB")
        size = image.size
        inputs, pos = prepare_processor_input(processor, image, prompt)
        image_info = processor.image_processor(images=[image], return_row_col_info=True, return_tensors="pt")
    n = len(pos.visual)
    rows, cols = int(image_info["rows"][0][0]), int(image_info["cols"][0][0])
    if n not in SUPPORTED_LAYOUTS or (rows, cols) not in (SUPPORTED_LAYOUTS[n], SUPPORTED_LAYOUTS[n][::-1]) or inputs["pixel_values"].shape != (1, n//64, 3, 512, 512):
        raise ValueError(f"Unsupported live visual layout: {n}, {rows}x{cols}")
    if not (inputs["attention_mask"] == 1).all() or not inputs["pixel_attention_mask"].bool().all():
        raise ValueError("Evaluation forbids padding")
    if not torch.equal(inputs["pixel_values"], image_info["pixel_values"]):
        raise ValueError("Processor crop/view drift")
    blocks = []
    for token in pos.visual:
        if not blocks or token != blocks[-1][-1] + 1:
            blocks.append([])
        blocks[-1].append(token)
    if len(blocks) != n//64 or any(len(block) != 64 for block in blocks):
        raise ValueError("Unexpected visual blocks")
    previous = 0
    for view, block in enumerate(blocks):
        tag = "<global-img>" if view == len(blocks)-1 else f"<row_{view//cols+1}_col_{view%cols+1}>"
        if tag not in processor.tokenizer.decode(inputs["input_ids"][0, previous:block[0]]):
            raise ValueError("Invalid view ordering")
        previous = block[-1] + 1
    # Reuse the same normalized P0 geometry with the live image's pixel dimensions.
    return inputs, pos, live_layout(n, rows, cols, size)


def live_layout(n, rows, cols, size):
    base = layout_from_p0(n)
    if (rows, cols) == (base.crop_rows, base.crop_cols):
        return replace(base, original_size=size)
    if (rows, cols) != (base.crop_cols, base.crop_rows):
        raise ValueError("Unsupported crop layout")
    # Transpose the audited normalized coordinates and reindex the crop views
    # and each projector's 8x8 rows to keep live native row-major order.
    order = []
    for view in range(rows*cols+1):
        if view == rows*cols:
            prior_view = view
        else:
            r, c = divmod(view, cols)
            prior_view = c*base.crop_cols+r
        order.extend(prior_view*64+u*8+v for v in range(8) for u in range(8))
    centers = base.centers_xy[order][:,::-1]
    prior_cells = base.cell_ids[order]
    cells = (prior_cells % 5)*5 + prior_cells//5
    expected = []
    owners = []
    for view in range(rows*cols+1):
        r, c = divmod(view, cols)
        global_view = view == rows*cols
        for v in range(8):
            for u in range(8):
                xn, xd = (2*u+1,16) if global_view else (2*(c*8+u)+1,16*cols)
                yn, yd = (2*v+1,16) if global_view else (2*(r*8+v)+1,16*rows)
                expected.append((xn/xd,yn/yd))
                owners.append(((5*yn)//yd)*5+(5*xn)//xd)
    counts = tuple(int(x) for x in np.bincount(cells,minlength=25))
    if not np.array_equal(centers,expected) or not np.array_equal(cells,owners) or min(counts)==0:
        raise ValueError("Transposed layout failed the independent rational mapping")
    return replace(base,crop_rows=rows,crop_cols=cols,original_size=size,centers_xy=readonly(centers,np.float64),
                   cell_ids=readonly(cells,np.int64),cell_counts=counts)

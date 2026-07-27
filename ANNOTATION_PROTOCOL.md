# Stomata Counting Pilot Protocol

## Primary output

For each image, record the total number of visible stomata. Do not convert the result to stomata per unit area for this pilot.

For detector training and validation, also draw one tight bounding box around
each countable stomatal complex. Image-level counts alone cannot teach the
model where a stoma is or distinguish a correct count from compensating false
positives and false negatives.

## What to count

- Count the coffee-bean-shaped stomatal complex: the paired guard cells and their central pore together.
- Do not count the pore alone, a single guard cell, or a generic dark outline.
- Count open and closed stomata alike unless the PI specifies otherwise.
- Do not count pavement-cell boundaries, isolated circular artifacts, bubbles, dust, glare, trichomes, or damaged tissue as stomata.

## Edge and quality rules

- Count a stoma touching an image edge only if more than half of the complex is visible.
- Exclude objects that are too blurred or occluded to classify confidently.
- If an object is ambiguous, flag it for review rather than silently including or excluding it.
- Record an image-quality note when focus, glare, debris, or uneven illumination may affect the count.

## Pilot record

Each image should have one row with:

`filename, expert_count, reviewer, ambiguous_objects, quality_flag, notes`

The expert count is the reference label. Automated methods should be evaluated against these labels, including the difficult images rather than only the clearest examples.

## Priority annotation subset

The first 20-image, morphology-diverse queue is recorded in
`annotation_priority_v1.csv`. It deliberately includes pale missed fields,
small-morphology fields, dense fields, debris-heavy fields, and possible true
zeros. Annotate the critical rows first if the full set cannot be completed in
one pass.

Do not assign `dataset_split` until the specimen relationships are known.
Images from the same plant, leaf, peel, or overlapping microscopic field must
remain in the same split. The PI or image provider should fill in
`specimen_id`, `leaf_id`, and `field_id` before the final train/validation/test
split is generated.

After boxes are marked `complete` and specimen IDs are filled, build the
leakage-safe tiled dataset with:

```bash
.venv/bin/python build_annotated_dataset.py
```

The exporter includes only completed annotations, includes negative
background tiles, writes a split manifest, and refuses missing specimen IDs by
default. `--allow-image-groups` exists only for a technical smoke test; it must
not be used for the final biological evaluation.

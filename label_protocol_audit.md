# Label Protocol Audit

## Scope

This audit traces where the damage labels come from and whether they support crack localization, sensor-level indication, or only damage-related transition detection.

## Direct Label Source

The operational label source in this workspace is the `Crack` column in [source/data.xlsx](C:/Users/540563/git/structural-health-monitoring-UMONS/source/data.xlsx).

A direct workbook inspection shows:

- `65` sheets total
- `17` sheets with at least one non-empty `Crack` entry
- the relevant headers are essentially `Air Pressure (bar)`, `Layers (#)`, `Distance (cm)`, `Force (N)`, `Displacement (mm)`, and `Crack`
- no workbook header was found for crack coordinates, crack maps, sensor-to-crack distance, microscope annotations, photo identifiers, or manual localization labels

The only location-like header found in the workbook is `Distance (cm)`, which is the support span or specimen geometry descriptor, not a crack coordinate.

## How The Old Pipeline Used Those Labels

In [merge_fbg.py](C:/Users/540563/git/structural-health-monitoring-UMONS/merge_fbg.py), lines 337-341 forward-fill metadata except for `Crack`, with an explicit comment saying the `Crack` column is preserved only where it is explicitly specified.

In [main_neural_network.py](C:/Users/540563/git/structural-health-monitoring-UMONS/main_neural_network.py), lines 275-276 make `Crack` a required column for model input. Then lines 329-332 fill missing values with zero and set the sequence label to the maximum `Crack` value within the grouped sequence.

In the rebuilt pipeline, [scripts/build_enriched_fbg_dataset.py](C:/Users/540563/git/structural-health-monitoring-UMONS/scripts/build_enriched_fbg_dataset.py) preserves the original workbook label as `label_crack_level` and defines the conservative binary target `label_damage_transition = int(Crack > 0)`.

## Answers To The Required Questions

### 1. Is the `Crack` column manually entered?

Most likely yes. The workspace evidence shows it exists as a metadata column in the Excel workbook rather than being computed automatically from raw FBG traces. No script was found that generates the original workbook `Crack` values.

### 2. Is it tied to force/loading step?

Yes. The `Crack` entries live on workbook rows that also carry force, displacement, air pressure, and specimen metadata. In practice, this is a load-step annotation scheme.

### 3. Is it tied to a repetition?

Not independently. In the old merge path, the workbook metadata row is aligned to repetition-level merged data, so a pressure-step label is inherited by multiple repetitions. That makes it a repeated label, not an independently observed repetition-specific event marker.

### 4. Is it tied to a physical crack location?

No evidence of that was found in this workspace.

### 5. Is it independent of FBG signal behavior?

It is independent in the narrow sense that it is not computed from the FBG signal in code. However, there is no independent supporting artifact here such as photos, microscopy, DIC, acoustic emission, or crack maps. So it is not strong enough to support a localization claim.

### 6. Are there photos, microscope images, post-test observations, or crack maps?

No linked ground-truth artifact was found that can be used as localization evidence. The workspace contains plots and result images, but no crack-coordinate or microscope annotation file tied to the workbook labels.

### 7. Can we assign crack proximity to `FBG1`, `FBG2`, or `FBG3`?

No. There is no independent file in this workspace mapping the observed crack state to a sensor-specific crack proximity label.

### 8. If not, can we only claim damage-state transition detection?

Yes. That is the highest defensible claim from the available evidence.

## Practical Dataset Consequences

The label problem is made worse by data attrition in the legacy path. The rebuilt dataset summary at [output/enriched_dataset_20260608_113447/dataset_summary.md](C:/Users/540563/git/structural-health-monitoring-UMONS/output/enriched_dataset_20260608_113447/dataset_summary.md) shows that several crack-positive files were effectively lost by the old `WL_ch2`-centric merge path.

The rebuilt dataset now contains `64` processed runs and `16` positive runs, but those positives are still load-step damage-transition labels, not localized crack coordinates.

## Defensibility Level

The correct recommendation is:

**Level 3: only damage-related transition detection is defensible.**

Level 1 is not supported because no independent crack localization evidence was found.

Level 2 is also too strong for the current workspace because there is no verified sensor-to-crack assignment that would justify sensor-specific damage indication claims.

## Recommendation For The Manuscript

Use language such as "mechanics-assisted detection of damage-related FBG signal transitions" or "grouped damage-state indication under progressive three-point bending."

Do not use the following phrases unless new ground-truth evidence is added outside this repo and integrated explicitly:

- independent crack localization
- direct crack localization
- sensor-to-crack proximity inference
- raw multiplexed FBG crack localization

The target variable in the revised paper should be presented plainly as a workbook-derived damage-state annotation collapsed into a binary transition label for grouped evaluation.

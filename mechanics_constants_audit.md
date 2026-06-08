# Mechanics Constants Audit

## Scope

This audit checks where mechanics-related constants are defined in the workspace and whether they are consistent enough to support mechanics-assisted features in the rebuilt Sensors revision pipeline.

The relevant new single source of truth is [config/mechanics_constants.yaml](C:/Users/540563/git/structural-health-monitoring-UMONS/config/mechanics_constants.yaml). The rebuilt dataset code reads those constants through [scripts/pipeline_common.py](C:/Users/540563/git/structural-health-monitoring-UMONS/scripts/pipeline_common.py).

## What Was Found

The workspace contains several conflicting or partially hardcoded mechanics assumptions.

| File | Evidence | Constants found | Assessment |
|---|---|---|---|
| [config/mechanics_constants.yaml](C:/Users/540563/git/structural-health-monitoring-UMONS/config/mechanics_constants.yaml) | Central YAML created for rebuilt pipeline | `E=18.6 GPa`, bounds `16.0-22.0`, `pe=0.22`, sensitivity `1.2 pm/microstrain`, width `34 mm`, layer thickness `0.3333333333 mm`, FBG `2` layers below top | Recommended source of truth |
| [scripts/pipeline_common.py](C:/Users/540563/git/structural-health-monitoring-UMONS/scripts/pipeline_common.py) | `parse_geometry_from_run()` at lines 308-337 | Reads width, thickness, `y_fbg`, `E`, `pe`, sensitivity from YAML | Consistent with rebuilt pipeline |
| [scripts/build_enriched_fbg_dataset.py](C:/Users/540563/git/structural-health-monitoring-UMONS/scripts/build_enriched_fbg_dataset.py) | mechanics feature block around lines 317-343 | Uses `M = F L / 4`, Euler-Bernoulli strain, WL2-derived observed strain, residuals | Consistent with rebuilt pipeline |
| [compute_mechanical_strain.py](C:/Users/540563/git/structural-health-monitoring-UMONS/compute_mechanical_strain.py) | lines 72-79 | `b=34 mm`, `h=4 mm`, `y=1.45 mm`, `E=22.0 GPa` | Conflicts with YAML defaults |
| [calculate_E_and_plot.py](C:/Users/540563/git/structural-health-monitoring-UMONS/calculate_E_and_plot.py) | lines 5-19 | `L=230 mm`, `b=34 mm`, `h=4 mm`, `y_fbg=1.333 mm`, sensitivity `1.2 pm/microstrain` | Partly compatible, but hardcoded to one geometry |
| [fbg_force_estimation.py](C:/Users/540563/git/structural-health-monitoring-UMONS/fbg_force_estimation.py) | header lines 17-22, executable lines 42-46 | Header says `E=18.6 GPa`, code sets `E_GPA=8.6`, `pe=0.22`, layer thickness `0.333 mm` | Internally inconsistent |

## Main Inconsistencies

The largest inconsistency is Young's modulus. The rebuilt pipeline uses `18.6 GPa` as the default, [compute_mechanical_strain.py](C:/Users/540563/git/structural-health-monitoring-UMONS/compute_mechanical_strain.py) uses `22.0 GPa`, and [fbg_force_estimation.py](C:/Users/540563/git/structural-health-monitoring-UMONS/fbg_force_estimation.py) documents `18.6 GPa` but executes `8.6 GPa`. That alone is enough to shift expected strain and all residual-based features materially.

The second inconsistency is the FBG distance from the neutral axis. [calculate_E_and_plot.py](C:/Users/540563/git/structural-health-monitoring-UMONS/calculate_E_and_plot.py) derives `y_fbg = 1.333 mm` for a 12-layer, 4 mm specimen with the FBG near the top. [compute_mechanical_strain.py](C:/Users/540563/git/structural-health-monitoring-UMONS/compute_mechanical_strain.py) instead hardcodes `y = 1.45 mm`. The rebuilt pipeline now computes `y_fbg` from the layer count and the assumption that the FBG sits two layers below the top surface.

The third issue is that older scripts often hardcode one specimen thickness or one span, while the actual runs vary by support span and layer count. Those scripts are not suitable as-is for a mixed-run grouped analysis.

## Recommended Single Source of Truth

Use [config/mechanics_constants.yaml](C:/Users/540563/git/structural-health-monitoring-UMONS/config/mechanics_constants.yaml) for all new experiments. The rebuilt code path through [scripts/pipeline_common.py](C:/Users/540563/git/structural-health-monitoring-UMONS/scripts/pipeline_common.py) and [scripts/build_enriched_fbg_dataset.py](C:/Users/540563/git/structural-health-monitoring-UMONS/scripts/build_enriched_fbg_dataset.py) already does that.

Recommended operational values for the conservative Sensors revision are:

| Quantity | Recommended value | Reason |
|---|---|---|
| Effective Young's modulus | `18.6 GPa` | Matches the stated paper-level value and sits inside the plausible audit range |
| Sensitivity bounds for audit | `16.0-22.0 GPa` | Captures the conflicting values present in the workspace |
| Photoelastic coefficient | `0.22` | Consistent across scripts |
| Strain sensitivity | `1.2 pm/microstrain` | Present in older scripts, but should still be described as approximate |
| Beam width | `34 mm` | Consistent across scripts |
| Layer thickness | `0.3333333333 mm` | Derived from the 12-layer, 4 mm specimen assumption already used in the workspace |
| FBG placement | `2` layers below top | Explicitly encoded in the rebuilt pipeline |

## What Is Defensible Now

Mechanics-derived features can be used as approximate covariates in the new pipeline. They should not be presented as a validated physics model. The constants have been reconciled enough for a conservative residual-feature experiment, but not enough for a strong physics-informed claim.

That is consistent with the grouped results already produced in [results/grouped_cv_summary.csv](C:/Users/540563/git/structural-health-monitoring-UMONS/results/grouped_cv_summary.csv). Removing mechanics residuals in [results/ablation_summary.md](C:/Users/540563/git/structural-health-monitoring-UMONS/results/ablation_summary.md) does not collapse performance, which means the current evidence does not support a claim that mechanics features are the dominant source of model skill.

## Recommendation

Proceed with the rebuilt dataset and grouped baselines using the YAML constants. In the manuscript, describe mechanics features as approximate beam-theory covariates and include a limitation paragraph stating that `E`, `y_fbg`, and the strain-wavelength mapping remain partially assumption-driven within this workspace.

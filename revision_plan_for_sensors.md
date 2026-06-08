# Revision Plan For Sensors

## Recommended Direction

The corrected pipeline supports a conservative Sensors paper on **mechanics-assisted detection of damage-related FBG signal transitions in GFRP beams under progressive three-point bending**.

The grouped evaluation does not support returning to the original crack-localization framing. The strongest evidence in [results/grouped_cv_summary.csv](C:/Users/540563/git/structural-health-monitoring-UMONS/results/grouped_cv_summary.csv) is window-level discrimination under grouped validation, but run-level detection remains weak. The best scalar models also perform similarly to each other, which argues against a deep-learning-centered manuscript at this stage.

The current decision is closest to **Case B moving toward Case D** from the requested framework: benchmark-style conservative comparison is still possible, but only if the paper openly states that generalization at run level remains limited.

## Recommended Title

Mechanics-Assisted Detection of Damage-Related FBG Signal Transitions in GFRP Beams Under Progressive Three-Point Bending

## Revised Abstract

This study investigates whether embedded fiber Bragg grating measurements exhibit reproducible damage-related signal transitions during progressive three-point bending of GFRP beams. We rebuilt the original analysis pipeline to preserve run identity, loading order, repetition structure, available multi-channel FBG summaries, and raw segmented interrogator windows while avoiding random splitting across correlated windows from the same run. Because the available labels come from workbook-level `Crack` annotations rather than independently localized crack-coordinate measurements, we frame the task as grouped damage-transition detection rather than crack localization. We generated multiple dataset variants spanning FBG-only features, mechanics-assisted residual features, protocol-aware loading features, and raw-window tensors, then evaluated grouped baselines with run-level metrics as the primary endpoint. The corrected results show moderate window-level discrimination but weak run-level F1, and mechanics-derived covariates act as approximate helpers rather than decisive physics constraints. These findings support a conservative interpretation of embedded FBG sensing as a damage-related transition indicator under the tested protocol, while highlighting the need for more independent damaged runs and external crack-location ground truth before stronger localization claims can be justified.

## Revised Contribution Statements

1. We provide a rebuilt dataset-generation pipeline that preserves run identity and repetition structure from raw interrogator files and workbook metadata.
2. We introduce grouped run-level evaluation to replace leakage-prone random splitting across correlated windows.
3. We compare FBG-only, mechanics-assisted, and protocol-aware feature sets under the same grouped validation design.
4. We show that the present workspace supports damage-transition detection claims only, not independent crack localization.

## Corrected Dataset Description

The rebuilt dataset is stored in [output/enriched_dataset_20260608_113447](C:/Users/540563/git/structural-health-monitoring-UMONS/output/enriched_dataset_20260608_113447). It contains `64` processed runs, `16` positive runs, `7413` repetition rows, and `471` positive repetition rows. Two interrogator files remain unmatched to workbook sheets. The dataset preserves `source_file`, `run_id`, loading-group identity, repetition identity, timestamps, sample boundaries, FBG summary statistics, cross-channel features, approximate mechanics residuals, and raw-window tensors in separate `.npz` artifacts.

The labels should be described as workbook-derived damage-state annotations. The binary target used for the corrected experiments is `label_damage_transition = 1` when workbook `Crack > 0`.

## Corrected Validation Description

All corrected experiments use grouped validation by run. Windows from the same run do not appear in both train and test folds. Run-level metrics are primary, and window-level metrics are reported only as correlated secondary summaries.

This change is necessary because the older pipeline grouped by pressure-defined sequences and then performed non-independent splitting. The revised analysis therefore measures a harder but scientifically defensible generalization target.

## What The Corrected Results Say

The main grouped results in [results/grouped_cv_summary.csv](C:/Users/540563/git/structural-health-monitoring-UMONS/results/grouped_cv_summary.csv) show the following pattern.

On window-level metrics, the best result is the full scalar feature set with logistic regression, with mean window F1 about `0.483` and mean window balanced accuracy about `0.771`. Dataset B and Dataset C are close, which means mechanics features and loading-sequence features each help somewhat, but neither creates a decisive separation.

On run-level metrics, all models remain weak. The best mean run F1 is only about `0.096` for the simple threshold-style rules and about `0.095` for full-feature logistic regression. That is the key scientific result: the corrected grouped setting is much less impressive than the old window-count story.

The ablations in [results/ablation_summary.md](C:/Users/540563/git/structural-health-monitoring-UMONS/results/ablation_summary.md) show shortcut-learning risk. Removing all loading-sequence features does not collapse performance, but neither do mechanics residuals emerge as dominant. `WL2` alone is clearly worse than multi-channel features. Mechanics-only features retain some discrimination, but not enough to justify a strong physics-driven claim.

## Recommended Results Tables

Table 1 should describe the dataset honestly: number of runs, positive runs, repetition rows, positive repetition rows, unmatched files, and missing-channel issues.

Table 2 should report grouped baseline performance for Datasets A-D, with mean and confidence interval for window F1, window balanced accuracy, run F1, run balanced accuracy, and run detection-at-least-once rate.

Table 3 should summarize the ablation study, especially `WL2` only, all channels together, loading-sequence removed, mechanics residuals removed, mechanics only, and summary-statistics only.

Table 4 should report per-run held-out behavior from [results/run_level_results.csv](C:/Users/540563/git/structural-health-monitoring-UMONS/results/run_level_results.csv), including delayed detections and false-positive-heavy runs.

## Recommended Methods Outline

The methods section should be reorganized around the corrected workflow.

Start with specimen and acquisition description. Then describe the raw interrogator files and workbook metadata. Then explain repetition/window extraction and the enriched dataset build. After that, describe feature construction in four layers: local FBG summaries, cross-channel features, loading-sequence covariates, and approximate mechanics residuals. Then describe grouped validation and run-level metrics. Only after that should the paper describe baseline models and the ablation design.

## Claims To Remove

Remove any claim of raw multiplexed FBG crack localization, direct crack localization, independent crack localization, or generalization based on hundreds of independent test samples.

Also remove any claim that the current pipeline is strongly physics-informed in a validated sense. The mechanics features are useful as approximate covariates, but the constant audit still requires caution.

## Claims That Remain Defensible

It remains defensible to claim that embedded FBG measurements contain damage-related transition information under the tested bending protocol. It is also defensible to claim that grouped validation gives a more realistic estimate than earlier random splitting. Finally, it is defensible to claim that multi-channel features outperform a `WL2`-only view and that mechanics-assisted covariates can be compared conservatively against simple baselines.

## Limitations Paragraph

The dataset does not contain independent crack-coordinate ground truth, microscope-confirmed crack maps, or sensor-specific proximity labels, so the task must be interpreted as damage-transition detection rather than crack localization. The number of positive runs remains small, run-level F1 under grouped validation is weak, and some mechanics parameters remain assumption-driven despite reconciliation into a single constants file. These limitations mean the present study should be read as an honest benchmark and reframing exercise, not as a validated general crack-localization system.

## CNN Decision

A grouped CNN run is not justified yet as the next primary experiment. The current grouped results already show that the main bottleneck is not lack of model complexity but limited independent signal at run level. Before adding [scripts/run_grouped_cnn.py](C:/Users/540563/git/structural-health-monitoring-UMONS/scripts/run_grouped_cnn.py), the paper should first stand on the rebuilt dataset, the grouped scalar baselines, and the label/mechanics audits.

If a CNN is explored later, it should be presented only as a secondary appendix experiment on Datasets E and F, with grouped folds identical to the scalar baselines and with learning-curve diagnostics reported openly.

## Submission Recommendation

Submission is only reasonable if the manuscript is rewritten around the conservative framing and if the grouped run-level weakness is stated directly. If the goal is still a crack-localization paper, do not submit yet. That would require additional independent ground truth and probably more positive runs.

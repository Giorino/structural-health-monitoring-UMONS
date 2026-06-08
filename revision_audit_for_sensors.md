# Revision Audit For MDPI Sensors

## 1. Executive summary

This workspace does not currently support an honest claim of independent crack localization from raw multiplexed FBG signals. The strongest evidence in the code points to a much narrower problem: classification of damage-related state transitions from per-repetition summarized FBG measurements plus load metadata.

My recommended claim level is **Level 3, conservative**.

The main reasons are straightforward. First, the crack labels come from the `Crack` column in [source/data.xlsx](source/data.xlsx), with no linked photo log, microscopy file, acoustic-emission record, DIC file, or manual annotation sheet giving physical crack coordinates. Second, the current model does not consume raw multiplexed FBG windows. In [main_neural_network.py](main_neural_network.py), `load_and_preprocess_data()` builds features from merged per-repetition CSV rows and uses only `WL_ch2`, `WL_ch2_std`, `Force (N)`, `Displacement (mm)`, `Air Pressure (bar)`, and an `is_small_sample` flag. Third, `create_sequences_and_labels()` groups measurements by air pressure across files and then randomly splits the resulting sequences, so the current evaluation is not run-independent.

The paper can still be salvaged for Sensors if the manuscript is reframed and the analysis is rebuilt. What is realistic before resubmission is a conservative paper on damage-related FBG transition detection under grouped validation, with explicit limitations and simpler baselines. What is **not** realistic without new experiments is a defensible claim of independent crack localization.

Evidence anchor: [merge_fbg.py](merge_fbg.py) lines 339-341; [main_neural_network.py](main_neural_network.py) lines 314, 317, 367, 384, 796-800; [run_cnn_kfold.py](run_cnn_kfold.py) line 42. Certainty: **high**.

## 2. Files inspected

The repository contains analysis code and generated outputs, but no manuscript source was found here. I found no `.tex`, `.docx`, `.pdf`, `.ipynb`, reviewer-response file, or revision notes file in this workspace.

| Path | What it appears to do | Notes |
| --- | --- | --- |
| [source/data.xlsx](source/data.xlsx) | Master metadata workbook with 65 sheets and columns such as `Air Pressure (bar)`, `Force (N)`, `Displacement (mm)`, and `Crack` | Primary label source |
| [interrogator-data/](interrogator-data) | Raw interrogator text files (`*-interrogator.txt`) | 67 files found |
| [output/20260508_111324/](output/20260508_111324) | Latest merged per-repetition CSV dataset | 56 merged CSVs found |
| [merge_fbg.py](merge_fbg.py) | Detects repetition segments in raw interrogator signals and merges them with Excel metadata | Core data-construction script |
| [batch_merge.py](batch_merge.py) | Batch runner that sweeps segmentation parameters and writes merged CSVs | Explicitly excludes `27cm*` files |
| [main_neural_network.py](main_neural_network.py) | Main ML pipeline, feature construction, sequence creation, random splitting, RF/KNN/CNN training | Primary evaluation script |
| [run_cnn_kfold.py](run_cnn_kfold.py) | 5-fold CNN cross-validation | Uses `StratifiedKFold`, not grouped CV |
| [analyze_dataset_pycaret.py](analyze_dataset_pycaret.py) | Flat tabular PyCaret classification experiment | Another baseline path, also ungrouped |
| [neural_network_results/training_percentage_sweep_20250917_094248.csv](neural_network_results/training_percentage_sweep_20250917_094248.csv) | Historical split sizes and test accuracies | Historical artifact, not fully reproducible from current latest output |
| [neural_network_results/CNN_training_history.png](neural_network_results/CNN_training_history.png) | Existing loss-curve figure | Loss only, no F1/recall/balanced accuracy logs |
| [compute_mechanical_strain.py](compute_mechanical_strain.py) | Euler-Bernoulli mechanical strain estimate on merged CSVs | Uses hardcoded geometry and modulus |
| [calculate_E_and_plot.py](calculate_E_and_plot.py) | Young's modulus back-calculation from merged CSV plus interrogator baseline | Uses `sensitivity = 1.2` pm/microstrain |
| [fbg_force_estimation.py](fbg_force_estimation.py) | End-to-end force estimation from raw interrogator data | Header says `E = 18.6 GPa`, code sets `E_GPA = 8.6` |
| [strain_data/manual_strain_test_data.csv](strain_data/manual_strain_test_data.csv) and [strain_data/manual_strain_test_data_corrected.csv](strain_data/manual_strain_test_data_corrected.csv) | Manual calibration dataset for strain-wavelength relation | Useful for calibration, not crack ground truth |
| [analyze_manual_strain_data.py](analyze_manual_strain_data.py) | Manual strain calibration analysis and plots | Independent calibration support only |
| [exercise_outputs/median_filter_dataset_summary.csv](exercise_outputs/median_filter_dataset_summary.csv) | Summary of signal filtering/segmentation behavior across interrogator files | Useful for data-quality discussion |

Two bookkeeping mismatches matter. There are 65 Excel sheets, 67 interrogator text files, and 56 latest merged CSVs. Several crack-labelled workbook sheets are not represented in the latest merged outputs.

Evidence anchor: repository file search plus [batch_merge.py](batch_merge.py) line 274. Certainty: **high**.

## 3. Labeling protocol findings

The current labels are not independently strong enough to support crack localization.

The direct label source is the `Crack` column in [source/data.xlsx](source/data.xlsx). In [merge_fbg.py](merge_fbg.py), the metadata sheet is read, non-`Crack` columns are forward-filled, and the comment explicitly says the `Crack` column is preserved without forward-fill so it only appears where explicitly specified (`columns_to_ffill`, lines 339-341). In repetition mode, that sheet row is then duplicated across the ten repetitions associated with that pressure row. In [main_neural_network.py](main_neural_network.py), `load_and_preprocess_data()` converts each repetition row into a sample and sets `crack_label = crack_values.max()` for that row.

There is no file in the workspace that records a physical crack location, crack coordinate, sensor-to-crack distance, microscope annotation, photo-based crack onset, or independent crack map. The labels are pressure-step annotations, not physical localization annotations.

Answers to the specific questions are as follows.

1. Independent ground truth was **not found** in this workspace. The only directly linked damage label source is the Excel `Crack` column. Certainty: **high**.

2. The labels are **not derived in code from FBG thresholds**, but they are also not backed here by an independent crack-location file. They appear to be manually entered state labels tied to metadata rows. Certainty: **medium**.

3. A defining manual annotation file for crack start windows was **not found**. The operative label table is [source/data.xlsx](source/data.xlsx). Certainty: **high**.

4. Physical crack locations are **not represented** in the available files. No `x`, `y`, `sensor`, `location`, `distance_to_crack`, or similar field exists in the workbook or merged outputs. Certainty: **high**.

5. The effective labels are pressure-row labels that become repetition-row labels in the merged CSVs. They are not true raw-window labels and not sensor-location labels. Certainty: **high**.

6. The task cannot honestly be called crack localization based on the current evidence. The safest reframing is either **damage-related signal transition detection** or **sensor-level damage-state indication under three-point bending**. Certainty: **high**.

There are 17 crack-labelled sheets in the workbook, but only 11 crack-labelled merged outputs in the latest batch, and only 7 of those 11 have the `WL_ch2` columns needed by the current model. That further weakens the practical positive dataset.

Suggested wording for the revised manuscript:

> We study whether embedded FBG measurements exhibit reproducible damage-related signal transitions during progressive three-point bending. The current labels are load-step damage-state annotations derived from the experimental metadata and should not be interpreted as independently verified crack coordinates.

Evidence anchor: [source/data.xlsx](source/data.xlsx); [merge_fbg.py](merge_fbg.py) lines 339-341; [main_neural_network.py](main_neural_network.py) lines 330-338. Certainty: **high**.

## 4. Dataset and split independence

The current pipeline does not create a train/validation/test split by run, specimen, or file. It creates sequences only after mixing repetition rows from multiple files that share the same air pressure.

The dataset construction path is the following. `batch_merge.py` writes one merged CSV per interrogator file, targeting about 120 rows per file (`TARGET_ROWS_WITH_HEADER = 121`, line 24). `main_neural_network.py` then groups each merged CSV by `(group_index, repetition_index)` and creates one measurement per repetition row. After that, `create_sequences_and_labels()` groups these measurements by `Air Pressure (bar)` alone (`air_pressure = measurement[5]`, line 367), sorts them by displacement (`measurements.sort(...)`, line 384), creates 50-step sliding windows, and only then applies `train_test_split()` randomly at sequence level (lines 796-800).

This means the split is not run-grouped. A single 50-step sequence routinely spans multiple source files. In a reconstruction from the current latest output directory, the number of distinct source files contributing to one sequence ranged from 2 to 6, with a mean of about 4.07 files per sequence.

### Available data counts

| Stage | Count | Comment |
| --- | ---: | --- |
| Excel sheets in [source/data.xlsx](source/data.xlsx) | 65 | Metadata units |
| Crack-labelled Excel sheets | 17 | Any non-null `Crack` cell |
| Raw interrogator files in [interrogator-data/](interrogator-data) | 67 | More raw files than sheets |
| `27cm*` raw files | 10 | Explicitly excluded by batch merge |
| Non-`27cm*` raw files | 57 | Candidate files for latest batch |
| Latest merged CSV files in [output/20260508_111324/](output/20260508_111324) | 56 | One expected file missing |
| Total repetition rows in latest merged outputs | 6374 | Mostly 120 rows per file |
| Files with `WL_ch2` and `WL_ch2_std` usable by current model | 47 | 9 files skipped silently |
| Usable repetition rows for current model | 5511 | After skipping 9 files |
| Usable crack-positive model files | 7 | Only 7 usable positive files |
| Rebuilt 50-step sequences from current latest output | 4717 | Current reproducible count |
| Historical total sequences in [neural_network_results/training_percentage_sweep_20250917_094248.csv](neural_network_results/training_percentage_sweep_20250917_094248.csv) | 4837 | Historical artifact mismatch |

### What the current split actually is

| Split artifact | Train | Validation | Test | Grouped by run/specimen? |
| --- | ---: | ---: | ---: | --- |
| Historical sweep CSV | 3387 | 724 | 726 | No |

There is no split file in the repository that preserves file identity for train, validation, or test. Because the split occurs after cross-file sequence construction, run-level and specimen-level split counts are not recoverable from the saved metrics alone.

### Effective sample size conclusion

The current effective sample size is far closer to the number of merged files, or perhaps the number of physically distinct specimens if those can be established, than to the number of sliding windows. Treating 726 test windows or 600 test windows as independent observations would be statistically misleading.

### Validation feasibility

With the present workspace, the only defensible immediate validation unit is the merged output file, that is, the run-level file in [output/20260508_111324/](output/20260508_111324). Leave-one-run-out or grouped K-fold by `source_file` is feasible **after** rebuilding the dataset so sequences do not mix files. Leave-one-specimen-out is **not yet justified**, because the workspace does not provide an unambiguous specimen identifier separate from run naming.

Evidence anchor: [batch_merge.py](batch_merge.py) lines 24, 274; [main_neural_network.py](main_neural_network.py) lines 367, 384, 796-800; [run_cnn_kfold.py](run_cnn_kfold.py) line 42. Certainty: **high**.

## 5. Feature leakage and shortcut-learning risks

The current pipeline contains several leakage and shortcut-learning risks, and one outright feature-construction error.

The error is that `delta_wl_ch2`, `delta_wl_rate`, and `delta_disp_rate` are computed after grouping by `(group_index, repetition_index)` in [main_neural_network.py](main_neural_network.py) lines 314 and 317. In the merged repetition CSVs, that group is one row wide, so these features collapse to zero. In a reconstruction from the latest output, all three features had exactly one unique value, zero. That means the model is not learning from local signal dynamics the way the manuscript implies.

The shortcut-learning risk is that the remaining nonzero features include `Force (N)`, `Displacement (mm)`, `Air Pressure (bar)`, and `is_small_sample`. Those are powerful proxies for the test protocol and specimen type. Because labels also live on metadata rows indexed by loading stage, the model can exploit the loading schedule rather than damage signatures.

The split leakage is even more serious. Sequences are grouped by air pressure across files and then randomly split. That allows shared protocol structure and near-duplicate metadata contexts to enter both train and test.

There are also data-quality shortcuts. `batch_merge.py` excludes all `27cm*` files by design, and the current model silently skips nine merged files missing `WL_ch2` or `WL_ch2_std`. Several of the skipped files are crack-positive, including `merged_23cm-16layers-5_20260508_1118.csv`.

### Potentially leakage-prone features or design choices

| Item | Why it is risky | Evidence |
| --- | --- | --- |
| `Force (N)` | Encodes loading progression directly | [main_neural_network.py](main_neural_network.py) feature list in `load_and_preprocess_data()` |
| `Displacement (mm)` | Same protocol-stage shortcut | Same function |
| `Air Pressure (bar)` | Used as the grouping key for sequence construction; also a direct progression cue | line 367 |
| `is_small_sample` | Encodes specimen category | line 320 and feature list |
| Cross-file sorting by displacement | Creates pseudo-trajectories from different runs | line 384 |
| Random split after mixed-sequence creation | Train/test contamination at run level | lines 796-800 |
| `delta_wl_ch2`, `delta_wl_rate`, `delta_disp_rate` | Currently degenerate and therefore misleading as claimed temporal descriptors | lines 314, 317 |

### Recommended ablation plan

Run the following after rebuilding the dataset with `source_file` retained as the grouping unit.

1. Original current feature set, but grouped by file and without cross-file sequence mixing.
2. Remove `Air Pressure (bar)`.
3. Remove `Force (N)` and `Displacement (mm)`.
4. Remove `is_small_sample`.
5. Remove all nonlocal or protocol descriptors and use only local FBG measurements.
6. Compare with a truly local single-channel baseline.

Evidence anchor: [main_neural_network.py](main_neural_network.py) lines 314, 317, 367, 384, 796-800; [batch_merge.py](batch_merge.py) line 274. Certainty: **high**.

## 6. Raw-signal claim assessment

The phrase **raw multiplexed FBG signals** is not accurate for the current model.

The raw interrogator text files are in [interrogator-data/](interrogator-data), but the learning pipeline does not train on those raw samples. It first segments repetitions, then takes per-segment median wavelength summaries in [merge_fbg.py](merge_fbg.py), then in [main_neural_network.py](main_neural_network.py) it builds 50-step sequences from repetition-level rows. Only `WL_ch2` is used by the model, not the full multiplexed set, and not the raw time-series within each repetition.

In other words, the actual representation is closer to:

> per-repetition summarized `WL_ch2` trajectories augmented with protocol metadata

than to raw multiplexed FBG windows.

Suggested revised title directions:

1. **Mechanics-assisted detection of damage-related FBG signal transitions in GFRP beams under three-point bending**
2. **Grouped evaluation of damage-state indication from summarized embedded FBG measurements in three-point bending**
3. **Conservative assessment of damage-related FBG state classification under progressive bending loads**

Suggested abstract wording:

> We analyze whether embedded FBG measurements exhibit reproducible damage-related transitions during progressive three-point bending of GFRP specimens. The present study uses per-repetition summarized FBG measurements and experimental metadata rather than independently localized crack-coordinate labels. We therefore frame the task as grouped damage-state indication, not independent crack localization. Under this framing, we compare mechanics-assisted and simpler baselines while emphasizing run-level validation and the limited number of independent damaged specimens.

Evidence anchor: [merge_fbg.py](merge_fbg.py); [main_neural_network.py](main_neural_network.py) feature list and sequence creation. Certainty: **high**.

## 7. Baseline comparison plan and existing baseline status

Some simple baselines already exist in code, but the currently saved results are not suitable as final manuscript evidence because they were evaluated under the same flawed sequence construction and ungrouped split.

### Baselines already present

| Baseline/model | Exists in workspace? | Where | Acceptable as-is for resubmission? |
| --- | --- | --- | --- |
| Random forest | Yes | [main_neural_network.py](main_neural_network.py) | No |
| KNN | Yes | [main_neural_network.py](main_neural_network.py) | No |
| CNN | Yes | [main_neural_network.py](main_neural_network.py) | No |
| PyCaret tabular comparison | Yes | [analyze_dataset_pycaret.py](analyze_dataset_pycaret.py) | No |
| Logistic regression | No explicit implementation found | - | Needs new analysis |
| Gradient boosting | No explicit implementation found | - | Needs new analysis |
| Mechanics threshold rule | No explicit implementation found | - | Needs new analysis |
| CNN using raw single-channel interrogator windows | No | - | Needs new analysis |

Historical test accuracies in [neural_network_results/training_percentage_sweep_20250917_094248.csv](neural_network_results/training_percentage_sweep_20250917_094248.csv) range from about 90.9% to 99.0%, but these numbers are not scientifically persuasive in their current form because the split is not run-independent.

### Realistic baseline plan before resubmission

If time is limited, I would implement only the minimum credible baseline set after rebuilding the dataset:

1. Logistic regression on repetition-level scalar features.
2. Random forest on the same scalar features.
3. A simple threshold rule based on `WL_ch2` shift or mechanics residual.
4. A simple 1D CNN on raw `WL 2` interrogator segments, grouped by file.

Anything more complex than that should only stay if it clearly beats those baselines under grouped validation.

Evidence anchor: [main_neural_network.py](main_neural_network.py); [analyze_dataset_pycaret.py](analyze_dataset_pycaret.py); [neural_network_results/training_percentage_sweep_20250917_094248.csv](neural_network_results/training_percentage_sweep_20250917_094248.csv). Certainty: **high**.

## 8. Grouped validation plan and current status

No valid grouped validation result currently exists in this workspace.

[run_cnn_kfold.py](run_cnn_kfold.py) uses `StratifiedKFold` rather than a grouped splitter, so it does not solve the dependence problem. There is also no artifact that reports run-level confusion counts, first-detection window, or confidence intervals over independent runs.

The realistic grouped validation plan is:

1. Rebuild the dataset with `source_file` preserved as the grouping unit.
2. Avoid cross-file sequence creation entirely.
3. Use leave-one-run-out if the number of crack-positive runs remains very small.
4. If enough positive runs exist after rebuilding, use `GroupKFold` by file.
5. Report window-level metrics only as correlated within-run metrics.
6. Make run-level metrics the primary table.

At the moment, leave-one-specimen-out is not justifiable because the workspace does not encode specimen identity separately from the run/file naming. That is an uncertainty the manuscript should state explicitly.

Evidence anchor: [run_cnn_kfold.py](run_cnn_kfold.py) line 42; [main_neural_network.py](main_neural_network.py) lines 367, 384, 796-800. Certainty: **high**.

## 9. Learning curve results

An existing training-history figure is available at [neural_network_results/CNN_training_history.png](neural_network_results/CNN_training_history.png). Visually, the training and validation losses both decrease and remain fairly close, with some mid-training noise rather than obvious classical divergence.

That said, this plot does **not** answer the reviewer request adequately. It only shows loss. The requested curves for F1, recall, and balanced accuracy are not logged in the current workspace. More importantly, because the validation split is not grouped, even a well-behaved loss curve does not establish run-level generalization.

The right response is therefore:

> Existing loss history suggests optimization was stable, but it is not sufficient evidence against overfitting under independent grouped evaluation. New grouped learning-curve runs are required.

Evidence anchor: [neural_network_results/CNN_training_history.png](neural_network_results/CNN_training_history.png); [main_neural_network.py](main_neural_network.py) `plot_training_history()`. Certainty: **high**.

## 10. Mechanics sensitivity analysis

The mechanics-informed part of the workspace is not internally consistent enough yet to support a strong physics-validation claim.

There are at least three incompatible modulus assumptions in the current scripts:

1. [compute_mechanical_strain.py](compute_mechanical_strain.py) uses `E_pa = 22.0e9` and `y_m = 1.45e-3`.
2. [fbg_force_estimation.py](fbg_force_estimation.py) says in the header that the paper uses `E = 18.6 GPa`, but the executable constant is `E_GPA = 8.6`.
3. [calculate_E_and_plot.py](calculate_E_and_plot.py) uses `sensitivity = 1.2` pm/microstrain and `y_fbg = 1.333 mm`.

There is also independent calibration evidence in [strain_data/manual_strain_test_data_corrected.csv](strain_data/manual_strain_test_data_corrected.csv), but it is not integrated into the model code. A quick linear fit on that manual calibration file gives wavelength sensitivities near `0.09 pm/microstrain` for the three channels, which is far from the hardcoded `1.2 pm/microstrain` used in [calculate_E_and_plot.py](calculate_E_and_plot.py). That discrepancy may reflect a unit issue, but until it is resolved, the physics pipeline should be described as approximate rather than validated.

The manuscript should therefore avoid claiming a validated mechanics model. A safer statement is:

> Mechanics-based descriptors were used as approximate physically motivated covariates under simplified Euler-Bernoulli assumptions. Their quantitative sensitivity to modulus, FBG depth, and calibration uncertainty remains to be systematically assessed.

A realistic sensitivity plan before resubmission is:

1. Sweep `E` over at least 18.6, 20, and 22 GPa.
2. Sweep `y_fbg` over a plausible layer-placement range.
3. Reconcile the strain-sensitivity units using the manual calibration dataset.
4. Report whether rankings or classifier outputs change materially under those perturbations.

Evidence anchor: [compute_mechanical_strain.py](compute_mechanical_strain.py) lines 76-77; [fbg_force_estimation.py](fbg_force_estimation.py) lines 18 and 42; [calculate_E_and_plot.py](calculate_E_and_plot.py) lines 14-19; [analyze_manual_strain_data.py](analyze_manual_strain_data.py) lines 237-241. Certainty: **high**.

## 11. Recommended reframing for MDPI Sensors

The recommended claim level is **Level 3, conservative**.

### New title options

1. Mechanics-assisted detection of damage-related FBG signal transitions in GFRP beams under three-point bending
2. Grouped evaluation of damage-state indication from embedded FBG measurements in progressive bending tests
3. Damage-state classification from summarized embedded FBG responses under three-point bending: a conservative grouped-validation study

### Abstract skeleton

Sentence 1: State the sensing setting and objective without saying localization.

Sentence 2: State clearly that the model uses summarized FBG measurements and experimental metadata from progressive three-point bending tests.

Sentence 3: State that labels are damage-state annotations tied to experimental load steps and that grouped run-level validation is used.

Sentence 4: State the contribution as a conservative benchmark of whether mechanics-assisted descriptors help damage-state indication relative to simpler baselines.

### Revised contribution statements

1. A transparent preprocessing pipeline from interrogator repetitions to analysis-ready FBG summaries.
2. A grouped validation framework that treats runs, not windows, as the relevant independent unit.
3. A comparison of simple baselines and mechanics-assisted variants under conservative reporting.
4. An explicit discussion of label, calibration, and physics-model limitations.

### Limitations paragraph

> The present dataset does not provide independently verified crack-coordinate ground truth, and the damage labels should be interpreted as load-step damage-state annotations rather than exact physical crack locations. The number of independent damaged runs is limited, some merged files are unusable because the central wavelength channel is missing, and several mechanics-related parameters remain approximate. Accordingly, the results should be interpreted as evidence for damage-related FBG signal discrimination under the tested protocol, not as a validated general crack-localization system.

### Reviewer-response strategy

| Reviewer criticism | What the workspace supports now | What to do |
| --- | --- | --- |
| Weak dataset quality | Supported | Report exclusions, missing channels, limited positive runs, and effective sample size honestly |
| Unclear crack ground truth | Supported | Admit label limitation and reframe task |
| Overclaiming localization | Supported | Remove localization claim |
| Lack of independent validation | Not supported | Needs new experiments or external validation data |
| Leakage from run-level descriptors | Supported | Rebuild grouped dataset and re-run |
| Modest improvement over baseline | Partially supported | Add grouped simple baselines |
| Insufficient comparison with simpler methods | Supported | Implement logistic regression, RF, threshold rule, simple raw-signal CNN |

Evidence anchor: all prior sections. Certainty: **high**.

## 12. Exact manuscript edits needed

1. Replace every use of **crack localization** with **damage-state indication**, **damage-related transition detection**, or similar conservative wording unless an independent location label source is added.

2. Remove **raw multiplexed FBG signals** from the title and abstract. The current model uses summarized repetition-level data and only the central FBG channel in the learning pipeline.

3. Add a dataset paragraph that states the actual available independent units, the number of crack-positive usable runs, the skipped files, and the absence of independent crack-coordinate labels.

4. Add a methods paragraph that explicitly says the previous random split was replaced by grouped run-level validation, if that re-analysis is completed.

5. Add a limitations paragraph on label provenance, specimen independence uncertainty, and mechanics-parameter uncertainty.

6. Replace any confidence interval or statistical interpretation that treats windows as independent samples.

7. Add a baseline table centered on simpler models.

8. Add a mechanics section that presents those descriptors as approximate guidance, not validated constitutive truth.

Evidence anchor: [main_neural_network.py](main_neural_network.py), [merge_fbg.py](merge_fbg.py), [compute_mechanical_strain.py](compute_mechanical_strain.py), [fbg_force_estimation.py](fbg_force_estimation.py). Certainty: **high**.

## 13. Remaining risks before submission

The largest remaining risk is that even after reframing, the number of independent damaged runs may still be too small for a persuasive Sensors paper. Right now only 7 usable crack-positive files survive the current `WL_ch2` requirement in the latest merged outputs.

The second risk is reproducibility drift. The historical training sweep reports 4837 sequences, while rebuilding from the current latest output gives 4717 sequences. That suggests the saved results are not tightly tied to the present data snapshot.

The third risk is that the mechanics descriptors may not survive scrutiny until the modulus, geometry, and strain-calibration assumptions are reconciled across scripts.

Evidence anchor: [neural_network_results/training_percentage_sweep_20250917_094248.csv](neural_network_results/training_percentage_sweep_20250917_094248.csv), [compute_mechanical_strain.py](compute_mechanical_strain.py), [fbg_force_estimation.py](fbg_force_estimation.py), [calculate_E_and_plot.py](calculate_E_and_plot.py). Certainty: **high**.

## 14. Priority checklist

1. Reframe the paper to Level 3 immediately.
2. Rebuild the dataset so sequences never mix files and retain `source_file` as the grouping key.
3. Re-run baselines under grouped run-level validation.
4. Report run-level metrics first and window-level metrics second.
5. Remove raw-signal and localization overclaims.
6. Reconcile mechanics constants and add a compact sensitivity analysis.
7. If grouped results remain weak, do not oversell; the honest fallback is a methods-oriented negative or conservative paper.
8. If independent crack-location evidence exists outside this repo, integrate it explicitly before using the word localization again.

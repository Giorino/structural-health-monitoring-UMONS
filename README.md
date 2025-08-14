### Run environment and script

This project uses a local virtual environment at `.venv` (arm64) to avoid architecture issues.

### 1) Activate the environment

```bash
source .venv/bin/activate
```

If you prefer not to activate it, call its Python directly:

```bash
.venv/bin/python -V
```

### 2) Recreate the environment (only if missing)

```bash
/usr/bin/arch -arm64 /opt/homebrew/bin/python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install numpy pandas scipy openpyxl
```

### 3) Run the script

With the environment activated:

```bash
python merge_fbg.py \
  --txt interrogator-data/20250808T093948Z.txt \
  --xlsx source/data.xlsx \
  --out output/merged.csv
```

Without activating the environment:

```bash
.venv/bin/python merge_fbg.py \
  --txt interrogator-data/20250808T093948Z.txt \
  --xlsx source/data.xlsx \
  --out output/merged.csv
```

### 4) Run the full pipeline (batch merge → plots → video)

Run the orchestrator that executes `batch_merge.py`, `plot_time_series.py`, and `multi_force_displacement_video.py` in sequence:

With the environment activated:

```bash
python main.py
```

Without activating the environment (exact command used):

```bash
.venv/bin/python main.py
```

### Per-repetition output (10 rows per air pressure)

One row per repetition (includes timestamp and first 3 wavelength channels):

```bash
.venv/bin/python merge_fbg.py \
  --txt interrogator-data/20250808T093948Z.txt \
  --xlsx source/data.xlsx \
  --sheet "27cm-12layers-2" \
  --out output/merged.csv \
  --output_mode repetition \
  --repeats_per_pressure 10 \
  --channels_to_use 3 \
  --timestamp_point mid \
  --include_std
```

### Choosing the Excel sheet

- By index (0-based): `--sheet 0` (first), `--sheet 1` (second), `--sheet 2` (third)
- By name: `--sheet "Sheet2"`

Example:

```bash
.venv/bin/python \
  merge_fbg.py \
  --txt interrogator-data/20250808T093948Z.txt \
  --xlsx source/data.xlsx \
  --sheet "27cm-12layers-2" \
  --out output/merged.csv

Note: The program will automatically append the sheet name and a date-time suffix to the output file name. For example, `--out .../merged.csv --sheet "27cm-12layers-2"` becomes `merged_27cm-12layers-2_YYYYMMDD_HHMM.csv` in the same directory.
```

### Key flags

- `--output_mode`: `group` (aggregate per pressure) or `repetition` (one row per repetition)
- `--repeats_per_pressure`: expected repetitions per pressure (default 10)
- `--channels_to_use`: number of WL channels to export (default 3)
- `--timestamp_point`: `start` | `mid` | `end` (default `mid`)
- `--include_std`: include per-channel standard deviation columns

### Output columns (repetition mode)

- `group_index`: 0-based index of the pressure row in the Excel sheet
- `repetition_index`: 0..9 within that pressure
- `segment_start_idx`, `segment_end_idx`, `segment_n_samples`: indices and sample count in the TXT data for the detected repetition
- `timestamp`: time corresponding to the chosen `--timestamp_point`
- `WL_ch1`, `WL_ch2`, `WL_ch3`: mean wavelength values for the first three channels (with optional `_std`)

View CLI help:

```bash
python merge_fbg.py --help
```

### Troubleshooting

- **NumPy import/architecture error**: Ensure you're using the `.venv` interpreter (arm64).
  - Check: ``.venv/bin/python -c "import platform; print(platform.machine())"`` should print `arm64`.
  - If not using the venv, activate it or run via `.venv/bin/python` as shown above.


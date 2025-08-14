#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
import re
from datetime import datetime

# Config (paths resolved relative to this file so the folder can be renamed freely)
PROJECT_ROOT = Path(__file__).resolve().parent

# Prefer the current Python interpreter; can be overridden by callers (e.g., main.py)
VENV_PY = Path(sys.executable)

MERGE_SCRIPT = PROJECT_ROOT / "merge_fbg.py"
INTERROGATOR_DIR = PROJECT_ROOT / "interrogator-data"
XLSX_PATH = PROJECT_ROOT / "source" / "data.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "output"
# Default threshold search parameters
DEFAULT_THRESH_START = 0.20
DEFAULT_THRESH_END = 0.35
DEFAULT_THRESH_STEP = 0.001
DEFAULT_SMOOTH_KERNEL = 7
TARGET_ROWS_WITH_HEADER = 121  # 120 data rows + header


def detect_sheet_name_from_filename(filename: str) -> str:
    # Map like '27cm-12layers-3-interrogator.txt' -> '27cm-12layers-3'
    m = re.match(r"^(.*?)-interrogator\.txt$", filename)
    if m:
        return m.group(1)
    # fallback: strip extension
    return Path(filename).stem


def count_lines(file_path: Path) -> int:
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return -1


def try_run(txt_path: Path, sheet: str, out_path: Path, diff_threshold: float, smooth_kernel: int) -> Path:
    cmd = [
        str(VENV_PY), str(MERGE_SCRIPT),
        "--txt", str(txt_path),
        "--xlsx", str(XLSX_PATH),
        "--sheet", sheet,
        "--out", str(out_path),
        "--diff_threshold", f"{diff_threshold}",
        "--smooth_kernel", str(smooth_kernel),
        "--output_mode", "repetition",
        "--include_std",
        "--rep_signal_mode", "maxvar",
    ]
    result = subprocess.run(
        cmd,
        check=True,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Parse the final output path from stdout
    m = re.search(r"Merged dataset saved to:\s*(.+)", result.stdout)
    if m:
        p = Path(m.group(1).strip())
        if p.exists():
            return p
    # Fallback to pattern search if parsing fails
    parent = out_path.parent
    stem = out_path.stem
    candidates = sorted(parent.glob(f"{stem}_*_{datetime.now().strftime('%Y%m%d')}*.csv"))
    return candidates[-1] if candidates else out_path


def parse_distance_cm(sheet: str) -> int | None:
    m = re.match(r"^(\d+)cm\b", sheet)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def choose_search_params(sheet: str):
    # Heuristic: shorter distances have smaller amplitude transitions → need lower thresholds and lighter smoothing
    dist = parse_distance_cm(sheet)
    if dist is not None and dist <= 15:
        return {
            "start": 0.05,
            "end": 0.5,
            "step": 0.001,
            "smooth_kernels": [5, 3, 7],
        }
    else:
        return {
            "start": DEFAULT_THRESH_START,
            "end": DEFAULT_THRESH_END,
            "step": DEFAULT_THRESH_STEP,
            "smooth_kernels": [DEFAULT_SMOOTH_KERNEL],
        }


def search_threshold_for_file(txt: Path, sheet: str, out_base: Path) -> Path | None:
    params = choose_search_params(sheet)
    start = params["start"]
    end = params["end"]
    step = params["step"]
    for sk in params["smooth_kernels"]:
        print(f"  Trying smooth_kernel={sk}")
        for i in range(int((end - start) / step) + 1):
            thr = start + i * step
            out_csv = out_base
            try:
                produced = try_run(txt, sheet, out_csv, thr, sk)
            except subprocess.CalledProcessError as e:
                err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
                print(f"  Threshold {thr:.3f} failed: {e}\n{err}")
                continue
            n_lines = count_lines(produced)
            print(f"  Threshold {thr:.3f} -> {n_lines} lines")
            if n_lines == TARGET_ROWS_WITH_HEADER:
                print(f"  Found optimal threshold {thr:.3f} yielding {n_lines} lines")
                return produced
    return None


def main():
    # Create timestamped subfolder for this batch run
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_output_dir = OUTPUT_DIR / batch_timestamp
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Batch output directory: {batch_output_dir}")

    files = sorted(INTERROGATOR_DIR.glob("*-interrogator.txt"))
    if not files:
        print(f"No interrogator files found in {INTERROGATOR_DIR}")
        sys.exit(1)

    for txt in files:
        sheet = detect_sheet_name_from_filename(txt.name)
        print(f"Processing: {txt.name} (sheet={sheet})")

        # Sweep thresholds with per-sheet parameters
        out_csv = batch_output_dir / "merged.csv"
        best_csv = search_threshold_for_file(txt, sheet, out_csv)
        if best_csv is None:
            print(f"  Did not reach {TARGET_ROWS_WITH_HEADER} lines for {txt.name}; leaving last produced file.")
        else:
            print(f"  Saved: {best_csv}")


if __name__ == "__main__":
    main()


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
DEFAULT_PEAK_HEIGHT_START = 0.05
DEFAULT_PEAK_HEIGHT_END = 0.8
DEFAULT_PEAK_HEIGHT_STEP = 0.01
DEFAULT_PEAK_PROMINENCE = 0.1
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


def try_run(
    txt_path: Path,
    sheet: str,
    out_path: Path,
    peak_height: float,
    peak_prominence: float,
    smooth_kernel: int,
) -> Path:
    cmd = [
        str(VENV_PY), str(MERGE_SCRIPT),
        "--txt", str(txt_path),
        "--xlsx", str(XLSX_PATH),
        "--sheet", sheet,
        "--out", str(out_path),
        "--peak_height", f"{peak_height}",
        "--peak_prominence", f"{peak_prominence}",
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
    if dist is not None and dist >= 23:
        # Larger distances might have more pronounced peaks but also more noise. Use a wider search.
        return {
            "h_start": 0.1,
            "h_end": 2.0,
            "h_step": 0.02,
            "p_start": 0.1,
            "p_end": 1.0,
            "p_step": 0.02,
            "smooth_kernels": [9, 7],
        }
    else:
        # Default for shorter distances
        return {
            "h_start": 0.05,
            "h_end": 0.8,
            "h_step": 0.01,
            "p_start": 0.05,
            "p_end": 0.4,
            "p_step": 0.02,
            "smooth_kernels": [7, 5],
        }


def search_threshold_for_file(txt: Path, sheet: str, out_base: Path) -> Path | None:
    params = choose_search_params(sheet)
    h_start, h_end, h_step = params["h_start"], params["h_end"], params["h_step"]
    p_start, p_end, p_step = params["p_start"], params["p_end"], params["p_step"]

    best_result = {"path": None, "lines": -1, "params": {}}
    min_dist = float("inf")

    for sk in params["smooth_kernels"]:
        print(f"  Trying smooth_kernel={sk}")
        # Iterate over a grid of peak height and prominence values
        for i in range(int((h_end - h_start) / h_step) + 1):
            h = h_start + i * h_step
            for j in range(int((p_end - p_start) / p_step) + 1):
                p = p_start + j * p_step
                out_csv = out_base
                try:
                    produced = try_run(txt, sheet, out_csv, h, p, sk)
                except subprocess.CalledProcessError:
                    # This combination is likely invalid for the signal, so we can skip it quietly
                    continue

                n_lines = count_lines(produced)
                print(f"  h={h:.3f}, p={p:.3f} -> {n_lines} lines")

                if n_lines == TARGET_ROWS_WITH_HEADER:
                    print(f"  Found optimal params: h={h:.3f}, p={p:.3f} yielding {n_lines} lines")
                    return produced
                
                # If not exact, find the closest within a tolerance
                dist = abs(n_lines - TARGET_ROWS_WITH_HEADER)
                # Prioritize results with more lines if distance is the same
                if dist < min_dist or (dist == min_dist and n_lines > best_result["lines"]):
                    min_dist = dist
                    best_result = {"path": produced, "lines": n_lines, "params": {"h": h, "p": p, "sk": sk}}

    # Accept the best result if it's within a reasonable tolerance (e.g., 15 rows)
    if best_result["path"] and min_dist <= 15:
        p = best_result['params']
        print(f"  No exact match. Using closest result: {best_result['lines']} lines (params: h={p['h']:.3f}, p={p['p']:.3f}, sk={p['sk']})")
        return best_result["path"]

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


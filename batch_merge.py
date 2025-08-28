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

# Exception files (cracked composites) - accept first working result
EXCEPTION_FILES = [
    "15cm-12layers-9",  # First exception: cracked composite
    # Add more exceptions here as needed, e.g.:
    # "19cm-12layers-2",
    # "23cm-12layers-4",
]


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
        "--absolute_wavelength",  # Preserve absolute wavelength values for proper FBG strain calculation
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
    # Optimized parameter ranges to reduce search time while maintaining functionality
    dist = parse_distance_cm(sheet)
    if dist is not None and dist >= 23:
        # Larger distances: reduced combinations from ~8800 to ~200
        return {
            "h_values": [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0],  # 10 values
            "p_values": [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0],  # 7 values  
            "smooth_kernels": [7, 9],  # Try 7 first as it's generally better
        }
    else:
        # Default for shorter distances: reduced combinations from ~2700 to ~140
        return {
            "h_values": [0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8],  # 11 values
            "p_values": [0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4],  # 7 values
            "smooth_kernels": [7, 5],  # Try 7 first as it's generally better
        }


def search_threshold_for_file(txt: Path, sheet: str, out_base: Path) -> Path | None:
    # Check if this is an exception file (cracked composite)
    is_exception = sheet in EXCEPTION_FILES
    if is_exception:
        print(f"  Exception file detected: {sheet} - using first working result")
        return search_threshold_for_exception_file(txt, sheet, out_base)
    
    params = choose_search_params(sheet)
    h_values = params["h_values"]
    p_values = params["p_values"]

    best_result = {"path": None, "lines": -1, "params": {}}
    min_dist = float("inf")
    temp_files = []  # Track temporary files for cleanup

    for sk in params["smooth_kernels"]:
        print(f"  Trying smooth_kernel={sk}")
        # Try parameter combinations, starting with more promising values first
        for h in h_values:
            for p in p_values:
                out_csv = out_base
                try:
                    produced = try_run(txt, sheet, out_csv, h, p, sk)
                    temp_files.append(produced)  # Track for potential cleanup
                except subprocess.CalledProcessError:
                    # This combination is likely invalid for the signal, so we can skip it quietly
                    continue

                n_lines = count_lines(produced)
                print(f"  h={h:.3f}, p={p:.3f} -> {n_lines} lines")

                if n_lines == TARGET_ROWS_WITH_HEADER:
                    print(f"  Found optimal params: h={h:.3f}, p={p:.3f} yielding {n_lines} lines")
                    # Clean up other temporary files but keep the successful one
                    _cleanup_temp_files([f for f in temp_files if f != produced])
                    return produced
                
                # If not exact, find the closest within a tolerance
                dist = abs(n_lines - TARGET_ROWS_WITH_HEADER)
                # Prioritize results with more lines if distance is the same
                if dist < min_dist or (dist == min_dist and n_lines > best_result["lines"]):
                    min_dist = dist
                    best_result = {"path": produced, "lines": n_lines, "params": {"h": h, "p": p, "sk": sk}}

                # Early exit if we find something very close (within 2 rows)
                if dist <= 2:
                    print(f"  Found very close match: {n_lines} lines (params: h={h:.3f}, p={p:.3f}, sk={sk})")
                    # Clean up other temporary files but keep the successful one
                    _cleanup_temp_files([f for f in temp_files if f != produced])
                    return produced

    # Clean up all temporary files except the best result
    if best_result["path"]:
        _cleanup_temp_files([f for f in temp_files if f != best_result["path"]])
    else:
        _cleanup_temp_files(temp_files)

    # Accept the best result if it's within a reasonable tolerance (e.g., 15 rows)
    if best_result["path"] and min_dist <= 15:
        p = best_result['params']
        print(f"  No exact match. Using closest result: {best_result['lines']} lines (params: h={p['h']:.3f}, p={p['p']:.3f}, sk={p['sk']})")
        return best_result["path"]

    return None


def search_threshold_for_exception_file(txt: Path, sheet: str, out_base: Path) -> Path | None:
    """Simplified search for exception files (cracked composites) - accept first working result."""
    params = choose_search_params(sheet)
    h_values = params["h_values"][:3]  # Only try first 3 height values
    p_values = params["p_values"][:3]  # Only try first 3 prominence values
    
    # Try only the first (most promising) smooth kernel
    sk = params["smooth_kernels"][0]
    print(f"  Using simplified search with smooth_kernel={sk}")
    
    for h in h_values:
        for p in p_values:
            out_csv = out_base
            try:
                produced = try_run(txt, sheet, out_csv, h, p, sk)
                n_lines = count_lines(produced)
                print(f"  h={h:.3f}, p={p:.3f} -> {n_lines} lines")
                
                # Accept any reasonable result (more than 20 lines)
                if n_lines > 20:
                    print(f"  Accepting first working result: {n_lines} lines (params: h={h:.3f}, p={p:.3f}, sk={sk})")
                    return produced
                    
            except subprocess.CalledProcessError:
                # This combination failed, try next
                continue
    
    print(f"  Warning: No working parameters found for exception file {sheet}")
    return None


def _cleanup_temp_files(temp_files):
    """Clean up temporary files to reduce I/O overhead."""
    for temp_file in temp_files:
        try:
            if temp_file and temp_file.exists():
                temp_file.unlink()
        except Exception:
            # Ignore cleanup errors
            pass


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


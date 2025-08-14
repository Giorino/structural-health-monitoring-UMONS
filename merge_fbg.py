# merge_fbg.py
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import medfilt
import argparse
from datetime import datetime

def find_wl_columns(df):
    # WL column patterns: 'WL 1[nm]', 'WL1[nm]', 'WL 1[nm]' or 'WL 1[nm]' variants or 'WL 1[nm]'
    candidates = [c for c in df.columns if re.search(r'\bWL\b|\bWL\s*\d+|\bWavelength', c, re.I)]
    # fallback: numeric-like columns near start
    if not candidates:
        candidates = [c for c in df.columns if df[c].dtype.kind in 'fif']
    return candidates

def select_wl_channels(df, wl_cols, channels_to_use=3):
    if not wl_cols:
        return []
    # Rank by number of non-null values (desc), then keep order stable
    non_null_counts = {c: int(df[c].notna().sum()) for c in wl_cols}
    ranked = sorted(wl_cols, key=lambda c: (-non_null_counts[c], wl_cols.index(c)))
    return ranked[:channels_to_use]

def find_time_column(df):
    # Try common time column names
    candidate_names = [
        c for c in df.columns
        if re.search(r"time|timestamp|date", str(c), flags=re.I)
    ]
    for c in candidate_names:
        # Accept numeric or datetime
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None

def load_interrogator(txt_path):
    # Robust loader that handles wrapped headers and a variable number of WL columns
    path = str(txt_path)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.read().splitlines()

    # find first data line (starts with ISO timestamp or numeric time)
    import re as _re
    data_start = 0
    iso_re = _re.compile(r"^\d{4}-\d{2}-\d{2}T")
    for i, line in enumerate(lines[:50]):  # scan first 50 lines
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if iso_re.match(stripped):
            # Expect at least 3 tokens: timestamp, time(s), wl1
            if len(parts) >= 3:
                data_start = i
                break
        else:
            # alternate: line begins with a number (time) followed by wl
            try:
                float(parts[0])
                if len(parts) >= 2:
                    data_start = i
                    break
            except Exception:
                pass

    # Determine number of columns from first data row
    first_data_tokens = lines[data_start].split()
    n_tokens = len(first_data_tokens)
    # Build names: assume first token is Timestamp (string) if it looks like ISO, otherwise Time_s
    if iso_re.match(lines[data_start]):
        names = ["Timestamp", "Time_s"]
        wl_count = max(0, n_tokens - 2)
    else:
        names = ["Time_s"]
        wl_count = max(0, n_tokens - 1)
    names += [f"WL {i}[nm]" for i in range(1, wl_count + 1)]

    # Read data from data_start with whitespace delimiter, no header
    df = pd.read_csv(
        path,
        sep=r'\s+',
        header=None,
        names=names,
        skiprows=data_start,
        engine='python'
    )
    # Try to coerce Timestamp to datetime if present
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    return df

def smooth_signal(signal, kernel_size=7):
    # median filter (odd kernel)
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return medfilt(signal, kernel_size=k)

def detect_segments(signal, diff_threshold=None, min_segment_length=5):
    # signal: 1D array (smoothed mean of chosen WL columns)
    grad = np.abs(np.gradient(signal))
    # threshold: if not provided use multiple of median noise
    if diff_threshold is None:
        diff_threshold = max(1e-6, np.median(grad) + 3 * np.std(grad))
    is_jump = grad > diff_threshold
    # turn points into boundaries (where jump is True). We'll group contiguous "not jump" areas as segments.
    boundaries = np.where(is_jump)[0]
    # build segments between boundaries
    segments = []
    start = 0
    for b in boundaries:
        end = b  # segment up to b
        if end - start >= min_segment_length:
            segments.append((start, end))
        start = b + 1
    # final segment
    if len(signal) - start >= min_segment_length:
        segments.append((start, len(signal)-1))
    # merge segments that are very small with neighbors
    merged = []
    for s in segments:
        if not merged:
            merged.append(list(s))
        else:
            prev = merged[-1]
            if s[0] - prev[1] <= 2:  # gap tiny -> merge
                merged[-1][1] = s[1]
            else:
                merged.append(list(s))
    # convert back to tuples
    merged = [(int(a), int(b)) for a,b in merged]
    return merged, diff_threshold

def summarize_segments(df, wl_cols, segments):
    summaries = []
    for (s,e) in segments:
        block = df.iloc[s:e+1]
        stats = {}
        stats['start_idx'] = s
        stats['end_idx'] = e
        stats['n_samples'] = len(block)
        for c in wl_cols:
            stats[f'{c}_mean'] = block[c].mean()
            stats[f'{c}_std']  = block[c].std()
        summaries.append(stats)
    return pd.DataFrame(summaries)

def merge_with_metadata(meta_df, seg_summary, repeats_per_pressure=10, verbose=True):
    # seg_summary rows correspond to repetitions; group them in consecutive groups of repeats_per_pressure
    n_segments = len(seg_summary)
    expected_groups = int(np.ceil(n_segments / repeats_per_pressure))
    if verbose:
        print(f"Detected {n_segments} repetition segments -> grouping into {expected_groups} air-pressure rows (group size {repeats_per_pressure})")
    groups = []
    for i in range(expected_groups):
        group = seg_summary.iloc[i*repeats_per_pressure:(i+1)*repeats_per_pressure]
        if len(group) == 0:
            break
        agg = {}
        agg['group_index'] = i
        # aggregate all WL stats by mean/std across repetitions
        wl_mean_cols = [c for c in seg_summary.columns if c.endswith('_mean')]
        wl_std_cols  = [c for c in seg_summary.columns if c.endswith('_std')]
        for c in wl_mean_cols:
            agg[f'{c}_mean'] = group[c].mean()
            agg[f'{c}_std']  = group[c].mean() if False else group[c].std()  # std of repetition means
        # also keep count
        agg['n_reps_found'] = len(group)
        groups.append(agg)
    groups_df = pd.DataFrame(groups)
    # Now merge: meta_df expected length should be at least groups_df length. We'll align by order.
    merged = pd.concat([meta_df.reset_index(drop=True).iloc[:len(groups_df)].reset_index(drop=True), groups_df.reset_index(drop=True)], axis=1)
    return merged

def merge_per_repetition(
    meta_df,
    df,
    seg_summary,
    wl_cols,
    repeats_per_pressure=10,
    time_col=None,
    channels_to_use=3,
    include_std=False,
    timestamp_point="mid",
    verbose=True,
):
    # Build one row per repetition (segment), mapped to each air-pressure row in order
    wl_cols_to_export = select_wl_channels(df, wl_cols, channels_to_use=channels_to_use)
    n_segments = len(seg_summary)
    expected_groups = int(np.ceil(n_segments / repeats_per_pressure))
    if verbose:
        print(f"Detected {n_segments} repetition segments -> will emit one row per segment across {expected_groups} air-pressure rows (group size {repeats_per_pressure})")

    rows = []
    group_to_emitted = {}
    for group_index in range(expected_groups):
        # Meta row aligned to this group
        meta_row = meta_df.reset_index(drop=True).iloc[group_index:group_index+1]
        if meta_row.empty:
            break
        for rep_index in range(repeats_per_pressure):
            seg_idx = group_index * repeats_per_pressure + rep_index
            if seg_idx >= n_segments:
                break
            seg = seg_summary.iloc[seg_idx]

            out = {}
            out['group_index'] = group_index
            out['repetition_index'] = rep_index
            out['segment_start_idx'] = int(seg['start_idx'])
            out['segment_end_idx'] = int(seg['end_idx'])
            out['segment_n_samples'] = int(seg['n_samples'])
            # timestamps
            if time_col is not None and time_col in df.columns:
                try:
                    t_start = df.iloc[out['segment_start_idx']][time_col]
                    t_end = df.iloc[out['segment_end_idx']][time_col]
                    if pd.api.types.is_numeric_dtype(df[time_col]):
                        # numeric seconds
                        t_mid = (float(t_start) + float(t_end)) / 2.0
                    else:
                        # datetime-like
                        t_start = pd.to_datetime(t_start, errors='coerce')
                        t_end = pd.to_datetime(t_end, errors='coerce')
                        t_mid = t_start + (t_end - t_start) / 2 if pd.notna(t_start) and pd.notna(t_end) else pd.NaT
                    chosen = {
                        'start': t_start,
                        'mid': t_mid,
                        'end': t_end,
                    }.get(timestamp_point, t_mid)
                    out['timestamp'] = chosen
                except Exception:
                    out['timestamp'] = np.nan
            # per-channel wavelengths (mean and std from segment summary)
            for i, c in enumerate(wl_cols_to_export, start=1):
                out[f'WL_ch{i}'] = seg.get(f'{c}_mean', np.nan)
                if include_std:
                    out[f'WL_ch{i}_std'] = seg.get(f'{c}_std', np.nan)

            # append combined row (meta + out)
            combined = pd.concat([meta_row.reset_index(drop=True), pd.DataFrame([out])], axis=1)
            rows.append(combined)
            group_to_emitted[group_index] = group_to_emitted.get(group_index, 0) + 1

    if verbose and group_to_emitted:
        print("Per-group repetition counts:")
        for gi in sorted(group_to_emitted.keys()):
            print(f"  group {gi}: {group_to_emitted[gi]} rows (expected {repeats_per_pressure})")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def main(args):
    txt_path = Path(args.txt)
    xlsx_path = Path(args.xlsx)
    out_csv = Path(args.out)

    print("Loading interrogator data...")
    df = load_interrogator(txt_path)
    wl_cols = find_wl_columns(df)
    if len(wl_cols) == 0:
        raise RuntimeError("No wavelength (WL) columns detected. Columns found: " + ", ".join(df.columns))
    print("Detected WL columns:", wl_cols)

    # choose a representative signal to detect transitions
    # Options: mean of all WL cols, or the single channel with maximum variance
    if args.rep_signal_mode == 'maxvar':
        per_channel_std = {c: float(df[c].std(skipna=True)) for c in wl_cols}
        best_col = max(per_channel_std, key=per_channel_std.get)
        rep_signal = df[best_col].values
    else:
        rep_signal = df[wl_cols].mean(axis=1).values
    sm = smooth_signal(rep_signal, kernel_size=args.smooth_kernel)

    print("Detecting segments (repetitions)...")
    segments, used_thr = detect_segments(sm, diff_threshold=args.diff_threshold, min_segment_length=args.min_segment_length)
    print(f"Used derivative threshold = {used_thr:.6g}. Found {len(segments)} segments")

    seg_summary = summarize_segments(df, wl_cols, segments)
    print("Segment summary head:")
    print(seg_summary.head())

    # load Excel metadata with robust sheet resolution
    xl = pd.ExcelFile(xlsx_path)
    resolved_sheet = None
    if isinstance(args.sheet, int):
        try:
            resolved_sheet = xl.sheet_names[args.sheet]
        except Exception:
            resolved_sheet = None
    else:
        # Exact match
        if args.sheet in xl.sheet_names:
            resolved_sheet = args.sheet
        else:
            # Try base name without trailing -N
            m = re.match(r"^(.*?)(?:-\d+)$", str(args.sheet))
            base = m.group(1) if m else None
            if base and base in xl.sheet_names:
                resolved_sheet = base
    if resolved_sheet is None:
        # Fallback: first sheet
        resolved_sheet = xl.sheet_names[0]
        print(f"Warning: sheet '{args.sheet}' not found. Falling back to '{resolved_sheet}'.")
    else:
        if str(resolved_sheet) != str(args.sheet):
            print(f"Info: resolved sheet '{args.sheet}' -> '{resolved_sheet}'")
    meta = pd.read_excel(xlsx_path, sheet_name=resolved_sheet)
    # Forward-fill metadata so repeated fields (e.g., Layers, Distance) are populated for all pressures
    meta = meta.ffill()
    print("Metadata columns:", list(meta.columns))
    # Output mode
    time_col = find_time_column(df)
    if args.output_mode == 'repetition':
        merged = merge_per_repetition(
            meta_df=meta,
            df=df,
            seg_summary=seg_summary,
            wl_cols=wl_cols,
            repeats_per_pressure=args.repeats_per_pressure,
            channels_to_use=args.channels_to_use,
            include_std=args.include_std,
            timestamp_point=args.timestamp_point,
            time_col=time_col,
            verbose=True,
        )
    else:
        merged = merge_with_metadata(meta, seg_summary, repeats_per_pressure=args.repeats_per_pressure)

    # decide final output filename: include sheet name and date-hour suffix
    try:
        xl = pd.ExcelFile(xlsx_path)
        sheet_label = None
        # Determine sheet label robustly
        if isinstance(args.sheet, int):
            try:
                sheet_label = xl.sheet_names[args.sheet]
            except Exception:
                sheet_label = str(args.sheet)
        else:
            # args.sheet may be a string name or a numeric-like index
            if args.sheet in xl.sheet_names:
                sheet_label = str(args.sheet)
            else:
                try:
                    idx = int(str(args.sheet))
                    sheet_label = xl.sheet_names[idx]
                except Exception:
                    sheet_label = str(args.sheet)
    except Exception:
        sheet_label = str(args.sheet)

    def _sanitize(name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "sheet")
        return safe.strip("._-") or "sheet"

    # Keep full sheet label so outputs differ per replica (e.g., '-1', '-2', '-3')
    timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M")
    base_out = out_csv
    parent = base_out.parent if base_out.parent else Path(".")
    stem = base_out.stem or "merged"
    ext = base_out.suffix or ".csv"
    final_name = f"{stem}_{_sanitize(sheet_label)}_{timestamp_suffix}{ext}"
    final_path = parent / final_name

    # save
    merged.to_csv(final_path, index=False)
    print(f"Merged dataset saved to: {final_path}")
    print("Merged head:")
    print(merged.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge FBG interrogator data (txt) with Excel metadata (xlsx)")
    parser.add_argument("--txt", required=True, help="path to interrogator txt file")
    parser.add_argument("--xlsx", required=True, help="path to metadata xlsx file (sheet has rows for each pressure)")
    parser.add_argument("--sheet", default=0, help="sheet name or index in xlsx")
    parser.add_argument("--out", default="merged_fbg.csv", help="output csv file")
    parser.add_argument("--repeats_per_pressure", type=int, default=10, help="how many repetition segments per pressure (default 10)")
    parser.add_argument("--smooth_kernel", type=int, default=7, help="median smoothing kernel size (odd, default 7)")
    parser.add_argument("--diff_threshold", type=float, default=None, help="derivative threshold for change detection (auto if None)")
    parser.add_argument("--min_segment_length", type=int, default=5, help="minimum number of samples in a repetition segment")
    parser.add_argument("--output_mode", choices=["group", "repetition"], default="repetition", help="output aggregation: group (group-of-repetitions per meta row) or repetition (one row per repetition)")
    parser.add_argument("--channels_to_use", type=int, default=3, help="number of WL channels to include (from left to right)")
    parser.add_argument("--include_std", action="store_true", help="include per-channel std columns alongside mean")
    parser.add_argument("--timestamp_point", choices=["start", "mid", "end"], default="mid", help="which segment timestamp to export")
    parser.add_argument("--rep_signal_mode", choices=["mean", "maxvar"], default="mean", help="how to build representative signal for segmentation")
    args = parser.parse_args()
    main(args)

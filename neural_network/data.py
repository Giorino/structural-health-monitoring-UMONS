import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_synthetic_data(num_samples=1000, sequence_length=25):
    """Generate synthetic data for testing when no real CSV files are available"""
    data = []
    for _ in range(num_samples):
        group_idx = np.random.randint(0, 10)
        rep_idx = np.random.randint(0, 5)

        sequence = []
        baseline_wl = 1550.0 + np.random.normal(0, 0.1)

        for t in range(sequence_length):
            time_factor = t / sequence_length
            wl_ch2 = baseline_wl + time_factor * np.random.uniform(0.1, 2.0) + np.random.normal(0, 0.01)
            wl_ch2_std = np.random.uniform(0.001, 0.01)
            force = time_factor * np.random.uniform(50, 500) + np.random.normal(0, 5)
            displacement = time_factor * np.random.uniform(1, 10) + np.random.normal(0, 0.1)
            air_pressure = np.random.uniform(1, 6)
            delta_wl_ch2 = wl_ch2 - baseline_wl
            if t > 0:
                prev_delta = sequence[-1][2]
                prev_disp = sequence[-1][4]
                delta_wl_rate = delta_wl_ch2 - prev_delta
                delta_disp_rate = displacement - prev_disp
            else:
                delta_wl_rate = 0
                delta_disp_rate = 0
            sequence.append([
                wl_ch2,
                wl_ch2_std,
                delta_wl_ch2,
                force,
                displacement,
                air_pressure,
                delta_wl_rate,
                delta_disp_rate,
            ])

        max_displacement = max([s[4] for s in sequence])
        if max_displacement < 2:
            crack_label = 0
        elif max_displacement < 5:
            crack_label = 1
        elif max_displacement < 8:
            crack_label = 2
        else:
            crack_label = 3

        data.append({
            'group_index': group_idx,
            'repetition_index': rep_idx,
            'sequence': np.array(sequence),
            'crack_label': crack_label,
        })
    return data


def load_and_preprocess_data(data_dir="./", prediction_horizon=5):
    """Load CSV files and preprocess data for neural network training"""

    def find_latest_output_dir(output_base_dir):
        if not os.path.exists(output_base_dir):
            return None
        subdirs = []
        for item in os.listdir(output_base_dir):
            subdir_path = os.path.join(output_base_dir, item)
            if os.path.isdir(subdir_path):
                subdirs.append((item, subdir_path))
        if not subdirs:
            return None
        latest_subdir = sorted(subdirs, key=lambda x: x[0], reverse=True)[0]
        return latest_subdir[1]

    csv_files = glob.glob(os.path.join(data_dir, "merged_*.csv"))

    output_dir = os.path.join(data_dir, "output")
    latest_dir = find_latest_output_dir(output_dir)
    if latest_dir:
        latest_csv_files = glob.glob(os.path.join(latest_dir, "merged_*.csv"))
        csv_files.extend(latest_csv_files)

    parent_output_dir = os.path.join("..", "output")
    latest_parent_dir = find_latest_output_dir(parent_output_dir)
    if latest_parent_dir and not latest_dir:
        latest_csv_files = glob.glob(os.path.join(latest_parent_dir, "merged_*.csv"))
        csv_files.extend(latest_csv_files)

    if not csv_files:
        return generate_synthetic_data()

    all_data = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            required_cols = [
                'group_index',
                'repetition_index',
                'WL_ch2',
                'WL_ch2_std',
                'Force (N)',
                'Displacement (mm)',
                'Air Pressure (bar)',
                'Crack',
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                continue

            groups = df.groupby(['group_index', 'repetition_index'])
            for (group_idx, rep_idx), group_df in groups:
                if len(group_df) < 1:
                    continue
                if 'timestamp' in group_df.columns:
                    group_df = group_df.sort_values('timestamp')
                elif 'segment_start_idx' in group_df.columns:
                    group_df = group_df.sort_values('segment_start_idx')

                baseline_wl = group_df['WL_ch2'].iloc[0]
                group_df = group_df.copy()
                group_df['delta_wl_ch2'] = group_df['WL_ch2'] - baseline_wl
                group_df['delta_wl_rate'] = group_df['delta_wl_ch2'].diff().fillna(0)
                group_df['delta_disp_rate'] = group_df['Displacement (mm)'].diff().fillna(0)

                features = [
                    'WL_ch2',
                    'WL_ch2_std',
                    'delta_wl_ch2',
                    'Force (N)',
                    'Displacement (mm)',
                    'Air Pressure (bar)',
                    'delta_wl_rate',
                    'delta_disp_rate',
                ]
                sequence = group_df[features].values

                crack_values = group_df['Crack'].fillna(0).astype(int)
                crack_label = crack_values.max()

                all_data.append({
                    'group_index': group_idx,
                    'repetition_index': rep_idx,
                    'sequence': sequence,
                    'crack_label': crack_label,
                })
        except Exception:
            continue

    if not all_data:
        return generate_synthetic_data()

    return all_data


def create_sequences_and_labels(data, sequence_length=50, prediction_horizon=5):
    """Create fixed-length sequences and labels for training using a sliding window approach."""
    sequences = []
    labels = []

    pressure_groups = {}
    for item in data:
        if item['sequence'].shape[0] > 0:
            measurement = item['sequence'][0]
            air_pressure = measurement[5]
            crack_label = item['crack_label']
            if air_pressure not in pressure_groups:
                pressure_groups[air_pressure] = []
            pressure_groups[air_pressure].append({
                'measurement': measurement,
                'crack_label': int(crack_label),
            })

    for _, measurements in pressure_groups.items():
        measurements.sort(key=lambda x: x['measurement'][4])
        num_measurements = len(measurements)
        if num_measurements >= sequence_length + prediction_horizon:
            for i in range(num_measurements - sequence_length - prediction_horizon + 1):
                feature_window = measurements[i: i + sequence_length]
                label_index = i + sequence_length + prediction_horizon - 1
                sequence_data = [m['measurement'] for m in feature_window]
                label = measurements[label_index]['crack_label']
                max_crack_in_sequence = max(m['crack_label'] for m in feature_window)
                final_label = max(label, max_crack_in_sequence)
                sequences.append(np.array(sequence_data))
                labels.append(final_label)

    return sequences, labels


def normalize_features(sequences):
    """Normalize features using StandardScaler"""
    all_features = np.concatenate([seq for seq in sequences], axis=0)
    scaler = StandardScaler()
    scaler.fit(all_features)
    normalized_sequences = []
    for seq in sequences:
        normalized_seq = scaler.transform(seq)
        normalized_sequences.append(normalized_seq)
    return normalized_sequences, scaler




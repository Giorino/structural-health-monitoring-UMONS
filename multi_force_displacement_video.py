
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import shutil
import glob

def find_latest_output_directory(base_dir=None):
    """Find the most recently modified subdirectory under the base output directory.

    If base_dir is None or a relative path, it is resolved relative to this file's directory.
    Returns an absolute path or None if not found.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir is None:
        resolved_base = os.path.join(script_dir, 'output')
    else:
        resolved_base = base_dir if os.path.isabs(base_dir) else os.path.join(script_dir, str(base_dir))

    if not os.path.isdir(resolved_base):
        return None

    output_dirs = [
        os.path.join(resolved_base, d)
        for d in os.listdir(resolved_base)
        if os.path.isdir(os.path.join(resolved_base, d))
    ]
    if not output_dirs:
        return None
    latest_dir = max(output_dirs, key=os.path.getmtime)
    return latest_dir

def create_multi_force_displacement_video(directory_path, output_video_name='multi_force_displacement_animation.mp4', frames_dir='temp_frames_mfd'):
    """
    Generates a single video animating the Force vs. Displacement curves from multiple CSV files.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return

    csv_files = sorted(glob.glob(os.path.join(directory_path, '*.csv')))
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return
        
    print(f"Found {len(csv_files)} CSV files to process in {directory_path}")

    # Load all dataframes and determine global plot limits
    dataframes = [pd.read_csv(f).dropna(how='all') for f in csv_files]
    max_len = max(len(df) for df in dataframes)
    
    # --- Determine dynamic plot limits ---
    min_displacement = min(df['Displacement (mm)'].min() for df in dataframes)
    max_displacement = max(df['Displacement (mm)'].max() for df in dataframes)
    min_force = min(df['Force (N)'].min() for df in dataframes)
    max_force = max(df['Force (N)'].max() for df in dataframes)

    # Add some padding
    disp_padding = (max_displacement - min_displacement) * 0.05
    force_padding = (max_force - min_force) * 0.05
    
    xlim_min = min_displacement - disp_padding
    xlim_max = max_displacement + disp_padding
    ylim_min = min_force - force_padding
    ylim_max = max_force + force_padding
    # ------------------------------------

    # Create a directory to store frames
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    # Get a color for each file
    colors = plt.cm.get_cmap('tab10', len(csv_files))

    # Generate a frame for each step in the longest test
    for i in range(max_len):
        plt.figure(figsize=(14, 10))

        for idx, df in enumerate(dataframes):
            # Only plot if the current frame index is within the length of this dataframe
            if i < len(df):
                current_df = df.iloc[:i+1]
                label = os.path.basename(csv_files[idx]).replace('.csv', '')

                # Plot the historical data
                plt.plot(current_df['Displacement (mm)'], current_df['Force (N)'], '-', color=colors(idx), label=label)
                # Highlight the current point
                plt.plot(current_df['Displacement (mm)'].iloc[-1], current_df['Force (N)'].iloc[-1], 'o', color=colors(idx), markersize=8)

        # Formatting
        plt.title(f'Comparative Force vs. Displacement (Frame {i+1}/{max_len})')
        plt.xlabel('Displacement (mm)')
        plt.ylabel('Force (N)')
        plt.xlim(xlim_min, xlim_max)
        plt.ylim(ylim_min, ylim_max)
        plt.legend()
        plt.grid(True)

        # Save frame
        frame_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
        plt.savefig(frame_path)
        plt.close()

    # --- Create Video from Frames ---
    frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.png')))
    if not frame_files:
        print("No frames were generated. Video creation aborted.")
        shutil.rmtree(frames_dir)
        return

    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(directory_path, output_video_name)
    video = cv2.VideoWriter(output_video_path, fourcc, 15, (width, height)) # 15 FPS

    for frame_file in frame_files:
        video.write(cv2.imread(frame_file))

    video.release()
    print(f"Video saved as {output_video_path}")

    # Clean up frames
    shutil.rmtree(frames_dir)
    print(f"Temporary directory {frames_dir} removed.")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    latest_dir = find_latest_output_directory(os.path.join(script_dir, 'output'))
    if latest_dir:
        create_multi_force_displacement_video(latest_dir)
    else:
        print("No output directories found.")
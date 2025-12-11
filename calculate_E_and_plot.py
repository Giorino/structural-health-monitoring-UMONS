import csv
import math

# --- PARAMETERS ---
L = 230.0   # Span length (mm)
b = 34.0    # Width (mm)
h = 4.0     # Thickness (mm)
# y_fbg Calculation
n_layers = 12
fbg_layer_interface = 10 # FBG is between 10th and 11th layer (0-indexed: on top of 10th)
# Distance from bottom to FBG:
y_from_bottom = (h / n_layers) * fbg_layer_interface
# Distance from neutral axis (center, h/2):
y_fbg = y_from_bottom - (h / 2)
# y_fbg should be positive if above neutral axis, negative if below.
# Here: h=4, n=12. layer_thickness = 0.333.
# y_from_bottom = 0.333 * 10 = 3.333
# y_fbg = 3.333 - 2.0 = 1.333 mm.
sensitivity = 1.2 # pm/microstrain

# Moment of Inertia (I)
I = (b * h**3) / 12

# Input/Output Files
INPUT_CSV = 'output/20250915_144000/merged_23cm-12layers-1_20250915_1443.csv'
OUTPUT_CSV = 'E_vs_Force.csv'
OUTPUT_HTML = 'E_vs_Force.html'

import os
import glob

# ... (existing imports)

# --- HELPER FUNCTIONS ---
def find_interrogator_file(csv_path):
    """Finds the corresponding interrogator file based on the CSV filename."""
    filename = os.path.basename(csv_path)
    # Expected format: merged_{IDENTIFIER}_{TIMESTAMP}.csv
    # e.g., merged_23cm-12layers-1_20250915_1443.csv -> 23cm-12layers-1
    try:
        parts = filename.split('_')
        if len(parts) >= 3:
            identifier = parts[1]
        else:
            print(f"Warning: Could not parse identifier from {filename}. Using manual fallback if needed.")
            return None
            
        print(f"Looking for interrogator file for identifier: {identifier}")
        
        # Search recursively for *{identifier}*interrogator.txt
        # Using glob with recursive flag (requires Python 3.5+)
        search_pattern = f"**/*{identifier}*interrogator.txt"
        found_files = glob.glob(search_pattern, recursive=True)
        
        if not found_files:
            print(f"Error: No interrogator file found for {identifier}")
            return None
            
        # Return the first match (or refine logic if multiple exist)
        print(f"Found interrogator file: {found_files[0]}")
        return found_files[0]
        
    except Exception as e:
        print(f"Error parsing filename: {e}")
        return None

def get_baseline_from_interrogator(file_path):
    """Reads the interrogator file and returns the minimum wavelength of WL 2."""
    try:
        min_wl = float('inf')
        with open(file_path, 'r') as f:
            # Skip header
            header = f.readline()
            # Identify column index for WL 2 (usually index 3 if splitting by tab)
            # Header: Timestamp Time [s] WL 1[nm] WL 2[nm] ...
            cols = header.split('\t')
            wl2_idx = -1
            for i, col in enumerate(cols):
                if "WL 2" in col:
                    wl2_idx = i
                    break
            
            if wl2_idx == -1:
                # Fallback to index 3 (standard format)
                wl2_idx = 3
                
            for line in f:
                parts = line.split('\t')
                if len(parts) > wl2_idx:
                    try:
                        val = float(parts[wl2_idx])
                        if val < min_wl and val > 1000: # Basic validity check
                            min_wl = val
                    except ValueError:
                        continue
        
        if min_wl == float('inf'):
            return None
            
        return min_wl
        
    except Exception as e:
        print(f"Error reading interrogator file: {e}")
        return None

# --- DATA PROCESSING ---
data_points = []

try:
    # 1. Find Baseline Automatically
    interrogator_file = find_interrogator_file(INPUT_CSV)
    baseline_wl = None
    
    if interrogator_file:
        baseline_wl = get_baseline_from_interrogator(interrogator_file)
        
    if baseline_wl:
        print(f"Automatically detected Baseline WL: {baseline_wl} nm")
    else:
        print("Could not detect baseline automatically. Using fallback/manual value.")
        baseline_wl = 1538.024 # Fallback
        print(f"Using Fallback Baseline WL: {baseline_wl} nm")

    with open(INPUT_CSV, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        # Group rows by Force
        force_groups = {}
        for row in rows:
            try:
                force = float(row['Force (N)'])
                if force not in force_groups:
                    force_groups[force] = []
                force_groups[force].append(float(row['WL_ch2']))
            except (ValueError, KeyError):
                continue
        
        # Process each group
        sorted_forces = sorted(force_groups.keys())
        
        print(f"{'Force (N)':<10} | {'Baseline (nm)':<15} | {'Mean E (GPa)':<12} | {'Std Dev (GPa)':<12}")
        print("-" * 60)

        for force in sorted_forces:
            wls = force_groups[force]
            
            # Calculate E for ALL iterations in the group to find Mean and Std Dev
            e_values = []
            for wl in wls:
                shift_nm = wl - baseline_wl
                
                # Avoid division by zero or negative shifts
                if shift_nm <= 0.0001:
                    continue

                # 1. Optical Strain Calculation
                measured_strain_ue = (shift_nm * 1000) / sensitivity
                measured_strain_unitless = measured_strain_ue * 1e-6
                
                # 2. Mechanical Values
                M = (force * L) / 4
                
                # 3. Young's Modulus (E) Calculation
                E_MPa = (M * y_fbg) / (measured_strain_unitless * I)
                E_GPa = E_MPa / 1000.0
                e_values.append(E_GPa)
            
            if not e_values:
                continue
                
            # Calculate Mean and Std Dev
            mean_e = sum(e_values) / len(e_values)
            variance = sum((x - mean_e) ** 2 for x in e_values) / len(e_values)
            std_dev = math.sqrt(variance)
            
            data_points.append({
                'force': force,
                'E_GPa': mean_e,
                'E_std': std_dev
            })
            
            print(f"{force:<10.1f} | {baseline_wl:<15.4f} | {mean_e:<12.2f} | {std_dev:<12.2f}")

except FileNotFoundError:
    print(f"Error: File {INPUT_CSV} not found.")
    exit(1)

# Sort by force for better plotting (already sorted by processing logic, but good to ensure)
data_points.sort(key=lambda x: x['force'])

# --- SAVE TO CSV ---
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['force', 'E_GPa', 'E_std'])
    writer.writeheader()
    writer.writerows(data_points)

print(f"\nProcessed {len(data_points)} force groups.")
print(f"Saved results to {OUTPUT_CSV}")

# --- GENERATE SVG PLOT ---
# Since matplotlib is not available, we generate an SVG manually.
OUTPUT_SVG = 'E_vs_Force_Plot.svg'

def create_svg(data, filename):
    width = 800
    height = 600
    margin = 80
    
    # Extract data
    x_vals = [d['force'] for d in data]
    y_vals = [d['E_GPa'] for d in data]
    
    x_min, x_max = min(x_vals), max(x_vals)
    # User request: "keep the range higher" to make it look more linear/stabilized.
    # Data is around 20-21 GPa. Setting range 0-30 GPa will show the stability.
    y_min, y_max = 16, 21
    
    # Pad x range only
    x_range = x_max - x_min
    x_min -= x_range * 0.1
    x_max += x_range * 0.1
    # y range is fixed now
    
    # Scale functions
    def get_x(val):
        return margin + (val - x_min) / (x_max - x_min) * (width - 2 * margin)
        
    def get_y(val):
        return height - margin - (val - y_min) / (y_max - y_min) * (height - 2 * margin)
    
    # Linear Regression REMOVED per user request to avoid misleading slope interpretation.
    
    # SVG Content
    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background-color:white; font-family:Arial, sans-serif;">']
    
    # Title
    svg.append(f'<text x="{width/2}" y="{margin/2}" text-anchor="middle" font-size="20" font-weight="bold">Young\'s Modulus (E) vs Force</text>')
    
    # Axes
    svg.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black" stroke-width="2"/>') # X Axis
    svg.append(f'<line x1="{margin}" y1="{height-margin}" x2="{margin}" y2="{margin}" stroke="black" stroke-width="2"/>') # Y Axis
    
    # Grid & Ticks (Simple)
    x_steps = 10
    y_steps = 10
    
    # X Grid
    for i in range(x_steps + 1):
        val = x_min + (x_max - x_min) * i / x_steps
        px = get_x(val)
        svg.append(f'<line x1="{px}" y1="{height-margin}" x2="{px}" y2="{margin}" stroke="#eee" stroke-width="1"/>')
        svg.append(f'<line x1="{px}" y1="{height-margin}" x2="{px}" y2="{height-margin+5}" stroke="black" stroke-width="1"/>')
        svg.append(f'<text x="{px}" y="{height-margin+20}" text-anchor="middle" font-size="12">{val:.0f}</text>')
    
    # Y Grid
    for i in range(y_steps + 1):
        val = y_min + (y_max - y_min) * i / y_steps
        py = get_y(val)
        svg.append(f'<line x1="{margin}" y1="{py}" x2="{width-margin}" y2="{py}" stroke="#eee" stroke-width="1"/>')
        svg.append(f'<line x1="{margin}" y1="{py}" x2="{margin-5}" y2="{py}" stroke="black" stroke-width="1"/>')
        svg.append(f'<text x="{margin-10}" y="{py+4}" text-anchor="end" font-size="12">{val:.1f}</text>')

    # Axis Labels
    svg.append(f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-size="14" font-weight="bold">Force (N)</text>')
    svg.append(f'<text x="{20}" y="{height/2}" text-anchor="middle" transform="rotate(-90 20,{height/2})" font-size="14" font-weight="bold">Young\'s Modulus (GPa)</text>')

    # Regression Line REMOVED

    # Data Points with Error Bars
    for d in data:
        cx = get_x(d['force'])
        cy = get_y(d['E_GPa'])
        std = d['E_std']
        
        # Error Bar (Vertical Line)
        y_top = get_y(d['E_GPa'] + std)
        y_bot = get_y(d['E_GPa'] - std)
        
        # Clamp error bars to plot area if needed, but SVG handles overflow by clipping usually.
        # Drawing Error Bar
        svg.append(f'<line x1="{cx}" y1="{y_top}" x2="{cx}" y2="{y_bot}" stroke="black" stroke-width="1.5"/>')
        # Error Bar Caps
        cap_width = 4
        svg.append(f'<line x1="{cx-cap_width}" y1="{y_top}" x2="{cx+cap_width}" y2="{y_top}" stroke="black" stroke-width="1.5"/>')
        svg.append(f'<line x1="{cx-cap_width}" y1="{y_bot}" x2="{cx+cap_width}" y2="{y_bot}" stroke="black" stroke-width="1.5"/>')
        
        # Point
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="blue" stroke="black" stroke-width="1"/>')

    svg.append('</svg>')
    
    with open(filename, 'w') as f:
        f.write('\n'.join(svg))

create_svg(data_points, OUTPUT_SVG)
print(f"Generated SVG plot at {OUTPUT_SVG}")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# 1. Prepare the Data
# Input signal: flat, a small bump, flat, a big spike, flat
np.random.seed(42)
signal = np.zeros(25)
signal[5:8] = [0.2, 0.5, 0.2]      # Small bump
signal[15:18] = [-0.5, 2.0, -0.5]  # Big peak pattern
signal += np.random.normal(0, 0.1, len(signal)) # Add slight noise

# Filter (Kernel): looking for a sharp peak
kernel = np.array([-1, 2, -1])
k_size = len(kernel)

# Output signal (Feature Map)
output_len = len(signal) - k_size + 1
output = np.zeros(output_len)
for i in range(output_len):
    output[i] = np.sum(signal[i:i+k_size] * kernel)

# 2. Setup the Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

# --- Top Plot: Input Signal ---
ax1.plot(signal, marker='o', color='royalblue', label='Input Sensor Data')
ax1.set_xlim(-1, len(signal))
ax1.set_ylim(-1.5, 2.5)
ax1.set_ylabel("Amplitude")
ax1.set_title("Input Data & Sliding Filter", fontsize=12)
ax1.grid(True, alpha=0.3)

# The Sliding Window Box
window_box = plt.Rectangle((0-0.5, -1.2), k_size, 3.5, fill=True, color='orange', alpha=0.3, edgecolor='red', lw=2)
ax1.add_patch(window_box)

# Text for calculation
calc_text = ax1.text(0.02, 0.85, '', transform=ax1.transAxes, fontsize=11, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# --- Bottom Plot: Feature Map ---
ax2.set_xlim(-1, len(signal))
ax2.set_ylim(min(output)-1, max(output)+1)
ax2.set_ylabel("Activation Score")
ax2.set_xlabel("Time Step (Index)")
ax2.set_title("Feature Map (Output)", fontsize=12)
ax2.grid(True, alpha=0.3)

# The Output Line being drawn
output_line, = ax2.plot([], [], marker='s', color='forestgreen', lw=2, label='Feature Map')
current_dot, = ax2.plot([], [], marker='o', color='red', markersize=10)

plt.tight_layout()

# 3. Animation Function
def animate(i):
    # Stop condition
    if i >= output_len:
        return window_box, calc_text, output_line, current_dot
    
    # Update Window Position
    window_box.set_xy((i - 0.25, -1.3))
    
    # Extract current window data
    window_data = signal[i:i+k_size]
    
    # Update Calculation Text
    calc_str = f"Filter: {kernel}\n"
    calc_str += f"Data:   [{window_data[0]:.1f}, {window_data[1]:.1f}, {window_data[2]:.1f}]\n"
    calc_str += f"Multiply & Add:\n"
    calc_str += f"({window_data[0]:.1f} * {kernel[0]}) + ({window_data[1]:.1f} * {kernel[1]}) + ({window_data[2]:.1f} * {kernel[2]}) = {output[i]:.2f}"
    calc_text.set_text(calc_str)
    
    # Update Output Line
    output_line.set_data(range(i+1), output[:i+1])
    current_dot.set_data([i], [output[i]])
    
    # Highlight the matching area (if score is high)
    if output[i] > 2.5:
        window_box.set_color('lime')
        window_box.set_edgecolor('green')
    else:
        window_box.set_color('orange')
        window_box.set_edgecolor('red')

    return window_box, calc_text, output_line, current_dot

# 4. Create and Save Animation
anim = animation.FuncAnimation(fig, animate, frames=output_len + 5, interval=600, blit=True)

out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exercise_outputs', 'convolution_animation.gif')
print(f"Saving GIF to {out_file}...")
# Use PillowWriter to save gif without needing ImageMagick
writer = animation.PillowWriter(fps=1.5)
anim.save(out_file, writer=writer)
print("Done!")

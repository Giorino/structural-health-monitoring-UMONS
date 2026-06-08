import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Function to draw a box with text
    def draw_box(x, y, w, h, text, facecolor, edgecolor='black', text_color='black', fontsize=10):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                     facecolor=facecolor, edgecolor=edgecolor, lw=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=fontsize, color=text_color, fontweight='bold')
        return x + w, y + h/2 # return right middle point for arrows

    # Colors
    c_input = '#E0F2FE'  # Light Blue
    c_cnn = '#BAE6FD'    # Blue
    c_phys = '#FED7AA'   # Orange
    c_merge = '#DDD6FE'  # Purple
    c_out = '#BBF7D0'    # Green

    # 1. Input Data
    x1, y1 = draw_box(0.5, 2.5, 1.5, 1.0, "Raw Sensor\nData\n(9 Channels)", c_input)

    # 2. CNN Branch (Top)
    x2, y2 = draw_box(3.0, 3.8, 1.8, 0.8, "CNN Layer 1\n(Extract Features)", c_cnn)
    x3, y3 = draw_box(5.3, 3.8, 1.8, 0.8, "CNN Layer 2\n(Combine Features)", c_cnn)
    x4, y4 = draw_box(7.6, 3.8, 1.5, 0.8, "Global\nPooling", c_cnn)

    # 3. Physics Branch (Bottom)
    x5, y5 = draw_box(3.0, 1.2, 2.5, 1.0, "Physics Head\n(Euler-Bernoulli Eq.)\nε = (P·L·y) / (4·E·I)", c_phys)

    # 4. Merge
    x6, y6 = draw_box(9.8, 2.5, 1.0, 1.0, "Concat", c_merge)

    # 5. Output
    x7, y7 = draw_box(11.3, 2.6, 0.5, 0.8, "FC\nLayer", c_out)

    # Arrows
    arrow_props = dict(facecolor='black', edgecolor='black', width=2, headwidth=8, headlength=10, shrink=0.05)
    
    # Input to branches
    ax.annotate("", xy=(3.0, 4.2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax.annotate("", xy=(3.0, 1.7), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="black", lw=2))

    # Between CNN layers
    ax.annotate("", xy=(5.3, y2), xytext=(x2, y2), arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax.annotate("", xy=(7.6, y3), xytext=(x3, y3), arrowprops=dict(arrowstyle="->", color="black", lw=2))

    # CNN and Physics to Merge
    ax.annotate("", xy=(9.8, 3.0), xytext=(x4, y4), arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax.annotate("", xy=(9.8, 2.0), xytext=(x5, y5), arrowprops=dict(arrowstyle="->", color="black", lw=2))

    # Merge to Output
    ax.annotate("", xy=(11.3, y6), xytext=(x6, y6), arrowprops=dict(arrowstyle="->", color="black", lw=2))

    # Branch labels
    ax.text(5.5, 5.0, "Data-Driven Branch (The Detective)", ha='center', fontsize=12, fontweight='bold', color='#0284C7')
    ax.text(4.2, 0.6, "Physics Branch (The Calculator)", ha='center', fontsize=12, fontweight='bold', color='#C2410C')

    # Save
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exercise_outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "physics_head_diagram_labeled.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved flowchart to {out_file}")

if __name__ == "__main__":
    draw_flowchart()

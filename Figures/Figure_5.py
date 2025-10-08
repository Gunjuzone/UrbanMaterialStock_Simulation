import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.lines import Line2D
import numpy as np 

# --- Global Style Settings ---
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 13
plt.rcParams["legend.title_fontsize"] = 14
plt.rcParams["figure.titlesize"] = 20
plt.rcParams["figure.titleweight"] = "bold"

# Data (anel a & b)
data = {
    "Material": ["Wood", "Brick", "Concrete", "Steel", "Glass", "Plastics", "Copper", "Aluminum"],
    "Building CV": [35.63, 33.19, 31.42, 28.52, 26.06, 23.58, 23.32, 23.29],
    "District CV": [2.03, 2.09, 1.40, 1.45, 1.37, 1.34, 1.33, 1.35],
    "Reduction Factor": [17.5, 15.9, 22.4, 19.7, 19.0, 17.6, 17.5, 17.3],
    "Building Rank": [1, 2, 3, 4, 5, 6, 7, 8],
    "District Rank": [2, 1, 5, 4, 6, 7, 8, 3]
}
df = pd.DataFrame(data)

# Data for Panel (c)

df_stats = pd.DataFrame({
    'sample_size': [100, 200, 400, 800, 1600],
    'Concrete_cv_mean': [6.073154866, 4.158969517, 3.008381612, 2.171579939, 1.552391037],
    'Concrete_cv_std': [0.511074977, 0.491976477, 0.370398839, 0.394675536, 0.03863303],
    'Brick_cv_mean': [7.318074946, 5.74890265, 5.046303392, 3.410896563, 2.843282497],
    'Brick_cv_std': [0.543135732, 0.51987515, 0.523698606, 0.443177844, 0.072189545],
    'Wood_cv_mean': [7.865020589, 5.349672268, 4.215281865, 2.95745547, 2.203018234],
    'Wood_cv_std': [2.687215748, 1.338345011, 0.773046257, 0.512835033, 0.054588096],
    'Steel_cv_mean': [6.401236537, 4.331545777, 3.062225489, 2.245271947, 1.590041812],
    'Steel_cv_std': [0.59693704, 0.504920133, 0.354681864, 0.439708492, 0.038901222],
    'Glass_cv_mean': [6.030288378, 4.086984491, 2.908393828, 2.112762019, 1.509277818],
    'Glass_cv_std': [0.552576304, 0.488319333, 0.353354182, 0.420802066, 0.039096598],
    'Plastics_cv_mean': [5.882512367, 3.984253786, 2.851162537, 2.05884641, 1.482210681],
    'Plastics_cv_std': [0.591155478, 0.511557112, 0.366680345, 0.419325545, 0.041187035],
    'Aluminium_cv_mean': [5.953398784, 4.018601952, 2.853112251, 2.070443943, 1.485256521],
    'Aluminium_cv_std': [0.591885097, 0.495160092, 0.353654875, 0.426509424, 0.04041885],
    'Copper_cv_mean': [5.866754691, 3.964429836, 2.8195658, 2.041568492, 1.46705516],
    'Copper_cv_std': [0.598590081, 0.495144398, 0.352170032, 0.417333505, 0.040473162]
})
materials = ['Concrete', 'Brick', 'Wood', 'Steel', 'Glass', 'Plastics', 'Aluminium', 'Copper']
x_values = [100, 200, 400, 800, 1600]


fig = plt.figure(figsize=(20, 18))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.4], width_ratios=[1.25, 1], wspace=0.4, hspace=0.25)


# Panel (a)

ax1 = fig.add_subplot(gs[0, 0])
x = range(len(df["Material"]))
building_color = "lightblue"
district_color = "#E69F00"

ax1.vlines(x, 0, df["Building CV"], color=building_color, linewidth=2)
ax1.scatter(x, df["Building CV"], color=building_color, edgecolor="black", s=120, label="Building CV (%)", zorder=3)
offset = 0.23
x_offset = [i + offset for i in x]
ax1.vlines(x_offset, 0, df["District CV"], color=district_color, linewidth=2)
ax1.scatter(x_offset, df["District CV"], color=district_color, edgecolor="black", marker="s", s=120, label="District CV (%)", zorder=3)

ax1.set_ylabel("Coefficient of Variation (%)")
ax1.set_xticks(x)
ax1.set_xticklabels(df["Material"], rotation=45)
ax1.set_ylim(-1, df["Building CV"].max() + 2.4)

# CV value labels
for i, val in enumerate(df["Building CV"]):
    ax1.text(i, val + 0.5, f"{val:.1f}", ha="center", va="bottom", fontsize=18, color="black")
for i, val in enumerate(df["District CV"]):
    ax1.text(x_offset[i], val + 0.5, f"{val:.2f}", ha="center", va="bottom", fontsize=18, color="black")

# Secondary axis for reduction factor
ax2 = ax1.twinx()
ax2.set_ylabel("Reduction Factor (×)", color="black")
ax2.tick_params(axis="y", colors="black")
ax2.set_ylim(ax1.get_ylim()[0], ax1.get_ylim()[1]) 


for i, val in enumerate(df["Reduction Factor"]):
    if df["Material"][i] == "Brick":
        ax1.text(i - 0.4, val, f"{val:.1f}x", fontsize=16, color="black", va="center")
    else:
        ax1.text(i, val, f"{val:.1f}x", fontsize=16, color="black", va="center", ha="center")


lines_labels = [ax.get_legend_handles_labels() for ax in [ax1]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
ax1.legend(lines, labels, loc="upper right", fontsize = 16)
ax1.set_title("(a)", loc="left")


# Panel (b)
ax3 = fig.add_subplot(gs[0, 1])
ax3.text(0.24, 0.7, "Building", ha="right", va="bottom", fontsize=18, weight="bold")
ax3.text(1, 0.7, "District", ha="left", va="bottom", fontsize=18, weight="bold")

ax3.scatter([0.27]*len(df), df["Building Rank"], color=building_color, s=80, edgecolor="black", marker="o")
for i, mat in enumerate(df["Material"]):
    ax3.text(0.05, df["Building Rank"][i], f"{df['Building Rank'][i]}. {mat}", ha="left", va="center", fontsize=16)

ax3.scatter([1]*len(df), df["District Rank"], color=district_color, s=80, edgecolor="black", marker="s")
for i, mat in enumerate(df["Material"]):
    ax3.text(1.05, df["District Rank"][i], f"{df['District Rank'][i]}. {mat}", ha="left", va="center", fontsize=16)

improved_color, worsened_color, no_change_color = "#009E73", "#D55E00", "#666666"
for i in range(len(df)):
    b_rank, d_rank = df["Building Rank"][i], df["District Rank"][i]
    if d_rank < b_rank:
        color, ls, lw = improved_color, "-", 2.0
    elif d_rank > b_rank:
        color, ls, lw = worsened_color, "--", 2.0
    else:
        color, ls, lw = no_change_color, ":", 2.0
    ax3.plot([0.27, 1], [b_rank, d_rank], color=color, linestyle=ls, linewidth=lw)

ax3.set_ylim(0.3, 8.5)
ax3.invert_yaxis()
ax3.set_title("(b)", loc="left")
ax3.axis("off")

legend_elements = [
    Line2D([0], [0], color=improved_color, lw=2, linestyle='-', label='Improved Rank'),
    Line2D([0], [0], color=worsened_color, lw=2, linestyle='--', label='Worsened Rank'),
    Line2D([0], [0], color=no_change_color, lw=1.5, linestyle=':', label='No Change')
]
ax3.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize = 16)


# Panel (c)
gs_c = gs[1, :].subgridspec(2, 4, wspace=0.25, hspace=0.25)
axes_c = [fig.add_subplot(gs_c[i, j]) for i in range(2) for j in range(4)]


ymins, ymaxs = [], []

for idx, (ax, material) in enumerate(zip(axes_c, materials)):
    cv_mean_col, cv_std_col = f'{material}_cv_mean', f'{material}_cv_std'
    if cv_mean_col in df_stats.columns and cv_std_col in df_stats.columns:
        y = df_stats[cv_mean_col]
        yerr = df_stats[cv_std_col]
        ax.plot(x_values, y, marker='o', color='steelblue', linewidth=2)
        ax.fill_between(x_values, y - yerr, y + yerr, alpha=0.2, color='steelblue')
        ax.set_title(material, fontsize=18, fontweight="bold")
                
        ymins.append((y - yerr).min())
        ymaxs.append((y + yerr).max())


    if idx >= 4:
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_values, rotation=90) 
        ax.set_xlabel("Number of Samples")
    else:
        ax.set_xticks(x_values)
        ax.set_xticklabels([])

if ymins and ymaxs:
    common_ymin = min(ymins)
    common_ymax = max(ymaxs)
    for ax in axes_c:
        ax.set_ylim(common_ymin, common_ymax)

fig.text(0.092, 0.30, 'Coefficient of Variation (%)', va='center', rotation='vertical', fontsize=20)

axes_c[0].set_title("(c)", loc="left")

plt.tight_layout()
output_dir = "Output"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "Figure_5.png"), dpi=300, bbox_inches="tight")
plt.show()
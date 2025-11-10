from main import load_db_data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import LogLocator, ScalarFormatter

df = load_db_data(
    f"SELECT scid,name,area,channel_connections,channel_length FROM side_channels WHERE scid IN (1,2,3,5,6,7,8,9,10,11,12,13,17,19,24,26,28,36,37,39,40,41,42,43,48,49);",
    index_col='scid')

plt.figure(figsize=(8, 6))

# Scatter plot
scatter = plt.scatter(
    df['area'],
    df['channel_length'],
    c=df['channel_connections'],
    cmap='viridis',
    s=30,
    alpha=1,
    edgecolor='k'
)

# Custom legend
connection_labels = {0: 'Zero-sided connected', 1: 'One-sided connected', 2: 'Two-sided connected'}
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=df['channel_connections'].min(), vmax=df['channel_connections'].max())
handles = [mpatches.Patch(color=cmap(norm(i)), label=label) for i, label in connection_labels.items()]
plt.legend(handles=handles, title="Channel Type", loc="upper left")

ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')

# ✅ Add *more visible ticks* on both major and minor positions
ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[2,4,6,8,10], numticks=15))
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[2,3,4,5,6,7,8,9,10], numticks=10))


# ✅ Plain number formatting, no “×10³”
formatter = ScalarFormatter()
formatter.set_scientific(False)
formatter.set_useOffset(False)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)

# Force update of tick labels
plt.draw()

# Grid & aesthetics
ax.grid(True, which='both', ls='--', lw=0.5, alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=9)

# Add labels for each point
for scid, row in df.iterrows():
    plt.text(
        row['area'] * 1.01,
        row['channel_length'] * 1.01,
        str(scid),
        fontsize=8,
        ha='left',
        va='bottom',
        alpha=0.7
    )

plt.xlabel('Area')
plt.ylabel('Channel Length')
plt.title('Channel Length vs Area')
plt.show()
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

# Create scatter plot, coloring by 'channel_connections'
scatter = plt.scatter(
    df['area'],
    df['channel_length'],
    c=df['channel_connections'],   # color values
    cmap='viridis',                # color map (try 'plasma', 'coolwarm', etc.)
    s=30,                          # point size
    alpha=1,                     # transparency
    edgecolor='k'                  # black edge for clarity
)

# Create custom legend
connection_labels = {
    0: 'Zero-sided connected',
    1: 'One-sided connected',
    2: 'Two-sided connected'
}

# Get the colormap and normalize values to match the color scale
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=df['channel_connections'].min(), vmax=df['channel_connections'].max())

# Create legend handles manually
handles = [
    mpatches.Patch(color=cmap(norm(i)), label=label)
    for i, label in connection_labels.items()
]

plt.legend(handles=handles, title="Channel Type", loc="upper left")

ax = plt.gca()

# Log scales
ax.set_xscale('log')
ax.set_yscale('log')

# More major and minor ticks
ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=100))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=100))

# Plain number formatting (no 10³)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(style='plain', axis='x')
ax.ticklabel_format(style='plain', axis='y')

# Grid & aesthetics
ax.grid(True, which='both', ls='--', lw=0.5, alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=9)

# Add labels using the index (scid)
for scid, row in df.iterrows():
    plt.text(
        row['area'] * 1.01,
        row['channel_length'] * 1.01,
        str(scid),     # use index label
        fontsize=8,
        ha='left',
        va='bottom',
        alpha=0.7
    )

# Labels and title
plt.xlabel('Area')
plt.ylabel('Channel Length')
plt.title('Channel Length vs Area')

plt.show()
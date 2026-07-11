#!/usr/bin/env python3
"""Figure 5: native MD cloud (both registers, 370K/50ns) + de novo designs + closest MD frame +
closest design. White background, no grid, no title, simple legend."""
import pickle, numpy as np, pandas as pd
import matplotlib.pyplot as plt

results = pickle.load(open('/tmp/md_50ns_results.pkl', 'rb'))
pool = pd.read_pickle('/tmp/pool_scored.pkl')  # forward-threaded flag etc. not required here
df = pd.read_csv('/home/ubuntu/if-mhc/outputs/denovo_scores/per_design.csv')
df = df[df.pep_len == 10].copy()
design_pool = df[~((df.source == 'rfd_recover') & (df.cond != 'fix0'))]

fig, ax = plt.subplots(figsize=(7.5, 7), dpi=200)
fig.patch.set_facecolor('white'); ax.set_facecolor('white')

gig_md, drg_md = results['6AM5'][0], results['6AMU'][0]
ax.scatter(gig_md.toGIG, gig_md.toDRG, s=10, alpha=0.35, color='#2a78d6', edgecolor='none', label='native GIG MD (370K)')
ax.scatter(drg_md.toGIG, drg_md.toDRG, s=10, alpha=0.35, color='#c0392b', edgecolor='none', label='native DRG MD (370K)')

ax.scatter(design_pool.toGIG, design_pool.toDRG, s=6, alpha=0.15, color='0.6', edgecolor='none', label='de novo designs')

# the closest crossing design and its closest MD frame
target_file = 'outputs/rfd_maxcond/pdb/6AM5_k18_44.pdb'
best = design_pool[design_pool.file == target_file].iloc[0]
ax.scatter([best.toGIG], [best.toDRG], s=140, color='magenta', edgecolor='black', linewidth=0.8, zorder=5, label='closest crossing design')

# closest individual MD frame to that design (from the earlier nearest-frame analysis: DRG frame at t=41.4ns)
frame_idx = (drg_md.time_ns - 41.4).abs().idxmin()
closest_frame = drg_md.loc[frame_idx]
ax.scatter([closest_frame.toGIG], [closest_frame.toDRG], s=140, marker='*', color='gold', edgecolor='black', linewidth=0.8, zorder=6, label='closest single MD frame')

ax.plot([best.toGIG, closest_frame.toGIG], [best.toDRG, closest_frame.toDRG], color='black', lw=0.8, ls=':', zorder=4)

ax.plot([0, 16], [0, 16], color='0.75', lw=0.8, ls='--', zorder=1)

for spine in ax.spines.values():
    spine.set_visible(True); spine.set_color('0.3')
ax.set_xlim(-0.5, 16); ax.set_ylim(-0.5, 16)
ax.set_xlabel('Cα RMSD to GIG (Å)')
ax.set_ylabel('Cα RMSD to DRG (Å)')
ax.grid(False)
ax.legend(fontsize=8.5, frameon=True, edgecolor='0.6', loc='upper right')

plt.tight_layout()
out = '/home/ubuntu/if-mhc/figures/fig5_md_cloud/fig5_md_cloud.png'
plt.savefig(out, dpi=200, facecolor='white', bbox_inches='tight')
print('wrote', out)

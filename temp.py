import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


wh = pd.read_csv(r"Desktop/undersampled-subspace-est/WeatherData/weatherHistory.csv")
wh = wh.drop(columns=['Formatted Date'])

wh = wh.dropna()

wh = pd.get_dummies(wh, columns=['Summary'], drop_first=False)
wh = pd.get_dummies(wh, columns=['Daily Summary'], drop_first=False)
wh = pd.get_dummies(wh, columns=['Precip Type'], drop_first=False)

np_wh = np.asarray(wh)

s = np.linalg.svd(np_wh)
fig, ax = plt.subplots(figsize=(6, 4))  # Width = 8, Height = 6 (inches)

# Plot the explained variance
ax.scatter(np.arange(s[1].shape[0]), s[1], marker='o', color='b')

# Add titles and labels
ax.set_xlabel(rf'Singular Value Index', fontsize=15)
ax.set_ylabel('Magnitude', fontsize=15)
# ax.legend(fontsize=12)
# fig.savefig('mouse_svals.png', dpi=500, bbox_inches="tight")
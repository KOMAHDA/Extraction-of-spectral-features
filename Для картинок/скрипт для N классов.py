import pandas as pd, numpy as np, matplotlib.pyplot as plt

path = r"D:\Projects\Extraction-of-spectral-features\Extraction-of-spectral-features\features\features_median_abcde.csv"
df = pd.read_csv(path, sep=';')
print("N =", len(df))
print(df['class'].value_counts())

cols = sorted([c for c in df.columns if c.startswith('wl_')], key=lambda c: int(c[3:]))
wl = 450 + 2 * np.arange(len(cols))
classes = {'NE': '#2a78b5', 'SK': '#3a9a6b', 'BCC': '#d9822b', 'MM': '#c0392b'}

plt.figure(figsize=(10, 5.6), dpi=200)
for cls, color in classes.items():
    x = df.loc[df['class'] == cls, cols].values
    med = np.median(x, axis=0)
    q1, q3 = np.percentile(x, [25, 75], axis=0)
    plt.plot(wl, med, color=color, lw=2.5, label=f"{cls} (n={len(x)})")
    plt.fill_between(wl, q1, q3, color=color, alpha=0.12)

plt.xlabel("Длина волны, нм", fontsize=16)
plt.ylabel("Интенсивность, отн. ед.", fontsize=16)
plt.tick_params(labelsize=14)
plt.legend(fontsize=14, frameon=False)
plt.tight_layout()
plt.savefig("spectra_by_class.png")
from pathlib import Path
import cv2, numpy as np, pandas as pd

ROOT = Path(r"D:\Projects\Целый датасет (Разложенный)")
CSV = r"D:\Projects\Extraction-of-spectral-features\Extraction-of-spectral-features\features\features_median_abcde.csv"
OUT = Path("slide_candidates"); OUT.mkdir(exist_ok=True)
WAVES, SIZE = [450, 544, 700], 700

def read(p, flag=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), flag)

df = pd.read_csv(CSV, sep=';', usecols=['Patient_ID', 'class', 'Area'])
top = df[df['class'].isin(['NE', 'MM', 'BCC', 'SK'])].sort_values('Area', ascending=False)
top = top.groupby('class').head(2)

for _, r in top.iterrows():
    pid, cls = str(r['Patient_ID']), r['class']
    mask = read(next((ROOT / "FINAL_MASKS_ALL").glob(f"{pid}_*")), cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    half = max(int(max(w, h) * 0.9), 60)
    H, W = mask.shape
    cx, cy = x + w // 2, y + h // 2
    x1, y1 = max(cx - half, 0), max(cy - half, 0)
    x2, y2 = min(cx + half, W), min(cy + half, H)
    panels = []
    for wl in WAVES:
        img = read(ROOT / "registered_cubes" / pid / f"{pid}_{wl:06.2f}_nm_aligned.tif")
        crop = img[y1:y2, x1:x2]
        lo, hi = np.percentile(crop, [1, 99])
        crop = np.uint8(np.clip((crop - lo) / max(hi - lo, 1) * 255, 0, 255))
        sx, sy = SIZE / (x2 - x1), SIZE / (y2 - y1)
        view = cv2.cvtColor(cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
        if wl == 544:
            pts = np.int32((c - [x1, y1]) * [sx, sy])
            cv2.drawContours(view, [pts], -1, (80, 220, 80), 3)
        panels.append(view)
    strip = np.hstack(panels)
    cv2.imencode(".png", strip)[1].tofile(str(OUT / f"{cls}_{pid}.png"))
    print(cls, pid, int(r['Area']))
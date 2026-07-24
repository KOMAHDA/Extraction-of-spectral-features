import os
import glob
import re
import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_mutual_information as nmi

# ==========================================
# 1. НАСТРОЙКИ
# ==========================================
RAW_DIR = r"C:\Users\artem\OneDrive\Рабочий стол\Учебка и внеучебка\Научка\Этапы научной работы\Выделение спектральных признаков\Тестовый датасет\693610"
ALIGNED_DIR = os.path.join(RAW_DIR, "registered_fft")

mask_files = glob.glob(os.path.join(RAW_DIR, "*_mask_otsu.png"))
mask_path = mask_files[0]
ref_wavelength_str = re.search(r"(\d+\.\d+_nm)", os.path.basename(mask_path)).group(1)
ref_tif_path = glob.glob(os.path.join(RAW_DIR, f"*{ref_wavelength_str}.tif"))[0]

# ==========================================
# 2. ОПРЕДЕЛЕНИЕ ROI
# ==========================================
mask_bytes = np.fromfile(mask_path, dtype=np.uint8)
mask_img = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
_, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

x, y, w, h = cv2.boundingRect(binary_mask)
PADDING = 40
h_full, w_full = mask_img.shape
x1, y1 = max(0, x - PADDING), max(0, y - PADDING)
x2, y2 = min(w_full, x + w + PADDING), min(h_full, y + h + PADDING)

ref_arr = tifffile.imread(ref_tif_path)
ref_roi = ref_arr[y1:y2, x1:x2].astype(np.float32)
ref_roi_norm = (ref_roi - ref_roi.min()) / (ref_roi.max() - ref_roi.min() + 1e-8)

# ==========================================
# 3. ПОДГОТОВКА И СОРТИРОВКА ФАЙЛОВ
# ==========================================
raw_files = glob.glob(os.path.join(RAW_DIR, "*.tif"))


# Функция для извлечения длины волны (для сортировки)
def extract_wl(filepath):
    match = re.search(r"(\d+\.\d+)_nm", filepath)
    return float(match.group(1)) if match else 0.0


# Сортируем файлы от 450 нм к концу спектра
raw_files = sorted(raw_files, key=extract_wl)

wavelengths = []
nmi_raw_list, nmi_aligned_list = [], []
ssim_raw_list, ssim_aligned_list = [], []

print("\n⚙️ Расчет метрик по каждому каналу...\n")
print(f"{'Длина волны':<13} | {'NMI (ДО -> ПОСЛЕ)':<22} | {'SSIM (ДО -> ПОСЛЕ)':<22}")
print("-" * 65)

# ==========================================
# 4. РАСЧЕТ И ВЫВОД В КОНСОЛЬ
# ==========================================
for raw_tif in raw_files:
    if raw_tif == ref_tif_path:
        continue  # Пропускаем сравнение эталона с самим собой

    filename = os.path.basename(raw_tif)
    aligned_tif = os.path.join(ALIGNED_DIR, filename)

    if not os.path.exists(aligned_tif):
        continue

    wl = extract_wl(filename)

    arr_raw = tifffile.imread(raw_tif)
    arr_aligned = tifffile.imread(aligned_tif)

    raw_roi = arr_raw[y1:y2, x1:x2].astype(np.float32)
    aligned_roi = arr_aligned[y1:y2, x1:x2].astype(np.float32)

    raw_roi_norm = (raw_roi - raw_roi.min()) / (raw_roi.max() - raw_roi.min() + 1e-8)
    aligned_roi_norm = (aligned_roi - aligned_roi.min()) / (aligned_roi.max() - aligned_roi.min() + 1e-8)

    # NMI
    n_raw = nmi(ref_roi, raw_roi, bins=100)
    n_align = nmi(ref_roi, aligned_roi, bins=100)

    # SSIM
    s_raw, _ = ssim(ref_roi_norm, raw_roi_norm, data_range=1.0, full=True)
    s_align, _ = ssim(ref_roi_norm, aligned_roi_norm, data_range=1.0, full=True)

    wavelengths.append(wl)
    nmi_raw_list.append(n_raw)
    nmi_aligned_list.append(n_align)
    ssim_raw_list.append(s_raw)
    ssim_aligned_list.append(s_align)

    # Печать строки таблицы
    print(f"{wl:>6.2f} нм      | {n_raw:.4f} -> {n_align:.4f}     | {s_raw:.4f} -> {s_align:.4f}")

print("-" * 65)

# ==========================================
# 5. ВИЗУАЛИЗАЦИЯ (ГРАФИКИ)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# График NMI
ax1.plot(wavelengths, nmi_raw_list, label='До выравнивания', color='crimson', alpha=0.7)
ax1.plot(wavelengths, nmi_aligned_list, label='После выравнивания (FFT)', color='mediumblue')
ax1.axvline(x=580.0, color='green', linestyle='--', label='Эталон (580 нм)', alpha=0.5)
ax1.set_title('Метрика NMI по длинам волн', fontweight='bold')
ax1.set_xlabel('Длина волны (нм)')
ax1.set_ylabel('Normalized Mutual Information')
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.7)

# График SSIM
ax2.plot(wavelengths, ssim_raw_list, label='До выравнивания', color='crimson', alpha=0.7)
ax2.plot(wavelengths, ssim_aligned_list, label='После выравнивания (FFT)', color='mediumblue')
ax2.axvline(x=580.0, color='green', linestyle='--', label='Эталон (580 нм)', alpha=0.5)
ax2.set_title('Метрика SSIM по длинам волн', fontweight='bold')
ax2.set_xlabel('Длина волны (нм)')
ax2.set_ylabel('Structural Similarity Index')
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()
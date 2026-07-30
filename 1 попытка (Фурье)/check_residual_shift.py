import os
import glob
import re
import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt

# ==========================================
# 1. НАСТРОЙКИ (Укажите пути для одного тестового пациента)
# ==========================================
PATIENT_DIR = r"D:\Projects\Целый датасет (Разложенный)\BCC - базальноклеточная карцинома\709722" 
ALIGNED_DIR = os.path.join(PATIENT_DIR, "registered_fft")
# ОБЯЗАТЕЛЬНО ПОМЕНЯЙТЕ ПУТЬ К МАСКЕ ДЛЯ ВЫБРАННОГО ПАЦИЕНТА!
MASK_PATH = r"D:\Projects\Целый датасет (Разложенный)\BCC - базальноклеточная карцинома\FINAL_MASKS\709722_1_580.00_nm_mask_otsu.png"
REF_WAVELENGTH = "580.00"

# ==========================================
# 2. ПОДГОТОВКА ЭТАЛОНА
# ==========================================
mask_bytes = np.fromfile(MASK_PATH, dtype=np.uint8)
mask_img = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
_, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

x, y, w, h = cv2.boundingRect(binary_mask)
PADDING = 100
h_full, w_full = mask_img.shape
x1, y1 = max(0, x - PADDING), max(0, y - PADDING)
x2, y2 = min(w_full, x + w + PADDING), min(h_full, y + h + PADDING)

ref_tif_path = glob.glob(os.path.join(PATIENT_DIR, f"*{REF_WAVELENGTH}_nm.tif"))[0]
ref_arr = tifffile.imread(ref_tif_path)
ref_roi = ref_arr[y1:y2, x1:x2].astype(np.float32)

# ==========================================
# 3. АНАЛИЗ СДВИГОВ (СЫРЫЕ vs ВЫРОВНЕННЫЕ)
# ==========================================
all_raw_files = glob.glob(os.path.join(PATIENT_DIR, "*.tif"))

def extract_wl(filepath):
    match = re.search(r"(\d+\.\d+)_nm", os.path.basename(filepath))
    return float(match.group(1)) if match else 0.0

all_raw_files = sorted(all_raw_files, key=extract_wl)

wavelengths = []
raw_shifts_x, raw_shifts_y = [], []
aligned_shifts_x, aligned_shifts_y = [], []

print("Вычисление сдвигов (в пикселях)...")

for raw_path in all_raw_files:
    filename = os.path.basename(raw_path)
    aligned_path = os.path.join(ALIGNED_DIR, filename)
    
    if not os.path.exists(aligned_path):
        continue
        
    wl = extract_wl(filename)
    wavelengths.append(wl)
    
    # Загружаем сырой и выровненный кадры
    raw_arr = tifffile.imread(raw_path)
    aligned_arr = tifffile.imread(aligned_path)
    
    raw_roi = raw_arr[y1:y2, x1:x2].astype(np.float32)
    aligned_roi = aligned_arr[y1:y2, x1:x2].astype(np.float32)
    
    # Считаем смещение сырого кадра (насколько пациент сдвинулся реально)
    shift_r, _ = cv2.phaseCorrelate(raw_roi, ref_roi)
    raw_shifts_x.append(shift_r[0])
    raw_shifts_y.append(shift_r[1])
    
    # Считаем смещение ВЫРОВНЕННОГО кадра (остаточный сдвиг)
    # В идеале алгоритм должен вернуть (0, 0)
    shift_a, _ = cv2.phaseCorrelate(aligned_roi, ref_roi)
    aligned_shifts_x.append(shift_a[0])
    aligned_shifts_y.append(shift_a[1])

# ==========================================
# 4. ПОСТРОЕНИЕ ГРАФИКОВ
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# График по оси X
ax1.plot(wavelengths, raw_shifts_x, label='Исходный сдвиг (Сырые данные)', color='crimson', alpha=0.6)
ax1.plot(wavelengths, aligned_shifts_x, label='Остаточный сдвиг (Выровненные данные)', color='mediumblue')
ax1.axhline(0, color='black', linestyle='--', linewidth=1)
ax1.set_title('Ось X: Анализ остаточного сдвига (в пикселях)')
ax1.set_ylabel('Сдвиг по X (пиксели)')
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.7)

# График по оси Y
ax2.plot(wavelengths, raw_shifts_y, label='Исходный сдвиг (Сырые данные)', color='crimson', alpha=0.6)
ax2.plot(wavelengths, aligned_shifts_y, label='Остаточный сдвиг (Выровненные данные)', color='mediumblue')
ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.set_title('Ось Y: Анализ остаточного сдвига (в пикселях)')
ax2.set_xlabel('Длина волны (нм)')
ax2.set_ylabel('Сдвиг по Y (пиксели)')
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()

# Вывод статистики в консоль
print("\nСтатистика остаточного сдвига:")
print(f"Средняя ошибка (X): {np.mean(np.abs(aligned_shifts_x)):.2f} пикс. (Было: {np.mean(np.abs(raw_shifts_x)):.2f})")
print(f"Средняя ошибка (Y): {np.mean(np.abs(aligned_shifts_y)):.2f} пикс. (Было: {np.mean(np.abs(raw_shifts_y)):.2f})")
print(f"Максимальная ошибка: {max(np.max(np.abs(aligned_shifts_x)), np.max(np.abs(aligned_shifts_y))):.2f} пикс.")
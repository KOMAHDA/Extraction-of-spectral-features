import os
import re
import glob
import cv2
import numpy as np
import tifffile

# ==========================================
# 1. НАСТРОЙКИ И ПУТИ
# ==========================================
DATA_DIR = r"C:\Users\artem\OneDrive\Рабочий стол\Учебка и внеучебка\Научка\Этапы научной работы\Выделение спектральных признаков\Тестовый датасет\693610"
OUTPUT_DIR = os.path.join(DATA_DIR, "registered_fft")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ищем маску и эталон
mask_files = glob.glob(os.path.join(DATA_DIR, "*_mask_otsu.png"))
if not mask_files:
    raise FileNotFoundError(f"❌ Маска не найдена в папке:\n{DATA_DIR}")

mask_path = mask_files[0]
mask_name = os.path.basename(mask_path)
ref_wavelength_str = re.search(r"(\d+\.\d+_nm)", mask_name).group(1)
ref_tif_path = glob.glob(os.path.join(DATA_DIR, f"*{ref_wavelength_str}.tif"))[0]

print(f"🎯 Опорный канал: {os.path.basename(ref_tif_path)}")

# ==========================================
# 2. ПОДГОТОВКА ЭТАЛОНА И ОПРЕДЕЛЕНИЕ ROI (Bounding Box)
# ==========================================
# Читаем маску через numpy (обход бага с кириллицей)
mask_bytes = np.fromfile(mask_path, dtype=np.uint8)
mask_img = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
_, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

# Находим координаты родинки (Bounding Box)
x, y, w, h = cv2.boundingRect(binary_mask)

# Добавляем "отступ" (padding) в 50 пикселей вокруг родинки,
# чтобы захватить кусок кожи для контекста, но обрезать статические углы
PADDING = 50
h_full, w_full = mask_img.shape
x1, y1 = max(0, x - PADDING), max(0, y - PADDING)
x2, y2 = min(w_full, x + w + PADDING), min(h_full, y + h + PADDING)

print(f"✂️ Вырезаем зону интереса для расчетов: X[{x1}:{x2}], Y[{y1}:{y2}]")

# Читаем эталонный кадр и вырезаем только ROI
ref_arr = tifffile.imread(ref_tif_path)
ref_float = ref_arr.astype(np.float32)
ref_roi = ref_float[y1:y2, x1:x2]

# Создаем окно Ханна для подавления краевых артефактов при FFT
# cv2.createHanningWindow принимает размер в формате (width, height)
hanning = cv2.createHanningWindow((ref_roi.shape[1], ref_roi.shape[0]), cv2.CV_32F)
ref_roi_hann = ref_roi * hanning

# ==========================================
# 3. ОСНОВНОЙ ЦИКЛ: ФАЗОВАЯ КОРРЕЛЯЦИЯ
# ==========================================
all_tif_files = glob.glob(os.path.join(DATA_DIR, "*.tif"))
print(f"\n🚀 Запуск выравнивания через FFT ({len(all_tif_files)} каналов)...\n")

for i, moving_tif_path in enumerate(all_tif_files, 1):
    filename = os.path.basename(moving_tif_path)
    output_path = os.path.join(OUTPUT_DIR, filename)

    moving_arr = tifffile.imread(moving_tif_path)
    max_do = moving_arr.max()

    if moving_tif_path == ref_tif_path:
        print(f"[{i}/{len(all_tif_files)}] 🟢 {filename} | ЭТАЛОН")
        tifffile.imwrite(output_path, moving_arr, photometric='minisblack')
        continue

    # Вырезаем ТУ ЖЕ самую зону интереса из сдвинутого канала
    moving_float = moving_arr.astype(np.float32)
    moving_roi = moving_float[y1:y2, x1:x2]
    moving_roi_hann = moving_roi * hanning

    # МАГИЯ: Фазовая корреляция считает чистый сдвиг (dx, dy)
    shift, response = cv2.phaseCorrelate(moving_roi_hann, ref_roi_hann)
    dx, dy = shift

    # Формируем матрицу аффинного сдвига: [[1, 0, dx], [0, 1, dy]]
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    # Применяем сдвиг ко ВСЕМУ оригинальному изображению
    aligned_arr = cv2.warpAffine(
        moving_arr,
        M,
        (w_full, h_full),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    max_posle = aligned_arr.max()
    print(
        f"[{i}/{len(all_tif_files)}] 🔄 {filename} | Сдвиг: X={dx:.1f}, Y={dy:.1f} | Max ДО: {max_do} -> ПОСЛЕ: {max_posle}")

    # Сохраняем физически точный TIF
    tifffile.imwrite(output_path, aligned_arr, photometric='minisblack')

print("\n🎉 Выравнивание FFT успешно завершено!")
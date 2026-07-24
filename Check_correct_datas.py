import os
import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt

# ==========================================
# 1. НАСТРОЙКИ ПУТЕЙ
# ==========================================
# Базовая директория
RAW_DIR = r"C:\Users\artem\OneDrive\Рабочий стол\Учебка и внеучебка\Научка\Этапы научной работы\Выделение спектральных признаков\Тестовый датасет\667160"
ALIGNED_DIR = os.path.join(RAW_DIR, "registered_fft")

# Пути к TIF-файлам (518 нм)
RAW_TIF_PATH = os.path.join(RAW_DIR, "667160_1_518.00_nm.tif")
ALIGNED_TIF_PATH = os.path.join(ALIGNED_DIR, "667160_1_518.00_nm.tif")

# Пути к конкретным маскам, которые ты сделал
MASK_RAW_PATH = r"C:\Users\artem\OneDrive\Рабочий стол\123\667160_1_518.00_nm.png"
MASK_ALIGNED_PATH = r"C:\Users\artem\OneDrive\Рабочий стол\123\667160_1_518.00_nm_sm.png"


# ==========================================
# 2. ФУНКЦИЯ ЗАГРУЗКИ МАСКИ (Обход кириллицы)
# ==========================================
def load_binary_mask(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Маска не найдена: {path}")

    mask_bytes = np.fromfile(path, dtype=np.uint8)
    mask_img = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask_img, 127, 1, cv2.THRESH_BINARY)
    return binary_mask.astype(bool)


# ==========================================
# 3. ВЫЧИСЛЕНИЯ
# ==========================================
print("⚙️ Загрузка изображений и масок...")
mask_raw = load_binary_mask(MASK_RAW_PATH)
mask_aligned = load_binary_mask(MASK_ALIGNED_PATH)

arr_raw = tifffile.imread(RAW_TIF_PATH)
arr_aligned = tifffile.imread(ALIGNED_TIF_PATH)

# Считаем среднее значение строго по площади масок
mean_raw = np.mean(arr_raw[mask_raw])
mean_aligned = np.mean(arr_aligned[mask_aligned])
diff = abs(mean_raw - mean_aligned)

print("\n" + "=" * 40)
print(f"📊 Средняя яркость ДО выравнивания: {mean_raw:.3f}")
print(f"📊 Средняя яркость ПОСЛЕ FFT:       {mean_aligned:.3f}")
print(f"📉 Абсолютная ошибка интерполяции:  {diff:.3f}")
print("=" * 40 + "\n")

# ==========================================
# 4. ВИЗУАЛИЗАЦИЯ (Столбчатая диаграмма)
# ==========================================
labels = ['Оригинал\n(Несмещенная маска)', 'После выравнивания\n(Смещенная маска)']
values = [mean_raw, mean_aligned]

plt.figure(figsize=(7, 5))
bars = plt.bar(labels, values, color=['crimson', 'mediumblue'], alpha=0.85, width=0.5)

# Добавляем точные значения над столбцами
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 2, f"{yval:.2f}",
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Оценка искажений радиометрии при FFT (518 нм)', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Средняя интенсивность пикселей', fontsize=12)
plt.ylim(0, max(values) * 1.15)  # Динамический отступ сверху для текста
plt.grid(axis='y', linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()
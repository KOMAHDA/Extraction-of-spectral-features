import os
import re
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# --- 1. Настройки путей ---
BASE_DIR = r"D:\Projects\Целый датасет (Разложенный)"
TEST_DISEASE = "BCC - базальноклеточная карцинома"

def apply_dog_filter(image, k_small=5, k_large=101):
    """
    Применяет Difference of Gaussians и усиливает локальный контраст (CLAHE).
    """
    img_float = image.astype(np.float32)
    
    # 1. DoG фильтрация
    blur_small = cv2.GaussianBlur(img_float, (k_small, k_small), 0)
    blur_large = cv2.GaussianBlur(img_float, (k_large, k_large), 0)
    dog = blur_small - blur_large
    
    # 2. Нормализация в 8-bit (фон станет примерно 128, т.е. серым)
    dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 3. Адаптивное усиление локального контраста
    # clipLimit контролирует силу контраста (можно увеличить до 3.0, если будет тускло)
    # tileGridSize разбивает изображение на квадраты 8x8 для локальной обработки
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    dog_boosted = clahe.apply(dog_norm)
    
    return dog_boosted

def parse_patient_data(disease_path):
    """
    Парсит маски и собирает пути к tiff-кадрам.
    """
    masks_dir = os.path.join(disease_path, "FINAL_MASKS")
    mask_files = glob(os.path.join(masks_dir, "*.png"))
    
    patients = {}
    
    # Регулярка для вытаскивания ID, опционального номера (capture) и длины волны
    # Поддерживает форматы: 10246_1_612.00_nm... и 41841_580.00_nm...
    regex = re.compile(r"(\d+)_?(?:(\d+)_)?(\d+\.\d+)_nm")

    for mask_path in mask_files:
        filename = os.path.basename(mask_path)
        match = regex.search(filename)
        
        if not match:
            continue
            
        patient_id = match.group(1)
        capture_id = match.group(2) if match.group(2) else "1"
        anchor_wave = float(match.group(3))
        
        # Ищем папку с кадрами пациента
        patient_folder = os.path.join(disease_path, patient_id)
        if not os.path.exists(patient_folder):
            continue
            
        # Собираем все tiff файлы и их длины волн
        tif_files = glob(os.path.join(patient_folder, "*.tif"))
        frames = []
        for tif in tif_files:
            tif_name = os.path.basename(tif)
            # Извлекаем длину волны из названия tif (например, 10246_1_450.00_nm.tif)
            wave_match = re.search(r"(\d+\.\d+)_nm", tif_name)
            if wave_match:
                wave = float(wave_match.group(1))
                frames.append((wave, tif))
                
        # Сортируем кадры по длине волны
        frames.sort(key=lambda x: x[0])
        
        patients[patient_id] = {
            "mask_path": mask_path,
            "anchor_wave": anchor_wave,
            "frames": frames
        }
        
    return patients

if __name__ == "__main__":
    disease_path = os.path.join(BASE_DIR, TEST_DISEASE)
    print(f"Парсинг папки: {disease_path}...")
    
    patients_data = parse_patient_data(disease_path)
    
    # Жестко задаем нужного пациента
    test_id = "709722"
    
    if test_id not in patients_data:
        print(f"Ошибка: Пациент {test_id} не найден или у него нет масок/кадров!")
        exit(1)
        
    test_data = patients_data[test_id]
    
    # 1. Находим экстремальные и опорную длины волн
    min_wave, min_frame_path = test_data["frames"][0]
    max_wave, max_frame_path = test_data["frames"][-1]
    anchor_wave = test_data["anchor_wave"]
    
    # Ищем путь к опорному кадру
    anchor_frame_path = next(f[1] for f in test_data["frames"] if f[0] == anchor_wave)
    
    print(f"\nСтресс-тест пациента {test_id}")
    print(f"Опорная волна: {anchor_wave} нм")
    print(f"Мин. волна: {min_wave} нм")
    print(f"Макс. волна: {max_wave} нм")
    
    def process_and_plot(ax_raw, ax_clean, file_path, wave_name):
        # Читаем файл с поддержкой кириллицы
        file_bytes = np.fromfile(file_path, dtype=np.uint8)
        raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        
        if raw_img is None:
            ax_raw.set_title(f"Ошибка чтения: {wave_name} нм")
            return
            
        # Применяем фильтр
        cleaned_img = apply_dog_filter(raw_img)
        
        # Отрисовка
        ax_raw.imshow(raw_img, cmap='gray')
        ax_raw.set_title(f"Сырой: {wave_name} нм")
        ax_raw.axis('off')
        
        ax_clean.imshow(cleaned_img, cmap='gray')
        ax_clean.set_title(f"DoG + CLAHE: {wave_name} нм")
        ax_clean.axis('off')

    # Создаем сетку 3x2 для сравнения трех кадров
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    
    # 1-й ряд - Опорная волна
    process_and_plot(axes[0, 0], axes[0, 1], anchor_frame_path, anchor_wave)
    # 2-й ряд - Минимальная волна
    process_and_plot(axes[1, 0], axes[1, 1], min_frame_path, min_wave)
    # 3-й ряд - Максимальная волна
    process_and_plot(axes[2, 0], axes[2, 1], max_frame_path, max_wave)
    
    plt.tight_layout()
    plt.show()
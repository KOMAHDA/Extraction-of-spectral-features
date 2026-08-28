import os
import glob
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- НАСТРОЙКИ ПУТЕЙ ---
CUBES_DIR = Path(r"D:\Projects\Целый датасет (Разложенный)\registered_cubes")
MASKS_DIR = Path(r"D:\Projects\Целый датасет (Разложенный)\FINAL_MASKS_ALL")
ABCDE_CSV = Path(r"D:\Projects\Extraction-of-spectral-features\Extraction-of-spectral-features\abcde_features_aligned.csv")
OUTPUT_DIR = Path(r"D:\Projects\Extraction-of-spectral-features\Extraction-of-spectral-features")

def cv2_imread_ru(path, flag=cv2.IMREAD_GRAYSCALE):
    """Чтение картинок с кириллицей в путях"""
    try:
        arr = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(arr, flag)
    except Exception:
        return None

def get_wavelength_from_filename(filename):
    """Извлекает длину волны для сортировки"""
    match = re.search(r'(\d+\.\d+)_nm', filename)
    return float(match.group(1)) if match else 0.0

def run_multi_extraction():
    if not ABCDE_CSV.exists():
        print(f"ОШИБКА: Файл {ABCDE_CSV} не найден!")
        return

    # Загружаем базу с ABCDE и диагнозами (используем разделитель ';')
    df_abcde = pd.read_csv(ABCDE_CSV, sep=';')
    df_abcde['Patient_ID'] = df_abcde['Patient_ID'].astype(str)
    
    print(f"Загружено {len(df_abcde)} пациентов из файла ABCDE.")
    
    results_median = []
    results_std = []
    results_deriv = []

    for _, row in tqdm(df_abcde.iterrows(), total=len(df_abcde), desc="Извлечение спектров"):
        pat_id = row['Patient_ID']
        diag_class = row['Disease_Class']
        
        # 1. Ищем маску
        mask_search = list(MASKS_DIR.glob(f"*{pat_id}*.*"))
        if not mask_search:
            continue
            
        mask = cv2_imread_ru(mask_search[0], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        # Бинаризуем маску
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 2. Ищем и сортируем кадры куба
        pat_folder = CUBES_DIR / pat_id
        if not pat_folder.exists():
            continue
            
        tif_files = list(pat_folder.glob("*_nm*.tif"))
        if len(tif_files) < 140: # Пропускаем, если кадров слишком мало
            continue
            
        tif_files = sorted(tif_files, key=lambda x: get_wavelength_from_filename(x.name))
        
        # 3. Извлекаем сырые спектральные значения
        median_spectrum = []
        std_spectrum = []
        
        for tif_path in tif_files:
            img = cv2_imread_ru(tif_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Берем пиксели только внутри маски
                pixels = img[mask_bin == 255]
                if len(pixels) > 0:
                    median_spectrum.append(np.median(pixels))
                    std_spectrum.append(np.std(pixels))
                else:
                    median_spectrum.append(0.0)
                    std_spectrum.append(0.0)
            else:
                median_spectrum.append(0.0)
                std_spectrum.append(0.0)
                
        # 4. Вычисляем первую производную (разница между соседними каналами)
        median_array = np.array(median_spectrum)
        deriv_spectrum = np.diff(median_array, prepend=median_array[0])
        
        # 5. Формируем базовый словарь с метаданными и ABCDE
        base_record = {
            'Patient_ID': pat_id,
            'class': diag_class,
            'A_Asymmetry': row['A_Asymmetry'],
            'B_Border': row['B_Border'],
            'C_Color_Std': row['C_Color_Std'],
            'D_Diameter': row['D_Diameter'],
            'E_Eccentricity': row['E_Eccentricity'],
            'Area': row['Area'],
            'frames_count': len(median_spectrum) # Служебная колонка для фильтрации
        }
        
        # Копируем базу для каждого метода
        rec_median = base_record.copy()
        rec_std = base_record.copy()
        rec_deriv = base_record.copy()
        
        # Разворачиваем 151 значение в колонки
        for i in range(len(median_spectrum)):
            rec_median[f'wl_{i+1}'] = round(median_spectrum[i], 4)
            rec_std[f'wl_{i+1}'] = round(std_spectrum[i], 4)
            rec_deriv[f'wl_{i+1}'] = round(deriv_spectrum[i], 4)
            
        results_median.append(rec_median)
        results_std.append(rec_std)
        results_deriv.append(rec_deriv)

    # 6. Создаем датафреймы и очищаем аномалии длины
    df_median = pd.DataFrame(results_median)
    df_std = pd.DataFrame(results_std)
    df_deriv = pd.DataFrame(results_deriv)
    
    if df_median.empty:
        print("Ошибка: не удалось извлечь данные.")
        return

    # Оставляем только те кубы, где количество кадров равно моде (убираем обрезки)
    mode_frames = df_median['frames_count'].mode()[0]
    
    for df, name in zip([df_median, df_std, df_deriv], ['median', 'std', 'deriv']):
        df_clean = df[df['frames_count'] == mode_frames].copy()
        df_clean.drop(columns=['frames_count'], inplace=True)
        
        out_path = OUTPUT_DIR / f"features_{name}_abcde.csv"
        df_clean.to_csv(out_path, index=False, sep=';')
        print(f"Сохранен файл {out_path.name} (Пациентов: {len(df_clean)}, Признаков: {df_clean.shape[1]-2})")

if __name__ == "__main__":
    run_multi_extraction()
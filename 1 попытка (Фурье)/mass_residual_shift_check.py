import os
import re
import glob
import cv2
import numpy as np
import tifffile
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

DATASET_ROOT = r"D:\Projects\Целый датасет (Разложенный)"
REPORT_PATH = os.path.join(DATASET_ROOT, "smart_residual_shift_report.csv")
RESIDUAL_THRESHOLD = 5.0

def process_single_patient(task_data):
    pathology_name, patient_id, wavelength, patient_dir, aligned_dir, mask_path = task_data
    
    mask_bytes = np.fromfile(mask_path, dtype=np.uint8)
    mask_img = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

    x, y, w, h = cv2.boundingRect(binary_mask)
    PADDING = 100
    h_full, w_full = mask_img.shape
    x1, y1 = max(0, x - PADDING), max(0, y - PADDING)
    x2, y2 = min(w_full, x + w + PADDING), min(h_full, y + h + PADDING)

    ref_tif_files = glob.glob(os.path.join(patient_dir, f"*{wavelength}_nm.tif"))
    if not ref_tif_files: 
        return None
    
    ref_tif_path = ref_tif_files[0]
    ref_arr = tifffile.imread(ref_tif_path)
    ref_roi = ref_arr[y1:y2, x1:x2].astype(np.float32)

    raw_tif_files = glob.glob(os.path.join(patient_dir, "*.tif"))
    
    raw_shifts_x, raw_shifts_y = [], []
    valid_aligned_x, valid_aligned_y = [], [] # Сюда пишем только валидные (видимые) каналы

    for raw_path in raw_tif_files:
        if raw_path == ref_tif_path: continue
            
        filename = os.path.basename(raw_path)
        aligned_path = os.path.join(aligned_dir, filename)
        
        if not os.path.exists(aligned_path): continue
            
        # Извлекаем длину волны текущего кадра
        match = re.search(r"(\d+\.\d+)_nm", filename)
        current_wl = float(match.group(1)) if match else 0.0

        raw_arr = tifffile.imread(raw_path)
        aligned_arr = tifffile.imread(aligned_path)
        
        raw_roi = raw_arr[y1:y2, x1:x2].astype(np.float32)
        aligned_roi = aligned_arr[y1:y2, x1:x2].astype(np.float32)
        
        shift_r, _ = cv2.phaseCorrelate(raw_roi, ref_roi)
        shift_a, _ = cv2.phaseCorrelate(aligned_roi, ref_roi)
        
        raw_shifts_x.append(abs(shift_r[0]))
        raw_shifts_y.append(abs(shift_r[1]))
        
        # УМНЫЙ ФИЛЬТР: Оцениваем остаточный сдвиг только там, где родинка физически видна (< 680 нм)
        # На каналах > 680 нм скрипт проверки ловит цифровой шум, мы это игнорируем
        if current_wl <= 680.0:
            valid_aligned_x.append(abs(shift_a[0]))
            valid_aligned_y.append(abs(shift_a[1]))

    if not valid_aligned_x:
        return None

    mean_raw_x = np.mean(raw_shifts_x)
    mean_raw_y = np.mean(raw_shifts_y)
    
    mean_align_x = np.mean(valid_aligned_x)
    mean_align_y = np.mean(valid_aligned_y)
    max_align_x = np.max(valid_aligned_x)
    max_align_y = np.max(valid_aligned_y)
    
    status = "ПРЕДУПРЕЖДЕНИЕ (> 5 пикс)" if (max_align_x > RESIDUAL_THRESHOLD or max_align_y > RESIDUAL_THRESHOLD) else "ОК"

    return [
        pathology_name, patient_id, wavelength, 
        f"{mean_raw_x:.2f}", f"{mean_raw_y:.2f}",
        f"{mean_align_x:.2f}", f"{mean_align_y:.2f}", 
        f"{max_align_x:.2f}", f"{max_align_y:.2f}", status
    ]

def run_smart_mass_check():
    print("Собираем список пациентов для УМНОЙ проверки...")
    tasks = []
    
    for pathology_name in os.listdir(DATASET_ROOT):
        pathology_dir = os.path.join(DATASET_ROOT, pathology_name)
        if not os.path.isdir(pathology_dir) or pathology_name == "FINAL_MASKS_ALL": continue
            
        masks_dir = os.path.join(pathology_dir, "FINAL_MASKS")
        if not os.path.exists(masks_dir): continue
            
        for mask_file in os.listdir(masks_dir):
            if mask_file.startswith('.'): continue
                
            match = re.search(r"^(\d+).*?(\d+\.\d+)_nm", mask_file)
            if not match: continue
                
            patient_id = match.group(1)
            wavelength = match.group(2)
            
            patient_dir = os.path.join(pathology_dir, patient_id)
            aligned_dir = os.path.join(patient_dir, "registered_fft")
            mask_path = os.path.join(masks_dir, mask_file)
            
            if os.path.exists(aligned_dir):
                tasks.append((pathology_name, patient_id, wavelength, patient_dir, aligned_dir, mask_path))

    print(f"Найдено пациентов: {len(tasks)}")
    print("Запускаем анализ...\n")

    with open(REPORT_PATH, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow([
            'Патология', 'Пациент', 'Эталон (нм)', 
            'Исходный сдвиг (X)', 'Исходный сдвиг (Y)',
            'Остаток Ср. X (<680нм)', 'Остаток Ср. Y (<680нм)', 
            'Остаток Макс. X (<680нм)', 'Остаток Макс. Y (<680нм)', 'Статус'
        ])

        processed_count = 0
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_single_patient, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                processed_count += 1
                if result:
                    writer.writerow(result)
                    f.flush()
                print(f"[ПРОГРЕСС] Обработано: {processed_count} / {len(tasks)}", end='\r')

    print(f"\n\nУмный анализ завершен! Отчет: {REPORT_PATH}")

if __name__ == "__main__":
    run_smart_mass_check()
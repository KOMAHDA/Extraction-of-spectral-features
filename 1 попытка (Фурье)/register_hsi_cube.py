import os
import re
import glob
import cv2
import numpy as np
import tifffile
from scipy.signal import medfilt

# ==========================================
# 1. НАСТРОЙКИ
# ==========================================
DATASET_ROOT = r"D:\Projects\Целый датасет (Разложенный)"

def extract_wl(filepath):
    match = re.search(r"(\d+\.\d+)_nm", os.path.basename(filepath))
    return float(match.group(1)) if match else 0.0

# ==========================================
# 2. ФУНКЦИЯ ДВУХПРОХОДНОГО ВЫРАВНИВАНИЯ
# ==========================================
def align_patient_cube_iterative(patient_dir, mask_path, ref_tif_path, wavelength):
    output_dir = os.path.join(patient_dir, "registered_fft")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Загрузка маски и определение ROI
    mask_bytes = np.fromfile(mask_path, dtype=np.uint8)
    mask_img = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

    x, y, w, h = cv2.boundingRect(binary_mask)
    PADDING = 100 
    h_full, w_full = mask_img.shape
    x1, y1 = max(0, x - PADDING), max(0, y - PADDING)
    x2, y2 = min(w_full, x + w + PADDING), min(h_full, y + h + PADDING)

    # 2. Подготовка эталона
    ref_arr = tifffile.imread(ref_tif_path)
    ref_roi = ref_arr[y1:y2, x1:x2].astype(np.float32)
    ref_roi_blurred = cv2.GaussianBlur(ref_roi, (15, 15), 0)
    hanning = cv2.createHanningWindow((ref_roi.shape[1], ref_roi.shape[0]), cv2.CV_32F)
    ref_roi_hann = ref_roi_blurred * hanning

    # 3. Загрузка всего куба в оперативную память для скорости
    all_tif_files = sorted(glob.glob(os.path.join(patient_dir, "*.tif")), key=extract_wl)
    cube_arrays = [tifffile.imread(p) for p in all_tif_files]
    
    # ==========================================
    # ПРОХОД 1: ГРУБОЕ ВЫРАВНИВАНИЕ (Coarse)
    # ==========================================
    raw_dx, raw_dy = [], []
    for arr in cube_arrays:
        moving_roi = arr[y1:y2, x1:x2].astype(np.float32)
        moving_roi_blurred = cv2.GaussianBlur(moving_roi, (15, 15), 0)
        moving_roi_hann = moving_roi_blurred * hanning
        
        shift, _ = cv2.phaseCorrelate(moving_roi_hann, ref_roi_hann)
        raw_dx.append(shift[0])
        raw_dy.append(shift[1])

    coarse_dx = medfilt(raw_dx, kernel_size=7)
    coarse_dy = medfilt(raw_dy, kernel_size=7)

    # ==========================================
    # ПРОХОД 2: ТОЧНОЕ ДОРАВНИВАНИЕ (Fine)
    # ==========================================
    fine_dx, fine_dy = [], []
    for i, arr in enumerate(cube_arrays):
        # Применяем грубый сдвиг
        M_coarse = np.float32([[1, 0, coarse_dx[i]], [0, 1, coarse_dy[i]]])
        coarse_aligned = cv2.warpAffine(arr, M_coarse, (w_full, h_full), flags=cv2.INTER_LINEAR)
        
        # Вырезаем ROI из УЖЕ сдвинутого изображения
        coarse_roi = coarse_aligned[y1:y2, x1:x2].astype(np.float32)
        coarse_roi_blurred = cv2.GaussianBlur(coarse_roi, (15, 15), 0)
        coarse_roi_hann = coarse_roi_blurred * hanning
        
        shift_fine, _ = cv2.phaseCorrelate(coarse_roi_hann, ref_roi_hann)
        
        fx, fy = shift_fine
        
        # ЗАЩИТА ОТ МУСОРА НА ТЕМНЫХ КАНАЛАХ
        # Так как кадр уже грубо выровнен, остаточный сдвиг физически не может превышать 15 пикселей.
        # Если FFT выдает больше 15 - это стопроцентный шум, обнуляем его.
        if abs(fx) > 15.0 or abs(fy) > 15.0:
            fx, fy = 0.0, 0.0
            
        fine_dx.append(fx)
        fine_dy.append(fy)
        
    fine_dx_filtered = medfilt(fine_dx, kernel_size=5)
    fine_dy_filtered = medfilt(fine_dy, kernel_size=5)

    # ==========================================
    # ПРОХОД 3: ФИНАЛЬНОЕ ПРИМЕНЕНИЕ И СОХРАНЕНИЕ
    # ==========================================
    for i, filepath in enumerate(all_tif_files):
        filename = os.path.basename(filepath)
        output_path = os.path.join(output_dir, filename)
        
        if filepath == ref_tif_path:
            tifffile.imwrite(output_path, cube_arrays[i], photometric='minisblack')
            continue
            
        # Складываем векторы
        total_dx = coarse_dx[i] + fine_dx_filtered[i]
        total_dy = coarse_dy[i] + fine_dy_filtered[i]
        
        M_final = np.float32([[1, 0, total_dx], [0, 1, total_dy]])
        
        aligned_arr = cv2.warpAffine(
            cube_arrays[i], 
            M_final, 
            (w_full, h_full), 
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        tifffile.imwrite(output_path, aligned_arr, photometric='minisblack')

# ==========================================
# 3. ГЛАВНЫЙ ЦИКЛ ПО ДАТАСЕТУ
# ==========================================
def process_entire_dataset():
    print(f"Запуск двухпроходной пакетной обработки датасета: {DATASET_ROOT}\n")
    
    for pathology_name in os.listdir(DATASET_ROOT):
        pathology_dir = os.path.join(DATASET_ROOT, pathology_name)
        
        if not os.path.isdir(pathology_dir) or pathology_name == "FINAL_MASKS_ALL":
            continue
            
        masks_dir = os.path.join(pathology_dir, "FINAL_MASKS")
        if not os.path.exists(masks_dir):
            continue
            
        print(f"Обработка класса: {pathology_name}")
        
        for mask_file in os.listdir(masks_dir):
            if mask_file.startswith('.'): continue
                
            match = re.search(r"^(\d+).*?(\d+\.\d+)_nm", mask_file)
            if not match: continue
                
            patient_id = match.group(1)
            wavelength = match.group(2)
            
            patient_dir = os.path.join(pathology_dir, patient_id)
            mask_path = os.path.join(masks_dir, mask_file)
            
            if not os.path.exists(patient_dir): continue
                
            ref_tif_files = glob.glob(os.path.join(patient_dir, f"*{wavelength}_nm.tif"))
            if not ref_tif_files: continue
                
            ref_tif_path = ref_tif_files[0]
            
            print(f"   - Пациент {patient_id} | Эталон: {wavelength} нм ...")
            
            try:
                align_patient_cube_iterative(patient_dir, mask_path, ref_tif_path, wavelength)
            except Exception as e:
                print(f"   [ОШИБКА] При обработке {patient_id}: {e}")
                
    print("\nОбработка всего датасета завершена!")

if __name__ == "__main__":
    process_entire_dataset()
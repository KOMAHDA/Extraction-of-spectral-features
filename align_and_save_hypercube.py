import os
import re
import cv2
import numpy as np
from glob import glob
import pandas as pd

# ==========================================
# 1. ФУНКЦИИ ПРЕДОБРАБОТКИ И ПАРСИНГА
# ==========================================

def apply_dog_filter(image, k_small=5, k_large=101):
    img_float = image.astype(np.float32)
    blur_small = cv2.GaussianBlur(img_float, (k_small, k_small), 0)
    blur_large = cv2.GaussianBlur(img_float, (k_large, k_large), 0)
    dog = blur_small - blur_large
    dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(dog_norm)

def parse_patient_data(disease_path):
    masks_dir = os.path.join(disease_path, "FINAL_MASKS")
    if not os.path.exists(masks_dir):
        return {}
    
    mask_files = glob(os.path.join(masks_dir, "*.png"))
    patients = {}
    regex = re.compile(r"(\d+)_?(?:(\d+)_)?(\d+\.\d+)_nm")

    for mask_path in mask_files:
        filename = os.path.basename(mask_path)
        match = regex.search(filename)
        if not match: continue
            
        patient_id = match.group(1)
        anchor_wave = float(match.group(3))
        
        patient_folder = os.path.join(disease_path, patient_id)
        if not os.path.exists(patient_folder): continue
            
        tif_files = glob(os.path.join(patient_folder, "*.tif"))
        frames = []
        for tif in tif_files:
            tif_name = os.path.basename(tif)
            wave_match = re.search(r"(\d+\.\d+)_nm", tif_name)
            if wave_match:
                wave = float(wave_match.group(1))
                frames.append((wave, tif))
                
        if not frames: continue
        frames.sort(key=lambda x: x[0])
        
        patients[patient_id] = {
            "mask_path": mask_path,
            "anchor_wave": anchor_wave,
            "frames": frames
        }
    return patients

# ==========================================
# 2. КЛАСС ТРЕКЕРА
# ==========================================

class HypercubeTracker:
    def __init__(self, initial_frame, initial_bbox):
        self.w, self.h = initial_bbox[2], initial_bbox[3]
        self.template = initial_frame[initial_bbox[1]:initial_bbox[1]+self.h, 
                                      initial_bbox[0]:initial_bbox[0]+self.w]
        
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        center_x = initial_bbox[0] + self.w / 2.0
        center_y = initial_bbox[1] + self.h / 2.0
        self.kalman.statePre = np.array([[center_x], [center_y], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[center_x], [center_y], [0], [0]], np.float32)
        
        self.global_anchor_center = (center_x, center_y)
        self.last_valid_center = (center_x, center_y)
        self.last_update_wave = None

    def track_next_frame(self, frame_clean, current_wave, fallback_threshold=0.40):
        prediction = self.kalman.predict()
        pred_x, pred_y = int(prediction[0][0]), int(prediction[1][0])
        
        pad = 40 
        H, W = frame_clean.shape
        
        roi_x1 = max(0, int(pred_x - self.w/2 - pad))
        roi_y1 = max(0, int(pred_y - self.h/2 - pad))
        roi_x2 = min(W, int(pred_x + self.w/2 + pad))
        roi_y2 = min(H, int(pred_y + self.h/2 + pad))
        
        roi = frame_clean[roi_y1:roi_y2, roi_x1:roi_x2]
        
        max_val = 0.0
        if roi.shape[0] >= self.h and roi.shape[1] >= self.w:
            res = cv2.matchTemplate(roi, self.template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val > fallback_threshold:
                found_center_x = roi_x1 + max_loc[0] + self.w/2.0
                found_center_y = roi_y1 + max_loc[1] + self.h/2.0
                self.last_valid_center = (found_center_x, found_center_y)
            else:
                found_center_x, found_center_y = self.last_valid_center
        else:
            found_center_x, found_center_y = self.last_valid_center
                
        if max_val > fallback_threshold:
            measurement = np.array([[np.float32(found_center_x)], [np.float32(found_center_y)]])
            self.kalman.correct(measurement)
        
        if max_val > 0.8:
            if self.last_update_wave is None or abs(current_wave - self.last_update_wave) > 20:
                top_left_x = int(found_center_x - self.w/2)
                top_left_y = int(found_center_y - self.h/2)
                if top_left_x >= 0 and top_left_y >= 0 and top_left_x+self.w <= W and top_left_y+self.h <= H:
                    self.template = frame_clean[top_left_y:top_left_y+self.h, top_left_x:top_left_x+self.w]
                    self.last_update_wave = current_wave
                
        dx = found_center_x - self.global_anchor_center[0]
        dy = found_center_y - self.global_anchor_center[1]
        
        return dx, dy, max_val

# ==========================================
# 3. АЛИГНМЕНТ И РАСЧЕТ МЕТРИК КАЧЕСТВА
# ==========================================

def process_and_evaluate_patient(patient_id, p_data, base_dir):
    frames = p_data["frames"]
    anchor_wave = p_data["anchor_wave"]
    mask_path = p_data["mask_path"]
    
    anchor_idx = next((i for i, f in enumerate(frames) if f[0] == anchor_wave), 0)
    anchor_path = frames[anchor_idx][1]
    
    # Загрузка маски и опорного кадра
    mask_bytes = np.fromfile(mask_path, dtype=np.uint8)
    mask = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
    if mask is None: return None
    
    anchor_bytes = np.fromfile(anchor_path, dtype=np.uint8)
    raw_anchor = cv2.imdecode(anchor_bytes, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if raw_anchor is None: return None
    
    clean_anchor = apply_dog_filter(raw_anchor)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    
    pad = 20
    H, W = clean_anchor.shape
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
    bbox = (x1, y1, x2 - x1, y2 - y1)
    
    # Эталонный шаблон родинки из опорного кадра для метрик
    reference_template = clean_anchor[y1:y1+h, x1:x1+w]
    
    # Трекинг вперед и назад от опорного кадра
    shifts_log = {anchor_wave: (0.0, 0.0)}
    
    def run_tracking(start_idx, step):
        tracker = HypercubeTracker(clean_anchor, bbox)
        curr = start_idx + step
        while 0 <= curr < len(frames):
            wave, path = frames[curr]
            f_bytes = np.fromfile(path, dtype=np.uint8)
            raw_f = cv2.imdecode(f_bytes, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
            if raw_f is not None:
                clean_f = apply_dog_filter(raw_f)
                thresh = 0.35 if (wave < 520 or wave > 720) else 0.50
                dx, dy, _ = tracker.track_next_frame(clean_f, wave, fallback_threshold=thresh)
                shifts_log[wave] = (dx, dy)
            curr += step

    run_tracking(anchor_idx, 1)
    run_tracking(anchor_idx, -1)
    
    # Выравнивание, сохранение и сбор метрик (NCC)
    output_dir = os.path.join(base_dir, "registered_cubes", patient_id)
    os.makedirs(output_dir, exist_ok=True)
    
    anchor_dx, anchor_dy = shifts_log.get(anchor_wave, (0.0, 0.0))
    ncc_scores = []
    
    for wave, path in frames:
        f_bytes = np.fromfile(path, dtype=np.uint8)
        raw_img = cv2.imdecode(f_bytes, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        if raw_img is None: continue
            
        dx, dy = shifts_log.get(wave, (0.0, 0.0))
        diff_dx = dx - anchor_dx
        diff_dy = dy - anchor_dy
        
        M = np.float32([[1, 0, -diff_dx], [0, 1, -diff_dy]])
        rows, cols = raw_img.shape
        aligned_img = cv2.warpAffine(raw_img, M, (cols, rows), 
                                     flags=cv2.INTER_LINEAR, 
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Считаем метрику качества (NCC) выровненного кропа по сравнению с эталоном
        clean_aligned = apply_dog_filter(aligned_img)
        aligned_roi = clean_aligned[y1:y1+h, x1:x1+w]
        if aligned_roi.shape == reference_template.shape and reference_template.size > 0:
            res = cv2.matchTemplate(aligned_roi, reference_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            ncc_scores.append(max_val)
        
        filename = f"{patient_id}_{wave:06.2f}_nm_aligned.tif"
        save_path = os.path.join(output_dir, filename)
        is_success, buffer = cv2.imencode(".tif", aligned_img)
        if is_success:
            buffer.tofile(save_path)
            
    mean_ncc = float(np.mean(ncc_scores)) if ncc_scores else 0.0
    min_ncc = float(np.min(ncc_scores)) if ncc_scores else 0.0
    
    return {
        "patient_id": patient_id,
        "total_frames": len(frames),
        "mean_ncc": mean_ncc,
        "min_ncc": min_ncc,
        "status": "SUCCESS" if mean_ncc > 0.65 else "WARNING_LOW_NCC"
    }

# ==========================================
# 4. ГЛАВНЫЙ КОНВЕЙЕР ПАКЕТНОЙ ОБРАБОТКИ
# ==========================================

if __name__ == "__main__":
    BASE_DIR = r"D:\Projects\Целый датасет (Разложенный)"
    
    # Находим все папки с болезнями в корне датасета
    disease_folders = [os.path.join(BASE_DIR, d) for d in os.listdir(BASE_DIR) 
                       if os.path.isdir(os.path.join(BASE_DIR, d)) and d != "registered_cubes"]
    
    quality_report = []
    
    for disease_path in disease_folders:
        disease_name = os.path.basename(disease_path)
        print(f"\nОбработка категории: {disease_name}")
        
        patients_data = parse_patient_data(disease_path)
        print(f"Найдено пациентов: {len(patients_data)}")
        
        for patient_id, p_data in patients_data.items():
            print(f" -> Обработка пациента {patient_id}...")
            try:
                res = process_and_evaluate_patient(patient_id, p_data, BASE_DIR)
                if res:
                    res["disease"] = disease_name
                    quality_report.append(res)
                    print(f"    [УСПЕХ] Средний NCC: {res['mean_ncc']:.3f} | Статус: {res['status']}")
            except Exception as e:
                print(f"    [ОШИБКА] Пациент {patient_id} упал с исключением: {e}")
                quality_report.append({
                    "patient_id": patient_id,
                    "disease": disease_name,
                    "total_frames": len(p_data["frames"]),
                    "mean_ncc": 0.0,
                    "min_ncc": 0.0,
                    "status": f"ERROR: {str(e)}"
                })
                
    # Сохраняем сводный отчет с метриками качества
    report_df = pd.DataFrame(quality_report)
    report_path = os.path.join(BASE_DIR, "alignment_quality_report.csv")
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"\nПакетная обработка завершена. Сводный отчет сохранен в:\n{report_path}")
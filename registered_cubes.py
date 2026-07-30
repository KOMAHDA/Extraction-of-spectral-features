import os
import re
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# ==========================================
# 1. ФУНКЦИИ ПРЕДОБРАБОТКИ И ПАРСИНГА
# ==========================================

def apply_dog_filter(image, k_small=5, k_large=101):
    """
    Удаляет виньетирование (Difference of Gaussians) и локально усиливает 
    контраст высокочастотных деталей (родинки, текстуры кожи) с помощью CLAHE.
    """
    img_float = image.astype(np.float32)
    
    blur_small = cv2.GaussianBlur(img_float, (k_small, k_small), 0)
    blur_large = cv2.GaussianBlur(img_float, (k_large, k_large), 0)
    dog = blur_small - blur_large
    
    dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(dog_norm)


def parse_patient_data(disease_path):
    masks_dir = os.path.join(disease_path, "FINAL_MASKS")
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
                
        frames.sort(key=lambda x: x[0])
        
        patients[patient_id] = {
            "mask_path": mask_path,
            "anchor_wave": anchor_wave,
            "frames": frames
        }
        
    return patients

# ==========================================
# 2. КЛАСС ПРЕДСКАЗАТЕЛЬНОГО ТРЕКЕРА (СТАБИЛЬНЫЙ БАЗИС)
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

    def track_next_frame(self, frame_clean, current_wave, fallback_threshold=0.50):
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
                # Если конфиденс низкий — удерживаем последнюю валидную позицию, не пуская вразнос
                found_center_x, found_center_y = self.last_valid_center
                found_center_x, found_center_y, max_val = self._global_search(frame_clean)
        else:
            found_center_x, found_center_y = self.last_valid_center
            found_center_x, found_center_y, max_val = self._global_search(frame_clean)
                
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

    def _global_search(self, frame_clean):
        res = cv2.matchTemplate(frame_clean, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        center_x = max_loc[0] + self.w/2.0
        center_y = max_loc[1] + self.h/2.0
        return center_x, center_y, max_val


# ==========================================
# 3. ГЛАВНЫЙ БЛОК ИСПОЛНЕНИЯ
# ==========================================

if __name__ == "__main__":
    BASE_DIR = r"D:\Projects\Целый датасет (Разложенный)"
    TEST_DISEASE = "BCC - базальноклеточная карцинома"
    TEST_PATIENT = "709722"

    print(f"Инициализация обработки пациента {TEST_PATIENT}...")
    
    disease_path = os.path.join(BASE_DIR, TEST_DISEASE)
    patients_data = parse_patient_data(disease_path)
    
    if TEST_PATIENT not in patients_data:
        print(f"Ошибка: Пациент {TEST_PATIENT} не найден или отсутствуют маски/кадры.")
        exit(1)
        
    p_data = patients_data[TEST_PATIENT]
    frames = p_data["frames"]
    anchor_wave = p_data["anchor_wave"]
    mask_path = p_data["mask_path"]
    
    anchor_idx = next(i for i, f in enumerate(frames) if f[0] == anchor_wave)
    anchor_path = frames[anchor_idx][1]
    
    mask_bytes = np.fromfile(mask_path, dtype=np.uint8)
    mask = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
    
    anchor_bytes = np.fromfile(anchor_path, dtype=np.uint8)
    raw_anchor = cv2.imdecode(anchor_bytes, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    
    clean_anchor = apply_dog_filter(raw_anchor)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    
    pad = 20
    H, W = clean_anchor.shape
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
    bbox = (x1, y1, x2 - x1, y2 - y1)
    
    print(f"Опорный кадр: {anchor_wave} нм. Базовый шаблон: {bbox[2]}x{bbox[3]} пикс.")
    
    shifts_log = {anchor_wave: (0.0, 0.0)}
    
    debug_dir = f"debug_tracking_{TEST_PATIENT}"
    os.makedirs(debug_dir, exist_ok=True)
    
    def process_sequence(start_idx, step, direction_name):
        print(f"\n--- Запуск трекинга: {direction_name} ---")
        tracker = HypercubeTracker(clean_anchor, bbox)
        
        current_idx = start_idx + step
        while 0 <= current_idx < len(frames):
            wave, path = frames[current_idx]
            
            f_bytes = np.fromfile(path, dtype=np.uint8)
            raw_f = cv2.imdecode(f_bytes, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
            clean_f = apply_dog_filter(raw_f)
            
            # Адаптивный порог: для синей/ИК зон снижаем порог до 0.40
            thresh = 0.40 if (wave < 520 or wave > 720) else 0.60
            dx, dy, conf = tracker.track_next_frame(clean_f, wave, fallback_threshold=thresh)
            shifts_log[wave] = (dx, dy)
            
            vis_img = cv2.normalize(raw_f, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
            
            center_x = tracker.global_anchor_center[0] + dx
            center_y = tracker.global_anchor_center[1] + dy
            box_x = int(center_x - tracker.w / 2)
            box_y = int(center_y - tracker.h / 2)
            
            color = (0, 255, 0) if conf >= thresh else (0, 0, 255)
            cv2.rectangle(vis_img, (box_x, box_y), (box_x + tracker.w, box_y + tracker.h), color, 3)
            
            status_text = "OK" if conf >= thresh else "BLIND/INERTIA"
            text = f"{wave}nm | Conf: {conf:.2f} | {status_text}"
            cv2.putText(vis_img, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            save_path = os.path.join(debug_dir, f"frame_{wave:05.1f}.jpg")
            is_success, buffer = cv2.imencode(".jpg", vis_img)
            if is_success:
                buffer.tofile(save_path)
            
            if current_idx % 10 == 0 or conf < thresh:
                status = "ЛОКАЛЬНЫЙ ПОИСК" if conf >= thresh else "ГЛОБАЛЬНЫЙ ОТКАТ"
                print(f"Волна {wave:6.1f} нм | dx: {dx:6.1f}, dy: {dy:6.1f} | Конфиденс: {conf:.2f} ({status})")
                
            current_idx += step

    process_sequence(anchor_idx, 1, "Вправо (к ИК спектру)")
    process_sequence(anchor_idx, -1, "Влево (к УФ спектру)")
    
    sorted_waves = sorted(shifts_log.keys())
    dx_vals = [shifts_log[w][0] for w in sorted_waves]
    dy_vals = [shifts_log[w][1] for w in sorted_waves]
    
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_waves, dx_vals, label='Сдвиг по X (dx)', color='blue', marker='.', markersize=4, linestyle='-')
    plt.plot(sorted_waves, dy_vals, label='Сдвиг по Y (dy)', color='red', marker='.', markersize=4, linestyle='-')
    plt.axvline(x=anchor_wave, color='green', linestyle='--', label=f'Опорный кадр ({anchor_wave} нм)')
    
    plt.title(f"Траектория макро/микро-сдвигов гиперкуба (Пациент {TEST_PATIENT})")
    plt.xlabel("Длина волны (нм)")
    plt.ylabel("Вектор сдвига (в пикселях)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
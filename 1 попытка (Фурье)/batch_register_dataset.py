import os
import re
import glob
import cv2
import numpy as np
import tifffile

# ==========================================
# 1. ПУТИ ДЛЯ ТЕСТОВОГО ПАЦИЕНТА (Подставьте вашего, например 709722)
# ==========================================
PATIENT_DIR = r"D:\Projects\Целый датасет (Разложенный)\BCC - базальноклеточная карцинома\709722" 
ALIGNED_DIR = os.path.join(PATIENT_DIR, "registered_masked")
MASK_PATH = r"D:\Projects\Целый датасет (Разложенный)\BCC - базальноклеточная карцинома\FINAL_MASKS\709722_1_580.00_nm_mask_otsu.png"
REF_WAVELENGTH = "580.00"

def extract_wl(filepath):
    match = re.search(r"(\d+\.\d+)_nm", os.path.basename(filepath))
    return float(match.group(1)) if match else 0.0

def process_single_patient():
    print(f"Запуск Маскированного Шаблонного Поиска: {PATIENT_DIR}")
    os.makedirs(ALIGNED_DIR, exist_ok=True)

    # 1. Читаем маску и находим СТРОГИЙ bounding box родинки
    mask_bytes = np.fromfile(MASK_PATH, dtype=np.uint8)
    mask_img = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(binary_mask)
    H_img, W_img = mask_img.shape

    # 2. Ищем эталонный файл
    all_tif_files = sorted(glob.glob(os.path.join(PATIENT_DIR, "*.tif")), key=extract_wl)
    ref_tif_files = glob.glob(os.path.join(PATIENT_DIR, f"*{REF_WAVELENGTH}_nm.tif"))
    if not ref_tif_files: return
    ref_tif_path = ref_tif_files[0]
    ref_idx = all_tif_files.index(ref_tif_path)

    # 3. Подготовка ШАБЛОНА И МАСКИ
    ref_arr = tifffile.imread(ref_tif_path).astype(np.float32)
    template = ref_arr[y:y+h, x:x+w]
    
    # Формируем альфа-маску (1.0 для родинки, 0.0 для всего остального)
    template_mask = binary_mask[y:y+h, x:x+w].astype(np.float32) / 255.0

    # Радиус поиска (максимально возможный физический сдвиг)
    PAD = 40
    S_x1, S_y1 = max(0, x - PAD), max(0, y - PAD)
    S_x2, S_y2 = min(W_img, x + w + PAD), min(H_img, y + h + PAD)

    final_dx = np.zeros(len(all_tif_files))
    final_dy = np.zeros(len(all_tif_files))

    # Вспомогательная функция поиска с откатом
    def get_shift(moving_arr, last_dx, last_dy):
        search_area = moving_arr[S_y1:S_y2, S_x1:S_x2].astype(np.float32)
        
        # Если родинка уехала за край картинки (защита от ошибок)
        if search_area.shape[0] < template.shape[0] or search_area.shape[1] < template.shape[1]:
            return last_dx, last_dy
            
        # Маскированный поиск (игнорируем тени и кожу!)
        res = cv2.matchTemplate(search_area, template, cv2.TM_CCORR_NORMED, mask=template_mask)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        dx = x - (S_x1 + max_loc[0])
        dy = y - (S_y1 + max_loc[1])
        
        # ЛОГИКА ФОЛБЭКА:
        # Если уверенность совпадения высокая (>0.85) и прыжок не безумный (<25 пикс)
        if max_val > 0.85 and abs(dx - last_dx) < 25 and abs(dy - last_dy) < 25:
            return dx, dy
        else:
            # Иначе берем ПОСЛЕДНИЙ ВАЛИДНЫЙ СДВИГ
            return last_dx, last_dy

    print("Шаг 1: Выравнивание от эталона ВПЕРЕД (>580 нм)...")
    curr_dx, curr_dy = 0.0, 0.0
    for i in range(ref_idx + 1, len(all_tif_files)):
        moving_arr = tifffile.imread(all_tif_files[i])
        curr_dx, curr_dy = get_shift(moving_arr, curr_dx, curr_dy)
        final_dx[i], final_dy[i] = curr_dx, curr_dy

    print("Шаг 2: Выравнивание от эталона НАЗАД (<580 нм)...")
    curr_dx, curr_dy = 0.0, 0.0
    for i in range(ref_idx - 1, -1, -1):
        moving_arr = tifffile.imread(all_tif_files[i])
        curr_dx, curr_dy = get_shift(moving_arr, curr_dx, curr_dy)
        final_dx[i], final_dy[i] = curr_dx, curr_dy

    print("Сохранение идеальных кадров...")
    for i, moving_tif_path in enumerate(all_tif_files):
        filename = os.path.basename(moving_tif_path)
        output_path = os.path.join(ALIGNED_DIR, filename)
        moving_arr = tifffile.imread(moving_tif_path)
        
        if moving_tif_path == ref_tif_path:
            tifffile.imwrite(output_path, moving_arr, photometric='minisblack')
            continue
            
        M = np.float32([[1, 0, final_dx[i]], [0, 1, final_dy[i]]])
        aligned_arr = cv2.warpAffine(
            moving_arr, M, (W_img, H_img), 
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        tifffile.imwrite(output_path, aligned_arr, photometric='minisblack')

    print("Готово. Пожалуйста, откройте папку registered_masked и посмотрите.")

if __name__ == "__main__":
    process_single_patient()
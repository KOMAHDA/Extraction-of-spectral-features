import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Импортируем нашу функцию из предыдущего шага
# (убедитесь, что она доступна в файле)
def apply_dog_filter(image, k_small=5, k_large=101):
    img_float = image.astype(np.float32)
    blur_small = cv2.GaussianBlur(img_float, (k_small, k_small), 0)
    blur_large = cv2.GaussianBlur(img_float, (k_large, k_large), 0)
    dog = blur_small - blur_large
    dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(dog_norm)

def get_template_from_mask(mask_path, raw_anchor_path):
    """
    Загружает маску, находит Bounding Box родинки и вырезает шаблон из очищенного кадра.
    """
    # Читаем маску (поддержка кириллицы)
    mask_bytes = np.fromfile(mask_path, dtype=np.uint8)
    mask = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Читаем сырой опорный кадр
    anchor_bytes = np.fromfile(raw_anchor_path, dtype=np.uint8)
    raw_anchor = cv2.imdecode(anchor_bytes, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    
    # Очищаем опорный кадр от статики
    clean_anchor = apply_dog_filter(raw_anchor)
    
    # Находим контуры на маске
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Ошибка: На маске не найдено контуров!")
        return None, None, None
        
    # Берем самый большой контур (это и есть родинка)
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # Добавляем небольшой отступ (padding), чтобы захватить текстуру кожи вокруг
    padding = 20
    H, W = clean_anchor.shape
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(W, x + w + padding)
    y2 = min(H, y + h + padding)
    
    # Вырезаем эталонный шаблон
    template = clean_anchor[y1:y2, x1:x2]
    bbox = (x1, y1, x2 - x1, y2 - y1) # (x, y, width, height)
    
    return clean_anchor, template, bbox

# --- ТЕСТОВЫЙ ЗАПУСК ---
if __name__ == "__main__":
    # Замените пути на ваши актуальные
    PATIENT_DIR = r"D:\Projects\Целый датасет (Разложенный)\BCC - базальноклеточная карцинома\101932"
    MASK_PATH = r"D:\Projects\Целый датасет (Разложенный)\BCC - базальноклеточная карцинома\FINAL_MASKS\101932_1_580.00_nm_mask_otsu.png"
    
    # Путь к опорному кадру (580 нм) и следующему соседнему кадру (например, 582 нм)
    ANCHOR_PATH = os.path.join(PATIENT_DIR, "101932_1_580.00_nm.tif")
    NEXT_FRAME_PATH = os.path.join(PATIENT_DIR, "101932_1_582.00_nm.tif") # Подставьте существующий соседний кадр
    
    print("Подготовка шаблона...")
    clean_anchor, template, bbox = get_template_from_mask(MASK_PATH, ANCHOR_PATH)
    
    x, y, w, h = bbox
    print(f"Размер вырезанного шаблона: {w}x{h} пикселей. Координаты: x={x}, y={y}")
    
    # --- Шаг поиска на следующем кадре ---
    # 1. Читаем и очищаем следующий кадр
    next_bytes = np.fromfile(NEXT_FRAME_PATH, dtype=np.uint8)
    raw_next = cv2.imdecode(next_bytes, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    clean_next = apply_dog_filter(raw_next)
    
    # 2. Выполняем нормализованную кросс-корреляцию (идеально работает на DoG изображениях)
    res = cv2.matchTemplate(clean_next, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # max_loc - это верхний левый угол найденного совпадения
    new_x, new_y = max_loc
    
    # Считаем физический вектор сдвига
    shift_dx = new_x - x
    shift_dy = new_y - y
    print(f"Уверенность (Correlation): {max_val:.4f}")
    print(f"Вектор сдвига: dx={shift_dx}, dy={shift_dy}")
    
    # --- Визуализация ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(clean_anchor, cmap='gray')
    # Рисуем рамку на опорном кадре
    rect1 = plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none', lw=2)
    axes[0].add_patch(rect1)
    axes[0].set_title("Опорный кадр (580 нм)")
    
    axes[1].imshow(template, cmap='gray')
    axes[1].set_title("Извлеченный шаблон")
    
    axes[2].imshow(clean_next, cmap='gray')
    # Рисуем рамку на найденном месте
    rect2 = plt.Rectangle((new_x, new_y), w, h, edgecolor='g', facecolor='none', lw=2)
    axes[2].add_patch(rect2)
    axes[2].set_title(f"Следующий кадр (сдвиг {shift_dx}, {shift_dy})")
    
    plt.tight_layout()
    plt.show()
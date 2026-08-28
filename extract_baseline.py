import os
import glob
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# --- Блок безопасного чтения путей с кириллицей ---
def cv2_imread_ru(path, flag=cv2.IMREAD_GRAYSCALE):
    try:
        img_array = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(img_array, flag)
    except Exception as e:
        print(f"Ошибка чтения {path}: {e}")
        return None

# --- Настройка путей ---
ORIGINAL_DIR = r"D:\Projects\Целый датасет (Разложенный)\Изначальный датасет"
CUBES_DIR = r"D:\Projects\Целый датасет (Разложенный)\registered_cubes"
MASKS_DIR = r"D:\Projects\Целый датасет (Разложенный)\FINAL_MASKS_ALL"
OUTPUT_DIR = r"D:\Projects\Extraction-of-spectral-features\Extraction-of-spectral-features"

# Списки злокачественных и доброкачественных (для мета-анализа)
MALIGNANT = ['MM', 'BCC', 'SCC']
BENIGN = ['DF', 'PP', 'HE', 'SK', 'NE']

def extract_labels_from_original_dataset():
    """Сканирует папки изначального датасета и собирает диагнозы."""
    labels_dict = {}
    print("Сканирование изначального датасета для сбора диагнозов...")
    
    if not os.path.exists(ORIGINAL_DIR):
        print(f"ОШИБКА: Папка {ORIGINAL_DIR} не найдена.")
        return labels_dict

    for class_folder in os.listdir(ORIGINAL_DIR):
        class_path = os.path.join(ORIGINAL_DIR, class_folder)
        if not os.path.isdir(class_path):
            continue
            
        # Извлекаем аббревиатуру (первое слово до пробела/тире, например "BCC")
        class_name = class_folder.split()[0].upper()
        
        # Получаем все папки пациентов внутри этого класса
        for patient_folder in os.listdir(class_path):
            if os.path.isdir(os.path.join(class_path, patient_folder)):
                labels_dict[patient_folder] = class_name
                
    print(f"Успешно собраны метки для {len(labels_dict)} пациентов.")
    return labels_dict

def get_wavelength_from_filename(filename):
    """Вытаскивает длину волны из имени файла (например, 1387_450.00_nm_aligned.tif -> 450.0)"""
    match = re.search(r'(\d+\.\d+)_nm', filename)
    if match:
        return float(match.group(1))
    return 0.0

def extract_spectral_signatures(labels_dict):
    """Извлекает средние значения пикселей по маске для каждого спектрального канала."""
    features = []
    
    patients = [d for d in os.listdir(CUBES_DIR) if os.path.isdir(os.path.join(CUBES_DIR, d))]
    print(f"Найдено папок кубов: {len(patients)}. Начинаем извлечение спектров...")
    
    for pat_id in patients:
        # Проверяем, есть ли диагноз для этого пациента
        if pat_id not in labels_dict:
            continue
            
        diag_class = labels_dict[pat_id]
        is_malignant = 1 if diag_class in MALIGNANT else 0
        
        # 1. Ищем маску (предполагаем, что в названии файла маски есть ID пациента)
        mask_search = glob.glob(os.path.join(MASKS_DIR, f"*{pat_id}*.*"))
        if not mask_search:
            print(f"Пропуск {pat_id}: не найдена маска.")
            continue
        mask_path = mask_search[0]
        
        mask = cv2_imread_ru(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        # Бинаризируем маску (защита от артефактов сжатия)
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Проверяем, не пустая ли маска
        if cv2.countNonZero(mask_bin) == 0:
            print(f"Пропуск {pat_id}: маска полностью черная.")
            continue
        
        # 2. Ищем и сортируем кадры куба
        base_dir = os.path.join(CUBES_DIR, pat_id)
        tif_files = glob.glob(os.path.join(base_dir, f"*_nm*.tif")) # Ищем файлы с длинами волн
        
        if len(tif_files) < 140: # Защита от неполных кубов (у вас их около 151)
            print(f"Пропуск {pat_id}: слишком мало кадров ({len(tif_files)} шт.)")
            continue
            
        # Сортируем строго по длине волны
        tif_files.sort(key=get_wavelength_from_filename)
        
        # 3. Извлекаем спектр
        spectrum = []
        for tif_path in tif_files:
            img = cv2_imread_ru(tif_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Считаем среднюю яркость только там, где маска белая (255)
                mean_val = cv2.mean(img, mask=mask_bin)[0]
                spectrum.append(mean_val)
            else:
                spectrum.append(0.0) # Заглушка на случай битого кадра
                
        # Сохраняем в список
        record = {
            'patient_id': pat_id,
            'class': diag_class,
            'is_malignant': is_malignant,
            'frames_count': len(spectrum)
        }
        
        # Разворачиваем спектр в колонки (wl_1, wl_2, ..., wl_151)
        for i, val in enumerate(spectrum):
            record[f'wl_{i+1}'] = round(val, 4)
            
        features.append(record)
        
    df = pd.DataFrame(features)
    
    # Отбрасываем кубы, у которых количество каналов отличается от самого частого (моды), 
    # чтобы матрица признаков была строго одинаковой ширины
    mode_frames = df['frames_count'].mode()[0]
    df = df[df['frames_count'] == mode_frames].copy()
    df.drop(columns=['frames_count'], inplace=True)
    
    return df

def run_ml_baseline(df):
    """Обучает классификаторы и визуализирует пространство признаков."""
    print(f"\n=== ЗАПУСК МАШИННОГО ОБУЧЕНИЯ (Baseline) ===")
    print(f"Размер итогового датасета: {df.shape[0]} изображений.")
    
    # Подготавливаем X и y
    feature_cols = [c for c in df.columns if c.startswith('wl_')]
    X = df[feature_cols].values
    y = df['class'].values
    
    # Стандартизация (очень важна для SVM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- 1. Визуализация t-SNE ---
    print("Генерация t-SNE визуализации...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=y, palette="Set1", s=100, alpha=0.8
    )
    plt.title("t-SNE проекция гиперспектральных признаков кожи", fontsize=16)
    plt.legend(title="Класс патологии", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    tsne_path = os.path.join(OUTPUT_DIR, "tsne_spectral_features.png")
    plt.savefig(tsne_path, dpi=300)
    print(f"График t-SNE сохранен: {tsne_path}")
    
    # --- 2. Кросс-валидация ---
    print("\nОценка моделей (Кросс-валидация Stratified 5-Fold):")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Модель 1: SVM
    svm_model = SVC(kernel='rbf', class_weight='balanced')
    svm_scores = cross_val_score(svm_model, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"SVM Accuracy: {svm_scores.mean():.3f} (±{svm_scores.std()*2:.3f})")
    
    # Модель 2: Random Forest
    rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf_scores = cross_val_score(rf_model, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"Random Forest Accuracy: {rf_scores.mean():.3f} (±{rf_scores.std()*2:.3f})")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    xgb_model = XGBClassifier(
        n_estimators=300,        # Количество деревьев
        learning_rate=0.05,      # Скорость обучения
        max_depth=5,             # Глубина деревьев
        subsample=0.8,           # Доля объектов для обучения одного дерева (защита от переобучения)
        colsample_bytree=0.8,    # Доля признаков для одного дерева
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1                # Использовать все ядра процессора для ускорения
    )
    
    xgb_scores = cross_val_score(xgb_model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
    print(f"XGBoost Accuracy: {xgb_scores.mean():.3f} (±{xgb_scores.std()*2:.3f})")


if __name__ == "__main__":
    # 1. Извлекаем метки
    labels_dict = extract_labels_from_original_dataset()
    
    # 2. Вытаскиваем спектральные векторы
    features_df = extract_spectral_signatures(labels_dict)
    
    if features_df.empty:
        print("Ошибка: не удалось извлечь спектральные признаки.")
    else:
        # Сохраняем датасет в CSV для нейросетей
        out_csv = os.path.join(OUTPUT_DIR, "spectral_signatures.csv")
        features_df.to_csv(out_csv, index=False, sep=';')
        print(f"\nИзвлечение завершено. Датасет сохранен в: {out_csv}")
        
        # 3. Запускаем Baseline
        run_ml_baseline(features_df)
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- 1. Пути к данным ---
SPECTRAL_CSV = r"D:\Projects\Extraction-of-spectral-features\Extraction-of-spectral-features\spectral_signatures.csv"
# ОБНОВЛЕННЫЙ ПУТЬ к новому файлу:
ABCDE_CSV = r"D:\Projects\Extraction-of-spectral-features\Extraction-of-spectral-features\abcde_features_aligned.csv"

def run_fusion_experiment():
    # --- 2. Загрузка данных ---
    print("Загрузка спектральных и морфологических данных...")
    df_spec = pd.read_csv(SPECTRAL_CSV, sep=';')
    
    # ИСПРАВЛЕННЫЙ РАЗДЕЛИТЕЛЬ: читаем через точку с запятой
    df_abcde = pd.read_csv(ABCDE_CSV, sep=';') 

    # Приводим ID к строковому типу для надежного слияния
    df_spec['patient_id'] = df_spec['patient_id'].astype(str)
    df_abcde['Patient_ID'] = df_abcde['Patient_ID'].astype(str)

    # --- 3. Слияние (Early Fusion) ---
    # Объединяем таблицы по ID пациента (inner join оставит только тех, кто есть в обеих базах)
    df_merged = pd.merge(df_spec, df_abcde, left_on='patient_id', right_on='Patient_ID', how='inner')
    print(f"Успешно объединено пациентов: {len(df_merged)}")

    if len(df_merged) == 0:
        print("Ошибка: Не удалось сопоставить ID пациентов. Проверьте форматы файлов.")
        return

    # --- 4. Подготовка признаков ---
    # Собираем спектральные колонки (wl_1 ... wl_151)
    spec_cols = [c for c in df_merged.columns if c.startswith('wl_')]
    # Собираем морфологические колонки
    abcde_cols = ['A_Asymmetry', 'B_Border', 'C_Color_Std', 'D_Diameter', 'E_Eccentricity', 'Area']
    
    # Итоговый вектор признаков (Спектр + Пространство)
    feature_cols = spec_cols + abcde_cols
    
    X = df_merged[feature_cols].values
    y = df_merged['class'].values # Метки берем из спектрального файла

    # Кодируем текстовые метки (BCC, MM) в числа для XGBoost
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Стандартизация (Уравнивает вес площади и отражения)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 5. Обучение и Оценка ---
    print(f"\n=== ЗАПУСК МАШИННОГО ОБУЧЕНИЯ (Слияние: Спектр + ABCDE) ===")
    print(f"Количество признаков у одного пациента: {X.shape[1]} ({len(spec_cols)} спектр + {len(abcde_cols)} геометрия)")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # SVM
    svm_model = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    svm_scores = cross_val_score(svm_model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
    print(f"SVM Accuracy:        {svm_scores.mean():.3f} (±{svm_scores.std()*2:.3f})")

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf_scores = cross_val_score(rf_model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
    print(f"Random Forest Acc:   {rf_scores.mean():.3f} (±{rf_scores.std()*2:.3f})")

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1
    )
    xgb_scores = cross_val_score(xgb_model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
    print(f"XGBoost Accuracy:    {xgb_scores.mean():.3f} (±{xgb_scores.std()*2:.3f})")

if __name__ == "__main__":
    run_fusion_experiment()
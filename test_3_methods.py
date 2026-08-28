import pandas as pd
import warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Игнорируем варнинги pandas и sklearn для чистоты консоли
warnings.filterwarnings('ignore')

# --- Настройки ---
DATA_DIR = Path(r"D:\Projects\Extraction-of-spectral-features\Extraction-of-spectral-features")

FILES_TO_TEST = {
    "1. МЕДИАНА (Защита от волос и бликов)": "features_median_abcde.csv",
    "2. СТАНДАРТНОЕ ОТКЛОНЕНИЕ (Внутренняя текстура)": "features_std_abcde.csv",
    "3. ПЕРВАЯ ПРОИЗВОДНАЯ (Устранение влияния освещения)": "features_deriv_abcde.csv"
}

def evaluate_dataset(file_path, method_name):
    print(f"\n{method_name}")
    print("-" * 50)
    
    if not file_path.exists():
        print(f"Файл {file_path.name} не найден!")
        return

    # Загружаем уже объединенные данные
    df = pd.read_csv(file_path, sep=';')
    
    # Отделяем признаки от правильных ответов
    # Берем всё, кроме ID пациента и метки класса
    feature_cols = [c for c in df.columns if c not in ['Patient_ID', 'class']]
    X = df[feature_cols].values
    y = df['class'].values

    # Кодируем метки и стандартизируем
    y_encoded = LabelEncoder().fit_transform(y)
    X_scaled = StandardScaler().fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. SVM
    svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    svm_acc = cross_val_score(svm, X_scaled, y_encoded, cv=cv, scoring='accuracy').mean()
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf_acc = cross_val_score(rf, X_scaled, y_encoded, cv=cv, scoring='accuracy').mean()

    # 3. XGBoost
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5, 
        subsample=0.8, colsample_bytree=0.8, objective='multi:softprob', 
        eval_metric='mlogloss', random_state=42, n_jobs=-1
    )
    xgb_acc = cross_val_score(xgb, X_scaled, y_encoded, cv=cv, scoring='accuracy').mean()

    print(f"SVM:           {svm_acc:.3f}")
    print(f"Random Forest: {rf_acc:.3f}")
    print(f"XGBoost:       {xgb_acc:.3f}")

if __name__ == "__main__":
    print("=== СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТОДОВ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ ===")
    for name, filename in FILES_TO_TEST.items():
        evaluate_dataset(DATA_DIR / filename, name)
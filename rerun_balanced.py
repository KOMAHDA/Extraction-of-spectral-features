import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"D:\Projects\Extraction-of-spectral-features\Extraction-of-spectral-features\features")
ABCDE = ["A_Asymmetry", "B_Border", "C_Color_Std", "D_Diameter", "E_Eccentricity", "Area"]

def wl_cols(df):
    return sorted([c for c in df.columns if c.startswith("wl_")], key=lambda c: int(c[3:]))

def load():
    sets = {}
    spec = pd.read_csv(DATA / "spectral_signatures.csv", sep=";")
    spec["patient_id"] = spec["patient_id"].astype(str)
    sets["Среднее (без ABCDE)"] = (spec[wl_cols(spec)].values, spec["class"].values)
    ab = pd.read_csv(DATA / "abcde_features_aligned.csv", sep=";")
    ab["Patient_ID"] = ab["Patient_ID"].astype(str)
    m = spec.merge(ab, left_on="patient_id", right_on="Patient_ID", how="inner")
    sets["Среднее + ABCDE"] = (m[wl_cols(spec) + ABCDE].values, m["class"].values)
    for name, f in [("Медиана + ABCDE", "features_median_abcde.csv"),
                    ("Ст. отклонение + ABCDE", "features_std_abcde.csv"),
                    ("Производная + ABCDE", "features_deriv_abcde.csv")]:
        d = pd.read_csv(DATA / f, sep=";")
        sets[name] = (d[wl_cols(d) + ABCDE].values, d["class"].values)
    return sets

def xgb():
    return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8,
                         colsample_bytree=0.8, objective="multi:softprob",
                         eval_metric="mlogloss", random_state=42, n_jobs=-1)

MODELS = {
    "SVM (веса)": (lambda: SVC(kernel="rbf", class_weight="balanced", random_state=42), False),
    "RF (веса)": (lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42), False),
    "XGBoost (без весов)": (xgb, False),
    "XGBoost (веса)": (xgb, True),
}

def evaluate(X, y, make, weighted):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc, bal, f1 = [], [], []
    oof = np.full(len(y), -1)
    for tr, te in cv.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        kw = {"sample_weight": compute_sample_weight("balanced", y[tr])} if weighted else {}
        p = make().fit(scaler.transform(X[tr]), y[tr], **kw).predict(scaler.transform(X[te]))
        oof[te] = p
        acc.append(accuracy_score(y[te], p))
        bal.append(balanced_accuracy_score(y[te], p))
        f1.append(f1_score(y[te], p, average="macro", zero_division=0))
    return acc, bal, f1, oof

rows, best = [], None
for sname, (X, labels) in load().items():
    le = LabelEncoder().fit(labels)
    y = le.transform(labels)
    print(f"\n{sname}  (N={len(y)}, признаков={X.shape[1]})")
    print(f"{'модель':22s}{'accuracy':>18s}{'balanced acc':>18s}{'macro-F1':>18s}")
    for mname, (make, w) in MODELS.items():
        a, b, f, oof = evaluate(X, y, make, w)
        s = lambda v: f"{np.mean(v):.3f} ±{2*np.std(v):.3f}"
        print(f"{mname:22s}{s(a):>18s}{s(b):>18s}{s(f):>18s}", flush=True)
        rows.append(dict(features=sname, model=mname, acc=np.mean(a), acc_2sd=2*np.std(a),
                         bal_acc=np.mean(b), bal_2sd=2*np.std(b), macro_f1=np.mean(f), f1_2sd=2*np.std(f)))
        if sname.startswith("Медиана") and mname == "XGBoost (без весов)":
            best = (y, oof, le.classes_)

pd.DataFrame(rows).to_csv(DATA / "rerun_results.csv", index=False, sep=";", encoding="utf-8-sig")
y, oof, names = best
cm = pd.DataFrame(confusion_matrix(y, oof), index=names, columns=names)
print("\nМатрица ошибок: Медиана + ABCDE, XGBoost без весов (строки — истина)\n", cm.to_string(), sep="")
cm.to_csv(DATA / "rerun_confusion_median_xgb.csv", sep=";", encoding="utf-8-sig")
print("\nСохранено: rerun_results.csv, rerun_confusion_median_xgb.csv")

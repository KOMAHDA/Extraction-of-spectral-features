"""Дополнительный анализ: злокачественные (MM, BCC, SCC) против доброкачественных.
Запуск: py rerun_binary.py  (в том же venv)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"D:\Projects\Extraction-of-spectral-features\Extraction-of-spectral-features\features")
ABCDE = ["A_Asymmetry", "B_Border", "C_Color_Std", "D_Diameter", "E_Eccentricity", "Area"]
MALIGNANT = {"MM", "BCC", "SCC"}

def wl_cols(df):
    return sorted([c for c in df.columns if c.startswith("wl_")], key=lambda c: int(c[3:]))

def load():
    sets = {}
    spec = pd.read_csv(DATA / "spectral_signatures.csv", sep=";")
    spec["patient_id"] = spec["patient_id"].astype(str)
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
                         colsample_bytree=0.8, objective="binary:logistic",
                         eval_metric="logloss", random_state=42, n_jobs=-1)

MODELS = {
    "SVM (веса)": (lambda: SVC(kernel="rbf", class_weight="balanced", random_state=42), False),
    "RF (веса)": (lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42), False),
    "XGBoost (веса)": (xgb, True),
}

def score(model, X):
    return model.decision_function(X) if hasattr(model, "decision_function") else model.predict_proba(X)[:, 1]

def evaluate(X, y, make, weighted):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    res = {"auc": [], "acc": [], "sens": [], "spec": []}
    oof = np.zeros(len(y), dtype=int)
    for tr, te in cv.split(X, y):
        sc = StandardScaler().fit(X[tr])
        kw = {"sample_weight": compute_sample_weight("balanced", y[tr])} if weighted else {}
        m = make().fit(sc.transform(X[tr]), y[tr], **kw)
        Xt = sc.transform(X[te])
        p = m.predict(Xt)
        oof[te] = p
        res["auc"].append(roc_auc_score(y[te], score(m, Xt)))
        res["acc"].append(accuracy_score(y[te], p))
        res["sens"].append(recall_score(y[te], p, pos_label=1))
        res["spec"].append(recall_score(y[te], p, pos_label=0))
    return res, oof

rows = []
for sname, (X, labels) in load().items():
    y = np.array([1 if c in MALIGNANT else 0 for c in labels])
    print(f"\n{sname}  (N={len(y)}, злокач.={y.sum()}, тривиальный прогноз accuracy={1-y.mean():.3f})")
    print(f"{'модель':16s}{'ROC-AUC':>16s}{'accuracy':>16s}{'чувствит.':>16s}{'специфичн.':>16s}{'меланомы':>10s}")
    for mname, (make, w) in MODELS.items():
        r, oof = evaluate(X, y, make, w)
        s = lambda k: f"{np.mean(r[k]):.3f} ±{2*np.std(r[k]):.3f}"
        mm = oof[labels == "MM"].mean()
        print(f"{mname:16s}{s('auc'):>16s}{s('acc'):>16s}{s('sens'):>16s}{s('spec'):>16s}{mm:>10.3f}", flush=True)
        rows.append({"features": sname, "model": mname, **{k: np.mean(v) for k, v in r.items()},
                     **{k + "_2sd": 2 * np.std(v) for k, v in r.items()}, "mm_sens": mm})
pd.DataFrame(rows).to_csv(DATA / "rerun_binary_results.csv", index=False, sep=";", encoding="utf-8-sig")
print("\nСохранено: rerun_binary_results.csv")

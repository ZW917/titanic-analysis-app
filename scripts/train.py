"""Reproduce notebook experiments offline; write only this project's artifacts.

Run from the project root: python scripts/train.py [--ablation]
Training never changes data/ or notebooks/. The Streamlit UI only reads results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from pipeline import AgeImputer, build_model, make_features

SCORING = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
AGE_LABELS = {"median": "整体中位数", "group": "分组中位数", "rf": "随机森林"}
MODEL_LABELS = {
    "LogisticRegression": "逻辑回归", "DecisionTree": "决策树",
    "RandomForest": "随机森林", "GradientBoosting": "梯度提升", "KNN": "K近邻",
}


def fingerprint(path):
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def score_summary(scores):
    return {
        "accuracy": float(np.mean(scores["test_accuracy"])),
        "accuracy_std": float(np.std(scores["test_accuracy"], ddof=1)),
        "f1": float(np.mean(scores["test_f1"])),
        "roc_auc": float(np.mean(scores["test_roc_auc"])),
        "fit_time": float(np.mean(scores["fit_time"])),
        "folds": {key: np.asarray(value).tolist() for key, value in scores.items()},
    }


def evaluate(candidate, X, y, cv_splits):
    return cross_validate(
        candidate, X, y, cv=cv_splits, scoring=SCORING,
        n_jobs=1, error_score="raise",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", action="store_true", help="附加三个工程特征的训练集对照实验")
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    timer = time.perf_counter()
    data_dir = ROOT / "data"
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    source_paths = [data_dir / "train.csv", data_dir / "test.csv", data_dir / "titanic_submission.csv"]
    source_hashes = {str(p.relative_to(ROOT)): fingerprint(p) for p in source_paths}
    notebook = ROOT / "notebooks" / "original_analysis.ipynb"
    if notebook.exists():
        source_hashes[str(notebook.relative_to(ROOT))] = fingerprint(notebook)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    reference = pd.read_csv(data_dir / "titanic_submission.csv")
    assert len(train) == 891 and len(test) == 418, "输入数据记录数与原项目不一致"
    assert train.PassengerId.is_unique and test.PassengerId.is_unique
    assert train.Survived.isin([0, 1]).all()
    X = train.drop(columns=["PassengerId", "Survived"])
    y = train["Survived"]
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(cv.split(X_train, y_train))
    train.loc[X_train.index].to_csv(out / "train_subset.csv", index=False, encoding="utf-8-sig")

    # Preview uses training records only and never supplies prefilled data to CV.
    print("[1/7] 生成训练部分预处理预览", flush=True)
    features = make_features(X_train)
    preview = train.loc[X_train.index, ["PassengerId", "Sex", "Pclass", "Age"]].copy()
    for strategy in AGE_LABELS:
        filled = AgeImputer(strategy).fit_transform(features)
        known = preview.Age.notna()
        assert filled.loc[known, "Age"].equals(preview.loc[known, "Age"])
        assert not filled.Age.isna().any()
        preview[f"Age_{strategy}"] = filled.Age
    preview.to_csv(out / "preprocessing_preview.csv", index=False, encoding="utf-8-sig")

    print("[2/7] 三种年龄填充方法五折比较", flush=True)
    age_scores = {}
    age_rows = []
    for strategy, label in AGE_LABELS.items():
        print(f"  年龄方法：{strategy}", flush=True)
        scores = evaluate(build_model(strategy), X_train, y_train, cv_splits)
        age_scores[strategy] = scores
        age_rows.append({"strategy": strategy, "label": label, **score_summary(scores)})
    best_age = max(age_rows, key=lambda row: row["accuracy"])["strategy"]

    print("[3/7] 五种分类模型比较", flush=True)
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }
    model_rows = []
    for name, classifier in classifiers.items():
        print(f"  分类模型：{name}", flush=True)
        scores = age_scores[best_age] if name == "LogisticRegression" else evaluate(
            build_model(best_age, classifier), X_train, y_train, cv_splits
        )
        model_rows.append({"model": name, "label": MODEL_LABELS[name], **score_summary(scores)})
    model_rows.sort(key=lambda row: row["accuracy"], reverse=True)

    print("[4/7] 逻辑回归 C 参数搜索", flush=True)
    search = GridSearchCV(
        build_model(best_age), param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
        scoring=SCORING, refit="accuracy", cv=cv_splits, n_jobs=1, error_score="raise",
    )
    search.fit(X_train, y_train)
    tuning = []
    for i, params in enumerate(search.cv_results_["params"]):
        accuracy_folds = [float(search.cv_results_[f"split{k}_test_accuracy"][i]) for k in range(5)]
        tuning.append({
            "C": float(params["model__C"]),
            "accuracy": float(search.cv_results_["mean_test_accuracy"][i]),
            "accuracy_std": float(np.std(accuracy_folds, ddof=1)),
            "f1": float(search.cv_results_["mean_test_f1"][i]),
            "roc_auc": float(search.cv_results_["mean_test_roc_auc"][i]),
            "fit_time": float(search.cv_results_["mean_fit_time"][i]),
            "fold_accuracy": accuracy_folds,
        })
    best_c = float(search.best_params_["model__C"])
    holdout_model = search.best_estimator_

    print("[5/7] 一次性评估固定方案的 179 人留出集", flush=True)
    predicted = holdout_model.predict(X_holdout).astype(int)
    probability = holdout_model.predict_proba(X_holdout)[:, list(holdout_model.classes_).index(1)]
    errors = (predicted != y_holdout.to_numpy()).astype(int)
    holdout_frame = train.loc[X_holdout.index].copy()
    holdout_frame["Predicted"] = predicted
    holdout_frame["Probability"] = probability
    holdout_frame["Error"] = errors
    holdout_frame["y_true"] = y_holdout
    holdout_frame["y_pred"] = predicted
    holdout_frame["y_proba"] = probability
    holdout_frame.to_csv(out / "holdout_predictions.csv", index=False, encoding="utf-8-sig")
    fpr, tpr, thresholds = roc_curve(y_holdout, probability)
    holdout_results = {
        "n": int(len(y_holdout)), "correct": int((1 - errors).sum()), "incorrect": int(errors.sum()),
        "accuracy": float(accuracy_score(y_holdout, predicted)),
        "precision": float(precision_score(y_holdout, predicted, zero_division=0)),
        "recall": float(recall_score(y_holdout, predicted, zero_division=0)),
        "f1": float(f1_score(y_holdout, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_holdout, probability)),
        "confusion_matrix": confusion_matrix(y_holdout, predicted, labels=[0, 1]).tolist(),
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                "thresholds": [float(t) if np.isfinite(t) else None for t in thresholds]},
        "training_rows": len(X_train),
        "majority_baseline_accuracy": float((y_holdout == 0).mean()),
    }

    print("[6/7] 全部 891 人重新训练并生成 418 人预测", flush=True)
    full_model = clone(holdout_model).fit(X, y)
    test_predictions = full_model.predict(test.loc[:, X.columns]).astype(int)
    submission = pd.DataFrame({"PassengerId": test.PassengerId.to_numpy(), "Survived": test_predictions})
    assert submission.columns.tolist() == ["PassengerId", "Survived"]
    assert len(submission) == 418 and submission.PassengerId.is_unique
    assert submission.PassengerId.tolist() == test.PassengerId.tolist()
    assert submission.notna().all().all() and submission.Survived.isin([0, 1]).all()
    reference_match = submission.equals(reference.astype(submission.dtypes.to_dict()))
    mismatch_count = int((submission.Survived.to_numpy() != reference.Survived.to_numpy()).sum()) if len(reference) == len(submission) else None
    submission.to_csv(out / "submission.csv", index=False, encoding="utf-8-sig")
    pd.testing.assert_frame_equal(pd.read_csv(out / "submission.csv"), submission)
    joblib.dump(full_model, out / "model.joblib", compress=3)
    reloaded = joblib.load(out / "model.joblib")
    np.testing.assert_array_equal(reloaded.predict(test.loc[:, X.columns]), test_predictions)

    ablation = None
    if args.ablation:
        print("[7/7] 可选特征对照：只使用 712 人，保持最终模型不变", flush=True)
        full_row = next(row for row in tuning if row["C"] == best_c)
        ablation = {
            "purpose": "Supplementary training-only feature removal comparison; final model unchanged",
            "rows": [{"feature_removed": None, "label": "完整特征", **full_row}],
        }
        for feature, label in [("FamilyGroup", "移除家庭类别"), ("TitleGroup", "移除称谓类别"), ("HasCabin", "移除舱号记录")]:
            print(f"  {label}", flush=True)
            candidate = build_model(best_age, excluded_features=(feature,))
            candidate.set_params(model__C=best_c)
            scores = evaluate(candidate, X_train, y_train, cv_splits)
            ablation["rows"].append({"feature_removed": feature, "label": label, **score_summary(scores)})

    results = {
        "schema_version": 1,
        "dataset": {"train_rows": len(train), "test_rows": len(test), "train_columns": list(train.columns),
                    "target_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
                    "missing_train": {k: int(v) for k, v in train.isna().sum().items()},
                    "missing_test": {k: int(v) for k, v in test.isna().sum().items()}},
        "split": {"train_n": len(X_train), "holdout_n": len(X_holdout), "train_survived": int(y_train.sum()),
                  "holdout_survived": int(y_holdout.sum()), "test_size": 0.2, "random_state": 42,
                  "stratified": True, "cv_folds": 5,
                  "train_passenger_ids": train.loc[X_train.index, "PassengerId"].tolist(),
                  "holdout_passenger_ids": train.loc[X_holdout.index, "PassengerId"].tolist(),
                  "cv_validation_passenger_ids": [train.loc[X_train.iloc[val].index, "PassengerId"].tolist() for _, val in cv_splits]},
        "age_comparison": age_rows,
        "model_comparison": model_rows,
        "tuning": tuning,
        "selection": {"age_strategy": best_age, "model": "LogisticRegression", "C": best_c,
                      "criterion": "Training five-fold mean accuracy; logistic regression tuning matches notebook",
                      "matches_notebook_configuration": best_age == "rf" and best_c == 1 and model_rows[0]["model"] == "LogisticRegression"},
        "holdout": holdout_results,
        "submission": {"n": len(submission), "survived": int(submission.Survived.sum()),
                       "not_survived": int((submission.Survived == 0).sum()),
                       "survival_rate": float(submission.Survived.mean()), "reference_match": bool(reference_match),
                       "reference_mismatches": mismatch_count, "reference_file": "data/titanic_submission.csv",
                       "has_true_labels": False},
        "ablation": ablation,
        "provenance": {"started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                       "runtime_seconds": time.perf_counter() - timer, "python": platform.python_version(),
                       "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "joblib"]},
                       "source_files": source_hashes,
                       "code_files": {str(p.relative_to(ROOT)): fingerprint(p) for p in [ROOT / "pipeline.py", Path(__file__).resolve()]},
                       "seeds": {"split": 42, "cv": 42, "age_random_forest": 0, "classifiers": 42},
                       "fit_time_scope": "Full pipeline mean fit time per fold; hardware and environment dependent",
                       "evaluation_scope": "Holdout is evaluated before full-data refit; ablation never changes final selection",
                       "roc_threshold_infinity_encoding": "null represents the initial infinite threshold"},
    }
    for relative, before in source_hashes.items():
        assert fingerprint(ROOT / relative) == before, f"只读源文件发生变化：{relative}"
    results["provenance"]["artifacts"] = {p.name: fingerprint(p) for p in out.iterdir() if p.is_file() and p.name != "results.json"}
    write_json(out / "results.json", results)
    print(json.dumps({"selection": results["selection"], "holdout_accuracy": holdout_results["accuracy"],
                      "submission": results["submission"], "seconds": results["provenance"]["runtime_seconds"]}, ensure_ascii=False), flush=True)
    if not reference_match:
        raise SystemExit("新预测与提供的历史CSV不完全一致：已保存真实新结果，请检查环境版本或历史文件来源，不能宣称复现一致。")


if __name__ == "__main__":
    main()

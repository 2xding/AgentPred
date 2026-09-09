"""Cross-validate a compact diverse portfolio and write ranked candidate CSVs."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    categorical_columns,
    emit_manifest,
    encoded_frames,
    load_task,
    native_frames,
    rank_unit,
    runtime_workdir,
    write_submission,
)

SEED = 20260710

def folds_for(y: np.ndarray, n_rows: int) -> StratifiedKFold:
    minority = int(np.bincount(y).min())
    return StratifiedKFold(n_splits=max(2, min(5, minority)), shuffle=True, random_state=SEED)

def score(y: np.ndarray, predictions: np.ndarray) -> float:
    return float(roc_auc_score(y, predictions))

def cv_catboost(x_train, y, x_test, cat_cols, splitter):
    from catboost import CatBoostClassifier
    oof = np.zeros(len(x_train))
    test_predictions = np.zeros(len(x_test))
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(x_train, y)):
        model = CatBoostClassifier(
            iterations=650 if len(x_train) >= 2000 else 420,
            depth=6 if len(x_train) >= 2000 else 5,
            learning_rate=0.04,
            loss_function="Logloss",
            eval_metric="AUC",
            l2_leaf_reg=7.0,
            random_strength=0.35,
            random_seed=SEED + fold,
            verbose=False,
            allow_writing_files=False,
            thread_count=3,
        )
        model.fit(
            x_train.iloc[fit_idx],
            y[fit_idx],
            cat_features=cat_cols,
            eval_set=(x_train.iloc[val_idx], y[val_idx]),
            early_stopping_rounds=70,
            use_best_model=True,
            verbose=False,
        )
        oof[val_idx] = model.predict_proba(x_train.iloc[val_idx])[:, 1]
        test_predictions += model.predict_proba(x_test)[:, 1] / splitter.n_splits
    return oof, test_predictions

def cv_lightgbm(x_train, y, x_test, categorical_cols, splitter):
    import lightgbm as lgb
    x_train = x_train.copy()
    x_test = x_test.copy()
    for column in categorical_cols:
        x_train[column] = x_train[column].round().astype("int64").astype("category")
        x_test[column] = x_test[column].round().astype("int64").astype("category")
    oof = np.zeros(len(x_train))
    test_predictions = np.zeros(len(x_test))
    small = len(x_train) < 2500
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(x_train, y)):
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=900 if not small else 500,
            learning_rate=0.025 if not small else 0.04,
            num_leaves=15 if small else 31,
            max_depth=-1,
            min_child_samples=20 if small else 35,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.15,
            reg_lambda=2.5,
            random_state=SEED + fold,
            n_jobs=3,
            verbosity=-1,
        )
        model.fit(
            x_train.iloc[fit_idx],
            y[fit_idx],
            eval_set=[(x_train.iloc[val_idx], y[val_idx])],
            eval_metric="auc",
            categorical_feature=categorical_cols,
            callbacks=[lgb.early_stopping(70, verbose=False), lgb.log_evaluation(0)],
        )
        oof[val_idx] = model.predict_proba(x_train.iloc[val_idx])[:, 1]
        test_predictions += model.predict_proba(x_test)[:, 1] / splitter.n_splits
    return oof, test_predictions

def cv_extra_trees(x_train, y, x_test, splitter):
    oof = np.zeros(len(x_train))
    test_predictions = np.zeros(len(x_test))
    leaf = 2 if len(x_train) < 3000 else 3
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(x_train, y)):
        model = ExtraTreesClassifier(
            n_estimators=500,
            max_features=0.8,
            min_samples_leaf=leaf,
            class_weight="balanced",
            random_state=SEED + fold,
            n_jobs=3,
        )
        model.fit(x_train.iloc[fit_idx], y[fit_idx])
        oof[val_idx] = model.predict_proba(x_train.iloc[val_idx])[:, 1]
        test_predictions += model.predict_proba(x_test)[:, 1] / splitter.n_splits
    return oof, test_predictions

def cv_logistic(train, y, test, features, splitter):
    cat_cols = categorical_columns(train, features)
    num_cols = [column for column in features if column not in cat_cols]
    oof = np.zeros(len(train))
    test_predictions = np.zeros(len(test))
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(train, y)):
        numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))])
        transformer = ColumnTransformer([("numeric", numeric, num_cols), ("categorical", categorical, cat_cols)])
        model = Pipeline([("features", transformer), ("model", LogisticRegression(C=0.25 if len(train) < 2000 else 0.7, max_iter=1500, solver="liblinear", random_state=SEED + fold))])
        model.fit(train.iloc[fit_idx][features], y[fit_idx])
        oof[val_idx] = model.predict_proba(train.iloc[val_idx][features])[:, 1]
        test_predictions += model.predict_proba(test[features])[:, 1] / splitter.n_splits
    return oof, test_predictions

def main() -> None:
    started = time.time()
    workdir = runtime_workdir()
    train, test, sample, id_col, target_col, features, y = load_task(workdir)
    native_train, native_test, cat_cols = native_frames(train, test, features)
    encoded_train, encoded_test, encoded_cat_cols = encoded_frames(train, test, features)
    splitter = folds_for(y, len(train))
    models: dict[str, dict] = {}
    errors: dict[str, str] = {}

    runners = [
        ("catboost", lambda: cv_catboost(native_train, y, native_test, cat_cols, splitter)),
        ("lightgbm", lambda: cv_lightgbm(encoded_train, y, encoded_test, encoded_cat_cols, splitter)),
        ("extra_trees", lambda: cv_extra_trees(encoded_train, y, encoded_test, splitter)),
        ("logistic", lambda: cv_logistic(train, y, test, features, splitter)),
    ]
    for name, runner in runners:
        model_started = time.time()
        try:
            oof, test_predictions = runner()
            if not np.isfinite(oof).all() or np.std(oof) <= 1e-9:
                raise ValueError("invalid or constant OOF predictions")
            models[name] = {"oof": oof, "test": test_predictions, "cv_auc": score(y, oof), "seconds": round(time.time() - model_started, 3)}
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}:{exc}"

    if not models:
        prior = float(np.mean(y))
        path = write_submission(workdir / "portfolio_prior.csv", np.full(len(test), prior), test, sample, id_col, target_col)
        emit_manifest({"candidates": [path], "errors": errors, "models": {}, "robust_choice": path, "total_seconds": round(time.time() - started, 3)})
        return

    ranked_names = sorted(models, key=lambda name: models[name]["cv_auc"], reverse=True)
    candidates: list[dict] = []
    for name in ranked_names:
        path = write_submission(workdir / f"portfolio_{name}.csv", models[name]["test"], test, sample, id_col, target_col)
        candidates.append({"path": path, "cv_auc": models[name]["cv_auc"], "kind": name})

    top_two = ranked_names[:2]
    if len(top_two) == 2:
        first_oof, second_oof = rank_unit(models[top_two[0]]["oof"]), rank_unit(models[top_two[1]]["oof"])
        first_test, second_test = rank_unit(models[top_two[0]]["test"]), rank_unit(models[top_two[1]]["test"])
        path = write_submission(workdir / "portfolio_rank_top2.csv", 0.5 * first_test + 0.5 * second_test, test, sample, id_col, target_col)
        candidates.append({"path": path, "cv_auc": score(y, 0.5 * first_oof + 0.5 * second_oof), "kind": "rank_top2"})

    weights = np.array([max(models[name]["cv_auc"] - 0.5, 0.005) ** 2 for name in ranked_names], dtype=float)
    weights /= weights.sum()
    all_oof = np.average(np.stack([rank_unit(models[name]["oof"]) for name in ranked_names]), axis=0, weights=weights)
    all_test = np.average(np.stack([rank_unit(models[name]["test"]) for name in ranked_names]), axis=0, weights=weights)
    robust_path = write_submission(workdir / "portfolio_rank_all.csv", all_test, test, sample, id_col, target_col)
    candidates.append({"path": robust_path, "cv_auc": score(y, all_oof), "kind": "rank_all"})

    # --- PHASE 1 HANDOVER LOGIC WITH SCAFFOLD GENERATOR ---
    try:
        import shutil
        common_path = Path(__file__).resolve().parent / "common.py"
        shutil.copy(common_path, workdir / "common.py")

        # 1. Write the Summary Document
        with open(workdir / "handover.md", "w") as f:
            f.write("# Phase 1 & 2 Summary (Baseline)\n")
            f.write(f"- **Data Shape:** {len(train)} rows, {len(features)} features.\n")
            f.write(f"- **Categorical Features:** {len(cat_cols)}\n")
            f.write("- **Baseline Models Ranked by OOF AUC:**\n")
            for name in ranked_names:
                f.write(f"  - {name}: {models[name]['cv_auc']:.6f}\n")
        
        # 2. Write the Pro Scaffold Script
        with open(workdir / "pro_opt.py", "w") as f:
            f.write(f"""import sys; sys.path.append("/work")
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from common import load_task, write_submission, native_frames

# 1. Load Data seamlessly
train, test, sample, id_col, target_col, features, y = load_task("/work")
x_train, x_test, cat_cols = native_frames(train, test, features)

# --- YOUR ADVANCED FEATURE ENGINEERING GOES HERE ---
# Example: Create robust mathematical interactions, target encoding, or cluster features.
# Make sure to update cat_cols if you add new categorical strings!


# 2. Train Model (Using the reliable baseline settings)
model = CatBoostClassifier(iterations=600, depth=6, learning_rate=0.04, loss_function="Logloss", eval_metric="AUC", random_seed=42, verbose=False)
model.fit(x_train, y, cat_features=cat_cols)

# 3. Predict & Submit
predictions = model.predict_proba(x_test)[:, 1]
write_submission("/work/pro_submission.csv", predictions, test, sample, id_col, target_col)
print("SUCCESS: Submission saved to /work/pro_submission.csv. Please submit this file now.")
""")
    except Exception as e:
        print(f"Handover error: {e}")

    candidates = sorted(candidates, key=lambda item: item["cv_auc"], reverse=True)
    emit_manifest({
        "candidates": [item["path"] for item in candidates[:8]],
        "candidate_metrics": candidates,
        "errors": errors,
        "robust_choice": robust_path,
        "total_seconds": round(time.time() - started, 3),
    })

if __name__ == "__main__":
    main()
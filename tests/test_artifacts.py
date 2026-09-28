"""Independently verify saved experiment boundaries, metrics and predictions.

These checks consume committed artifacts; they never retrain or modify data.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

import pipeline  # noqa: F401 -- imports the trusted joblib artifact's classes


class ArtifactIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads((ROOT / "artifacts" / "results.json").read_text(encoding="utf-8"))
        cls.raw_train = pd.read_csv(ROOT / "data" / "train.csv")
        cls.raw_test = pd.read_csv(ROOT / "data" / "test.csv")
        cls.subset = pd.read_csv(ROOT / "artifacts" / "train_subset.csv")
        cls.holdout = pd.read_csv(ROOT / "artifacts" / "holdout_predictions.csv")
        cls.submission = pd.read_csv(ROOT / "artifacts" / "submission.csv")

    def test_train_holdout_reproduce_original_stratified_split(self):
        train, holdout = train_test_split(
            self.raw_train, test_size=0.2, random_state=42, stratify=self.raw_train.Survived
        )
        pd.testing.assert_frame_equal(self.subset, train.reset_index(drop=True))
        pd.testing.assert_frame_equal(
            self.holdout[self.raw_train.columns], holdout.reset_index(drop=True)
        )
        self.assertEqual(len(self.subset), 712)
        self.assertEqual(len(self.holdout), 179)
        self.assertEqual(self.subset.Survived.sum(), 273)
        self.assertEqual(self.holdout.Survived.sum(), 69)
        self.assertFalse(set(self.subset.PassengerId) & set(self.holdout.PassengerId))
        self.assertEqual(set(self.subset.PassengerId) | set(self.holdout.PassengerId), set(self.raw_train.PassengerId))
        metadata = self.results["split"]
        self.assertEqual(metadata["train_passenger_ids"], self.subset.PassengerId.tolist())
        self.assertEqual(metadata["holdout_passenger_ids"], self.holdout.PassengerId.tolist())
        self.assertEqual(metadata["train_n"], len(train))
        self.assertEqual(metadata["holdout_n"], len(holdout))

    def test_cv_validation_folds_cover_training_once_and_exclude_holdout(self):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        expected = [self.subset.iloc[validation].PassengerId.tolist()
                    for _, validation in cv.split(self.subset, self.subset.Survived)]
        saved = self.results["split"]["cv_validation_passenger_ids"]
        self.assertEqual(saved, expected)
        counts = Counter(passenger for fold in saved for passenger in fold)
        self.assertEqual(counts, Counter(self.subset.PassengerId))
        for fold in saved:
            self.assertFalse(set(fold) & set(self.holdout.PassengerId))

    def test_holdout_metrics_match_independent_recalculation(self):
        y = self.holdout.Survived.to_numpy()
        predicted = self.holdout.Predicted.to_numpy()
        probability = self.holdout.Probability.to_numpy()
        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())
        self.assertTrue(np.isin(predicted, [0, 1]).all())
        np.testing.assert_array_equal(self.holdout.Error, (predicted != y).astype(int))
        np.testing.assert_array_equal(self.holdout.y_true, y)
        np.testing.assert_array_equal(self.holdout.y_pred, predicted)
        np.testing.assert_allclose(self.holdout.y_proba, probability)
        expected = {
            "accuracy": accuracy_score(y, predicted),
            "precision": precision_score(y, predicted, zero_division=0),
            "recall": recall_score(y, predicted, zero_division=0),
            "f1": f1_score(y, predicted, zero_division=0),
            "roc_auc": roc_auc_score(y, probability),
        }
        saved = self.results["holdout"]
        for metric, score in expected.items():
            self.assertAlmostEqual(saved[metric], score, places=12, msg=metric)
        self.assertEqual(saved["n"], len(y))
        self.assertEqual(saved["correct"], int((predicted == y).sum()))
        self.assertEqual(saved["incorrect"], int((predicted != y).sum()))
        self.assertEqual(saved["training_rows"], len(self.subset))
        self.assertAlmostEqual(saved["majority_baseline_accuracy"], float((y == 0).mean()))
        np.testing.assert_array_equal(saved["confusion_matrix"], confusion_matrix(y, predicted, labels=[0, 1]))
        fpr, tpr, thresholds = roc_curve(y, probability)
        np.testing.assert_allclose(saved["roc"]["fpr"], fpr)
        np.testing.assert_allclose(saved["roc"]["tpr"], tpr)
        saved_thresholds = [np.inf if value is None else value for value in saved["roc"]["thresholds"]]
        np.testing.assert_allclose(saved_thresholds, thresholds, rtol=1e-12, atol=1e-12)

    def test_cv_summaries_are_derived_from_saved_fold_scores(self):
        for row in self.results["age_comparison"] + self.results["model_comparison"]:
            for metric in ["accuracy", "f1", "roc_auc"]:
                folds = row["folds"][f"test_{metric}"]
                self.assertEqual(len(folds), 5)
                self.assertAlmostEqual(row[metric], float(np.mean(folds)), places=12)
            self.assertAlmostEqual(row["accuracy_std"], float(np.std(row["folds"]["test_accuracy"], ddof=1)), places=12)
            self.assertAlmostEqual(row["fit_time"], float(np.mean(row["folds"]["fit_time"])), places=12)
        tuning = self.results["tuning"]
        self.assertEqual([row["C"] for row in tuning], [0.01, 0.1, 1, 10, 100])
        for row in tuning:
            self.assertAlmostEqual(row["accuracy"], float(np.mean(row["fold_accuracy"])), places=12)
            self.assertAlmostEqual(row["accuracy_std"], float(np.std(row["fold_accuracy"], ddof=1)), places=12)
        self.assertEqual(self.results["selection"]["age_strategy"], max(self.results["age_comparison"], key=lambda row: row["accuracy"])["strategy"])
        self.assertEqual(self.results["selection"]["C"], max(tuning, key=lambda row: row["accuracy"])["C"])

    def test_preview_preserves_observed_ages_and_training_membership(self):
        preview = pd.read_csv(ROOT / "artifacts" / "preprocessing_preview.csv")
        pd.testing.assert_frame_equal(preview[["PassengerId", "Sex", "Pclass", "Age"]], self.subset[["PassengerId", "Sex", "Pclass", "Age"]])
        self.assertEqual(preview.Age.isna().sum(), 137)
        known = preview.Age.notna()
        for column in ["Age_median", "Age_group", "Age_rf"]:
            self.assertFalse(preview[column].isna().any())
            np.testing.assert_array_equal(preview.loc[known, column], preview.loc[known, "Age"])

    def test_submission_matches_reference_and_reloaded_full_model(self):
        self.assertEqual(self.submission.columns.tolist(), ["PassengerId", "Survived"])
        self.assertEqual(len(self.submission), 418)
        self.assertTrue(self.submission.PassengerId.is_unique)
        self.assertFalse(self.submission.isna().any().any())
        self.assertTrue(self.submission.Survived.isin([0, 1]).all())
        self.assertEqual(self.submission.PassengerId.tolist(), self.raw_test.PassengerId.tolist())
        reference = pd.read_csv(ROOT / "data" / "titanic_submission.csv")
        pd.testing.assert_frame_equal(self.submission, reference)
        full_model = joblib.load(ROOT / "artifacts" / "model.joblib")
        raw_columns = self.raw_train.drop(columns=["PassengerId", "Survived"]).columns
        np.testing.assert_array_equal(full_model.predict(self.raw_test[raw_columns]), self.submission.Survived)
        self.assertEqual(full_model.named_steps["age"].strategy, self.results["selection"]["age_strategy"])
        self.assertEqual(full_model.named_steps["model"].C, self.results["selection"]["C"])
        self.assertEqual(full_model.named_steps["age"].age_median_, self.raw_train.Age.median())
        self.assertEqual(full_model.named_steps["encode"].named_transformers_["num"].n_samples_seen_, 891)
        saved = self.results["submission"]
        self.assertEqual(saved["n"], 418)
        self.assertEqual(saved["survived"], int(self.submission.Survived.sum()))
        self.assertEqual(saved["not_survived"], int(self.submission.Survived.eq(0).sum()))
        self.assertAlmostEqual(saved["survival_rate"], self.submission.Survived.mean())
        self.assertTrue(saved["reference_match"])
        self.assertEqual(saved["reference_mismatches"], 0)
        self.assertFalse(saved["has_true_labels"])

    def test_source_inputs_match_recorded_hashes(self):
        for relative, recorded in self.results["provenance"]["source_files"].items():
            # Normalize the metadata paths for Windows and Linux clones alike.
            normalized = relative.replace("\\", "/")
            path = ROOT.joinpath(*normalized.split("/"))
            # The original notebook is an optional local audit copy. Public
            # releases omit it; every required data input remains mandatory.
            if normalized == "notebooks/original_analysis.ipynb" and not path.exists():
                continue
            content = path.read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), recorded["sha256"])
            self.assertEqual(len(content), recorded["bytes"])


if __name__ == "__main__":
    unittest.main()

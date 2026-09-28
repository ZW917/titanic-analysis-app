"""Targeted contracts: no mutation/leakage, unseen groups, saved model use."""

import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline import AgeImputer, FareEmbarkedImputer, build_model, make_features


class PipelineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / ".local").mkdir(exist_ok=True)
        cls.train = pd.read_csv(ROOT / "data" / "train.csv")
        cls.test = pd.read_csv(ROOT / "data" / "test.csv")
        cls.X = cls.train.drop(columns=["PassengerId", "Survived"])
        cls.X_train, cls.X_holdout, cls.y_train, cls.y_holdout = train_test_split(
            cls.X, cls.train.Survived, test_size=0.2, random_state=42, stratify=cls.train.Survived
        )

    def test_features_preserve_source_and_alignment(self):
        before = self.X_train.copy(deep=True)
        feature = make_features(self.X_train)
        pd.testing.assert_frame_equal(self.X_train, before)
        self.assertTrue(feature.index.equals(before.index))
        self.assertEqual(feature.HasCabin.value_counts().to_dict(), {0: 552, 1: 160})
        self.assertEqual(feature.FamilyGroup.value_counts().to_dict(), {"Alone": 434, "Small": 228, "Medium": 41, "Large": 9})

    def test_group_imputer_uses_fit_data_and_preserves_known_age(self):
        feature = make_features(self.X_train)
        before = feature.copy(deep=True)
        imputer = AgeImputer("group").fit(feature)
        filled = imputer.transform(feature)
        self.assertEqual(imputer.age_median_, 28.5)
        self.assertFalse(filled.Age.isna().any())
        known = feature.Age.notna()
        pd.testing.assert_series_equal(filled.loc[known, "Age"], feature.loc[known, "Age"])
        pd.testing.assert_frame_equal(feature, before)
        unseen = feature.iloc[:1].copy()
        unseen.loc[:, "Sex"] = "unseen"
        unseen.loc[:, "Age"] = np.nan
        self.assertEqual(imputer.transform(unseen).Age.iloc[0], 28.5)
        self.assertEqual(imputer.age_median_, 28.5)

    def test_fare_missing_group_falls_back_to_training_global(self):
        feature = make_features(self.X_train)
        imputer = FareEmbarkedImputer().fit(feature)
        test_feature = make_features(self.test)
        result = imputer.transform(test_feature)
        self.assertEqual(result.loc[self.test.PassengerId == 1044, "Fare"].iloc[0], 8.05)
        unseen = feature.iloc[:1].copy()
        unseen.loc[:, "Pclass"] = 99
        unseen.loc[:, "Fare"] = np.nan
        unseen.loc[:, "Embarked"] = np.nan
        output = imputer.transform(unseen)
        self.assertEqual(output.Fare.iloc[0], feature.Fare.median())
        self.assertEqual(output.Embarked.iloc[0], "S")

    def test_pipeline_saved_reload_and_unknown_categories(self):
        model = build_model("group").fit(self.X_train, self.y_train)
        record = self.X_holdout.iloc[:1].copy()
        record.loc[:, "Name"] = "Example, UnseenTitle. Passenger"
        record.loc[:, "Embarked"] = "NEW"
        record.loc[:, "Age"] = np.nan
        prediction = model.predict(record)
        self.assertIn(prediction[0], [0, 1])
        with tempfile.TemporaryDirectory(dir=ROOT / ".local") as directory:
            filename = Path(directory) / "roundtrip.joblib"
            joblib.dump(model, filename)
            reloaded = joblib.load(filename)
            np.testing.assert_array_equal(reloaded.predict(record), prediction)

    def test_ablation_removes_only_requested_classifier_feature(self):
        model = build_model("median", excluded_features=("TitleGroup",)).fit(self.X_train, self.y_train)
        columns = model.named_steps["encode"].get_feature_names_out()
        self.assertFalse(any("TitleGroup" in name for name in columns))
        self.assertTrue(any("FamilyGroup" in name for name in columns))


if __name__ == "__main__":
    unittest.main()

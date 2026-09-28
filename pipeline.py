"""Reusable Titanic pipeline, transcribed from the original 41-step notebook.

Keep this module importable when loading the trusted, locally generated joblib
artifact. All learned preprocessing is fitted inside each cross-validation fold.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


TITLE_MAP = {
    "Mr": "Mr", "Mrs": "Ms", "Miss": "Ms", "Ms": "Ms", "Mme": "Ms",
    "Mlle": "Ms", "Master": "Child", "Capt": "Officer", "Col": "Officer",
    "Major": "Officer", "Dr": "Officer", "Rev": "Officer", "Don": "Royalty",
    "Dona": "Royalty", "Sir": "Royalty", "Lady": "Royalty",
    "Jonkheer": "Royalty", "the Countess": "Royalty",
}
NUMERIC_COLUMNS = ["Age", "Fare"]
CATEGORICAL_COLUMNS = [
    "Pclass", "Sex", "Embarked", "HasCabin", "FamilyGroup", "TitleGroup"
]
FEATURE_COLUMNS = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
    "HasCabin", "FamilyGroup", "TitleGroup",
]
RAW_REQUIRED_COLUMNS = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Name", "Cabin"
]


def make_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return notebook features without mutating X or using target labels."""
    missing = set(RAW_REQUIRED_COLUMNS) - set(X.columns)
    if missing:
        raise ValueError(f"缺少输入字段：{', '.join(sorted(missing))}")
    df = X.copy()
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    family_size = df["SibSp"] + df["Parch"] + 1
    df["FamilyGroup"] = pd.cut(
        family_size,
        bins=[0, 1, 4, 7, float("inf")],
        labels=["Alone", "Small", "Medium", "Large"],
        right=True,
    )
    titles = df["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    df["TitleGroup"] = titles.map(TITLE_MAP).fillna("Unknown")
    return df[FEATURE_COLUMNS]


class AgeImputer(TransformerMixin, BaseEstimator):
    """Learn median/group medians/random forest from training ages only."""

    def __init__(self, strategy="median"):
        self.strategy = strategy

    def fit(self, X, y=None):
        if self.strategy not in ["median", "group", "rf"]:
            raise ValueError("strategy必须为median、group或rf")
        self.age_median_ = X["Age"].median()
        if pd.isna(self.age_median_):
            raise ValueError("训练数据至少需要一条已知年龄记录")
        if self.strategy == "group":
            self.group_age_ = (
                X.groupby(["Sex", "Pclass"])["Age"].median().rename("GroupAge")
            )
        elif self.strategy == "rf":
            self.age_features_ = ["Pclass", "SibSp", "Parch"]
            known = X["Age"].notna()
            self.age_model_ = RandomForestRegressor(
                n_estimators=2000, random_state=0, n_jobs=-1
            )
            self.age_model_.fit(X.loc[known, self.age_features_], X.loc[known, "Age"])
        return self

    def transform(self, X):
        check_is_fitted(self, "age_median_")
        df = X.copy()
        missing = df["Age"].isna()
        if self.strategy == "group":
            group_values = df.join(self.group_age_, on=["Sex", "Pclass"])["GroupAge"]
            df["Age"] = df["Age"].fillna(group_values)
        elif self.strategy == "rf" and missing.any():
            df.loc[missing, "Age"] = self.age_model_.predict(
                df.loc[missing, self.age_features_]
            )
        df["Age"] = df["Age"].fillna(self.age_median_)
        return df


class FareEmbarkedImputer(TransformerMixin, BaseEstimator):
    """Use training class/global fare medians and training embarkation mode."""

    def fit(self, X, y=None):
        self.fare_by_class_ = X.groupby("Pclass")["Fare"].median()
        self.fare_median_ = X["Fare"].median()
        modes = X["Embarked"].mode()
        if modes.empty or pd.isna(self.fare_median_):
            raise ValueError("训练数据必须包含已知票价和登船港口")
        self.embarked_mode_ = modes.iloc[0]
        return self

    def transform(self, X):
        check_is_fitted(self, ["fare_median_", "embarked_mode_"])
        df = X.copy()
        df["Fare"] = (
            df["Fare"].fillna(df["Pclass"].map(self.fare_by_class_))
            .fillna(self.fare_median_)
        )
        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode_)
        return df


def build_model(age_strategy="rf", classifier=None, excluded_features=()):
    """Construct a fresh pipeline. Exclusions affect only classifier encoding.

    The optional ablation does not remove fields used to estimate missing age;
    this keeps the imputation method fixed when comparing engineered features.
    """
    unknown = set(excluded_features) - set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)
    if unknown:
        raise ValueError(f"无法移除未知模型特征：{sorted(unknown)}")
    numeric = [c for c in NUMERIC_COLUMNS if c not in excluded_features]
    categorical = [c for c in CATEGORICAL_COLUMNS if c not in excluded_features]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )
    if classifier is None:
        classifier = LogisticRegression(max_iter=2000, random_state=42, C=1)
    return Pipeline([
        ("features", FunctionTransformer(make_features, validate=False)),
        ("age", AgeImputer(strategy=age_strategy)),
        ("basic", FareEmbarkedImputer()),
        ("encode", preprocessor),
        ("model", clone(classifier)),
    ])

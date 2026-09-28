"""Small end-to-end UI checks against prepared, real project artifacts.

Run after ``python scripts/train.py`` with ``python -m unittest discover -s tests``.
These checks do not retrain models, edit data, or start a browser server.
"""
import json
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
READY = all((ROOT / "artifacts" / name).is_file() for name in [
    "results.json", "model.joblib", "train_subset.csv", "submission.csv",
    "holdout_predictions.csv", "preprocessing_preview.csv",
])


@unittest.skipUnless(READY, "Prepare the offline artifacts with python scripts/train.py first")
class StreamlitAppTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assert_clean()

    def assert_clean(self):
        self.assertEqual([e.message for e in self.app.exception], [])

    def page(self, label):
        self.app.radio(key="page").set_value(label).run()
        self.assert_clean()

    def metrics(self):
        return {m.label: m.value for m in self.app.metric}

    def test_six_pages_render_saved_results(self):
        for page in ["项目概览", "数据质量与处理", "交互式生存分析", "模型实验", "留出评估", "预测体验"]:
            with self.subTest(page=page):
                self.page(page)
                self.assertGreater(len(self.app.get("plotly_chart")), 0)
        self.page("留出评估")
        result = json.loads((ROOT / "artifacts" / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(self.metrics()["准确率"], f"{result['holdout']['accuracy']:.2%}")
        for grouping in ["性别", "舱位", "原始年龄组"]:
            self.app.radio(key="error_group").set_value(grouping).run()
            self.assert_clean()

    def test_filter_denominator_and_empty_selection(self):
        self.page("交互式生存分析")
        self.app.multiselect(key="eda_sex").set_value(["女性"])
        self.app.multiselect(key="eda_class").set_value([1])
        self.app.slider(key="eda_age").set_value((20, 40))
        self.app.checkbox(key="eda_unknown").set_value(False).run()
        self.assert_clean()
        train = pd.read_csv(ROOT / "artifacts" / "train_subset.csv")
        expected = train.loc[train.Sex.eq("female") & train.Pclass.eq(1) & train.Age.between(20, 40)]
        self.assertEqual(self.metrics()["当前样本"], str(len(expected)))
        self.assertEqual(self.metrics()["当前生存率"], f"{expected.Survived.mean():.1%}")
        self.assertEqual(self.metrics()["原始年龄缺失"], "0")
        for topic in ["年龄与家庭", "票价分布", "补充特征"]:
            self.app.radio(key="eda_topic").set_value(topic).run()
            self.assert_clean()
        self.app.multiselect(key="eda_class").set_value([]).run()
        self.assert_clean()
        self.assertTrue(any("没有乘客记录" in item.value for item in self.app.info))
        self.assertNotIn("当前生存率", self.metrics())

    def test_unknown_age_form_matches_saved_pipeline(self):
        self.page("预测体验")
        self.app.checkbox(key="predict_age_unknown").set_value(True)
        self.app.selectbox(key="predict_embarked").set_value("未知")
        next(b for b in self.app.button if b.label == "预测这名乘客").click().run()
        self.assert_clean()
        record = self.app.session_state["single_prediction"]
        source = pd.DataFrame([{
            "PassengerId": 0, "Pclass": 3, "Name": "Example, Miss. Passenger",
            "Sex": "female", "Age": np.nan, "SibSp": 0, "Parch": 0,
            "Ticket": "DEMO", "Fare": 15.0, "Cabin": np.nan, "Embarked": np.nan,
        }])
        model = joblib.load(ROOT / "artifacts" / "model.joblib")
        positive_index = list(model.classes_).index(1)
        self.assertEqual(record["prediction"], int(model.predict(source)[0]))
        self.assertAlmostEqual(record["probability"], float(model.predict_proba(source)[0, positive_index]), places=10)
        self.assertIn("未知", record["summary"]["原始年龄"])

    def test_test_prediction_query_and_invalid_id(self):
        self.page("预测体验")
        source = pd.read_csv(ROOT / "artifacts" / "submission.csv")
        self.assertEqual(self.metrics()["测试乘客"], str(len(source)))
        self.assertEqual(self.metrics()["预测生存"], str(int(source.Survived.sum())))
        self.app.text_input(key="test_id").set_value("893").run()
        self.assert_clean()
        tables = [table.value for table in self.app.dataframe if "预测标签" in table.value.columns]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["乘客编号"].tolist(), [893])
        self.assertEqual(tables[0]["预测标签"].iloc[0], source.loc[source.PassengerId.eq(893), "Survived"].iloc[0])
        self.app.text_input(key="test_id").set_value("not-a-number").run()
        self.assert_clean()
        self.assertTrue(any("请输入整数" in info.value for info in self.app.info))


if __name__ == "__main__":
    unittest.main()

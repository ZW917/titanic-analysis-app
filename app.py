"""Read-only Streamlit presentation of the Titanic analysis artifacts.

Offline model training is deliberately separate. All paths are project-relative.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui import (
    AGE_LABELS, CORAL, FAMILY_LABELS, GOLD, MUTED, NAVY, OUTCOME_COLORS, PALETTE, TEAL,
    age_groups, chart, footer, forest_plot, hero, metrics, note, rate_summary,
    section, setup_style, step_card,
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
PAGES = ["项目概览", "数据质量与处理", "交互式生存分析", "模型实验", "留出评估", "预测体验"]
METRIC_LABELS = {"accuracy": "准确率", "f1": "F1", "roc_auc": "ROC-AUC"}
FIELD_LABELS = {
    "PassengerId": "乘客编号", "Survived": "实际生存标签", "Pclass": "舱位", "Name": "姓名",
    "Sex": "性别", "Age": "原始年龄", "SibSp": "同行兄弟姐妹/配偶", "Parch": "同行父母/子女",
    "Ticket": "船票编号", "Fare": "票价", "Cabin": "舱号", "Embarked": "登船港口",
    "Predicted": "预测标签", "Probability": "预测生存概率", "Error": "是否预测错误",
}


def file_stamp(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_data(show_spinner=False)
def read_csv(path: str, stamp: tuple[int, int]) -> pd.DataFrame:
    return pd.read_csv(path)


def csv(name: str, folder: Path = ARTIFACTS) -> pd.DataFrame:
    path = folder / name
    return read_csv(str(path), file_stamp(path))


@st.cache_data(show_spinner=False)
def read_results(path: str, stamp: tuple[int, int]) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="正在加载完整预测流水线…")
def read_model(path: str, stamp: tuple[int, int]):
    # The repository's pipeline module defines the trusted local model classes.
    import pipeline  # noqa: F401
    return joblib.load(path)


def required(names: list[str]) -> bool:
    missing = [name for name in names if not (ARTIFACTS / name).is_file()]
    if missing:
        st.info("此页面所需的离线分析成果尚未就绪。准备完成后刷新页面即可。")
        st.caption("待准备文件：" + "、".join(missing))
        st.code("python scripts/train.py", language="bash")
        st.caption("上述命令由项目维护者在部署前运行；网页交互不会触发训练。")
        return False
    return True


def get_results() -> dict:
    path = ARTIFACTS / "results.json"
    return read_results(str(path), file_stamp(path)) if path.is_file() else {}


def display_table(df: pd.DataFrame, key: str | None = None, **kwargs) -> None:
    st.dataframe(df, hide_index=True, width="stretch", key=key, **kwargs)


def download_csv(df: pd.DataFrame, label: str, filename: str, key: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8-sig"), file_name=filename,
                       mime="text/csv", key=key)


def derived(df: pd.DataFrame) -> pd.DataFrame:
    from pipeline import make_features
    transformed = make_features(df)
    out = df.copy()
    for name in ["HasCabin", "FamilyGroup", "TitleGroup"]:
        out[name] = transformed[name].to_numpy()
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["AgeGroup"] = age_groups(out["Age"])
    out["生存结果"] = out["Survived"].map({0: "未生存", 1: "生存"})
    return out


def overview(results: dict) -> None:
    hero("一段航程，一次数据探索", "从乘客信息出发，理解生存差异；用可复现的实验，建立并检验预测模型。",
         "TITANIC  /  ANALYSIS & PREDICTION", ("数据清洗", "交互探索", "模型比较", "独立评估"))
    holdout = results.get("holdout", {})
    metrics([
        ("有标签乘客", "891", "712人训练 · 179人留出评估"),
        ("待预测乘客", "418", "测试文件不含真实生存标签"),
        ("留出准确率", f"{holdout['accuracy']:.2%}" if "accuracy" in holdout else "待载入", "正式评估基于179名留出乘客"),
        ("交叉验证折数", "5", "所有方案使用相同分层划分"),
    ])
    section("围绕三个问题展开", "RESEARCH QUESTIONS")
    for col, content in zip(st.columns(3), [
        ("01 · 发现", "谁的生存率更高？", "探索性别、舱位、年龄和家庭规模，结合样本量理解群体差异。"),
        ("02 · 选择", "哪些处理与模型有效？", "比较三种年龄填充、五种分类模型，以及逻辑回归正则化参数。"),
        ("03 · 检验", "模型能预测新乘客吗？", "在未参与选型的留出数据上评估，再生成418人的无标签测试预测。"),
    ]):
        with col:
            step_card(*content)
    section("让数据边界清晰可见", "DATA JOURNEY")
    fig = go.Figure(go.Sankey(
        arrangement="snap", node=dict(pad=24, thickness=16, color=[NAVY, TEAL, GOLD, "#8099B0", CORAL],
            label=["有标签数据 · 891", "训练与五折验证 · 712", "独立留出评估 · 179", "无标签测试 · 418", "最终测试预测 · 418"]),
        link=dict(source=[0, 0, 3], target=[1, 2, 4], value=[712, 179, 418],
                  color=["rgba(22,140,137,.23)", "rgba(217,164,65,.28)", "rgba(128,153,176,.27)"]),
    ))
    chart(fig, "overview_flow", 280)
    note("最终采用的方案", "随机森林回归器填补缺失年龄 → 数值标准化与类别编码 → 逻辑回归分类（C＝1）。方案确定并完成留出评估后，再使用全部891条有标签数据重训用于测试预测。")
    if holdout:
        a, b = st.columns([1.15, 1])
        with a:
            section("结论有多可靠？")
            st.write(f"在{holdout.get('n',179)}名留出乘客中，预测正确{holdout.get('correct',149)}人。F1为{holdout.get('f1',0):.4f}，ROC-AUC为{holdout.get('roc_auc',0):.4f}。")
            st.write("交互探索展示关联；交叉验证用于比较方案；留出数据用于最后的独立检验。三者承担不同任务。")
        with b:
            with st.container(border=True):
                st.markdown("**建议浏览路径**")
                st.write("数据质量与处理 → 交互式生存分析 → 模型实验 → 留出评估 → 预测体验")
                st.caption("左侧切换页面。图表可悬停查看数值，点击图例筛选，使用图表工具栏下载PNG。")
    with st.expander("数据来源、项目范围与复现"):
        st.markdown("原实验使用[和鲸社区的 Titanic 数据集](https://www.heywhale.com/mw/dataset/5f69d01971c700003078960e/file)。这是经典泰坦尼克号乘客数据；字段说明见“数据质量与处理”页面。")
        st.write("本应用整理并复用课程项目中的数据处理和实验成果，使用固定随机种子、相同分层交叉验证及完整预处理流水线。网页只读取项目内数据和离线成果，不会在浏览页面时重新训练模型。")
        st.caption("原实验步骤已保留；运行方法、资料引用与工具辅助说明见项目仓库。")


def quality(results: dict) -> None:
    hero("先理解缺失，再选择处理", "每一个填充值都有来源。对原始字段进行检查，并比较三种年龄处理方式。",
         "02 / DATA QUALITY", ("整体文件检查", "训练部分学习规则", "保留原始年龄"))
    raw = csv("train.csv", ROOT / "data")
    test = csv("test.csv", ROOT / "data")
    source = st.radio("检查哪个原始文件", ["有标签数据 · 891人", "无标签测试 · 418人"], horizontal=True, key="quality_source")
    current = raw if source.startswith("有") else test
    missing = current.isna().sum().rename("缺失数").to_frame()
    missing["缺失率"] = missing["缺失数"] / len(current)
    missing = missing.query("缺失数 > 0").sort_values("缺失率", ascending=True).reset_index(names="字段")
    left, right = st.columns([1.15, 1])
    with left:
        section("缺失数据分布")
        fig = px.bar(missing, x="缺失率", y="字段", orientation="h", text="缺失数", color_discrete_sequence=[TEAL])
        fig.update_traces(texttemplate="%{text}人", textposition="outside")
        fig.update_xaxes(range=[0, 1], tickformat=".0%", title="原始文件中的缺失比例")
        chart(fig, "missing_chart", 300)
    with right:
        section("检查结果")
        display_table(pd.DataFrame({"检查项": ["记录数", "完全重复记录", "重复乘客编号", "包含缺失的字段数"],
                                    "结果": [len(current), int(current.duplicated().sum()), int(current.PassengerId.duplicated().sum()), int(current.isna().any().sum())]}))
        st.caption("缺失不等于错误。年龄和舱号采用不同策略；极端票价也不因数值较高而直接删除。")
    age_tab, other_tab, fields_tab = st.tabs(["年龄：比较三种填充", "其他字段与特征构造", "字段说明与原始数据"])
    with age_tab:
        if required(["preprocessing_preview.csv"]):
            preview = csv("preprocessing_preview.csv")
            methods = {"整体中位数": "Age_median", "性别＋舱位分组中位数": "Age_group", "随机森林回归": "Age_rf"}
            selected = st.selectbox("年龄填充方法", list(methods), key="age_method")
            column = methods[selected]
            unknown = preview.Age.isna()
            metrics([("原始年龄缺失", str(int(unknown.sum())), "仅712人的训练部分"),
                     ("处理后年龄缺失", str(int(preview[column].isna().sum())), selected),
                     ("已知年龄保持不变", "是" if np.allclose(preview.loc[~unknown,"Age"], preview.loc[~unknown,column]) else "请核查", "仅对缺失项填充")])
            col1, col2 = st.columns(2)
            with col1:
                section("观测年龄与估计年龄")
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=preview.loc[~unknown, "Age"], name="原始已知年龄", opacity=.75, marker_color=NAVY, xbins=dict(start=0,end=85,size=5)))
                fig.add_trace(go.Histogram(x=preview.loc[unknown, column], name="缺失项的估计年龄", opacity=.72, marker_color=GOLD, xbins=dict(start=0,end=85,size=5)))
                fig.update_layout(barmode="overlay")
                fig.update_xaxes(title="年龄（岁）")
                fig.update_yaxes(title="人数")
                chart(fig, "age_preview_hist")
                st.caption("两种颜色对应不同记录。估计年龄的分布不能作为真实年龄分布解释。")
            with col2:
                section("同一批缺失记录如何填充")
                comparison = preview.loc[unknown, ["PassengerId", "Sex", "Pclass", "Age", "Age_median", "Age_group", "Age_rf"]].head(12)
                display_table(comparison.rename(columns={"PassengerId":"编号", "Sex":"性别", "Pclass":"舱位", "Age":"原始年龄", "Age_median":"整体中位数", "Age_group":"分组中位数", "Age_rf":"随机森林"}).round(2))
            explanations = {
                "整体中位数": "使用训练部分已知年龄的中位数。方法简单、计算快，但不同群体使用同一个填充值。",
                "性别＋舱位分组中位数": "根据性别和舱位匹配训练部分的组内中位数；无法匹配时回退到整体中位数。",
                "随机森林回归": "只使用年龄已知的训练记录，以Pclass、SibSp、Parch预测年龄；不使用生存标签。最终通过生存预测的交叉验证比较填充方案。",
            }
            note(selected, explanations[selected])
    with other_tab:
        display_table(pd.DataFrame([
            ["Cabin", "构造HasCabin", "1＝有舱号记录；0＝记录缺失", "记录缺失不代表没有住舱"],
            ["Embarked", "训练部分众数填充", "使用出现最多的港口代码", "本次训练部分众数为S"],
            ["Fare", "同舱位中位数填充", "无法匹配时使用整体中位数", "统计值只在训练数据学习"],
            ["SibSp＋Parch＋1", "构造家庭类别", "1人、2–4人、5–7人、8人及以上", "包括乘客本人"],
            ["Name", "提取称谓并合并", "Mr、Ms、Child、Officer、Royalty、Unknown", "Child来源于Master，不等于所有儿童"],
        ], columns=["字段", "处理方法", "规则", "解释边界"]))
        if required(["train_subset.csv"]):
            train = csv("train_subset.csv")
            fare = train.groupby("Pclass").Fare.median().rename("训练部分票价中位数").reset_index()
            display_table(fare)
            st.caption("先划分数据，再学习填充规则。正式交叉验证中，每折分别学习填充值与标准化参数。")
    with fields_tab:
        display_table(pd.DataFrame({"字段": list(current.columns), "含义": [FIELD_LABELS.get(c,c) for c in current.columns],
                                    "存储类型": [str(current[c].dtype) for c in current.columns],
                                    "缺失数量": [int(current[c].isna().sum()) for c in current.columns]}))
        st.caption("Pclass虽以整数保存，但在最终模型中作为类别编码。PassengerId用于对应结果，不作为预测特征。")
        display_table(current.head(30), "raw_preview")


def exploration(results: dict) -> None:
    hero("让问题带着图表变化", "筛选乘客，比较群体。人数、比例与分布始终来自同一批选择后的记录。",
         "03 / INTERACTIVE EXPLORATION", ("仅训练部分 n＝712", "原始年龄", "关联分析"))
    if not required(["train_subset.csv"]):
        return
    train = derived(csv("train_subset.csv"))
    with st.container(border=True):
        cols = st.columns([1, 1, 1.5])
        with cols[0]:
            sex = st.multiselect("性别", ["女性", "男性"], default=["女性", "男性"], key="eda_sex")
        with cols[1]:
            classes = st.multiselect("舱位", [1, 2, 3], default=[1, 2, 3], format_func=lambda v: f"{v}等舱", key="eda_class")
        with cols[2]:
            ages = st.slider("已知年龄范围（岁）", 0, 80, (0, 80), key="eda_age")
        ca, cb = st.columns([1, 2])
        with ca:
            include_unknown = st.checkbox("包含原始年龄缺失者", value=True, key="eda_unknown")
        with cb:
            families = st.multiselect("家庭类别", list(FAMILY_LABELS), default=list(FAMILY_LABELS), format_func=FAMILY_LABELS.get, key="eda_family")
    sex_values = [{"女性":"female", "男性":"male"}[s] for s in sex]
    age_mask = train.Age.between(*ages) | (train.Age.isna() & include_unknown)
    filtered = train.loc[train.Sex.isin(sex_values) & train.Pclass.isin(classes) & train.FamilyGroup.isin(families) & age_mask].copy()
    if filtered.empty:
        st.info("当前筛选条件下没有乘客记录。请增加性别、舱位、家庭类别，或扩大年龄范围。")
        return
    n = len(filtered)
    survivors = int(filtered.Survived.sum())
    metrics([("当前样本", str(n), f"占训练部分 {n / len(train):.1%}"),
             ("实际生存人数", str(survivors), f"未生存 {n-survivors} 人"),
             ("当前生存率", f"{survivors / n:.1%}", "分母为当前筛选后的样本数"),
             ("原始年龄缺失", str(int(filtered.Age.isna().sum())), "不以估计年龄替代观测年龄")])
    theme = st.radio("探索主题", ["性别与舱位", "年龄与家庭", "票价分布", "补充特征"], horizontal=True, key="eda_topic")
    if theme == "性别与舱位":
        left, right = st.columns(2)
        with left:
            section("各舱位的性别构成")
            counts = filtered.assign(性别=filtered.Sex.map({"female":"女性", "male":"男性"})).groupby(["Pclass", "性别"]).size().reset_index(name="人数")
            fig = px.bar(counts, x="Pclass", y="人数", color="性别", text="人数", color_discrete_map={"女性":TEAL,"男性":NAVY})
            fig.update_xaxes(title="舱位", tickvals=[1,2,3], ticktext=["一等舱","二等舱","三等舱"])
            chart(fig, "eda_composition")
        with right:
            section("同一舱位，生存率有何差异？")
            means = filtered.pivot_table(index="Sex", columns="Pclass", values="Survived", aggfunc="mean").reindex(index=["female","male"], columns=[1,2,3])
            counts = filtered.pivot_table(index="Sex", columns="Pclass", values="Survived", aggfunc="size").reindex(index=["female","male"], columns=[1,2,3]).fillna(0)
            texts = [[f"{means.iloc[i,j]:.1%}<br>n={int(counts.iloc[i,j])}" if counts.iloc[i,j] else "无样本" for j in range(3)] for i in range(2)]
            fig = go.Figure(go.Heatmap(z=means.to_numpy(), x=["一等舱","二等舱","三等舱"], y=["女性","男性"], text=texts, texttemplate="%{text}",
                colorscale=[[0,"#EDF4F2"],[.5,"#72B9AF"],[1,NAVY]], zmin=0,zmax=1, colorbar=dict(tickformat=".0%",title="生存率"), hovertemplate="%{y} · %{x}<br>%{text}<extra></extra>", hoverongaps=False))
            fig.update_yaxes(autorange="reversed")
            chart(fig, "eda_heatmap")
        st.caption("空白单元表示没有样本；其他单元显示该组合的实际生存率和人数。群体差异不等于因果效应。")
    elif theme == "年龄与家庭":
        left, right = st.columns(2)
        with left:
            section("原始年龄分布")
            known = filtered.dropna(subset=["Age"])
            if known.empty:
                st.info("当前记录的原始年龄全部缺失。")
            else:
                fig = px.histogram(known, x="Age", color="生存结果", color_discrete_map=OUTCOME_COLORS, nbins=20, barmode="overlay", opacity=.7)
                fig.update_xaxes(title="原始年龄（岁）")
                fig.update_yaxes(title="人数")
                chart(fig, "eda_age_hist")
        with right:
            section("各年龄组生存率")
            chart(forest_plot(rate_summary(filtered,"AgeGroup",order=AGE_LABELS),"AgeGroup"), "eda_age_rate")
        section("家庭规模与生存率")
        family = filtered.assign(家庭类别=filtered.FamilyGroup.astype(object).map(FAMILY_LABELS))
        chart(forest_plot(rate_summary(family,"家庭类别",order=list(FAMILY_LABELS.values())),"家庭类别",GOLD), "eda_family_rate", 320)
        st.caption("横线为95% Wilson区间，悬停查看生存人数/组内人数。样本少的组往往有更宽的区间。")
    elif theme == "票价分布":
        fare = filtered.dropna(subset=["Fare"]).copy()
        fare["log(1＋票价)"] = np.log1p(fare.Fare)
        left, right = st.columns(2)
        with left:
            section("舱位票价：箱线图与观测点")
            fare["舱位"] = fare.Pclass.map({1:"一等舱",2:"二等舱",3:"三等舱"})
            fig = px.box(fare, x="舱位", y="log(1＋票价)", color="舱位", points="all", color_discrete_sequence=PALETTE, category_orders={"舱位":["一等舱","二等舱","三等舱"]})
            fig.update_traces(marker=dict(size=4, opacity=.35), jitter=.35)
            fig.update_layout(showlegend=False)
            chart(fig, "eda_fare_box")
        with right:
            section("两类结果的票价累积分布")
            fig = go.Figure()
            for outcome in ["未生存","生存"]:
                values = np.sort(fare.loc[fare["生存结果"].eq(outcome),"log(1＋票价)"].to_numpy())
                if len(values):
                    unique, counts = np.unique(values, return_counts=True)
                    cumulative = np.cumsum(counts) / len(values)
                    # Preserve ties as one jump, include the zero baseline, and
                    # keep a single-observation group visible via a marker.
                    end = float(fare["log(1＋票价)"].max()) + .08
                    x = np.r_[0.0, unique, end]
                    y = np.r_[0.0, cumulative, 1.0]
                    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", marker=dict(size=4), line=dict(shape="hv",color=OUTCOME_COLORS[outcome],width=2.5),name=f"{outcome}（n={len(values)}）",hovertemplate="log(1＋票价)=%{x:.3f}<br>累计比例 %{y:.1%}<extra>%{fullData.name}</extra>"))
            fig.update_xaxes(title="log(1＋票价)")
            fig.update_yaxes(title="组内累计比例",tickformat=".0%",range=[0,1.02])
            chart(fig,"eda_fare_ecdf")
        st.caption("对数转换只用于清楚展示偏斜分布；本项目分类模型中的票价采用标准化。每条累积曲线的分母是该结果组的人数。")
    else:
        group_choice = st.selectbox("查看补充特征",["称谓类别","舱号记录状态","登船港口"],key="eda_extra")
        group_map = {"称谓类别":"TitleGroup","舱号记录状态":"HasCabin","登船港口":"Embarked"}
        group = group_map[group_choice]
        extra = filtered.copy()
        if group == "HasCabin":
            extra[group] = extra[group].map({0:"舱号记录缺失",1:"有舱号记录"})
        elif group == "Embarked":
            mode = train.Embarked.mode().iloc[0]
            extra[group] = extra[group].fillna(mode)
            st.caption(f"此分析按原项目规则，以训练部分港口众数 {mode} 填充缺失港口。")
        chart(forest_plot(rate_summary(extra,group),group),"eda_extra_rate",420)
        st.caption("点为观察到的生存率，横线为95% Wilson区间。Child类别来自Master称谓；HasCabin表示记录状态。")
    with st.expander("查看当前筛选明细与统计口径"):
        st.write(f"当前筛选包括{len(filtered)}名训练乘客；生存率为{survivors}÷{len(filtered)}。年龄范围筛选只作用于已知年龄，缺失年龄由单独的选项控制。")
        display_table(filtered[["PassengerId","Sex","Pclass","Age","Fare","FamilySize","Survived"]].rename(columns=FIELD_LABELS))
        download_csv(filtered[["PassengerId","Sex","Pclass","Age","Fare","FamilySize","Survived"]],"下载当前筛选数据","training_selection.csv","download_eda")


def score_table(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["label","accuracy","accuracy_std","f1","roc_auc","fit_time"] if c in frame]
    return frame[cols].rename(columns={"label":"方案", "accuracy":"准确率", "accuracy_std":"准确率折间标准差", "f1":"F1", "roc_auc":"ROC-AUC", "fit_time":"完整流水线平均拟合秒"})


def experiments(results: dict) -> None:
    hero("每一次选择，都有对照", "固定分层划分，逐步比较填充方法、分类模型和参数，依据实验结果确定最终方案。",
         "04 / MODEL EXPERIMENTS", ("训练部分 n＝712", "五折分层交叉验证", "预处理在每折内部学习"))
    if not results:
        required(["results.json"])
        return
    age_tab, model_tab, tune_tab = st.tabs(["年龄填充对照", "五种分类模型", "逻辑回归参数"])
    with age_tab:
        frame = pd.DataFrame(results.get("age_comparison",[]))
        if not frame.empty:
            a,b=st.columns(2)
            with a:
                section("平均准确率与折间波动")
                fig=go.Figure(go.Scatter(x=frame.label,y=frame.accuracy,mode="markers",marker=dict(color=TEAL,size=12),error_y=dict(type="data",array=frame.accuracy_std,thickness=1.6,width=6),customdata=frame.accuracy_std,hovertemplate="%{x}<br>平均准确率 %{y:.2%}<br>折间标准差 %{customdata:.4f}<extra></extra>"))
                fig.update_yaxes(title="准确率（局部范围）",tickformat=".0%",range=[.75,.9])
                chart(fig,"exp_age_accuracy")
            with b:
                section("完整流水线拟合时间")
                fig=px.bar(frame,x="label",y="fit_time",color="label",color_discrete_sequence=PALETTE,text="fit_time")
                fig.update_traces(texttemplate="%{text:.3f}秒",textposition="outside")
                fig.update_xaxes(title=None)
                fig.update_yaxes(title="每折平均拟合时间（秒）")
                fig.update_layout(showlegend=False)
                chart(fig,"exp_age_cost")
            display_table(score_table(frame).round(4))
            note("如何做出选择", "按平均准确率选择随机森林填充。它相比分组中位数的提升幅度较小，而计算成本更高；本次结果不能证明其在所有划分上都显著优于简单方法。")
            st.caption("误差条是5折准确率的样本标准差，不是置信区间。拟合时间包括整条流水线，数值依运行环境而变化。")
    with model_tab:
        frame=pd.DataFrame(results.get("model_comparison",[]))
        if not frame.empty:
            sort_label=st.selectbox("比较与排序指标",list(METRIC_LABELS.values()),key="model_metric")
            metric=next(k for k,v in METRIC_LABELS.items() if v==sort_label)
            frame=frame.sort_values(metric,ascending=False)
            left,right=st.columns([1,1.2])
            with left:
                section(f"按{sort_label}排序")
                fig=go.Figure(go.Scatter(x=frame[metric],y=frame.label,mode="markers",marker=dict(size=12,color=TEAL),
                    error_x=dict(type="data",array=frame.accuracy_std,thickness=1.4,width=5) if metric=="accuracy" else None,
                    hovertemplate="%{y}<br>%{x:.4f}<extra></extra>"))
                fig.update_yaxes(autorange="reversed")
                fig.update_xaxes(title=sort_label,range=[.65,1],tickformat=".0%" if metric=="accuracy" else ".2f")
                chart(fig,"exp_models_rank")
            with right:
                section("不同指标，提供不同视角")
                z=frame[["accuracy","f1","roc_auc"]].to_numpy()
                fig=go.Figure(go.Heatmap(z=z,x=["Accuracy","F1","ROC-AUC"],y=frame.label,text=np.round(z,4),texttemplate="%{text:.4f}",colorscale=[[0,"#F2F6F5"],[.6,"#76BAB1"],[1,NAVY]],zmin=0,zmax=1,colorbar=dict(title="指标值")))
                fig.update_yaxes(autorange="reversed")
                chart(fig,"exp_models_heat")
            display_table(score_table(frame).round(4))
            note("最终分类器：逻辑回归", "逻辑回归取得本次最高准确率和F1；梯度提升的ROC-AUC更高。项目预先以准确率作为主要选择指标，因此选择逻辑回归。年龄填充用的随机森林回归器，与这里比较的随机森林分类器是两个模型。")
    with tune_tab:
        frame=pd.DataFrame(results.get("tuning",[]))
        if not frame.empty:
            frame=frame.sort_values("C")
            section("C＝1在本次参数范围中表现最好")
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=frame.C,y=frame.accuracy+frame.accuracy_std,mode="lines",line=dict(width=0),showlegend=False,hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=frame.C,y=frame.accuracy-frame.accuracy_std,mode="lines",line=dict(width=0),fill="tonexty",fillcolor="rgba(22,140,137,.15)",name="均值±折间样本标准差",hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=frame.C,y=frame.accuracy,mode="lines+markers",line=dict(color=TEAL,width=2.5),marker=dict(size=10),name="平均准确率",hovertemplate="C=%{x}<br>准确率 %{y:.2%}<extra></extra>"))
            fig.add_vline(x=float(results.get("selection",{}).get("C",1)),line_dash="dash",line_color=GOLD)
            fig.update_xaxes(type="log",title="正则化参数 C（对数刻度）",tickvals=frame.C)
            fig.update_yaxes(title="准确率（局部范围）",tickformat=".0%",range=[.74,.9])
            chart(fig,"exp_tuning",420)
            display_table(frame.rename(columns={"accuracy":"准确率","accuracy_std":"准确率折间标准差","f1":"F1","roc_auc":"ROC-AUC"}).round(4))
            st.caption("C越小，对模型系数的约束越强。本次调参确认C＝1，没有使准确率超出前面的逻辑回归结果。阴影表示折间标准差。")
    ablation=results.get("ablation")
    if ablation and ablation.get("rows"):
        with st.expander("补充实验：特征对照"):
            ablation_frame=pd.DataFrame(ablation["rows"])
            baseline=ablation_frame.loc[ablation_frame.feature_removed.isna(),"accuracy"]
            if not baseline.empty:
                changes=ablation_frame.loc[ablation_frame.feature_removed.notna()].copy()
                changes["准确率变化（百分点）"]=(changes.accuracy-float(baseline.iloc[0]))*100
                fig=go.Figure(go.Bar(x=changes["准确率变化（百分点）"],y=changes.label,orientation="h",
                    marker_color=[CORAL if v<0 else TEAL for v in changes["准确率变化（百分点）"]],
                    text=[f"{v:+.2f}" for v in changes["准确率变化（百分点）"]],textposition="outside",
                    hovertemplate="%{y}<br>相对完整特征变化 %{x:+.2f} 个百分点<extra></extra>"))
                fig.add_vline(x=0,line_color=NAVY,line_width=1)
                fig.update_xaxes(title="相对完整特征方案的平均准确率变化（百分点）")
                fig.update_yaxes(autorange="reversed")
                chart(fig,"exp_ablation",290)
            display_table(score_table(ablation_frame).round(4))
            st.caption("每次从分类器输入中移除一项构造特征，使用相同训练部分、相同五折划分与固定模型参数。这里比较的是本次验证表现，不是因果贡献或显著性结论；不改变固定留出评估的归属。")


def evaluation(results: dict) -> None:
    hero("用未参与选型的数据检验", "模型选择完成后，固定在179名留出乘客上评估。总体表现与分组错误，一起呈现。",
         "05 / HOLDOUT EVALUATION", ("独立留出 n＝179", "固定模型与判定规则", "不用于再次调参"))
    if not required(["holdout_predictions.csv","results.json"]):
        return
    data=csv("holdout_predictions.csv")
    holdout=results["holdout"]
    metrics([("准确率",f"{holdout['accuracy']:.2%}",f"{holdout['correct']} / {holdout['n']} 人预测正确"),
             ("精确率",f"{holdout['precision']:.2%}","预测为生存的人中，有多少实际生存"),
             ("召回率",f"{holdout['recall']:.2%}","实际生存的人中，模型找到了多少"),
             ("F1",f"{holdout['f1']:.4f}","综合精确率与召回率"),
             ("ROC-AUC",f"{holdout['roc_auc']:.4f}","反映概率排序的区分能力")])
    left,right=st.columns(2)
    with left:
        section("每一类错误，具体有多少人")
        cm=np.asarray(holdout["confusion_matrix"])
        fig=go.Figure(go.Heatmap(z=cm,x=["预测未生存","预测生存"],y=["实际未生存","实际生存"],text=cm,texttemplate="%{text}人",colorscale=[[0,"#ECF4F1"],[1,NAVY]],showscale=False,hovertemplate="%{y}<br>%{x}<br>%{z}人<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        chart(fig,"eval_confusion")
        st.caption(f"对角线为正确预测：{int(cm[0,0])}＋{int(cm[1,1])}＝{int(np.trace(cm))}人。另有{int(cm[0,1])}人与{int(cm[1,0])}人分别属于两种错误。")
    with right:
        section("改变阈值时，区分能力如何")
        roc=holdout["roc"]
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=roc["fpr"],y=roc["tpr"],mode="lines",line=dict(color=TEAL,width=2.5),name=f"逻辑回归 · AUC={holdout['roc_auc']:.4f}"))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(color=MUTED,dash="dash"),name="随机区分参考线"))
        fig.update_xaxes(title="假阳性率",range=[0,1])
        fig.update_yaxes(title="真阳性率（召回率）",range=[0,1.02])
        chart(fig,"eval_roc")
        st.caption("ROC-AUC不等于分类准确率。它衡量模型给生存者较高分数的排序能力。")
    baseline=float(data.Survived.eq(0).mean())
    note("对照一个简单参考",f"如果全部预测为未生存，本次留出集准确率为{baseline:.2%}。当前模型准确率为{holdout['accuracy']:.2%}，提高了{(holdout['accuracy']-baseline)*100:.2f}个百分点。")
    section("哪些群体更值得关注？", "ERROR ANALYSIS")
    group_label=st.radio("按什么分组查看错误",["性别","舱位","原始年龄组"],horizontal=True,key="error_group")
    grouped=data.copy()
    if group_label=="性别":
        grouped["群体"]=grouped.Sex.map({"female":"女性","male":"男性"})
        order=["女性","男性"]
    elif group_label=="舱位":
        grouped["群体"]=grouped.Pclass.map({1:"一等舱",2:"二等舱",3:"三等舱"})
        order=["一等舱","二等舱","三等舱"]
    else:
        grouped["群体"]=age_groups(grouped.Age)
        order=AGE_LABELS
    summary=rate_summary(grouped,"群体","Error",order)
    left,right=st.columns([1.4,1])
    with left:
        chart(forest_plot(summary,"群体",CORAL,"错误率"),"eval_group_errors",390)
    with right:
        display_table(summary.rename(columns={"n":"人数","k":"错误人数","rate":"错误率","lower":"95%区间下限","upper":"95%区间上限"}).round(4))
        st.caption("横线为95% Wilson区间。小样本组的比例更容易受少数记录影响。各分组视角都在重新统计同一批错误，不可跨表相加。")
    with st.expander("查看预测错误的乘客记录"):
        errors=data.loc[data.Error.eq(1),["PassengerId","Sex","Pclass","Age","Fare","Survived","Predicted","Probability"]]
        display_table(errors.rename(columns=FIELD_LABELS).round(4),"error_records")
        download_csv(errors,"下载错误记录","holdout_errors.csv","download_errors")
        st.caption("这里的年龄保留原始缺失状态；模型内部已执行年龄填充。此表用于解释本次评估，不用于继续调整已报告的模型。")


def prediction(results: dict) -> None:
    hero("把分析变成一次预测", "输入一名乘客的信息，体验完整流水线；也可以查询并下载418名测试乘客的预测结果。",
         "06 / PREDICTION EXPERIENCE", ("全量891人重新训练", "模型输出为估计", "不含真实测试标签"))
    one_tab,batch_tab=st.tabs(["单名乘客预测","418人测试预测"])
    with one_tab:
        st.caption("表单只使用称谓类别，不需要输入真实姓名；舱号用“是否有记录”表示。程序据此构造与训练时一致的原始字段。")
        with st.form("passenger_form"):
            a,b,c=st.columns(3)
            with a:
                sex=st.selectbox("性别",["女性","男性"],key="predict_sex")
                pclass=st.selectbox("舱位等级",[1,2,3],index=2,format_func=lambda x:f"{x}等舱",key="predict_class")
                title=st.selectbox("姓名中的称谓",["Miss","Mrs","Mr","Master","Dr","Rev","Sir","未知称谓"],key="predict_title",help="请按记录选择。Mr通常为男性，Miss/Mrs通常为女性，Master为男童称谓。")
            with b:
                age_unknown=st.checkbox("年龄未知，交给模型填充",value=False,key="predict_age_unknown")
                age=st.number_input("年龄（岁）",min_value=0.0,max_value=100.0,value=28.0,step=1.0,key="predict_age",help="勾选年龄未知时，此数值不会使用。")
                fare=st.number_input("票价",min_value=0.0,max_value=1000.0,value=15.0,step=1.0,key="predict_fare")
            with c:
                sibsp=st.number_input("同行兄弟姐妹及配偶人数",min_value=0,max_value=8,value=0,step=1,key="predict_sibsp")
                parch=st.number_input("同行父母及子女人数",min_value=0,max_value=6,value=0,step=1,key="predict_parch")
                embarked=st.selectbox("登船港口",["S","C","Q","未知"],key="predict_embarked")
                cabin=st.checkbox("有舱号记录",value=False,key="predict_cabin")
            submitted=st.form_submit_button("预测这名乘客",type="primary",width="stretch")
        if submitted:
            if required(["model.joblib"]):
                if (sex=="男性" and title in ["Miss","Mrs"]) or (sex=="女性" and title in ["Mr","Master","Sir"]):
                    st.warning("本次性别与称谓的组合不符合数据中的常见记录方式，请核对这两项输入。以下仍按你提交的原始组合计算，程序不会自动改动称谓。")
                chosen_title="Unlisted" if title=="未知称谓" else title
                inputs=pd.DataFrame([{"PassengerId":0,"Pclass":pclass,"Name":f"Example, {chosen_title}. Passenger",
                    "Sex":"female" if sex=="女性" else "male","Age":np.nan if age_unknown else age,
                    "SibSp":sibsp,"Parch":parch,"Ticket":"DEMO","Fare":fare,"Cabin":"C0" if cabin else np.nan,
                    "Embarked":np.nan if embarked=="未知" else embarked}])
                model_path=ARTIFACTS/"model.joblib"
                try:
                    model=read_model(str(model_path),file_stamp(model_path))
                    pred=int(model.predict(inputs)[0])
                    class_index=list(model.classes_).index(1)
                    probability=float(model.predict_proba(inputs)[0,class_index])
                    st.session_state["single_prediction"]={"prediction":pred,"probability":probability,
                        "summary":{ "性别":sex,"舱位":f"{pclass}等舱","原始年龄":"未知（由流水线估计）" if age_unknown else f"{age:g}岁",
                                    "称谓":title,"票价":f"{fare:g}","家庭规模":str(sibsp+parch+1),"港口":embarked,"舱号记录":"有" if cabin else "缺失"}}
                except (ValueError,TypeError,KeyError,AttributeError,ImportError,ModuleNotFoundError,OSError) as exc:
                    st.error("模型暂时无法完成预测。请核对部署依赖版本，或由维护者重新生成离线模型。")
                    with st.expander("诊断信息"):
                        st.code(f"{type(exc).__name__}: {exc}")
        last=st.session_state.get("single_prediction")
        if last:
            section("上一次提交的预测结果")
            metrics([("预测类别","生存" if last["prediction"] else "未生存","使用模型默认分类判定"),
                     ("估计生存概率",f"{last['probability']:.1%}","这是模型分数，不是真实群体生存率")])
            st.progress(last["probability"])
            display_table(pd.DataFrame([last["summary"]]))
            st.caption("结果对应上一次点击预测时的输入。修改表单后需要重新提交。该模型用于历史数据学习与课程展示；改变输入得到的差异不能解释为因果效应。")
    with batch_tab:
        if required(["submission.csv"]):
            submissions=csv("submission.csv")
            positive=int(submissions.Survived.sum())
            metrics([("测试乘客",str(len(submissions)),"无真实生存标签"),("预测生存",str(positive),f"{positive/len(submissions):.2%}"),
                     ("预测未生存",str(len(submissions)-positive),f"{1-positive/len(submissions):.2%}")])
            left,right=st.columns([.8,1.4])
            with left:
                fig=go.Figure(go.Pie(labels=["预测生存","预测未生存"],values=[positive,len(submissions)-positive],hole=.68,marker=dict(colors=[TEAL,NAVY]),textinfo="percent",sort=False))
                fig.update_layout(annotations=[dict(text=f"{len(submissions)}<br>条预测",x=.5,y=.5,font_size=22,showarrow=False)])
                chart(fig,"test_prediction_donut",320)
            with right:
                result_filter=st.selectbox("显示哪些预测",["全部","预测生存","预测未生存"],key="test_outcome")
                id_query=st.text_input("按乘客编号查询",placeholder="例如 893；留空显示全部",key="test_id")
                view=submissions.copy()
                if result_filter!="全部":
                    view=view.loc[view.Survived.eq(1 if result_filter=="预测生存" else 0)]
                if id_query.strip():
                    try:
                        pid=int(id_query.strip())
                        view=view.loc[view.PassengerId.eq(pid)]
                    except ValueError:
                        st.info("乘客编号请输入整数。")
                        view=view.iloc[:0]
                if view.empty:
                    st.info("没有符合当前查询条件的预测记录。")
                else:
                    display_table(view.assign(预测结果=view.Survived.map({0:"未生存",1:"生存"})).rename(columns={"PassengerId":"乘客编号","Survived":"预测标签"}),"test_prediction_table",height=265)
                download_csv(submissions,"下载完整418人预测CSV","titanic_submission.csv","download_submission")
                if not view.empty and len(view)!=len(submissions):
                    download_csv(view,"下载当前查询结果","titanic_filtered_predictions.csv","download_filtered")
            accuracy=results.get("holdout",{}).get("accuracy")
            evaluation_text=f"{accuracy:.2%}的准确率来自前面的179人留出评估。" if accuracy is not None else "分类准确率来自前面的179人留出评估。"
            note("评估结果与测试预测分开解读", evaluation_text+f"这里的{len(submissions)}人没有真实标签，不能计算其预测准确率；{positive}人、{len(submissions)-positive}人描述的是模型预测分布。")


def main() -> None:
    st.set_page_config(page_title="Titanic · 乘客生存分析",page_icon="◈",layout="wide",initial_sidebar_state="expanded")
    setup_style()
    with st.sidebar:
        st.markdown('<div class="project-brand">TITANIC / DATA STUDIO</div>',unsafe_allow_html=True)
        st.markdown("### 乘客生存分析")
        st.caption("从数据问题，走向可解释的预测")
        st.divider()
        page=st.radio("探索项目",PAGES,key="page",label_visibility="collapsed")
        st.divider()
        st.markdown("**数据使用范围**")
        st.caption("训练与交叉验证：712人\n\n独立留出评估：179人\n\n无标签测试预测：418人")
        st.caption("所有数据与模型均从本项目目录读取。")
    results=get_results()
    renderers={"项目概览":overview,"数据质量与处理":quality,"交互式生存分析":exploration,"模型实验":experiments,"留出评估":evaluation,"预测体验":prediction}
    try:
        renderers[page](results)
    except FileNotFoundError:
        st.info("此页面所需数据尚未准备好。请按README中的步骤准备项目数据与离线成果后刷新。")
    footer()


if __name__=="__main__":
    main()

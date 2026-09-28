"""Shared presentation helpers; no data mutation or model fitting."""
from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

NAVY = "#163B53"
TEAL = "#168C89"
GOLD = "#D9A441"
CORAL = "#D87763"
MUTED = "#647887"
PALETTE = [TEAL, NAVY, GOLD, CORAL, "#8099B0"]
OUTCOME_COLORS = {"未生存": NAVY, "生存": TEAL}
FAMILY_LABELS = {
    "Alone": "独自出行 · 1人", "Small": "小家庭 · 2–4人",
    "Medium": "中等家庭 · 5–7人", "Large": "大家庭 · 8人及以上",
}
AGE_LABELS = ["0–<12岁", "12–<18岁", "18–<35岁", "35–<60岁", "60岁及以上", "原始年龄缺失"]


def setup_style() -> None:
    # Only our own HTML classes are styled; native widgets keep Streamlit styling.
    st.markdown("""
    <style>
    .project-brand {font-size:.78rem; font-weight:750; letter-spacing:.17em; color:#168C89; margin:.2rem 0 .7rem;}
    .hero {background:linear-gradient(115deg,#14374D 0%,#20566A 72%,#237875 100%); padding:2.1rem 2.2rem; border-radius:20px; color:white; margin:0 0 1.5rem;}
    .hero .eyebrow {font-size:.74rem; letter-spacing:.14em; color:#BCE4DC; font-weight:700;}
    .hero h1 {font-size:clamp(1.75rem,3vw,2.65rem); line-height:1.3; color:white; margin:.55rem 0 .8rem; padding:0;}
    .hero p {font-size:.97rem; line-height:1.8; color:#DAE9EB; margin:0; max-width:920px;}
    .hero .tag {display:inline-block; border:1px solid #7BA7AE; padding:.22rem .7rem; border-radius:20px; color:#E4F0EC; font-size:.75rem; margin:1rem .35rem 0 0;}
    .note-card {background:#EDF5F3; border-left:3px solid #168C89; border-radius:0 12px 12px 0; padding:1rem 1.2rem; margin:.5rem 0 1.2rem; color:#214B56; line-height:1.7;}
    .note-card strong {color:#163B53;}
    .section-kicker {color:#168C89; font-size:.75rem; font-weight:750; letter-spacing:.12em; margin:1.1rem 0 .25rem;}
    .page-foot {font-size:.76rem; color:#73858D; border-top:1px solid #DCE5E6; padding-top:1rem; margin-top:2rem; line-height:1.7;}
    .step-card {background:white; border:1px solid #DFE8E9; border-radius:14px; padding:1.1rem; min-height:150px;}
    .step-card .num {color:#168C89; font-size:.78rem; font-weight:750; letter-spacing:.1em;}
    .step-card h4 {color:#163B53; font-size:1.03rem; padding:.5rem 0; margin:0;}
    .step-card p {color:#647887; font-size:.86rem; line-height:1.6; margin:0;}
    @media(max-width:640px){.hero{padding:1.5rem;}.step-card{min-height:auto;}}
    </style>
    """, unsafe_allow_html=True)


def hero(title: str, subtitle: str, eyebrow: str, tags: tuple[str, ...] = ()) -> None:
    chips = "".join(f'<span class="tag">{escape(t)}</span>' for t in tags)
    st.markdown(
        f'<section class="hero"><div class="eyebrow">{escape(eyebrow)}</div>'
        f'<h1>{escape(title)}</h1><p>{escape(subtitle)}</p>{chips}</section>',
        unsafe_allow_html=True,
    )


def note(title: str, text: str) -> None:
    st.markdown(f'<div class="note-card"><strong>{escape(title)}</strong><br>{escape(text)}</div>', unsafe_allow_html=True)


def step_card(number: str, title: str, body: str) -> None:
    st.markdown(f'<div class="step-card"><div class="num">{escape(number)}</div><h4>{escape(title)}</h4><p>{escape(body)}</p></div>', unsafe_allow_html=True)


def section(title: str, eyebrow: str | None = None) -> None:
    if eyebrow:
        st.markdown(f'<div class="section-kicker">{escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.subheader(title)


def metrics(items: list[tuple[str, str, str]]) -> None:
    for col, (label, value, caption) in zip(st.columns(len(items)), items):
        with col, st.container(border=True):
            st.metric(label, value)
            st.caption(caption)


def style_figure(fig: go.Figure, height: int = 370) -> go.Figure:
    fig.update_layout(
        template="plotly_white", font=dict(family="Microsoft YaHei, PingFang SC, Arial, sans-serif", size=12, color=NAVY),
        colorway=PALETTE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(l=22, r=22, t=28, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(gridcolor="#E7EEEF", zerolinecolor="#DCE5E6", automargin=True)
    fig.update_yaxes(gridcolor="#E7EEEF", zerolinecolor="#DCE5E6", automargin=True)
    return fig


def chart(fig: go.Figure, key: str, height: int = 370) -> None:
    st.plotly_chart(style_figure(fig, height), width="stretch", theme=None, key=key,
                    config={"displaylogo": False, "scrollZoom": False,
                            "toImageButtonOptions": {"format": "png", "scale": 2}})


def age_groups(age: pd.Series) -> pd.Series:
    groups = pd.cut(age, bins=[0, 12, 18, 35, 60, np.inf], labels=AGE_LABELS[:-1], right=False)
    return groups.astype(object).where(age.notna(), AGE_LABELS[-1])


def rate_summary(df: pd.DataFrame, group: str, target: str = "Survived", order: list | None = None) -> pd.DataFrame:
    """Observed proportions with Wilson intervals (including k=0 and k=n)."""
    summary = df.groupby(group, observed=True, dropna=False)[target].agg(["count", "sum"]).rename(columns={"count": "n", "sum": "k"})
    if order is not None:
        summary = summary.reindex(order).dropna(subset=["n"])
    summary = summary[summary["n"] > 0].copy()
    if summary.empty:
        return pd.DataFrame(columns=[group, "n", "k", "rate", "lower", "upper"])
    n = summary["n"].to_numpy(dtype=float)
    p = summary["k"].to_numpy(dtype=float) / n
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    summary["rate"] = p
    summary["lower"] = np.maximum(0, center - half)
    summary["upper"] = np.minimum(1, center + half)
    return summary.reset_index()


def forest_plot(summary: pd.DataFrame, group: str, color: str = TEAL, xlabel: str = "生存率") -> go.Figure:
    fig = go.Figure()
    if summary.empty:
        return fig
    labels = [f"{g}  ·  n={int(n)}" for g, n in zip(summary[group], summary["n"])]
    fig.add_trace(go.Scatter(
        x=summary["rate"], y=labels, mode="markers", marker=dict(size=10, color=color),
        error_x=dict(type="data", symmetric=False, array=np.maximum(summary["upper"]-summary["rate"], 0),
                     arrayminus=np.maximum(summary["rate"]-summary["lower"], 0), thickness=1.6, width=4, color=color),
        customdata=np.column_stack([summary["k"], summary["n"], summary["lower"], summary["upper"]]),
        hovertemplate="%{y}<br>比例：%{x:.1%}<br>人数：%{customdata[0]:.0f}/%{customdata[1]:.0f}<br>95% Wilson区间：%{customdata[2]:.1%}–%{customdata[3]:.1%}<extra></extra>",
        showlegend=False,
    ))
    fig.update_xaxes(title=xlabel, range=[-.02, 1.04], tickformat=".0%")
    fig.update_yaxes(autorange="reversed", title=None)
    return fig


def footer() -> None:
    st.markdown('<div class="page-foot">TITANIC · 数据分析与机器学习课程项目<br>探索分析使用训练部分；正式评估使用固定留出部分；测试集结果为模型预测。图表支持悬停查看与PNG下载。</div>', unsafe_allow_html=True)

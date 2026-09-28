# 泰坦尼克号乘客生存分析与预测

用 Streamlit 将原有 41 步 Notebook 的数据检查、特征分析、建模比较、独立评估和预测结果组织成可交互的课程项目。网页读取已保存的实验成果；切换页面和筛选条件不会反复训练模型。

在线演示：[Titanic 乘客生存分析](https://zw917-titanic-analysis.streamlit.app/)。公开代码仓库：[ZW917/titanic-analysis-app](https://github.com/ZW917/titanic-analysis-app)。2026-09-28 已完成云端部署，并实际检查六页导航、交互筛选、模型预测与 CSV 下载。

本目录是独立创建的网页项目。原始 Notebook、CSV 和报告所在目录不属于本项目的写入目标。`notebooks/original_analysis.ipynb` 如存在，仅是原 Notebook 的参考副本，默认不进入 Git 仓库。

## 项目内容

| 页面 | 展示目的 |
|---|---|
| 项目概览 | 研究问题、数据范围、工作流程和主要结果 |
| 数据质量与处理 | 缺失值检查、三种年龄处理方案、填充前后核对 |
| 交互式生存分析 | 按性别、舱位、年龄等筛选训练数据，查看联动图表 |
| 模型实验 | 年龄填充方案、分类模型、逻辑回归参数的真实比较 |
| 留出评估 | 留出指标、混淆矩阵、ROC、分组错误率与错误记录 |
| 预测体验 | 输入乘客信息调用已训练流水线，查询和下载测试预测 |

六个页面与应用左侧导航一一对应。

## 数据和结果的范围

| 数据部分 | 人数 | 用途 |
|---|---:|---|
| 原始有标签数据 | 891 | 先划分，再进行训练和独立评估 |
| 训练部分 | 712 | 详细探索、学习预处理规则、五折比较和调参 |
| 留出部分 | 179 | 方案选定后的最终评估，不参与选型与调参 |
| 无标签测试数据 | 418 | 生成预测，没有真实标签可用于计算准确率 |

最终方案为 **随机森林回归器填补年龄 + 逻辑回归分类器（C=1）**。年龄模型使用 `Pclass`、`SibSp`、`Parch`，2000 棵树；它与模型比较中的随机森林分类器是不同任务。

新项目已经完成完整离线训练，复现了原 Notebook 的留出结果：Accuracy **83.24%（149/179）**、Precision **79.10%**、Recall **76.81%**、F1 **0.7794**、ROC-AUC **0.8715**；混淆矩阵为 `[[96, 14], [16, 53]]`，行表示真实类别，列表示预测类别，类别顺序均为“未生存、生存”。五折平均准确率为 **83.43%**。

方案确定后，在全部 891 条有标签数据上重新训练，用于最终测试和单人预测。新项目的 418 条预测与原 CSV 逐行一致：171 人预测生存、247 人预测未生存。网页数字从实际生成的 `artifacts` 读取。83.24% 属于前面 179 人的留出评估，不能用作全量重训模型对 418 人的准确率。

## Windows 本地启动

在存放项目的父文件夹打开 PowerShell，执行以下命令进入项目并启动。若已在项目根目录，跳过第一行。新建虚拟环境只安装本项目依赖，不更改电脑现有 Python 环境。

```powershell
Set-Location -LiteralPath '.\titanic_streamlit'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

按照终端显示的本地网址在浏览器中打开。退出时在启动终端按 `Ctrl+C`。如果当前终端已经使用本项目的虚拟环境，也可执行：

```powershell
python -m streamlit run app.py
```

依赖版本以 `requirements.txt` 为准；不要直接将现有电脑所有包导出后替换它。云端也应使用 Python 3.12 和同一份依赖清单。

## 文件说明

```text
titanic_streamlit/
├── app.py                         Streamlit 入口
├── ui.py                          页面和图表
├── pipeline.py                    特征构造、自定义处理器和模型流水线
├── requirements.txt               已验证的依赖版本
├── .streamlit/config.toml          应用显示设置
├── scripts/train.py               离线复现训练和导出结果
├── data/
│   ├── README.md                   数据来源和字段说明
│   ├── train.csv                   有标签数据副本
│   └── test.csv                    无标签数据副本
├── artifacts/
│   ├── results.json                模型比较、参数比较和评估结果
│   ├── train_subset.csv            训练部分及分析字段
│   ├── holdout_predictions.csv     留出真实标签、预测和概率
│   ├── preprocessing_preview.csv   缺失处理预览
│   ├── submission.csv              418 条最终预测
│   └── model.joblib                全量重训后的完整流水线
├── docs/
│   ├── DEPLOYMENT.md               GitHub 与云端部署步骤
│   └── DEFENSE.md                  演示路线、常见问题和验收清单
└── notebooks/original_analysis.ipynb  原 Notebook 参考副本（默认忽略）
```

应用通过项目内路径读取文件，不依赖原 Notebook 所在目录。复制整个项目到其他位置后，保持上述目录关系即可。不要在应用代码中加入个人电脑的绝对路径。

## 复现实验

只查看网页时，使用已经生成的 `artifacts` 即可。需要重新验证训练过程时，在本项目目录和对应环境中执行：

```powershell
.\.venv\Scripts\python.exe scripts/train.py
```

包含特征对照实验的复现命令：

```powershell
.\.venv\Scripts\python.exe scripts/train.py --ablation
```

三个特征移除对照均已完成，结果保存在 `artifacts/results.json`，也显示在“模型实验”页面。实验限定在 712 人的训练部分，使用同样的交叉验证划分，不根据 179 人留出结果调整模型。

| 特征方案 | 五折平均准确率 | 相比完整特征 |
|---|---:|---:|
| 完整特征 | 83.43% | 基准 |
| 移除家庭类别 | 81.32% | 下降 2.10 个百分点 |
| 移除称谓类别 | 81.74% | 下降 1.69 个百分点 |
| 移除舱号记录 | 82.30% | 下降 1.12 个百分点 |

差值按未四舍五入的结果计算。这些对照支持上述特征在本次模型和划分中的预测作用，不代表因果效应或统计显著性结论。三个对照是补充分析，原最终方案保持不变。

训练会写入本项目 `artifacts`。请在需要更新这些新项目成果时执行；它不应覆盖原项目目录里的任何文件。完整训练包含多个 2000 棵树的年龄模型，比打开网页更耗时，建议汇报前离线完成。

如果加载模型出现 scikit-learn 版本不一致警告，应恢复训练时依赖版本，或在目标环境重新训练并整体更新模型和结果。不要通过屏蔽警告把不兼容的加载视为验证通过。仅加载本项目可信流程生成的 `model.joblib`。

## 发布与交付

GitHub 保存代码和项目文件；Streamlit Community Cloud 运行 Python 应用。GitHub Pages 不能直接运行本项目的 Streamlit 服务。代码与[云端网页](https://zw917-titanic-analysis.streamlit.app/)均已发布。部署使用 `main` 分支、入口 `app.py`、Python 3.12，详细操作见 [部署说明](docs/DEPLOYMENT.md)。

项目采用公开 GitHub 仓库交付。上传内容排除 `.venv`、`.local`、本地日志、源文件完整性检查报告和包含个人路径的 Notebook 副本。数据来源及原发布方声明说明见 [数据说明](data/README.md)。

课程仍需按教师要求提交数据集、项目代码、实验报告并参加中期检查；网页是展示与复现补充。答辩建议见 [演示与讲解](docs/DEFENSE.md)。

## 项目来源与贡献说明

本项目题目由学生自行选择，网页在已有 41 步实验基础上进行工程化整理。数据来源沿用原项目的 [Heywhale 数据页面](https://www.heywhale.com/mw/dataset/5f69d01971c700003078960e/file)。来源链接不等于授权声明，本仓库不为原数据补充或变更许可证。

开发中使用 AI 辅助整理网页程序、交付文档和实现检查。学生应如实说明自己完成并能核验的选题、理解、设计判断、修改、运行、分析和讲解，不将参考资料或 AI 辅助部分表述为全部独立原创。最终结论以实际代码、数据和运行结果为依据，不保证某个评分等级。

技术依据： [Streamlit 部署文档](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)、[项目文件组织](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)。

本次检查记录见 [验证记录](docs/VERIFICATION.md)。当前电脑也可双击 start_app.cmd 启动网页（首次下载者先按上文安装依赖）。

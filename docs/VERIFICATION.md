# 项目验证记录

验证日期：2026-09-23。环境：Windows、Python 3.12.10、项目独立虚拟环境，依赖版本见 requirements.txt。

- 从复制的数据重新执行 `python scripts/train.py --ablation`，完整流程成功。
- 最终方案与原实验一致：随机森林填补年龄、逻辑回归分类、C=1。
- 留出部分179人，149人预测正确，Accuracy=0.8324022346；混淆矩阵为 `[[96,14],[16,53]]`。
- 全部891人重训的模型生成418条测试预测，与参考CSV逐条相同；171人预测生存，247人预测未生存。
- 新增三个训练集特征移除对照；数值来自 artifacts/results.json，不改变最终选定方案。
- `python -m pytest -q`：16项测试及6项子测试通过。包括六页导航、筛选分母、空筛选、年龄缺失预测、预测查询、保存模型重载、数据划分和指标独立重算。
- 将发布ZIP解压到不含原Notebook及本地配置的新目录，重复上述测试，16项测试及6项子测试全部通过。
- `python -m pip check`：无损坏依赖。
- 本地Streamlit健康检查返回HTTP 200；浏览器实际打开页面并检查首页、交互分析和留出评估的排版及图表。
- 受保护的8个原文件SHA256均与开始工作时相同；原Python环境的包版本和包集合未改变。含个人路径的校验记录仅保留在本地，不进入公开仓库。

以上为 2026-09-23 的本地验证记录；云端验证见下方 2026-09-28 补充。手机布局及异地网络访问未做专项测试。

## 2026-09-28 部署前复核

- `python -B -m pytest -q`：16 项测试、6 项子测试再次全部通过。
- `python -m pip check`：无损坏依赖；本地 Streamlit 健康检查返回 `ok`。
- 再次核对 8 个受保护原文件，SHA256 全部保持不变。
- 已发布公开仓库 [ZW917/titanic-analysis-app](https://github.com/ZW917/titanic-analysis-app)，未登录访问可读取项目文件。
- 首次发布的 `main` 提交为 `98f687a931568d20e1b4762de2504e72b6bf7dd4`；`git ls-remote` 与本地提交一致。
- 部署前检查通过后继续进行云端部署，结果见下一节。

## 2026-09-28 云端部署与实际操作验证

- 正式网页：[Titanic 乘客生存分析](https://zw917-titanic-analysis.streamlit.app/)。平台为 Streamlit Community Cloud，仓库为 `ZW917/titanic-analysis-app`，分支 `main`，入口 `app.py`。
- 高级设置选择 Python 3.12；构建日志确认实际运行 Python 3.12.14，Streamlit 1.64.0、scikit-learn 1.9.0，应用成功启动。
- 不带账号凭据的 HTTP 请求访问应用健康接口 `/~/+/_stcore/health`，返回 HTTP 200 和 `ok`。
- 另开无痕浏览器窗口访问正式网页，未登录 Streamlit 账号即可加载首页、图表及主要指标，确认公开访问正常。
- 浏览器实际切换六个页面，图表和数据正常显示。模型比较显示逻辑回归五折平均准确率 83.43%；留出评估显示 149/179 正确、83.24%、F1=0.7794、ROC-AUC=0.8715。
- 交互分析中筛选女性，显示 253 人、188 人生存、生存率 74.3%、39 人原始年龄缺失，与本地训练部分重新统计一致。
- 单人预测提交示例：女性、三等舱、Miss、年龄未知、无同行亲属、票价15、港口S、无舱号记录。云端返回“生存”、估计生存概率 63.8%；本地已保存流水线的独立计算为 0.6383035680521998，显示精度内一致。
- 测试预测页显示 418 人、171 人预测生存、247 人预测未生存；查询 893 号仅返回对应记录，标签为1。
- 从线上“下载完整418人预测CSV”取得 CSV，重新读取后确认列为 `PassengerId,Survived`，共418行，与 `artifacts/submission.csv` 全部逐行一致。
- 再次核对受保护的8个原文件，内容保持不变。

重跑测试：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

仅在原始构建电脑可执行 `python scripts/verify_originals.py`，它读取被忽略的 `.local/source_integrity.json`；其他下载者不需要此文件或此项检查。

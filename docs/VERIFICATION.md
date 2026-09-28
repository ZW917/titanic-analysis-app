# 本地验证记录

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

自动测试不等于云端已通过验收。手机布局、云端部署及异地访问仍需在实际环境验证。公开仓库和云端状态以README中的实际链接及后续验证记录为准。

## 2026-09-28 部署前复核

- `python -B -m pytest -q`：16 项测试、6 项子测试再次全部通过。
- `python -m pip check`：无损坏依赖；本地 Streamlit 健康检查返回 `ok`。
- 再次核对 8 个受保护原文件，SHA256 全部保持不变。
- 已发布公开仓库 [ZW917/titanic-analysis-app](https://github.com/ZW917/titanic-analysis-app)，未登录访问可读取项目文件。
- 首次发布的 `main` 提交为 `98f687a931568d20e1b4762de2504e72b6bf7dd4`；`git ls-remote` 与本地提交一致。
- 云端部署尚待完成，不能将本地检查或 GitHub 上传视作网页上线验收。

重跑测试：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

仅在原始构建电脑可执行 `python scripts/verify_originals.py`，它读取被忽略的 `.local/source_integrity.json`；其他下载者不需要此文件或此项检查。

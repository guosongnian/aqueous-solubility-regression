# 数据与内容来源

本项目使用 `tomonari-masada/course2026-sml` 仓库中的水溶解度数据，并将数据版本固定在提交：

`0884fce9ab63aab630e4e9d066c4c317dbb54ad4`

程序会从该固定版本下载以下文件，并通过 SHA-256 校验完整性：

- `solTrainX.csv`
- `solTrainY.csv`
- `solTestX.csv`
- `solTestY.csv`

原始课程材料：<https://github.com/tomonari-masada/course2026-sml>

数据可追溯至 Kuhn 与 Johnson 的 *Applied Predictive Modeling* 配套数据，以及 Delaney 等人的水溶解度建模工作。本仓库重新组织并独立实现分析流程，未复制原课程 Notebook 的文字内容。

请在继续分发或用于其他场景前，自行核对上游数据及其引用要求。

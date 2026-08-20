# 水溶解度回归：正则化与分子指纹降维

这个项目使用分子描述符和二进制分子指纹预测水溶解度 `logS`，重点比较普通线性回归、Ridge、Lasso、Elastic Net，以及“指纹 TruncatedSVD 降维 + Ridge”。

项目将原本分散的课程练习整理为一个可复现的完整实验：固定数据版本、自动校验文件、在交叉验证管道内完成预处理和调参，并且只在模型选择完成后评估一次测试集。

## 结果

数据包含 951 条训练样本、316 条测试样本和 228 个特征，其中包括 208 个二进制分子指纹与 20 个分子描述符。

| 模型 | 5 折交叉验证 RMSE |
|---|---:|
| Ridge | **0.649** |
| Elastic Net | 0.651 |
| Lasso | 0.653 |
| 指纹 TruncatedSVD + Ridge | 0.655 |
| 普通线性回归 | 0.732 |
| 中位数基线 | 2.064 |

交叉验证最终选择 `Ridge(alpha=10)`。在未参与调参的测试集上：

| RMSE | MAE | R² |
|---:|---:|---:|
| **0.752** | **0.567** | **0.869** |

正则化明显优于普通最小二乘；降维模型表现接近，但没有超过直接使用全部指纹的 Ridge。测试 RMSE 高于交叉验证 RMSE，说明独立测试集仍然更难，这也是比单独报告 R² 更重要的结果。

## 方法

- 使用上游仓库已经给定的训练集和测试集，不重新随机切分数据。
- 仅在训练集上执行 5 折交叉验证与超参数选择。
- 连续/计数描述符在每个交叉验证折内进行中位数填补和标准化。
- 208 个二进制指纹保持原始 0/1 表示；另一条候选管道在折内执行 TruncatedSVD。
- 使用 RMSE 作为模型选择指标，同时报告测试集 MAE 和 R²。
- 数据下载固定到上游提交，并使用 SHA-256 防止文件静默变化。

## 运行

建议使用 Python 3.11 或更高版本；GitHub Actions 使用 Python 3.12 自动检查依赖、模块导入和语法。

```bash
git clone https://github.com/guosongnian/aqueous-solubility-regression.git
cd aqueous-solubility-regression
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python solubility_regression.py
```

第一次运行会自动下载并校验四个 CSV 文件。完整参数搜索通常只需要几十秒；若只想检查环境，可运行：

```bash
python solubility_regression.py --quick
```

程序生成的模型比较、测试指标、预测结果和诊断图保存在本地 `outputs/` 目录。完整分析过程也可以直接阅读：

- [`notebooks/aqueous_solubility_regression.ipynb`](notebooks/aqueous_solubility_regression.ipynb)
- [`solubility_regression.py`](solubility_regression.py)

## 项目结构

```text
.
├── notebooks/
│   └── aqueous_solubility_regression.ipynb
├── .github/workflows/ci.yml
├── solubility_regression.py
├── requirements.txt
├── NOTICE.md
└── README.md
```

`data/` 与 `outputs/` 均由程序在本地创建，不提交生成文件。

## 局限与后续方向

- 当前样本量较小，分子指纹维度相对较高；结果可能随化学空间变化而改变。
- 随机 K 折不能保证结构相近的分子被分到同一折。更严格的评估可以使用基于分子骨架的分组划分，但需要额外的结构标识（例如 SMILES）。
- 当前项目专注于可解释的线性与正则化方法；树模型或图神经网络可以作为后续比较，但应继续遵守同一测试集隔离原则。

数据来源与引用说明见 [`NOTICE.md`](NOTICE.md)。

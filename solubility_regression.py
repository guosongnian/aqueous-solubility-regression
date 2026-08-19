"""水溶解度回归：线性模型、正则化与指纹降维的可复现实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
SOURCE_COMMIT = "0884fce9ab63aab630e4e9d066c4c317dbb54ad4"
RAW_BASE = (
    "https://raw.githubusercontent.com/tomonari-masada/course2026-sml/"
    f"{SOURCE_COMMIT}"
)
DATA_FILES = {
    "solTrainX.csv": "bb2cc4f9d25d40026db2a1687c536bf16356735d2f7c63d7078e9d856179f88f",
    "solTrainY.csv": "0a6da0c0523a37c9393f85fda9c41685c8a8f4b4918fcb823bcf0523cf4f24c1",
    "solTestX.csv": "3a4ed91bd42eac637652a6e4a9fded46f167a8dd1331fd8bb4a574386b45ceba",
    "solTestY.csv": "66cc9fc6e13c44265a6c2dc6c711698d6c4787522ab042a733a16aaad825d751",
}


def file_sha256(path: Path) -> str:
    """计算文件的 SHA-256，用于确认实验数据没有发生变化。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_data(data_dir: Path) -> None:
    """下载缺失的数据，并校验固定版本的文件哈希。"""
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, expected_hash in DATA_FILES.items():
        path = data_dir / filename
        if not path.exists():
            url = f"{RAW_BASE}/data/{filename}"
            print(f"下载 {filename} ...")
            urllib.request.urlretrieve(url, path)
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{filename} 校验失败：期望 {expected_hash}，实际 {actual_hash}"
            )


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """读取官方划分的训练集和测试集，并进行基本一致性检查。"""
    ensure_data(data_dir)
    x_train = pd.read_csv(data_dir / "solTrainX.csv", index_col=0).reset_index(drop=True)
    y_train = pd.read_csv(data_dir / "solTrainY.csv", index_col=0).iloc[:, 0].reset_index(drop=True)
    x_test = pd.read_csv(data_dir / "solTestX.csv", index_col=0).reset_index(drop=True)
    y_test = pd.read_csv(data_dir / "solTestY.csv", index_col=0).iloc[:, 0].reset_index(drop=True)

    if list(x_train.columns) != list(x_test.columns):
        raise ValueError("训练集和测试集的特征列不一致。")
    if x_train.isna().any().any() or x_test.isna().any().any():
        print("提示：特征中存在缺失值，将由管道内的中位数填补处理。")
    return x_train, y_train, x_test, y_test


def make_preprocessors(
    descriptor_columns: list[str], fingerprint_columns: list[str]
) -> tuple[ColumnTransformer, ColumnTransformer]:
    """构造直接建模与指纹降维两种预处理器。"""
    descriptors = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    direct = ColumnTransformer(
        [
            ("descriptors", descriptors, descriptor_columns),
            # 指纹均为 0/1 变量，保持原始尺度便于比较不同正则化方法。
            ("fingerprints", "passthrough", fingerprint_columns),
        ]
    )
    reduced = ColumnTransformer(
        [
            ("descriptors", descriptors, descriptor_columns),
            (
                "fingerprints",
                TruncatedSVD(random_state=RANDOM_STATE),
                fingerprint_columns,
            ),
        ]
    )
    return direct, reduced


def build_searches(
    descriptor_columns: list[str],
    fingerprint_columns: list[str],
    quick: bool = False,
) -> tuple[GridSearchCV, GridSearchCV]:
    """建立候选模型搜索；每一步预处理都在交叉验证折内拟合。"""
    direct_preprocessor, reduced_preprocessor = make_preprocessors(
        descriptor_columns, fingerprint_columns
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    common = {
        "scoring": "neg_root_mean_squared_error",
        "cv": cv,
        "n_jobs": -1,
        "return_train_score": False,
    }

    direct_pipeline = Pipeline(
        [("preprocess", direct_preprocessor), ("model", Ridge())]
    )
    if quick:
        ridge_alphas = [0.1, 1.0, 10.0]
        lasso_alphas = [0.003, 0.03]
        elastic_alphas = [0.003, 0.03]
        elastic_ratios = [0.2, 0.8]
    else:
        ridge_alphas = np.logspace(-3, 3, 13).tolist()
        lasso_alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
        elastic_alphas = [0.001, 0.003, 0.01, 0.03, 0.1]
        elastic_ratios = [0.2, 0.5, 0.8]

    direct_grid = [
        {"model": [DummyRegressor(strategy="median")]},
        {"model": [LinearRegression()]},
        {"model": [Ridge()], "model__alpha": ridge_alphas},
        {
            "model": [Lasso(max_iter=30_000, random_state=RANDOM_STATE)],
            "model__alpha": lasso_alphas,
        },
        {
            "model": [ElasticNet(max_iter=30_000, random_state=RANDOM_STATE)],
            "model__alpha": elastic_alphas,
            "model__l1_ratio": elastic_ratios,
        },
    ]
    direct_search = GridSearchCV(direct_pipeline, direct_grid, **common)

    reduced_pipeline = Pipeline(
        [("preprocess", reduced_preprocessor), ("model", Ridge())]
    )
    component_grid = [20, 40, 80] if quick else [20, 40, 80, 120]
    alpha_grid = [0.1, 1.0, 10.0] if quick else np.logspace(-2, 2, 9).tolist()
    reduced_search = GridSearchCV(
        reduced_pipeline,
        {
            "preprocess__fingerprints__n_components": component_grid,
            "model__alpha": alpha_grid,
        },
        **common,
    )
    return direct_search, reduced_search


def model_family(model: object) -> str:
    """将 sklearn 对象映射为便于展示的模型名称。"""
    names = {
        "DummyRegressor": "中位数基线",
        "LinearRegression": "普通线性回归",
        "Ridge": "Ridge",
        "Lasso": "Lasso",
        "ElasticNet": "Elastic Net",
    }
    return names.get(type(model).__name__, type(model).__name__)


def summarize_searches(
    direct_search: GridSearchCV, reduced_search: GridSearchCV
) -> pd.DataFrame:
    """汇总每类模型在交叉验证中的最优结果。"""
    direct = pd.DataFrame(direct_search.cv_results_)
    direct["模型"] = direct["param_model"].map(model_family)
    direct["CV_RMSE"] = -direct["mean_test_score"]
    best_direct = (
        direct.sort_values("CV_RMSE")
        .groupby("模型", as_index=False)
        .first()[["模型", "CV_RMSE", "params"]]
    )

    reduced = pd.DataFrame(reduced_search.cv_results_)
    best_reduced = reduced.loc[reduced["mean_test_score"].idxmax()]
    reduced_row = pd.DataFrame(
        [
            {
                "模型": "指纹 SVD + Ridge",
                "CV_RMSE": -best_reduced["mean_test_score"],
                "params": best_reduced["params"],
            }
        ]
    )
    return (
        pd.concat([best_direct, reduced_row], ignore_index=True)
        .sort_values("CV_RMSE")
        .reset_index(drop=True)
    )


def test_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """计算一次最终测试评估指标。"""
    return {
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def save_diagnostics(
    y_true: pd.Series, y_pred: np.ndarray, output_dir: Path
) -> None:
    """保存预测值和残差诊断图。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    residuals = np.asarray(y_true) - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7, ax=axes[0])
    limits = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(limits, limits, "--", color="black", linewidth=1)
    axes[0].set(xlabel="Observed logS", ylabel="Predicted logS", title="Observed vs predicted")

    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7, ax=axes[1])
    axes[1].axhline(0, linestyle="--", color="black", linewidth=1)
    axes[1].set(xlabel="Predicted logS", ylabel="Residual", title="Residual diagnostics")
    fig.tight_layout()
    fig.savefig(output_dir / "test_diagnostics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(project_root: Path, quick: bool = False) -> dict[str, object]:
    """运行完整实验并将结果写入 outputs 目录。"""
    sns.set_theme(style="whitegrid")
    data_dir = project_root / "data"
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_test, y_test = load_data(data_dir)

    fingerprint_columns = [c for c in x_train.columns if c.startswith("FP")]
    descriptor_columns = [c for c in x_train.columns if c not in fingerprint_columns]
    print(
        f"训练集: {x_train.shape}；测试集: {x_test.shape}；"
        f"分子指纹: {len(fingerprint_columns)}；描述符: {len(descriptor_columns)}"
    )

    direct_search, reduced_search = build_searches(
        descriptor_columns, fingerprint_columns, quick=quick
    )
    print("搜索线性与正则化模型 ...")
    direct_search.fit(x_train, y_train)
    print("搜索指纹降维模型 ...")
    reduced_search.fit(x_train, y_train)

    comparison = summarize_searches(direct_search, reduced_search)
    print("\n交叉验证结果（越低越好）:")
    print(comparison[["模型", "CV_RMSE"]].to_string(index=False))

    searches = [direct_search, reduced_search]
    selected_search = max(searches, key=lambda search: search.best_score_)
    selected_name = (
        "指纹 SVD + Ridge"
        if selected_search is reduced_search
        else model_family(selected_search.best_estimator_.named_steps["model"])
    )

    # 模型选择到此结束；测试集只在选定模型上评估一次。
    predictions = selected_search.best_estimator_.predict(x_test)
    metrics = test_metrics(y_test, predictions)
    result = {
        "selected_model": selected_name,
        "best_params": {
            key: (type(value).__name__ if key == "model" else value)
            for key, value in selected_search.best_params_.items()
        },
        "cross_validation_rmse": float(-selected_search.best_score_),
        "test_metrics": metrics,
    }
    print(f"\n最终模型: {selected_name}")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    comparison.assign(params=comparison["params"].astype(str)).to_csv(
        output_dir / "model_comparison.csv", index=False
    )
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    pd.DataFrame(
        {"observed_logS": y_test, "predicted_logS": predictions, "residual": y_test - predictions}
    ).to_csv(output_dir / "test_predictions.csv", index=False)
    save_diagnostics(y_test, predictions, output_dir)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="使用较小的参数网格，适合快速检查运行环境。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(Path(__file__).resolve().parent, quick=arguments.quick)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from app.utils import logger, CHARTS_DIR, load_config

config = load_config()
chart_config = config.get("chart", {})

_chart_counter = 0

def _reset_chart_counter():
    global _chart_counter
    _chart_counter = 0

def _next_chart_number() -> int:
    global _chart_counter
    _chart_counter += 1
    return _chart_counter

def _get_chart_caption(chart_type: str, chart_number: int, details: Dict[str, Any] = None) -> Tuple[str, str]:
    captions = {
        "histogram": {
            "title": f"图{chart_number}",
            "note": f"图{chart_number} 各变量分布的直方图。横轴为表达量，纵轴为频数，红色虚线为均值，绿色虚线为中位数。",
        },
        "boxplot": {
            "title": f"图{chart_number}",
            "note": f"图{chart_number} 各变量在不同条件下的箱线图。箱体表示四分位距，须线表示1.5倍IQR范围，圆点为异常值。",
        },
        "violin": {
            "title": f"图{chart_number}",
            "note": f"图{chart_number} 各变量分布的小提琴图。核密度估计曲线展示数据分布形态，宽度表示概率密度。",
        },
        "correlation": {
            "title": f"图{chart_number}",
            "note": f"图{chart_number} 变量间相关系数热图。颜色越深表示相关性越强，红色表示正相关，蓝色表示负相关。",
        },
        "bar": {
            "title": f"图{chart_number}",
            "note": f"图{chart_number} 各组均值的柱状图。误差棒表示标准差，用于比较不同组间的差异。",
        },
        "interactive_scatter": {
            "title": f"图{chart_number}",
            "note": f"图{chart_number} 两变量散点图。每个点代表一个样本，用于观察变量间的线性或非线性关系。",
        },
        "dose_response": {
            "title": f"图{chart_number}",
            "note": f"图{chart_number} 剂量-效应曲线。红色曲线为四参数Logistic拟合，绿色虚线标注IC50值。",
        },
        "survival_curve": {
            "title": f"图{chart_number}",
            "note": f"图{chart_number} Kaplan-Meier生存曲线。阶梯线表示生存概率随时间的变化，红色虚线标注中位生存时间。",
        },
        "residuals": {
            "title": f"图{chart_number}",
            "note": f"图{chart_number} 残差诊断图。左图为残差vs拟合值散点图（检验同方差性），右图为残差Q-Q图（检验正态性）。",
        },
    }
    cap = captions.get(chart_type, {"title": f"图{chart_number}", "note": f"图{chart_number} 数据分析图表。"})
    return cap["title"], cap["note"]

def set_chinese_font():
    plt.rcParams["font.sans-serif"] = ["SimSun", "DejaVu Sans", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

def save_chart(filename: str, session_id: Optional[str] = None) -> str:
    if session_id:
        session_dir = CHARTS_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        filepath = session_dir / filename
    else:
        filepath = CHARTS_DIR / filename
    plt.savefig(filepath, dpi=chart_config.get("dpi", 300), bbox_inches="tight")
    plt.close()
    logger.info(f"Chart saved: {filepath}")
    if session_id:
        return f"{session_id}/{filename}"
    return filename

def plot_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, error_bars: bool = True,
                   group_col: Optional[str] = None, session_id: Optional[str] = None) -> str:
    set_chinese_font()
    fig, ax = plt.subplots(figsize=tuple(chart_config.get("figure_size", [10, 6])))

    if group_col and group_col in df.columns:
        grouped = df.groupby([x_col, group_col])[y_col].mean().reset_index()
        if error_bars:
            errors = df.groupby([x_col, group_col])[y_col].std().reset_index()
            errors.columns = [x_col, group_col, "error"]
            grouped = grouped.merge(errors, on=[x_col, group_col])
            grouped["error"] = grouped["error"].fillna(0)
        pivot = grouped.pivot(index=x_col, columns=group_col, values=y_col)
        pivot.plot(kind="bar", ax=ax, color=chart_config.get("colors", None))
        if error_bars and "error" in grouped.columns:
            error_pivot = grouped.pivot(index=x_col, columns=group_col, values="error")
            ax.errorbar(range(len(pivot)), pivot.values.T, yerr=error_pivot.values.T, fmt="none", c="black", capsize=5)
    else:
        grouped = df.groupby(x_col)[y_col].mean().reset_index()
        if error_bars:
            errors = df.groupby(x_col)[y_col].std().fillna(0)
            ax.bar(grouped[x_col].astype(str), grouped[y_col], yerr=errors, capsize=5, color=chart_config.get("colors", ["#4C72B0"]))
        else:
            ax.bar(grouped[x_col].astype(str), grouped[y_col], color=chart_config.get("colors", ["#4C72B0"]))

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} by {x_col}")
    plt.tight_layout()

    filename = f"bar_{x_col}_{y_col}.png"
    return save_chart(filename, session_id)

def plot_line_chart(df: pd.DataFrame, x_col: str, y_col: str, group_col: Optional[str] = None,
                    session_id: Optional[str] = None) -> str:
    set_chinese_font()
    fig, ax = plt.subplots(figsize=tuple(chart_config.get("figure_size", [10, 6])))

    if group_col and group_col in df.columns:
        for name, group in df.groupby(group_col):
            sorted_group = group.sort_values(x_col)
            ax.plot(sorted_group[x_col], sorted_group[y_col], marker="o", label=str(name))
    else:
        sorted_df = df.sort_values(x_col)
        ax.plot(sorted_df[x_col], sorted_df[y_col], marker="o", color=chart_config.get("colors", ["#4C72B0"])[0])

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} over {x_col}")
    ax.legend()
    plt.tight_layout()

    filename = f"line_{x_col}_{y_col}.png"
    return save_chart(filename, session_id)

def plot_boxplot(df: pd.DataFrame, value_col: str, group_col: Optional[str] = None,
                 session_id: Optional[str] = None) -> str:
    set_chinese_font()
    fig, ax = plt.subplots(figsize=tuple(chart_config.get("figure_size", [10, 6])))

    if group_col and group_col in df.columns:
        data = [df[df[group_col] == g][value_col].dropna() for g in df[group_col].unique()]
        labels = [str(g) for g in df[group_col].unique()]
        ax.boxplot(data, labels=labels)
    else:
        ax.boxplot(df[value_col].dropna())

    ax.set_ylabel(value_col)
    ax.set_title(f"Boxplot of {value_col}")
    plt.tight_layout()

    filename = f"boxplot_{value_col}.png"
    return save_chart(filename, session_id)

def plot_violin(df: pd.DataFrame, value_col: str, group_col: Optional[str] = None,
                session_id: Optional[str] = None) -> str:
    set_chinese_font()
    fig, ax = plt.subplots(figsize=tuple(chart_config.get("figure_size", [10, 6])))

    if group_col and group_col in df.columns:
        sns.violinplot(data=df, x=group_col, y=value_col, ax=ax, palette=chart_config.get("colors", None))
    else:
        sns.violinplot(data=df, y=value_col, ax=ax, palette=chart_config.get("colors", None))

    ax.set_title(f"Violin plot of {value_col}")
    plt.tight_layout()

    filename = f"violin_{value_col}.png"
    return save_chart(filename, session_id)

def plot_histogram(df: pd.DataFrame, col: str, bins: int = 30, session_id: Optional[str] = None) -> str:
    set_chinese_font()
    fig, ax = plt.subplots(figsize=tuple(chart_config.get("figure_size", [10, 6])))

    ax.hist(df[col].dropna(), bins=bins, edgecolor="black", alpha=0.7, color=chart_config.get("colors", ["#4C72B0"])[0])
    ax.axvline(df[col].mean(), color="red", linestyle="--", label=f"Mean: {df[col].mean():.2f}")
    ax.axvline(df[col].median(), color="green", linestyle="--", label=f"Median: {df[col].median():.2f}")

    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of {col}")
    ax.legend()
    plt.tight_layout()

    filename = f"histogram_{col}.png"
    return save_chart(filename, session_id)

def plot_correlation_heatmap(df: pd.DataFrame, method: str = "pearson", session_id: Optional[str] = None) -> str:
    set_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 8))

    numeric_df = df.select_dtypes(include=[np.number])
    if method == "pearson":
        corr = numeric_df.corr(method="pearson")
    else:
        corr = numeric_df.corr(method="spearman")

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                fmt=".2f", ax=ax, square=True, linewidths=0.5)

    ax.set_title(f"Correlation Heatmap ({method})")
    plt.tight_layout()

    filename = f"correlation_heatmap_{method}.png"
    return save_chart(filename, session_id)

def plot_dose_response(df: pd.DataFrame, dose_col: str, response_col: str,
                       fitted_params: Optional[Dict[str, Any]] = None,
                       session_id: Optional[str] = None) -> str:
    set_chinese_font()
    fig, ax = plt.subplots(figsize=tuple(chart_config.get("figure_size", [10, 6])))

    data = df[[dose_col, response_col]].dropna().sort_values(by=dose_col)
    ax.scatter(data[dose_col], data[response_col], color="blue", label="Data", zorder=5)

    if fitted_params:
        from scipy.optimize import curve_fit
        def four_param_logistic(x, bottom, top, log_ic50, hill):
            return bottom + (top - bottom) / (1 + 10 ** ((log_ic50 - x) * hill))

        log_x = np.log10(data[dose_col].values)
        y_pred = four_param_logistic(log_x, *fitted_params.get("parameters", [0, 100, 1, 1]))
        ax.plot(data[dose_col], y_pred, color="red", linewidth=2, label="4PL Fit")

        ic50 = fitted_params.get("ic50", None)
        if ic50:
            ax.axvline(ic50, color="green", linestyle="--", label=f"IC50 = {ic50:.2f}")

    ax.set_xlabel(dose_col)
    ax.set_ylabel(response_col)
    ax.set_title("Dose-Response Curve")
    ax.legend()
    ax.set_xscale("log")
    plt.tight_layout()

    filename = "dose_response.png"
    return save_chart(filename, session_id)

def plot_survival_curve(survival_data: Dict[str, Any], session_id: Optional[str] = None) -> str:
    set_chinese_font()
    fig, ax = plt.subplots(figsize=tuple(chart_config.get("figure_size", [10, 6])))

    curve = survival_data.get("survival_curve", {})
    times = curve.get("times", [])
    probs = curve.get("survival_prob", [])

    if times and probs:
        ax.step(times, probs, where="post", label="Kaplan-Meier", color="blue")
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival Probability")
        ax.set_title("Kaplan-Meier Survival Curve")

        median = survival_data.get("median_survival")
        if median:
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
            ax.axvline(median, color="red", linestyle="--", label=f"Median: {median:.1f}")

        ax.legend()
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "No survival data available", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()

    filename = "survival_curve.png"
    return save_chart(filename, session_id)

def plot_residual_diagnostics(regression_result: Dict[str, Any], df: pd.DataFrame,
                               target_col: str, feature_cols: List[str],
                               session_id: Optional[str] = None) -> str:
    set_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    data = df[[target_col] + feature_cols].dropna()
    if data.empty:
        fig.suptitle("数据不足，无法生成残差图")
        filename = "residual_diagnostics.png"
        return save_chart(filename, session_id)
    
    try:
        from sklearn.linear_model import LinearRegression
        X = data[feature_cols].values
        y = data[target_col].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        axes[0].scatter(y_pred, residuals, alpha=0.6, color="#4C72B0")
        axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
        axes[0].set_xlabel("拟合值")
        axes[0].set_ylabel("残差")
        axes[0].set_title("残差 vs 拟合值")
        axes[0].grid(True, alpha=0.3)
        
        import scipy.stats as stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title("残差 Q-Q 图")
        axes[1].grid(True, alpha=0.3)
    except Exception as e:
        logger.error(f"Residual plot error: {e}")
        fig.suptitle("残差图生成失败")
    
    fig.suptitle("回归模型残差诊断", fontsize=14, y=1.02)
    plt.tight_layout()
    
    filename = "residual_diagnostics.png"
    return save_chart(filename, session_id)

def plot_interactive_bar(df: pd.DataFrame, x_col: str, y_col: str, group_col: Optional[str] = None) -> str:
    if group_col and group_col in df.columns:
        fig = px.bar(df, x=x_col, y=y_col, color=group_col, barmode="group", title=f"{y_col} by {x_col}")
    else:
        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
    return plotly.io.to_json(fig)

def plot_interactive_line(df: pd.DataFrame, x_col: str, y_col: str, group_col: Optional[str] = None) -> str:
    if group_col and group_col in df.columns:
        fig = px.line(df, x=x_col, y=y_col, color=group_col, markers=True, title=f"{y_col} over {x_col}")
    else:
        fig = px.line(df, x=x_col, y=y_col, markers=True, title=f"{y_col} over {x_col}")
    return plotly.io.to_json(fig)

def plot_interactive_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: Optional[str] = None) -> str:
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
    return plotly.io.to_json(fig)

def plot_interactive_heatmap(df: pd.DataFrame, method: str = "pearson") -> str:
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(method=method)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title=f"Correlation Heatmap ({method})")
    return plotly.io.to_json(fig)

def plot_interactive_boxplot(df: pd.DataFrame, value_col: str, group_col: Optional[str] = None) -> str:
    if group_col and group_col in df.columns:
        fig = px.box(df, x=group_col, y=value_col, color=group_col, title=f"Boxplot of {value_col}")
    else:
        fig = px.box(df, y=value_col, title=f"Boxplot of {value_col}")
    return plotly.io.to_json(fig)

def generate_all_charts(df: pd.DataFrame, session_id: str, analysis_results: Dict[str, Any],
                         chart_captions: Dict[str, str] = None) -> Dict[str, str]:
    _reset_chart_counter()
    charts = {}
    chart_caption_map = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    try:
        if numeric_cols:
            first_num = numeric_cols[0]
            charts["histogram"] = plot_histogram(df, first_num, session_id=session_id)
            charts["boxplot"] = plot_boxplot(df, first_num, session_id=session_id)
            charts["violin"] = plot_violin(df, first_num, session_id=session_id)

        if len(numeric_cols) >= 2:
            charts["correlation"] = plot_correlation_heatmap(df, session_id=session_id)
            charts["interactive_correlation"] = plot_interactive_heatmap(df)

        if numeric_cols and categorical_cols:
            charts["bar"] = plot_bar_chart(df, categorical_cols[0], numeric_cols[0], session_id=session_id)
            charts["interactive_bar"] = plot_interactive_bar(df, categorical_cols[0], numeric_cols[0])

        if len(numeric_cols) >= 2:
            charts["interactive_scatter"] = plot_interactive_scatter(df, numeric_cols[0], numeric_cols[1])

        if "dose_response" in analysis_results:
            dr = analysis_results["dose_response"]
            if "error" not in dr:
                charts["dose_response"] = plot_dose_response(df, dr.get("dose_col", ""), dr.get("response_col", ""), dr, session_id=session_id)

        if "survival_analysis" in analysis_results:
            sa = analysis_results["survival_analysis"]
            if "error" not in sa:
                charts["survival_curve"] = plot_survival_curve(sa, session_id=session_id)

        if "regression" in analysis_results or "regression_analysis" in analysis_results:
            reg_data = analysis_results.get("regression") or analysis_results.get("regression_analysis", {})
            if isinstance(reg_data, dict) and "error" not in reg_data and "model" in reg_data:
                target = reg_data.get("target", "")
                features = reg_data.get("features", [])
                if target and features:
                    charts["residual_diagnostics"] = plot_residual_diagnostics(
                        reg_data, df, target, features, session_id=session_id
                    )

    except Exception as e:
        logger.error(f"Error generating charts: {e}")

    for chart_key in list(charts.keys()):
        if not chart_key.startswith("interactive_"):
            _next_chart_number()
            caption_title, caption_note = _get_chart_caption(chart_key, _chart_counter)
            chart_caption_map[chart_key] = {"title": caption_title, "note": caption_note}

    charts["chart_captions"] = chart_caption_map
    return charts

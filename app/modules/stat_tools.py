import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import (
    ttest_ind, ttest_rel, mannwhitneyu, shapiro, levene,
    f_oneway, pearsonr, spearmanr, kruskal, norm
)
from typing import Dict, Any, List, Optional, Tuple
from app.utils import logger
import warnings
warnings.filterwarnings("ignore")

def descriptive_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return {"error": "没有数值列可用于描述统计"}

    result = {}
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) < 2:
            continue
        result[col] = {
            "count": int(len(data)),
            "mean": float(data.mean()),
            "std": float(data.std()),
            "median": float(data.median()),
            "min": float(data.min()),
            "max": float(data.max()),
            "q1": float(data.quantile(0.25)),
            "q3": float(data.quantile(0.75)),
            "iqr": float(data.quantile(0.75) - data.quantile(0.25)),
            "skewness": float(data.skew()),
            "kurtosis": float(data.kurtosis()),
            "missing": int(df[col].isnull().sum()),
            "missing_pct": round(float(df[col].isnull().sum() / len(df[col]) * 100), 2),
        }
    return result

def check_normality(data: pd.Series) -> Dict[str, Any]:
    if len(data) < 3:
        return {"is_normal": False, "statistic": None, "p_value": None, "suggestion": "样本量过小，无法进行正态性检验"}
    if len(data) > 5000:
        data = data.sample(5000, random_state=42)
    try:
        stat, p_value = shapiro(data)
        is_normal = p_value > 0.05
        suggestion = "数据服从正态分布" if is_normal else "数据不服从正态分布，建议使用非参数检验"
        return {
            "test": "Shapiro-Wilk",
            "is_normal": is_normal,
            "statistic": float(stat),
            "p_value": float(p_value),
            "suggestion": suggestion,
        }
    except Exception as e:
        return {"test": "Shapiro-Wilk", "is_normal": False, "error": str(e)}

def t_test_independent(df: pd.DataFrame, col: str, group_col: str, group1: str, group2: str) -> Dict[str, Any]:
    try:
        g1 = df[df[group_col] == group1][col].dropna()
        g2 = df[df[group_col] == group2][col].dropna()
        if len(g1) < 2 or len(g2) < 2:
            return {"error": "每组至少需要2个样本"}

        normality_1 = check_normality(g1)
        normality_2 = check_normality(g2)
        both_normal = normality_1.get("is_normal", False) and normality_2.get("is_normal", False)

        _, levene_p = levene(g1, g2)
        equal_var = levene_p > 0.05

        stat, p_value = ttest_ind(g1, g2, equal_var=equal_var)

        pooled_std = np.sqrt(((len(g1)-1)*g1.std()**2 + (len(g2)-1)*g2.std()**2) / (len(g1)+len(g2)-2))
        cohens_d = float((g1.mean() - g2.mean()) / pooled_std) if pooled_std > 0 else 0

        result = {
            "test": "Independent t-test",
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "equal_variance": equal_var,
            "levene_p": float(levene_p),
            "group1_mean": float(g1.mean()),
            "group2_mean": float(g2.mean()),
            "group1_n": len(g1),
            "group2_n": len(g2),
            "effect_size": {"cohens_d": cohens_d},
            "normality": {"group1": normality_1, "group2": normality_2},
        }
        if not both_normal:
            result["suggestion"] = "数据非正态，建议使用Mann-Whitney U检验"
        return result
    except Exception as e:
        return {"error": str(e)}

def t_test_paired(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    try:
        data = df[[col1, col2]].dropna()
        if len(data) < 2:
            return {"error": "至少需要2对配对样本"}
        stat, p_value = ttest_rel(data[col1], data[col2])
        diff = data[col1] - data[col2]
        cohens_d = float(diff.mean() / diff.std()) if diff.std() > 0 else 0
        return {
            "test": "Paired t-test",
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "n_pairs": len(data),
            "mean_diff": float(diff.mean()),
            "effect_size": {"cohens_d": cohens_d},
        }
    except Exception as e:
        return {"error": str(e)}

def mann_whitney_u(df: pd.DataFrame, col: str, group_col: str, group1: str, group2: str) -> Dict[str, Any]:
    try:
        g1 = df[df[group_col] == group1][col].dropna()
        g2 = df[df[group_col] == group2][col].dropna()
        if len(g1) < 2 or len(g2) < 2:
            return {"error": "每组至少需要2个样本"}
        stat, p_value = mannwhitneyu(g1, g2, alternative="two-sided")
        n1, n2 = len(g1), len(g2)
        r_squared = float(1 - (6 * stat) / (n1 * n2 * (n1 + n2 + 1)))
        return {
            "test": "Mann-Whitney U",
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "group1_median": float(g1.median()),
            "group2_median": float(g2.median()),
            "effect_size": {"r_squared": max(0, r_squared)},
        }
    except Exception as e:
        return {"error": str(e)}

def one_way_anova(df: pd.DataFrame, col: str, group_col: str) -> Dict[str, Any]:
    """执行单因素方差分析，包含分组变量合理性检测"""
    try:
        # 分组变量合理性检测
        unique_groups = df[group_col].dropna().unique()
        n_groups = len(unique_groups)
        
        # 检查组数是否有效
        if n_groups < 2:
            return {
                "error": "不适用（分组无效）",
                "reason": f"{group_col} 只有 {n_groups} 个有效组，无法进行组间比较",
                "n_groups": n_groups,
                "groups": [str(g) for g in unique_groups],
            }
        
        # 检查每组样本数
        group_sizes = {str(g): len(df[df[group_col] == g][col].dropna()) for g in unique_groups}
        min_group_size = min(group_sizes.values())
        
        if min_group_size < 2:
            small_groups = [g for g, size in group_sizes.items() if size < 2]
            return {
                "error": "不适用（分组无效）",
                "reason": f"以下组样本数不足2个：{', '.join(small_groups)}，无法进行方差分析",
                "n_groups": n_groups,
                "group_sizes": group_sizes,
            }
        
        # 检查是否为唯一标识符（每组只出现一次）
        if all(size == 1 for size in group_sizes.values()):
            return {
                "error": "不适用（分组无效）",
                "reason": f"{group_col} 为唯一标识符，每个值只出现一次，无法进行组间比较",
                "n_groups": n_groups,
                "group_sizes": group_sizes,
            }
        
        groups = [df[df[group_col] == g][col].dropna() for g in unique_groups]
        groups = [g for g in groups if len(g) >= 2]
        
        if len(groups) < 2:
            return {
                "error": "不适用（分组无效）",
                "reason": f"{group_col} 只有 {len(groups)} 个有效组（每组至少2个样本）",
                "n_groups": n_groups,
                "group_sizes": group_sizes,
            }

        normality_results = {}
        all_normal = True
        for g_name, g_data in zip(df[group_col].unique(), groups):
            nr = check_normality(g_data)
            normality_results[str(g_name)] = nr
            if not nr.get("is_normal", False):
                all_normal = False

        _, levene_p = levene(*groups)
        equal_variance = levene_p > 0.05

        # 根据正态性和方差齐性选择检验方法
        use_nonparametric = not all_normal or not equal_variance
        
        if use_nonparametric:
            # 使用Kruskal-Wallis非参数检验
            stat, p_value = kruskal(*groups)
            test_name = "Kruskal-Wallis H"
            eta_squared = None
            interpretation = "数据不满足正态性或方差齐性假设，采用非参数Kruskal-Wallis检验"
        else:
            # 使用标准方差分析
            stat, p_value = f_oneway(*groups)
            test_name = "One-way ANOVA"
            
            ss_between = sum(len(g) * (g.mean() - np.mean([g.mean() for g in groups]))**2 for g in groups)
            ss_total = sum(((g - g.mean())**2).sum() for g in groups) + ss_between
            eta_squared = float(ss_between / ss_total) if ss_total > 0 else 0
            interpretation = "数据满足正态性和方差齐性假设，采用参数检验One-way ANOVA"

        # 效应量解读
        effect_interpretation = None
        if eta_squared is not None:
            if eta_squared < 0.01:
                effect_interpretation = "效应量极小（η² < 0.01），组间差异可能无实际意义"
            elif eta_squared < 0.06:
                effect_interpretation = "小效应（0.01 ≤ η² < 0.06），组间存在轻微差异"
            elif eta_squared < 0.14:
                effect_interpretation = "中等效应（0.06 ≤ η² < 0.14），组间存在明显差异"
            else:
                effect_interpretation = "大效应（η² ≥ 0.14），组间存在显著差异"

        result = {
            "test": test_name,
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "n_groups": len(groups),
            "group_means": {str(g.name): float(g.mean()) for g in groups},
            "levene_p": float(levene_p),
            "equal_variance": equal_variance,
            "effect_size": {"eta_squared": eta_squared} if eta_squared is not None else {"note": "非参数检验不计算η²"},
            "normality": normality_results,
            "all_normal": all_normal,
            "interpretation": interpretation,
            "effect_interpretation": effect_interpretation,
        }

        # 异常统计值解释
        if np.isinf(stat) or stat > 1e10:
            result["statistic_note"] = "F值极大，表明组间差异极为显著（组间方差远大于组内方差）"
        if p_value == 0:
            result["p_value_note"] = "P值显示为0，实际为极小值（< 1e-300），表明结果高度显著"

        # 事后检验
        if p_value < 0.05:
            if use_nonparametric:
                result["post_hoc"] = _dunn_test(df, col, group_col)
            else:
                result["post_hoc"] = _tukey_hsd(df, col, group_col)
            
            # 矛盾结果解释
            post_hoc = result.get("post_hoc", {})
            comparisons = post_hoc.get("comparisons", [])
            if comparisons:
                sig_comparisons = [c for c in comparisons if c.get("significant")]
                if not sig_comparisons:
                    result["contradiction_note"] = (
                        "注意：整体检验显著但事后两两比较均不显著。"
                        "可能原因：整体检验检验的是组间总体差异，事后检验进行了多重比较校正（如Bonferroni），"
                        "提高了显著性阈值；样本量较小导致事后检验效能不足；③差异分布在多个组的组合中而非单一组间。"
                        "建议：增加样本量或关注效应量较大的组间比较。"
                    )

        return result
    except Exception as e:
        return {"error": str(e)}

def _dunn_test(df: pd.DataFrame, col: str, group_col: str) -> Dict[str, Any]:
    """Dunn事后检验（非参数）"""
    try:
        from itertools import combinations
        from scipy.stats import rankdata
        
        groups = df[group_col].unique()
        comparisons = []
        n_comparisons = len(groups) * (len(groups) - 1) // 2
        
        for g1, g2 in combinations(groups, 2):
            d1 = df[df[group_col] == g1][col].dropna()
            d2 = df[df[group_col] == g2][col].dropna()
            _, p = mannwhitneyu(d1, d2, alternative="two-sided")
            # Bonferroni校正
            adjusted_p = min(p * n_comparisons, 1.0)
            comparisons.append({
                "group1": str(g1),
                "group2": str(g2),
                "p_value": float(p),
                "adjusted_p_value": float(adjusted_p),
                "significant": adjusted_p < 0.05,
            })
        
        return {
            "method": "Dunn检验 (Bonferroni校正)",
            "comparisons": comparisons,
            "significant": any(c["significant"] for c in comparisons),
        }
    except Exception as e:
        return {"method": "Dunn检验", "error": str(e)}

def _tukey_hsd(df: pd.DataFrame, col: str, group_col: str) -> Dict[str, Any]:
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        groups = df[group_col].unique()
        if len(groups) < 3:
            from itertools import combinations
            comparisons = []
            for g1, g2 in combinations(groups, 2):
                d1 = df[df[group_col] == g1][col].dropna()
                d2 = df[df[group_col] == g2][col].dropna()
                _, p = ttest_ind(d1, d2)
                comparisons.append({
                    "group1": str(g1),
                    "group2": str(g2),
                    "p_value": float(p),
                    "significant": p < 0.05 / (len(groups) * (len(groups) - 1) / 2),
                })
            return {"method": "Bonferroni-corrected t-tests", "comparisons": comparisons}

        data = df[[col, group_col]].dropna()
        tukey = pairwise_tukeyhsd(endog=data[col], groups=data[group_col], alpha=0.05)
        
        # Parse Tukey results from summary table
        comparisons = []
        summary_data = tukey.summary().data
        for row in summary_data[1:]:  # Skip header row
            g1, g2 = row[0], row[1]
            p_val = float(row[4])
            reject = row[5] == "True" or row[5] is True
            comparisons.append({
                "group1": str(g1),
                "group2": str(g2),
                "mean_diff": float(row[2]),
                "p_value": p_val,
                "significant": reject,
            })
        
        return {
            "method": "Tukey HSD",
            "comparisons": comparisons,
            "significant": any(c["significant"] for c in comparisons),
        }
    except Exception as e:
        return {"method": "Tukey HSD (fallback)", "error": str(e)}

def correlation_analysis(df: pd.DataFrame, method: str = "pearson") -> Dict[str, Any]:
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return {"error": "至少需要2个数值列"}

        columns = numeric_df.columns.tolist()
        corr_matrix = np.zeros((len(columns), len(columns)))
        p_matrix = np.zeros((len(columns), len(columns)))

        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_matrix[i, j] = 0.0
                elif j > i:
                    data = df[[col1, col2]].dropna()
                    if len(data) < 3:
                        corr_matrix[i, j] = corr_matrix[j, i] = np.nan
                        p_matrix[i, j] = p_matrix[j, i] = np.nan
                        continue
                    if method == "pearson":
                        r, p = pearsonr(data[col1], data[col2])
                    else:
                        r, p = spearmanr(data[col1], data[col2])
                    corr_matrix[i, j] = corr_matrix[j, i] = r
                    p_matrix[i, j] = p_matrix[j, i] = p

        return {
            "method": method,
            "columns": columns,
            "correlation_matrix": corr_matrix.tolist(),
            "p_value_matrix": p_matrix.tolist(),
            "significant_pairs": [
                {
                    "var1": columns[i],
                    "var2": columns[j],
                    "correlation": float(corr_matrix[i, j]),
                    "p_value": float(p_matrix[i, j]),
                    "significant": p_matrix[i, j] < 0.05,
                }
                for i in range(len(columns))
                for j in range(i + 1, len(columns))
                if not np.isnan(p_matrix[i, j])
            ],
        }
    except Exception as e:
        return {"error": str(e)}

def _calculate_vif(df: pd.DataFrame, features: List[str]) -> Dict[str, float]:
    """计算方差膨胀因子(VIF)用于共线性诊断"""
    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        X = sm.add_constant(df[features].dropna())
        vif_data = {}
        for i, feat in enumerate(features):
            vif_data[feat] = float(variance_inflation_factor(X.values, i + 1))
        return vif_data
    except Exception:
        return {}

def _interpret_vif(vif_data: Dict[str, float]) -> str:
    """解读VIF结果"""
    if not vif_data:
        return ""
    max_vif = max(vif_data.values())
    high_vif_vars = [k for k, v in vif_data.items() if v > 10]
    
    if max_vif > 10:
        return f"存在严重多重共线性（VIF > 10）：{', '.join(high_vif_vars)}。建议使用岭回归或剔除高相关变量。"
    elif max_vif > 5:
        return f"存在中度多重共线性（最大VIF = {max_vif:.2f}）。建议关注相关变量或考虑正则化方法。"
    else:
        return f"多重共线性在可接受范围内（最大VIF = {max_vif:.2f} < 5）。"

def _interpret_r_squared(r2: float) -> str:
    """解读R²效应量"""
    if r2 < 0.02:
        return "R²极小（< 0.02），模型解释力极弱，自变量几乎无法解释因变量变异"
    elif r2 < 0.13:
        return "小效应（0.02 ≤ R² < 0.13），模型解释力较弱"
    elif r2 < 0.26:
        return "中等效应（0.13 ≤ R² < 0.26），模型具有一定解释力"
    else:
        return "大效应（R² ≥ 0.26），模型解释力较强"

def linear_regression(df: pd.DataFrame, target: str, features: List[str], method: str = "ols") -> Dict[str, Any]:
    try:
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report

        data = df[[target] + features].dropna()
        if len(data) < len(features) + 2:
            return {"error": "样本量不足以进行回归分析"}

        X = data[features].values
        y = data[target].values

        # VIF共线性诊断（仅对OLS、岭回归、Lasso、弹性网）
        vif_data = {}
        vif_interpretation = ""
        if method in ["ols", "ridge", "lasso", "elasticnet"]:
            vif_data = _calculate_vif(data, features)
            vif_interpretation = _interpret_vif(vif_data)

        # OLS 线性回归
        if method == "ols":
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(features) - 1)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            f_stat = (r2 / len(features)) / ((1 - r2) / (len(y) - len(features) - 1)) if (1 - r2) > 0 else np.inf
            f_p_value = 1 - stats.f.cdf(f_stat, len(features), len(y) - len(features) - 1)
            
            # 异常值处理
            f_stat_display = float(f_stat) if not np.isinf(f_stat) else None
            f_stat_note = ""
            if np.isinf(f_stat):
                f_stat_note = "F值为无穷大，表明模型拟合完美（残差为0或极小），可能存在过拟合"
            
            return {
                "model": "多元线性回归 (OLS)",
                "target": target,
                "features": features,
                "r_squared": float(r2),
                "adjusted_r_squared": float(adj_r2),
                "rmse": float(rmse),
                "f_statistic": f_stat_display,
                "f_p_value": float(f_p_value),
                "coefficients": {
                    "intercept": float(model.intercept_),
                    **{feat: float(coef) for feat, coef in zip(features, model.coef_)},
                },
                "effect_size": {"r_squared": float(r2), "adjusted_r_squared": float(adj_r2)},
                "n_samples": len(data),
                "vif": vif_data,
                "vif_interpretation": vif_interpretation,
                "r_squared_interpretation": _interpret_r_squared(r2),
                "f_statistic_note": f_stat_note,
            }

        # 岭回归
        elif method == "ridge":
            model = Ridge(alpha=1.0)
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(features) - 1)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            return {
                "model": "岭回归 (Ridge)",
                "target": target,
                "features": features,
                "alpha": 1.0,
                "r_squared": float(r2),
                "adjusted_r_squared": float(adj_r2),
                "rmse": float(rmse),
                "coefficients": {
                    "intercept": float(model.intercept_),
                    **{feat: float(coef) for feat, coef in zip(features, model.coef_)},
                },
                "n_samples": len(data),
                "vif": vif_data,
                "vif_interpretation": vif_interpretation,
                "r_squared_interpretation": _interpret_r_squared(r2),
                "note": "岭回归通过L2正则化处理多重共线性，系数被压缩但不为0",
            }

        # Lasso 回归
        elif method == "lasso":
            model = Lasso(alpha=0.1, max_iter=10000)
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(features) - 1)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            n_nonzero = sum(1 for c in model.coef_ if abs(c) > 1e-6)
            selected = [f for f, c in zip(features, model.coef_) if abs(c) > 1e-6]
            return {
                "model": "Lasso 回归",
                "target": target,
                "features": features,
                "alpha": 0.1,
                "r_squared": float(r2),
                "adjusted_r_squared": float(adj_r2),
                "rmse": float(rmse),
                "n_features_selected": n_nonzero,
                "selected_features": selected,
                "coefficients": {
                    "intercept": float(model.intercept_),
                    **{feat: float(coef) for feat, coef in zip(features, model.coef_)},
                },
                "n_samples": len(data),
                "vif": vif_data,
                "vif_interpretation": vif_interpretation,
                "r_squared_interpretation": _interpret_r_squared(r2),
                "note": f"Lasso回归通过L1正则化进行特征筛选，{n_nonzero}/{len(features)}个变量被保留",
            }

        # 弹性网回归
        elif method == "elasticnet":
            model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(features) - 1)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            n_nonzero = sum(1 for c in model.coef_ if abs(c) > 1e-6)
            selected = [f for f, c in zip(features, model.coef_) if abs(c) > 1e-6]
            return {
                "model": "弹性网回归 (Elastic Net)",
                "target": target,
                "features": features,
                "alpha": 0.1,
                "l1_ratio": 0.5,
                "r_squared": float(r2),
                "adjusted_r_squared": float(adj_r2),
                "rmse": float(rmse),
                "n_features_selected": n_nonzero,
                "selected_features": selected,
                "coefficients": {
                    "intercept": float(model.intercept_),
                    **{feat: float(coef) for feat, coef in zip(features, model.coef_)},
                },
                "n_samples": len(data),
                "vif": vif_data,
                "vif_interpretation": vif_interpretation,
                "r_squared_interpretation": _interpret_r_squared(r2),
                "note": f"弹性网结合L1+L2正则化，{n_nonzero}/{len(features)}个变量被保留",
            }

        # 多项式回归
        elif method == "polynomial":
            degree = 2
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_poly = poly.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)
            y_pred = model.predict(X_poly)
            r2 = r2_score(y, y_pred)
            n_params = X_poly.shape[1]
            adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - n_params - 1)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            feature_names = poly.get_feature_names_out(features)
            return {
                "model": f"多项式回归 (degree={degree})",
                "target": target,
                "features": features,
                "degree": degree,
                "r_squared": float(r2),
                "adjusted_r_squared": float(adj_r2),
                "rmse": float(rmse),
                "coefficients": {
                    "intercept": float(model.intercept_),
                    **{feat: float(coef) for feat, coef in zip(feature_names, model.coef_)},
                },
                "n_samples": len(data),
                "r_squared_interpretation": _interpret_r_squared(r2),
                "note": "多项式回归用于拟合非线性关系，高次项可能增加过拟合风险",
            }

        # 逻辑回归（二分类）
        elif method == "logistic":
            y_binary = (y > y.median()).astype(int)
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y_binary)
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]
            accuracy = accuracy_score(y_binary, y_pred)
            coef_exp = np.exp(model.coef_[0])
            return {
                "model": "逻辑回归 (Logistic)",
                "target": target,
                "features": features,
                "accuracy": float(accuracy),
                "coefficients": {
                    "intercept": float(model.intercept_[0]),
                    **{feat: float(coef) for feat, coef in zip(features, model.coef_[0])},
                },
                "odds_ratio": {
                    feat: float(odds) for feat, odds in zip(features, coef_exp)
                },
                "n_samples": len(data),
                "n_positive": int(y_binary.sum()),
                "n_negative": int((1 - y_binary).sum()),
                "note": "逻辑回归用于二分类预测，OR值>1表示正相关，OR值<1表示负相关",
            }

        else:
            return {"error": f"不支持的回归方法: {method}"}

    except Exception as e:
        return {"error": str(e)}

def stepwise_regression(df: pd.DataFrame, target: str, features: List[str], direction: str = "both") -> Dict[str, Any]:
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error
        import statsmodels.api as sm

        data = df[[target] + features].dropna()
        if len(data) < len(features) + 2:
            return {"error": "样本量不足以进行逐步回归分析"}

        y = data[target].values
        X = data[features].values

        # 向前逐步回归
        if direction == "forward":
            selected = []
            remaining = list(features)
            best_score = -np.inf

            while remaining:
                best_feat = None
                best_new_score = best_score

                for feat in remaining:
                    current_features = selected + [feat]
                    X_current = sm.add_constant(data[current_features].values)
                    model = sm.OLS(y, X_current).fit()
                    score = model.rsquared_adj

                    if score > best_new_score:
                        best_new_score = score
                        best_feat = feat

                if best_feat is not None and best_new_score > best_score:
                    selected.append(best_feat)
                    remaining.remove(best_feat)
                    best_score = best_new_score
                else:
                    break

        # 向后逐步回归
        elif direction == "backward":
            selected = list(features)
            best_score = -np.inf

            while len(selected) > 1:
                worst_feat = None
                best_new_score = best_score

                for feat in selected:
                    current_features = [f for f in selected if f != feat]
                    X_current = sm.add_constant(data[current_features].values)
                    model = sm.OLS(y, X_current).fit()
                    score = model.rsquared_adj

                    if score > best_new_score:
                        best_new_score = score
                        worst_feat = feat

                if worst_feat is not None and best_new_score > best_score:
                    selected.remove(worst_feat)
                    best_score = best_new_score
                else:
                    break

        # 双向逐步回归
        else:
            selected = []
            remaining = list(features)
            best_score = -np.inf

            while remaining or len(selected) > 0:
                # 向前步骤
                best_feat = None
                best_new_score = best_score

                for feat in remaining:
                    current_features = selected + [feat]
                    X_current = sm.add_constant(data[current_features].values)
                    model = sm.OLS(y, X_current).fit()
                    score = model.rsquared_adj

                    if score > best_new_score:
                        best_new_score = score
                        best_feat = feat

                if best_feat is not None and best_new_score > best_score:
                    selected.append(best_feat)
                    remaining.remove(best_feat)
                    best_score = best_new_score
                else:
                    break

                # 向后步骤
                while len(selected) > 1:
                    worst_feat = None
                    best_remove_score = best_score

                    for feat in selected:
                        current_features = [f for f in selected if f != feat]
                        X_current = sm.add_constant(data[current_features].values)
                        model = sm.OLS(y, X_current).fit()
                        score = model.rsquared_adj

                        if score > best_remove_score:
                            best_remove_score = score
                            worst_feat = feat

                    if worst_feat is not None and best_remove_score > best_score:
                        selected.remove(worst_feat)
                        remaining.append(worst_feat)
                        best_score = best_remove_score
                    else:
                        break

        # 构建最终模型
        if not selected:
            return {"error": "逐步回归未选择任何变量"}

        X_final = sm.add_constant(data[selected].values)
        model = sm.OLS(y, X_final).fit()

        return {
            "model": f"逐步回归 ({direction})",
            "selected_features": selected,
            "n_features_selected": len(selected),
            "r_squared": float(model.rsquared),
            "adjusted_r_squared": float(model.rsquared_adj),
            "f_statistic": float(model.fvalue),
            "f_p_value": float(model.f_pvalue),
            "coefficients": {
                "intercept": float(model.params[0]),
                **{feat: float(coef) for feat, coef in zip(selected, model.params[1:])},
            },
            "p_values": {
                "intercept": float(model.pvalues[0]),
                **{feat: float(p) for feat, p in zip(selected, model.pvalues[1:])},
            },
            "n_samples": len(data),
            "direction": direction,
        }

    except Exception as e:
        return {"error": str(e)}

def dose_response(df: pd.DataFrame, dose_col: str, response_col: str) -> Dict[str, Any]:
    try:
        from scipy.optimize import curve_fit

        data = df[[dose_col, response_col]].dropna().sort_values(by=dose_col)
        if len(data) < 4:
            return {"error": "至少需要4个数据点进行剂量-效应拟合"}

        x = data[dose_col].values
        y = data[response_col].values

        def four_param_logistic(x, bottom, top, log_ic50, hill):
            return bottom + (top - bottom) / (1 + 10 ** ((log_ic50 - x) * hill))

        y_min, y_max = y.min(), y.max()
        x_min, x_max = np.log10(x.min()), np.log10(x.max())
        p0 = [y_min, y_max, (x_min + x_max) / 2, 1.0]

        log_x = np.log10(x)
        try:
            popt, pcov = curve_fit(four_param_logistic, log_x, y, p0=p0, maxfev=10000)
            bottom, top, log_ic50, hill = popt
            ic50 = 10 ** log_ic50

            y_pred = four_param_logistic(log_x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            return {
                "model": "Four-parameter logistic",
                "bottom": float(bottom),
                "top": float(top),
                "log_ic50": float(log_ic50),
                "ic50": float(ic50),
                "hill_slope": float(hill),
                "r_squared": float(r2),
                "parameters": popt.tolist(),
            }
        except Exception as fit_error:
            return {"error": f"曲线拟合失败: {str(fit_error)}"}
    except Exception as e:
        return {"error": str(e)}

def survival_analysis(df: pd.DataFrame, time_col: str, event_col: str, group_col: Optional[str] = None) -> Dict[str, Any]:
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test

        data = df[[time_col, event_col]].dropna()
        data[event_col] = data[event_col].astype(int)

        kmf = KaplanMeierFitter()
        kmf.fit(durations=data[time_col], event_observed=data[event_col])

        median_survival = float(kmf.median_survival_time_) if np.isfinite(kmf.median_survival_time_) else None

        result = {
            "test": "Kaplan-Meier",
            "median_survival": median_survival,
            "n_events": int(data[event_col].sum()),
            "n_total": len(data),
            "survival_curve": {
                "times": kmf.survival_function_.index.tolist(),
                "survival_prob": kmf.survival_function_[kmf.event_observed.name].tolist(),
            },
        }

        if group_col and group_col in df.columns:
            groups = df[group_col].unique()
            if len(groups) >= 2:
                group_data = []
                for g in groups:
                    gdf = df[df[group_col] == g][[time_col, event_col]].dropna()
                    gdf[event_col] = gdf[event_col].astype(int)
                    group_data.append((g, gdf))

                if len(group_data) >= 2:
                    g1_name, g1_data = group_data[0]
                    g2_name, g2_data = group_data[1]
                    results_lr = logrank_test(
                        g1_data[time_col], g2_data[time_col],
                        event_observed_A=g1_data[event_col],
                        event_observed_B=g2_data[event_col],
                    )
                    result["log_rank_test"] = {
                        "test_statistic": float(results_lr.test_statistic),
                        "p_value": float(results_lr.p_value),
                        "significant": results_lr.p_value < 0.05,
                        "groups_compared": [str(g1_name), str(g2_name)],
                    }

        return result
    except ImportError:
        return {"error": "需要安装 lifelines 包: pip install lifelines"}
    except Exception as e:
        return {"error": str(e)}

def recommend_test(df: pd.DataFrame, col: str, group_col: Optional[str] = None) -> Dict[str, Any]:
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        recommendations = []

        if group_col:
            n_groups = df[group_col].nunique()
            if n_groups == 2:
                data = df[col].dropna()
                normality = check_normality(data)
                if normality.get("is_normal", False):
                    recommendations.append({
                        "test": "Independent t-test",
                        "reason": "数据正态分布，两组比较",
                        "api": "/api/v2/t-test",
                    })
                else:
                    recommendations.append({
                        "test": "Mann-Whitney U",
                        "reason": "数据非正态，两组比较",
                        "api": "/api/v2/mann-whitney",
                    })
            elif n_groups >= 3:
                recommendations.append({
                    "test": "One-way ANOVA",
                    "reason": f"{n_groups}组比较",
                    "api": "/api/v2/anova",
                })
        else:
            if len(numeric_cols) >= 2:
                recommendations.append({
                    "test": "Correlation Analysis",
                    "reason": "多变量相关性分析",
                    "api": "/api/v2/correlation",
                })
            if len(numeric_cols) >= 1:
                recommendations.append({
                    "test": "Linear Regression",
                    "reason": "回归分析",
                    "api": "/api/v2/linear-regression",
                })

        return {"recommendations": recommendations}
    except Exception as e:
        return {"error": str(e)}

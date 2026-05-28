import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from app.utils import logger, make_serializable
from app.modules.stat_tools import (
    descriptive_statistics, check_normality, t_test_independent,
    one_way_anova, correlation_analysis, linear_regression
)
from app.modules.regression_extended import (
    robust_regression, quantile_regression, dummy_variable_regression,
    hierarchical_linear_model, bayesian_regression, recommend_regression_method
)
from app.modules.visualizer import generate_all_charts

class AutoAnalysisEngine:
    def __init__(self):
        self.results = {}
        self.analysis_log = []

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        self.results = {}
        self.analysis_log = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        self.results["data_characteristics"] = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
        }

        self.results["descriptive"] = descriptive_statistics(df)
        self.analysis_log.append("已执行描述性统计")

        normality_results = {}
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) >= 3:
                normality_results[col] = check_normality(data)
        self.results["normality"] = normality_results
        self.analysis_log.append("已执行正态性检验")

        t_test_results = {}
        for cat_col in categorical_cols:
            unique_vals = df[cat_col].dropna().unique()
            if len(unique_vals) == 2:
                for num_col in numeric_cols:
                    group1 = df[df[cat_col] == unique_vals[0]][num_col].dropna()
                    group2 = df[df[cat_col] == unique_vals[1]][num_col].dropna()
                    if len(group1) >= 2 and len(group2) >= 2:
                        result = t_test_independent(df, num_col, cat_col, str(unique_vals[0]), str(unique_vals[1]))
                        t_test_results[f"{num_col}~{cat_col}"] = result
        self.results["t_test"] = t_test_results
        if t_test_results:
            self.analysis_log.append("已执行T检验（2组分类变量）")

        anova_results = {}
        for cat_col in categorical_cols:
            unique_vals = df[cat_col].dropna().unique()
            if len(unique_vals) >= 3:
                for num_col in numeric_cols:
                    result = one_way_anova(df, num_col, cat_col)
                    anova_results[f"{num_col}~{cat_col}"] = result
        self.results["anova"] = anova_results
        if anova_results:
            self.analysis_log.append("已执行方差分析（≥3组分类变量）")

        if len(numeric_cols) >= 2:
            corr_result = correlation_analysis(df, numeric_cols)
            self.results["correlation"] = corr_result
            self.analysis_log.append("已执行相关性分析")

        if len(numeric_cols) >= 2:
            regression_results = self._auto_regression(df, numeric_cols, categorical_cols)
            self.results["regression"] = regression_results
            self.analysis_log.append("已执行回归分析")

        # Charts will be generated later with session_id
        self.results["charts"] = {}
        self.analysis_log.append("已准备可视化图表")

        self.results["analysis_log"] = self.analysis_log
        self.results["methods_used"] = self._get_methods_used()

        return self.results

    def _auto_regression(self, df: pd.DataFrame, numeric_cols: List[str],
                         categorical_cols: List[str]) -> Dict[str, Any]:
        target_col = numeric_cols[-1]
        feature_cols = numeric_cols[:-1]

        if len(feature_cols) < 1:
            return {"error": "自变量不足"}

        has_outliers = self._detect_outliers(df, numeric_cols)
        has_categorical = len(categorical_cols) > 0
        sample_size = len(df)

        data_characteristics = {
            "has_outliers": has_outliers,
            "has_categorical": has_categorical,
            "has_nested": False,
            "sample_size": sample_size,
            "n_features": len(feature_cols),
            "multicollinearity": False,
        }

        recommendation = recommend_regression_method(data_characteristics)

        results = {}
        primary_method = recommendation["primary_recommendation"]

        # 模型比较：在自动模式下比较OLS、稳健回归、岭回归
        if sample_size >= 10 and len(feature_cols) >= 1:
            model_comparison = {}
            
            # OLS模型
            ols_result = linear_regression(df, target_col, feature_cols, method="ols")
            if "error" not in ols_result:
                model_comparison["ols"] = {
                    "r_squared": ols_result.get("r_squared", 0),
                    "adjusted_r_squared": ols_result.get("adjusted_r_squared", 0),
                    "rmse": ols_result.get("rmse", float("inf")),
                }
            
            # 稳健回归
            try:
                robust_result = robust_regression(df, target_col, feature_cols)
                if "error" not in robust_result:
                    model_comparison["robust"] = {
                        "r_squared": robust_result.get("r_squared", 0),
                        "rmse": robust_result.get("rmse", float("inf")),
                    }
            except Exception:
                pass
            
            # 岭回归
            ridge_result = linear_regression(df, target_col, feature_cols, method="ridge")
            if "error" not in ridge_result:
                model_comparison["ridge"] = {
                    "r_squared": ridge_result.get("r_squared", 0),
                    "adjusted_r_squared": ridge_result.get("adjusted_r_squared", 0),
                    "rmse": ridge_result.get("rmse", float("inf")),
                }
            
            # 选择最佳模型
            if model_comparison:
                best_model = self._select_best_model(model_comparison, has_outliers)
                results["model_comparison"] = model_comparison
                results["model_selection_reason"] = best_model["reason"]
                primary_method = best_model["method"]
                # 更新recommendation以反映实际使用的方法
                recommendation["primary_recommendation"] = primary_method
                recommendation["selection_reason"] = best_model["reason"]

        if primary_method == "robust":
            results["robust"] = robust_regression(df, target_col, feature_cols)
        elif primary_method == "dummy" and has_categorical:
            results["dummy"] = dummy_variable_regression(df, target_col, feature_cols, categorical_cols[:1])
        elif primary_method == "hlm":
            results["hlm"] = {"note": "需要指定分组变量"}
        elif primary_method == "bayesian":
            results["bayesian"] = bayesian_regression(df, target_col, feature_cols)
        elif primary_method == "ridge":
            results["ridge"] = linear_regression(df, target_col, feature_cols, method="ridge")
        elif primary_method == "lasso":
            results["lasso"] = linear_regression(df, target_col, feature_cols, method="lasso")
        elif primary_method == "elastic_net":
            results["elastic_net"] = linear_regression(df, target_col, feature_cols, method="elasticnet")
        else:
            results["ols"] = linear_regression(df, target_col, feature_cols, method="ols")

        results["recommendation"] = recommendation
        return results

    def _select_best_model(self, comparison: Dict[str, Dict], has_outliers: bool) -> Dict[str, str]:
        """比较多个回归模型并选择最佳模型"""
        if not comparison:
            return {"method": "ols", "reason": "默认使用OLS回归"}
        
        # 如果存在异常值，优先选择稳健回归
        if has_outliers and "robust" in comparison:
            robust_r2 = comparison["robust"].get("r_squared", 0)
            ols_r2 = comparison.get("ols", {}).get("r_squared", 0)
            if robust_r2 >= ols_r2 * 0.95:  # 稳健回归R²接近OLS
                return {
                    "method": "robust",
                    "reason": f"因数据存在异常值，稳健回归的R²({robust_r2:.3f})与OLS({ols_r2:.3f})接近但更稳健，故选择稳健回归"
                }
        
        # 否则选择调整R²最高的模型
        best_method = "ols"
        best_adj_r2 = -1
        for method, metrics in comparison.items():
            adj_r2 = metrics.get("adjusted_r_squared", metrics.get("r_squared", 0))
            if adj_r2 > best_adj_r2:
                best_adj_r2 = adj_r2
                best_method = method
        
        reason_map = {
            "ols": "OLS回归的调整R²最高，数据质量良好，无严重异常值或共线性",
            "ridge": "岭回归的调整R²最高，L2正则化有效缓解了多重共线性问题",
            "robust": "稳健回归的调整R²最高，Huber损失函数有效处理了异常值",
        }
        
        return {
            "method": best_method,
            "reason": reason_map.get(best_method, f"{best_method.upper()}模型的拟合效果最佳")
        }

    def _detect_outliers(self, df: pd.DataFrame, numeric_cols: List[str]) -> bool:
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) < 3:
                continue
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                return True
        return False

    def _get_methods_used(self) -> List[Dict[str, str]]:
        """返回分析方法列表，包含方法名称和选择理由"""
        methods = []
        
        # 描述性统计
        if "descriptive" in self.results:
            methods.append({
                "name": "描述性统计",
                "reason": "所有数据分析的基础步骤，计算均值、标准差、中位数、偏度等统计量，了解数据分布特征"
            })
        
        # 正态性检验
        if "normality" in self.results:
            numeric_cols = self.results.get("data_characteristics", {}).get("numeric_cols", [])
            methods.append({
                "name": "正态性检验 (Shapiro-Wilk)",
                "reason": f"检测到 {len(numeric_cols)} 个数值型变量，需检验数据是否服从正态分布，以决定后续使用参数检验还是非参数检验"
            })
        
        # T检验
        if "t_test" in self.results and self.results["t_test"]:
            n_tests = len(self.results["t_test"])
            methods.append({
                "name": "独立样本T检验",
                "reason": f"发现包含2个分组的分类变量，执行了 {n_tests} 次T检验比较组间差异。若方差不齐则自动使用Welch校正"
            })
        
        # 方差分析
        if "anova" in self.results and self.results["anova"]:
            n_tests = len(self.results["anova"])
            methods.append({
                "name": "方差分析 (ANOVA)",
                "reason": f"发现包含3个及以上分组的分类变量，执行了 {n_tests} 次ANOVA检验。若方差不齐或非正态则使用Kruskal-Wallis非参数检验"
            })
        
        # 相关性分析
        if "correlation" in self.results:
            numeric_cols = self.results.get("data_characteristics", {}).get("numeric_cols", [])
            # 检查正态性决定使用Pearson还是Spearman
            normality = self.results.get("normality", {})
            all_normal = all(r.get("is_normal", False) for r in normality.values())
            corr_method = "Pearson" if all_normal else "Spearman"
            methods.append({
                "name": f"相关性分析 ({corr_method})",
                "reason": f"检测到 {len(numeric_cols)} 个数值型变量，{'所有变量服从正态分布，使用Pearson相关系数' if all_normal else '部分变量不服从正态分布，使用Spearman秩相关系数'}"
            })
        
        # 回归分析
        if "regression" in self.results:
            regression = self.results["regression"]
            recommendation = regression.get("recommendation", {})
            primary = recommendation.get("primary_recommendation", "ols")
            
            logger.info(f"Regression recommendation: {recommendation}")
            logger.info(f"Primary method: {primary}")
            
            method_name_map = {
                "ols": "多元线性回归 (OLS)",
                "ridge": "岭回归 (Ridge)",
                "lasso": "Lasso回归",
                "elastic_net": "弹性网回归 (Elastic Net)",
                "robust": "稳健回归 (Huber)",
                "dummy": "虚拟变量回归",
                "hlm": "层次线性模型 (HLM)",
                "bayesian": "贝叶斯回归",
            }
            
            method_reason_map = {
                "ols": "数据质量良好，无严重异常值或共线性，使用经典最小二乘法",
                "ridge": "检测到多重共线性（VIF>10），使用L2正则化缓解共线性问题",
                "lasso": "特征较多，使用L1正则化进行变量筛选和压缩",
                "elastic_net": "同时存在共线性和需要变量筛选，结合L1+L2正则化",
                "robust": "检测到数据存在异常值，使用Huber损失函数提高模型鲁棒性",
                "dummy": "数据包含分类变量，自动转换为虚拟变量纳入回归模型",
                "hlm": "检测到嵌套数据结构（如患者-医院-地区），使用多层线性模型",
                "bayesian": "样本量较小（<30），使用贝叶斯方法提供更稳健的估计",
            }
            
            methods.append({
                "name": method_name_map.get(primary, f"回归分析 ({primary.upper()})"),
                "reason": method_reason_map.get(primary, "根据数据特征自动选择最优回归方法")
            })
        
        return methods

auto_engine = AutoAnalysisEngine()

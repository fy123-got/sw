import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from app.utils import logger, make_serializable

def robust_regression(df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> Dict[str, Any]:
    """稳健回归（HuberRegressor）：抗异常值干扰"""
    try:
        from sklearn.linear_model import HuberRegressor
        from sklearn.preprocessing import StandardScaler
        from scipy import stats

        X = df[feature_cols].dropna()
        y = df.loc[X.index, target_col].dropna()
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        if len(X) < 10:
            return {"error": "样本量不足（需≥10）"}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = HuberRegressor(max_iter=1000)
        model.fit(X_scaled, y)

        y_pred = model.predict(X_scaled)
        residuals = y - y_pred

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        n = len(y)
        p = len(feature_cols)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared

        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))

        coef_df = pd.DataFrame({
            "feature": ["Intercept"] + feature_cols,
            "coefficient": [model.intercept_] + model.coef_.tolist(),
        })

        t_values = []
        p_values = []
        for i, feat in enumerate(["Intercept"] + feature_cols):
            if i == 0:
                coef = model.intercept_
            else:
                coef = model.coef_[i - 1]
            se = np.sqrt(mse / n) if n > 0 else 0
            t_val = coef / se if se > 0 else 0
            p_val = 2 * (1 - stats.t.cdf(abs(t_val), df=n - p - 1)) if n > p + 1 else 1.0
            t_values.append(float(t_val))
            p_values.append(float(p_val))

        coef_df["t_value"] = t_values
        coef_df["p_value"] = p_values
        coef_df["significant"] = coef_df["p_value"] < 0.05

        result = {
            "method": "稳健回归 (Huber)",
            "description": "使用Huber损失函数，对异常值具有鲁棒性",
            "r_squared": float(r_squared),
            "adj_r_squared": float(adj_r_squared),
            "rmse": float(rmse),
            "mae": float(mae),
            "n_samples": int(n),
            "n_features": int(p),
            "coefficients": make_serializable(coef_df.to_dict("records")),
            "recommendation": "数据存在异常值时，稳健回归比OLS更可靠",
        }

        return result
    except ImportError:
        return {"error": "sklearn未安装，无法执行稳健回归"}
    except Exception as e:
        logger.error(f"稳健回归失败: {e}")
        return {"error": str(e)}


def quantile_regression(df: pd.DataFrame, target_col: str, feature_cols: List[str],
                        quantiles: List[float] = None) -> Dict[str, Any]:
    """分位数回归：分析不同分位数下的变量关系"""
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import quantreg

        if quantiles is None:
            quantiles = [0.25, 0.50, 0.75]

        X = df[feature_cols].dropna()
        y = df.loc[X.index, target_col].dropna()
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        if len(X) < 10:
            return {"error": "样本量不足（需≥10）"}

        X_sm = sm.add_constant(X)
        formula = f"{target_col} ~ {' + '.join(feature_cols)}"

        quantile_results = []
        for q in quantiles:
            try:
                model = quantreg(formula, df.loc[common_idx])
                result = model.fit(q=q)

                coef_dict = {}
                for var in ["const"] + feature_cols:
                    coef_dict[var] = {
                        "coefficient": float(result.params.get(var, 0)),
                        "std_err": float(result.bse.get(var, 0)),
                        "t_value": float(result.tvalues.get(var, 0)),
                        "p_value": float(result.pvalues.get(var, 1)),
                        "ci_lower": float(result.conf_int().loc[var, 0]) if var in result.conf_int().index else None,
                        "ci_upper": float(result.conf_int().loc[var, 1]) if var in result.conf_int().index else None,
                    }

                quantile_results.append({
                    "quantile": q,
                    "coefficients": coef_dict,
                    "aic": float(result.aic),
                    "bic": float(result.bic),
                })
            except Exception as e:
                logger.warning(f"分位数 {q} 回归失败: {e}")
                quantile_results.append({"quantile": q, "error": str(e)})

        result = {
            "method": "分位数回归",
            "description": "分析不同分位数（25%、50%、75%）下的变量关系，不依赖正态分布假设",
            "quantiles": quantiles,
            "results": make_serializable(quantile_results),
            "recommendation": "分位数回归可揭示变量在不同分布位置的影响差异",
        }

        return result
    except ImportError:
        return {"error": "statsmodels未安装，无法执行分位数回归"}
    except Exception as e:
        logger.error(f"分位数回归失败: {e}")
        return {"error": str(e)}


def dummy_variable_regression(df: pd.DataFrame, target_col: str, feature_cols: List[str],
                              categorical_cols: List[str] = None) -> Dict[str, Any]:
    """虚拟变量回归：处理分类自变量"""
    try:
        import statsmodels.api as sm

        if categorical_cols is None:
            categorical_cols = []

        df_model = df[[target_col] + feature_cols + categorical_cols].dropna()

        if len(df_model) < 10:
            return {"error": "样本量不足（需≥10）"}

        df_dummies = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

        y = df_dummies[target_col]
        X = df_dummies.drop(columns=[target_col])
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        coef_df = pd.DataFrame({
            "feature": model.params.index.tolist(),
            "coefficient": model.params.values,
            "std_err": model.bse.values,
            "t_value": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_lower": model.conf_int()[0].values,
            "ci_upper": model.conf_int()[1].values,
        })
        coef_df["significant"] = coef_df["p_value"] < 0.05

        result = {
            "method": "虚拟变量回归",
            "description": f"将分类变量（{', '.join(categorical_cols)}）转换为虚拟变量后回归",
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "f_statistic": float(model.fvalue),
            "f_pvalue": float(model.f_pvalue),
            "n_samples": int(model.nobs),
            "n_features": int(len(model.params) - 1),
            "coefficients": make_serializable(coef_df.to_dict("records")),
            "categorical_vars": categorical_cols,
            "recommendation": "分类变量已自动转换为虚拟变量，首个类别作为参照组",
        }

        return result
    except ImportError:
        return {"error": "statsmodels未安装，无法执行虚拟变量回归"}
    except Exception as e:
        logger.error(f"虚拟变量回归失败: {e}")
        return {"error": str(e)}


def hierarchical_linear_model(df: pd.DataFrame, target_col: str, feature_cols: List[str],
                              group_col: str) -> Dict[str, Any]:
    """层次线性模型（HLM / MixedLM）：处理嵌套数据结构"""
    try:
        import statsmodels.api as sm
        from statsmodels.regression.mixed_linear_model import MixedLM

        df_model = df[[target_col] + feature_cols + [group_col]].dropna()

        if len(df_model) < 20:
            return {"error": "样本量不足（需≥20）"}

        groups = df_model[group_col]
        y = df_model[target_col]
        X = df_model[feature_cols]
        X = sm.add_constant(X)

        model = MixedLM(y, X, groups)
        result = model.fit()

        fixed_ef = pd.DataFrame({
            "feature": result.fe_params.index.tolist(),
            "coefficient": result.fe_params.values,
            "std_err": result.bse_fe.values,
            "z_value": result.tvalues.values,
            "p_value": result.pvalues.values,
        })
        fixed_ef["significant"] = fixed_ef["p_value"] < 0.05

        re_var = float(result.cov_re.values.flatten()[0]) if hasattr(result, 'cov_re') else None

        result_dict = {
            "method": "层次线性模型 (HLM / MixedLM)",
            "description": f"处理嵌套数据（分组变量：{group_col}），考虑组内相关性",
            "fixed_effects": make_serializable(fixed_ef.to_dict("records")),
            "random_effect_variance": float(re_var) if re_var is not None else None,
            "aic": float(result.aic),
            "bic": float(result.bic),
            "n_groups": int(result.n_groups),
            "n_samples": int(result.nobs),
            "recommendation": "层次模型适用于嵌套数据（如患者-医院-地区），可控制组间差异",
        }

        return result_dict
    except ImportError:
        return {"error": "statsmodels未安装，无法执行层次线性模型"}
    except Exception as e:
        logger.error(f"层次线性模型失败: {e}")
        return {"error": str(e)}


def bayesian_regression(df: pd.DataFrame, target_col: str, feature_cols: List[str],
                        draws: int = 2000) -> Dict[str, Any]:
    """贝叶斯回归：提供参数的后验分布估计"""
    try:
        import pymc as pm
        import arviz as az

        df_model = df[[target_col] + feature_cols].dropna()

        if len(df_model) < 10:
            return {"error": "样本量不足（需≥10）"}

        X = df_model[feature_cols].values
        y = df_model[target_col].values

        with pm.Model() as model:
            sigma = pm.HalfCauchy("sigma", beta=10)
            intercept = pm.Normal("intercept", mu=0, sigma=10)

            betas = []
            for i, feat in enumerate(feature_cols):
                beta = pm.Normal(f"beta_{feat}", mu=0, sigma=10)
                betas.append(beta)

            mu = intercept + sum(b * X[:, i] for i, b in enumerate(betas))
            likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

            trace = pm.sample(draws=draws, tune=1000, target_accept=0.95, return_inferencedata=True)

        summary = az.summary(trace)

        coef_results = []
        for var in summary.index:
            coef_results.append({
                "parameter": var,
                "mean": float(summary.loc[var, "mean"]),
                "sd": float(summary.loc[var, "sd"]),
                "hdi_3%": float(summary.loc[var, "hdi_3%"]),
                "hdi_97%": float(summary.loc[var, "hdi_97%"]),
                "r_hat": float(summary.loc[var, "r_hat"]),
            })

        result = {
            "method": "贝叶斯回归",
            "description": "使用MCMC采样估计参数后验分布，提供不确定性量化",
            "n_draws": draws,
            "n_tune": 1000,
            "coefficients": make_serializable(coef_results),
            "recommendation": "贝叶斯回归适用于小样本或需要不确定性量化的场景",
        }

        return result
    except ImportError:
        return {"error": "PyMC未安装，无法执行贝叶斯回归。安装命令：pip install pymc arviz"}
    except Exception as e:
        logger.error(f"贝叶斯回归失败: {e}")
        return {"error": str(e)}


def recommend_regression_method(data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
    """根据数据特征自动推荐回归方法"""
    recommendations = []
    reasons = []

    has_outliers = data_characteristics.get("has_outliers", False)
    has_categorical = data_characteristics.get("has_categorical", False)
    has_nested = data_characteristics.get("has_nested", False)
    sample_size = data_characteristics.get("sample_size", 100)
    n_features = data_characteristics.get("n_features", 1)
    multicollinearity = data_characteristics.get("multicollinearity", False)

    if has_outliers:
        recommendations.append("robust")
        reasons.append("数据存在异常值，稳健回归（Huber）对异常值具有鲁棒性")

    if has_categorical:
        recommendations.append("dummy")
        reasons.append("数据包含分类变量，虚拟变量回归可处理分类自变量")

    if has_nested:
        recommendations.append("hlm")
        reasons.append("数据具有嵌套结构，层次线性模型（HLM）可控制组间差异")

    if sample_size < 30:
        recommendations.append("bayesian")
        reasons.append("样本量较小（<30），贝叶斯回归可提供更稳健的参数估计")

    if multicollinearity:
        recommendations.append("ridge")
        reasons.append("存在多重共线性，岭回归（Ridge）可有效处理共线性问题")

    if not recommendations:
        recommendations.append("ols")
        reasons.append("数据无明显特殊特征，使用经典最小二乘回归（OLS）")

    return {
        "recommended_methods": recommendations,
        "reasons": reasons,
        "primary_recommendation": recommendations[0] if recommendations else "ols",
    }

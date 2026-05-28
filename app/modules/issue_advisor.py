import numpy as np
from typing import Dict, Any, List, Optional
from app.utils import logger

class IssueAdvisor:
    def __init__(self):
        self.issues = []

    def analyze(self, analysis_results: Dict[str, Any], cleaning_report: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        self.issues = []

        self._check_descriptive_issues(analysis_results)
        self._check_normality_issues(analysis_results)
        self._check_ttest_issues(analysis_results)
        self._check_anova_issues(analysis_results)
        self._check_correlation_issues(analysis_results)
        self._check_regression_issues(analysis_results)
        self._check_cleaning_issues(cleaning_report)

        return self.issues

    def _add_issue(self, category: str, severity: str, problem: str, suggestion: str, details: str = ""):
        self.issues.append({
            "category": category,
            "severity": severity,
            "problem": problem,
            "suggestion": suggestion,
            "details": details,
        })

    def _check_descriptive_issues(self, results: Dict[str, Any]):
        descriptive = results.get("descriptive", {})
        if not descriptive or "error" in descriptive:
            return

        for col, stats in descriptive.items():
            skewness = stats.get("skewness", 0)
            kurtosis = stats.get("kurtosis", 0)
            missing_pct = stats.get("missing_pct", 0)

            if abs(skewness) > 1:
                direction = "右偏" if skewness > 0 else "左偏"
                self._add_issue(
                    category="描述性统计",
                    severity="中等",
                    problem=f"变量「{col}」偏度为 {skewness:.2f}，呈明显{direction}分布",
                    suggestion="建议对数变换（右偏）或平方根变换后再分析，或使用稳健回归方法",
                    details=f"偏度 {skewness:.2f}，峰度 {kurtosis:.2f}"
                )

            if abs(kurtosis) > 2:
                direction = "尖峰" if kurtosis > 0 else "平峰"
                self._add_issue(
                    category="描述性统计",
                    severity="低",
                    problem=f"变量「{col}」峰度为 {kurtosis:.2f}，呈{direction}分布",
                    suggestion="峰度过高可能存在极端值，建议检查异常值或使用稳健方法",
                    details=f"峰度 {kurtosis:.2f}"
                )

            if missing_pct > 20:
                self._add_issue(
                    category="缺失值",
                    severity="高",
                    problem=f"变量「{col}」缺失率高达 {missing_pct:.1f}%",
                    suggestion="考虑删除该变量，或使用多重插补（Multiple Imputation）方法",
                    details=f"缺失率 {missing_pct:.1f}%"
                )

    def _check_normality_issues(self, results: Dict[str, Any]):
        normality = results.get("normality", {})
        if not normality:
            return

        non_normal_cols = []
        for col, nr in normality.items():
            if not nr.get("is_normal", True):
                non_normal_cols.append(col)

        if non_normal_cols:
            self._add_issue(
                category="正态性检验",
                severity="中等",
                problem=f"以下变量不服从正态分布：{', '.join(non_normal_cols)}",
                suggestion="已自动使用非参数检验（如Kruskal-Wallis），但事后检验效能较低；或尝试Box-Cox变换使数据正态化",
                details=f"非正态变量：{', '.join(non_normal_cols)}"
            )

    def _check_ttest_issues(self, results: Dict[str, Any]):
        t_tests = results.get("t_test", {})
        if not t_tests:
            return

        for key, tt in t_tests.items():
            if "error" in tt:
                continue

            equal_var = tt.get("equal_variance", True)
            if not equal_var:
                self._add_issue(
                    category="T检验",
                    severity="低",
                    problem=f"「{key}」方差不齐（Levene检验P < 0.05）",
                    suggestion="已自动使用Welch校正T检验，结果更可靠",
                    details=f"Levene P = {tt.get('levene_p', 'N/A')}"
                )

    def _check_anova_issues(self, results: Dict[str, Any]):
        anovas = results.get("anova", {})
        if not anovas:
            return

        for key, anova in anovas.items():
            if "error" in anova:
                continue

            overall_p = anova.get("p_value", 1)
            posthoc = anova.get("posthoc", [])

            if overall_p < 0.05 and posthoc:
                significant_pairs = [p for p in posthoc if p.get("p_adj", 1) < 0.05]
                if not significant_pairs:
                    self._add_issue(
                        category="方差分析",
                        severity="中等",
                        problem=f"「{key}」整体检验显著（P = {overall_p:.4f}），但事后检验所有组间差异均不显著",
                        suggestion="可能原因：(1) 样本量不足导致检验效能低；(2) 多重比较校正过于严格。建议增加样本量或合并相似组别",
                        details=f"整体P = {overall_p:.4f}，事后检验无显著组间差异"
                    )

            equal_var = anova.get("equal_variance", True)
            if not equal_var:
                self._add_issue(
                    category="方差分析",
                    severity="中等",
                    problem=f"「{key}」方差不齐（Levene检验P = {anova.get('levene_p', 'N/A'):.4f}）",
                    suggestion="已自动使用Kruskal-Wallis非参数检验，但检验效能低于参数检验",
                    details=f"Levene P = {anova.get('levene_p', 'N/A')}"
                )

    def _check_correlation_issues(self, results: Dict[str, Any]):
        corr = results.get("correlation", {})
        if not corr or "error" in corr:
            return

        corr_matrix_data = corr.get("correlation_matrix", {})
        columns = corr.get("columns", [])
        if not corr_matrix_data or not columns:
            return

        # Handle both dict and list formats
        if isinstance(corr_matrix_data, list):
            # Convert list of lists to dict format
            corr_dict = {}
            for i, col1 in enumerate(columns):
                corr_dict[col1] = {}
                for j, col2 in enumerate(columns):
                    corr_dict[col1][col2] = corr_matrix_data[i][j]
            corr_matrix = corr_dict
        else:
            corr_matrix = corr_matrix_data

        for var1, row in corr_matrix.items():
            for var2, corr_val in row.items():
                if var1 >= var2:
                    continue
                if abs(corr_val) > 0.9:
                    self._add_issue(
                        category="相关性分析",
                        severity="高",
                        problem=f"「{var1}」与「{var2}」高度相关（r = {corr_val:.3f}），存在严重多重共线性",
                        suggestion="建议在回归分析中删除其中一个变量，或使用岭回归/Lasso回归处理共线性",
                        details=f"相关系数 r = {corr_val:.3f}"
                    )

    def _check_regression_issues(self, results: Dict[str, Any]):
        regression = results.get("regression", {})
        if not regression or "error" in regression:
            return

        ols_result = regression.get("ols", {})
        if ols_result and "error" not in ols_result:
            r_squared = ols_result.get("r_squared", 0)
            if r_squared < 0.2:
                self._add_issue(
                    category="回归分析",
                    severity="高",
                    problem=f"回归模型R² = {r_squared:.3f}，模型解释力较弱",
                    suggestion="可能原因：(1) 遗漏重要预测变量；(2) 变量间关系非线性。建议尝试多项式回归或增加特征变量",
                    details=f"R² = {r_squared:.3f}"
                )

            vif = ols_result.get("vif", {})
            if vif:
                for var, vif_val in vif.items():
                    if vif_val > 10:
                        self._add_issue(
                            category="回归分析",
                            severity="高",
                            problem=f"变量「{var}」VIF = {vif_val:.2f} > 10，存在严重多重共线性",
                            suggestion="建议删除该变量或改用岭回归（Ridge）处理共线性问题",
                            details=f"VIF = {vif_val:.2f}"
                        )
                    elif vif_val > 5:
                        self._add_issue(
                            category="回归分析",
                            severity="中等",
                            problem=f"变量「{var}」VIF = {vif_val:.2f} > 5，存在中度多重共线性",
                            suggestion="建议关注共线性影响，或使用岭回归/Lasso回归",
                            details=f"VIF = {vif_val:.2f}"
                        )

    def _check_cleaning_issues(self, cleaning_report: Dict[str, Any]):
        if not cleaning_report:
            return

        outlier_info = cleaning_report.get("outliers", {})
        if outlier_info:
            total_outliers = outlier_info.get("total_outliers", 0)
            if total_outliers > 0:
                outlier_cols = outlier_info.get("outlier_columns", {})
                col_names = list(outlier_cols.keys())[:5]
                self._add_issue(
                    category="异常值检测",
                    severity="中等",
                    problem=f"检测到 {total_outliers} 个异常值，涉及变量：{', '.join(col_names)}",
                    suggestion="建议核对原始数据；若为真实值，可使用稳健回归或分位数回归降低异常值影响",
                    details=f"异常值总数：{total_outliers}"
                )

        missing_info = cleaning_report.get("missing", {})
        if missing_info:
            total_missing = missing_info.get("total_missing", 0)
            if total_missing > 0:
                self._add_issue(
                    category="缺失值处理",
                    severity="低",
                    problem=f"共处理 {total_missing} 个缺失值",
                    suggestion="已自动填充缺失值（数值列用中位数，分类列用众数）。若缺失率过高，建议谨慎解读结果",
                    details=f"缺失值总数：{total_missing}"
                )

issue_advisor = IssueAdvisor()

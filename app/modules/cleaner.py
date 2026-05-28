import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from app.utils import logger

def detect_outliers(df: pd.DataFrame, method: str = "both") -> Dict[str, Any]:
    outlier_report = {
        "method": method,
        "outlier_columns": {},
        "total_outliers": 0,
        "outlier_percentage": 0,
        "outlier_details": [],
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    total_cells = len(df) * len(numeric_cols) if numeric_cols else 0
    total_outlier_count = 0
    outlier_details = []
    
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) < 3:
            continue
        
        col_outliers = {"indices": [], "values": [], "method": method}
        
        if method == "zscore" or method == "both":
            mean = data.mean()
            std = data.std()
            if std > 0:
                z_scores = np.abs((data - mean) / std)
                z_outliers = data[z_scores > 3]
                col_outliers["zscore_outliers"] = {
                    "count": len(z_outliers),
                    "values": z_outliers.tolist(),
                    "threshold": "±3σ",
                }
                for idx, val in z_outliers.items():
                    outlier_details.append({
                        "column": col,
                        "row_index": int(idx),
                        "value": float(val),
                        "method": "3σ原则",
                        "threshold": f"均值±3σ ({mean:.2f}±{3*std:.2f})",
                    })
        
        if method == "iqr" or method == "both":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
            col_outliers["iqr_outliers"] = {
                "count": len(iqr_outliers),
                "values": iqr_outliers.tolist(),
                "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)},
            }
            for idx, val in iqr_outliers.items():
                outlier_details.append({
                    "column": col,
                    "row_index": int(idx),
                    "value": float(val),
                    "method": "箱线图法(IQR)",
                    "threshold": f"[{lower_bound:.2f}, {upper_bound:.2f}]",
                })
        
        if method == "both":
            z_count = col_outliers.get("zscore_outliers", {}).get("count", 0)
            iqr_count = col_outliers.get("iqr_outliers", {}).get("count", 0)
            col_outliers["total_count"] = max(z_count, iqr_count)
        elif method == "zscore":
            col_outliers["total_count"] = col_outliers.get("zscore_outliers", {}).get("count", 0)
        else:
            col_outliers["total_count"] = col_outliers.get("iqr_outliers", {}).get("count", 0)
        
        if col_outliers["total_count"] > 0:
            outlier_report["outlier_columns"][col] = col_outliers
            total_outlier_count += col_outliers["total_count"]
    
    outlier_report["total_outliers"] = total_outlier_count
    outlier_report["outlier_percentage"] = round(
        (total_outlier_count / total_cells * 100) if total_cells > 0 else 0, 2
    )
    outlier_report["outlier_details"] = outlier_details
    
    return outlier_report


def _handle_missing_knn(df: pd.DataFrame, numeric_cols: List[str], object_cols: List[str]) -> pd.DataFrame:
    try:
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        for col in object_cols:
            mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_value)
        return df
    except ImportError:
        logger.warning("sklearn未安装，KNN插补不可用，退化为中位数填充")
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        return df


def clean_data(
    df: pd.DataFrame,
    fill_strategy: str = "mean",
    remove_empty_rows: bool = True,
    remove_empty_cols: bool = True,
    remove_duplicates: bool = True,
    detect_outliers_flag: bool = True,
    missing_strategy: str = "median",
    outlier_strategy: str = "detect_only",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cleaning_log: Dict[str, Any] = {
        "original_shape": df.shape,
        "original_rows": len(df),
        "original_cols": len(df.columns),
        "steps": [],
        "removed_data": [],
        "missing_strategy": missing_strategy,
        "outlier_strategy": outlier_strategy,
    }

    if df.empty:
        cleaning_log["steps"].append({"step": "empty_check", "message": "数据为空，跳过清洗"})
        return df, cleaning_log

    if remove_empty_cols:
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            cleaning_log["removed_data"].append({
                "type": "empty_columns",
                "columns": empty_cols,
            })
            df = df.drop(columns=empty_cols)
            cleaning_log["steps"].append({
                "step": "remove_empty_columns",
                "removed": empty_cols,
                "count": len(empty_cols),
            })
            logger.info(f"Removed {len(empty_cols)} empty columns")

    if remove_empty_rows:
        initial_rows = len(df)
        df = df.dropna(how="all")
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            cleaning_log["steps"].append({
                "step": "remove_empty_rows",
                "removed_count": removed_rows,
            })
            logger.info(f"Removed {removed_rows} empty rows")

    if remove_duplicates:
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_dups = initial_rows - len(df)
        if removed_dups > 0:
            cleaning_log["removed_data"].append({
                "type": "duplicate_rows",
                "count": removed_dups,
            })
            cleaning_log["steps"].append({
                "step": "remove_duplicates",
                "removed_count": removed_dups,
            })
            logger.info(f"Removed {removed_dups} duplicate rows")

    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if missing_strategy == "drop":
            initial_rows = len(df)
            df = df.dropna()
            removed_missing = initial_rows - len(df)
            cleaning_log["steps"].append({
                "step": "drop_missing_rows",
                "removed_count": removed_missing,
            })
            logger.info(f"Dropped {removed_missing} rows with missing values")
        elif missing_strategy == "knn":
            df = _handle_missing_knn(df, numeric_cols, object_cols)
            filled_count = missing_before - df.isnull().sum().sum()
            cleaning_log["steps"].append({
                "step": "fill_missing_knn",
                "filled_count": int(filled_count),
            })
            logger.info(f"KNN imputed {filled_count} missing values")
        else:
            strategy_fill = {"median": "median", "mean": "mean", "mode": "mode"}.get(missing_strategy, "median")
            for col in numeric_cols:
                col_missing = df[col].isnull().sum()
                if col_missing > 0:
                    if missing_strategy == "median":
                        fill_value = df[col].median()
                    elif missing_strategy == "mean":
                        fill_value = df[col].mean()
                    elif missing_strategy == "mode":
                        fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                    else:
                        fill_value = df[col].median()

                    df[col] = df[col].fillna(fill_value)
                    cleaning_log["steps"].append({
                        "step": "fill_missing",
                        "column": col,
                        "strategy": missing_strategy,
                        "filled_count": int(col_missing),
                        "fill_value": float(fill_value) if isinstance(fill_value, (np.floating, np.integer, float)) else fill_value,
                    })

            for col in object_cols:
                col_missing = df[col].isnull().sum()
                if col_missing > 0:
                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                    df[col] = df[col].fillna(mode_value)
                    cleaning_log["steps"].append({
                        "step": "fill_missing_categorical",
                        "column": col,
                        "strategy": "mode",
                        "filled_count": int(col_missing),
                        "fill_value": mode_value,
                    })

    missing_after = df.isnull().sum().sum()
    cleaning_log["steps"].append({
        "step": "missing_summary",
        "before": int(missing_before),
        "after": int(missing_after),
    })

    if detect_outliers_flag:
        outlier_report = detect_outliers(df, method="both")
        cleaning_log["outlier_detection"] = outlier_report
        
        if outlier_strategy == "remove":
            outlier_rows = set()
            for col in numeric_cols:
                data = df[col].dropna()
                if len(data) < 3:
                    continue
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier_idx = df.index[(df[col] < lower) | (df[col] > upper)]
                outlier_rows.update(outlier_idx)
            removed_count = len(outlier_rows)
            df = df.drop(index=list(outlier_rows))
            df = df.reset_index(drop=True)
            cleaning_log["steps"].append({
                "step": "remove_outliers",
                "method": "IQR",
                "count": removed_count,
            })
            logger.info(f"Removed {removed_count} rows with outlier values")
        elif outlier_strategy == "winsorize":
            for col in numeric_cols:
                data = df[col].dropna()
                if len(data) < 3:
                    continue
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                winsorized_count = int(((df[col] < lower) | (df[col] > upper)).sum())
                if winsorized_count > 0:
                    df[col] = df[col].clip(lower=lower, upper=upper)
            cleaning_log["steps"].append({
                "step": "winsorize_outliers",
                "method": "IQR边界缩尾",
            })
            logger.info("Winsorized outlier values to IQR bounds")
        elif outlier_strategy == "mark_na":
            for col in numeric_cols:
                data = df[col].dropna()
                if len(data) < 3:
                    continue
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                marked = int(((df[col] < lower) | (df[col] > upper)).sum())
                if marked > 0:
                    df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan
            missing_recount = df.isnull().sum().sum()
            cleaning_log["steps"].append({
                "step": "mark_outliers_as_na",
                "marked_count": int((~outlier_report["outlier_details"]).sum() if hasattr(outlier_report["outlier_details"], "sum") else outlier_report["total_outliers"]),
            })
            if missing_recount > 0:
                for col in numeric_cols:
                    col_missing = df[col].isnull().sum()
                    if col_missing > 0:
                        df[col] = df[col].fillna(df[col].median())
                for col in object_cols:
                    col_missing = df[col].isnull().sum()
                    if col_missing > 0:
                        mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                        df[col] = df[col].fillna(mode_value)
            cleaning_log["steps"].append({
                "step": "fill_missing_after_outlier_marking",
            })
            logger.info("Marked outliers as NA and re-filled missing values")
        else:
            cleaning_log["steps"].append({
                "step": "outlier_detection",
                "method": "3σ原则 + 箱线图法（IQR）",
                "total_outliers": outlier_report["total_outliers"],
                "outlier_percentage": f"{outlier_report['outlier_percentage']}%",
                "outlier_columns": list(outlier_report["outlier_columns"].keys()),
            })

    cleaning_log["final_shape"] = df.shape
    cleaning_log["final_rows"] = len(df)
    cleaning_log["final_cols"] = len(df.columns)
    cleaning_log["column_types"] = {
        col: str(dtype) for col, dtype in df.dtypes.items()
    }
    cleaning_log["numeric_columns"] = df.select_dtypes(include=[np.number]).columns.tolist()
    cleaning_log["categorical_columns"] = df.select_dtypes(include=["object", "category"]).columns.tolist()

    return df, cleaning_log

def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
    }

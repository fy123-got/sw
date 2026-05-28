import os
import sys
import time
import uuid
import json
import argparse
import threading
from pathlib import Path
from urllib.parse import unquote
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request, Response, Cookie
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.utils import (
    logger, load_config, cache, generate_session_id, get_session_dir,
    generate_cache_key, cleanup_old_files, make_serializable, BASE_DIR, DATA_DIR, CHARTS_DIR
)
from app.modules.cleaner import clean_data, detect_column_types
from app.modules.stat_tools import (
    descriptive_statistics, check_normality, t_test_independent, t_test_paired,
    mann_whitney_u, one_way_anova, correlation_analysis, linear_regression,
    dose_response, survival_analysis, recommend_test, stepwise_regression
)
from app.modules.ai_client import ai_client
from app.modules.visualizer import generate_all_charts
from app.modules.report_builder import create_word_report
from app.modules.data_fetcher import data_fetcher
from app.modules.auto_analysis_engine import auto_engine
from app.modules.issue_advisor import issue_advisor
from app.modules.bio_db_adapter import BioDatabaseAdapter

config = load_config()

app = FastAPI(title="生物医药数据智能分析平台", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/charts", StaticFiles(directory=str(CHARTS_DIR)), name="charts")

async_tasks: Dict[str, Dict[str, Any]] = {}
session_analysis_cache: Dict[str, Dict[str, Any]] = {}
latest_auto_analysis: Dict[str, Any] = {}  # 存储最新的自动分析结果
bio_adapter = BioDatabaseAdapter()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Biomed Platform...")
    cleanup_thread = threading.Thread(target=_periodic_cleanup, daemon=True)
    cleanup_thread.start()
    data_fetcher.start_all()
    data_fetcher.on_new_data(_on_new_data_callback)

def _on_new_data_callback(filepath: str, record: dict):
    logger.info(f"New data received: {filepath}")
    # 在新线程中自动分析新获取的数据
    import threading
    t = threading.Thread(target=_sync_auto_analyze_file, args=(filepath,), daemon=True)
    t.start()

def _sync_auto_analyze_file(filepath: str):
    """同步版本的自动分析（用于后台线程调用）"""
    try:
        import time
        time.sleep(0.5)
        
        logger.info(f"Starting auto analysis for watched file: {filepath}")
        
        df = _read_file(filepath)
        if df.empty:
            logger.error("Data is empty after reading")
            return
        
        logger.info(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
        
        cleaned_df, cleaning_log = clean_data(df)
        if cleaned_df.empty:
            logger.error("Data is empty after cleaning")
            return
        
        logger.info(f"Data cleaned: {len(cleaned_df)} rows remaining")
        
        # 获取或创建 session_id
        sid = generate_session_id()
        get_session_dir(sid)
        
        results = auto_engine.analyze(cleaned_df)
        results["cleaning_log"] = cleaning_log
        results["filename"] = os.path.basename(filepath)
        
        # Generate charts
        try:
            charts = generate_all_charts(cleaned_df, sid, results)
            # Keep chart_captions for report generation
            results["charts"] = {k: v for k, v in charts.items() if not k.startswith("interactive_")}
            results["interactive_charts"] = {k: v for k, v in charts.items() if k.startswith("interactive_")}
            logger.info("Charts generated successfully")
        except Exception as chart_err:
            logger.error(f"Chart generation error: {chart_err}")
            results["charts"] = {}
            results["interactive_charts"] = {}
        
        issues = issue_advisor.analyze(results, cleaning_log)
        results["issues"] = issues
        
        session_analysis_cache[sid] = results
        latest_auto_analysis.update(results)  # 同时存储到全局变量
        
        logger.info(f"Auto analysis completed for watched file: {filepath}")
        logger.info(f"Results keys: {list(results.keys())}")
    except Exception as e:
        import traceback
        logger.error(f"Auto analyze after file watch error: {e}")
        logger.error(traceback.format_exc())

def _periodic_cleanup():
    while True:
        time.sleep(3600)
        try:
            cleanup_old_files(max_age_hours=config.get("cleanup", {}).get("max_age_hours", 24))
            cache.cleanup()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def _get_session_id(request: Request, session_id: Optional[str] = None) -> str:
    if session_id:
        return session_id
    sid = request.cookies.get("session_id")
    if not sid:
        sid = generate_session_id()
    get_session_dir(sid)
    return sid

def _read_file(filepath: str) -> pd.DataFrame:
    ext = Path(filepath).suffix.lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(filepath)
    elif ext == ".csv":
        return pd.read_csv(filepath)
    elif ext == ".pdf":
        try:
            import PyPDF2
            import tabula
            dfs = tabula.read_pdf(filepath, pages="all", multiple_tables=True)
            if dfs:
                return pd.concat(dfs, ignore_index=True)
            return pd.DataFrame()
        except ImportError:
            raise HTTPException(status_code=400, detail="PDF解析需要安装tabula-py: pip install tabula-py")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF解析失败: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}")

@app.get("/", response_class=HTMLResponse)
async def index():
    try:
        template_path = BASE_DIR / "templates" / "index.html"
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to load index.html: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load page: {str(e)}")

@app.post("/api/upload")
async def upload_file(request: Request, file: UploadFile = File(...), session_id: Optional[str] = Cookie(None)):
    try:
        sid = _get_session_id(request, session_id)
        session_dir = get_session_dir(sid)

        original_filename = unquote(file.filename) if file.filename else "unnamed"
        safe_name = Path(original_filename).name

        filepath = session_dir / safe_name
        counter = 1
        while filepath.exists():
            stem = Path(safe_name).stem
            suffix = Path(safe_name).suffix
            filepath = session_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)

        df = _read_file(str(filepath))
        col_types = detect_column_types(df)

        response = JSONResponse({
            "message": "上传成功",
            "filename": filepath.name,
            "filepath": str(filepath),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "column_types": col_types,
            "session_id": sid,
        })
        response.set_cookie(key="session_id", value=sid, max_age=86400, httponly=True)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze(request: Request, session_id: Optional[str] = Cookie(None)):
    try:
        sid = _get_session_id(request, session_id)
        session_dir = get_session_dir(sid)

        files = list(session_dir.glob("*"))
        if not files:
            raise HTTPException(status_code=400, detail="请先上传数据文件")

        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        df = _read_file(str(latest_file))

        if df.empty:
            return {"error": "数据为空，无法进行分析"}

        # Get analysis options from request body
        try:
            body = await request.json()
        except Exception:
            body = {}

        analysis_mode = body.get("analysis_mode", "auto")
        analysis_config = body.get("analysis_config", {})

        if analysis_mode == "manual" and analysis_config:
            options = {
                "descriptive": analysis_config.get("descriptive", True),
                "ttest": analysis_config.get("ttest", False),
                "anova": analysis_config.get("anova", False),
                "correlation": analysis_config.get("correlation", True),
                "regression": analysis_config.get("regression", False),
                "regression_method": analysis_config.get("regression_method", "ols"),
                "normality": analysis_config.get("normality", True),
                "correction_method": analysis_config.get("correction_method", "bonferroni"),
                "missing_strategy": analysis_config.get("missing_strategy", "median"),
                "outlier_strategy": analysis_config.get("outlier_strategy", "detect_only"),
            }
        else:
            options = {
                "descriptive": body.get("descriptive", True),
                "ttest": body.get("ttest", False),
                "anova": body.get("anova", False),
                "correlation": body.get("correlation", True),
                "regression": body.get("regression", False),
                "regression_method": body.get("regression_method", "ols"),
                "normality": body.get("normality", False),
                "correction_method": "bonferroni",
                "missing_strategy": "median",
                "outlier_strategy": "detect_only",
            }

        missing_strategy = options.get("missing_strategy", "median")
        outlier_strategy = options.get("outlier_strategy", "detect_only")
        cleaned_df, cleaning_log = clean_data(
            df,
            missing_strategy=missing_strategy,
            outlier_strategy=outlier_strategy,
        )

        if cleaned_df.empty:
            return {"error": "清洗后数据为空"}

        results = {
            "filename": latest_file.name,
            "cleaning_log": cleaning_log,
        }

        # Descriptive statistics
        if options["descriptive"]:
            results["descriptive_statistics"] = descriptive_statistics(cleaned_df)

        # Normality test
        if options["normality"]:
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
            normality_results = {}
            for col in numeric_cols:
                data = cleaned_df[col].dropna()
                if len(data) >= 3:
                    normality_results[col] = check_normality(data)
            results["normality_tests"] = normality_results

        # T-test (auto-detect groups if categorical column exists)
        if options["ttest"]:
            categorical_cols = cleaned_df.select_dtypes(include=["object", "category"]).columns.tolist()
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
            if categorical_cols and numeric_cols:
                group_col = categorical_cols[0]
                groups = cleaned_df[group_col].unique()
                if len(groups) == 2:
                    ttest_results = {}
                    for num_col in numeric_cols[:3]:
                        result = t_test_independent(cleaned_df, num_col, group_col, str(groups[0]), str(groups[1]))
                        ttest_results[f"{num_col}_by_{group_col}"] = result
                    results["t_tests"] = ttest_results

        # ANOVA
        if options["anova"]:
            categorical_cols = cleaned_df.select_dtypes(include=["object", "category"]).columns.tolist()
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
            if categorical_cols and numeric_cols:
                group_col = categorical_cols[0]
                n_groups = cleaned_df[group_col].nunique()
                if n_groups >= 3:
                    anova_results = {}
                    for num_col in numeric_cols[:3]:
                        result = one_way_anova(cleaned_df, num_col, group_col)
                        anova_results[f"{num_col}_by_{group_col}"] = result
                    results["anova_tests"] = anova_results

        # Correlation analysis
        if options["correlation"]:
            results["correlation_analysis"] = correlation_analysis(cleaned_df)

        # Regression analysis
        if options["regression"]:
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                target = numeric_cols[0]
                features = numeric_cols[1:3]
                regression_method = options.get("regression_method", "ols")
                if regression_method == "stepwise":
                    results["regression_analysis"] = stepwise_regression(cleaned_df, target, features)
                else:
                    results["regression_analysis"] = linear_regression(cleaned_df, target, features, method=regression_method)

        charts = generate_all_charts(cleaned_df, sid, results)
        results["charts"] = {k: v for k, v in charts.items() if not k.startswith("interactive_") and k != "chart_captions"}
        results["interactive_charts"] = {k: v for k, v in charts.items() if k.startswith("interactive_")}

        cache_key = generate_cache_key(latest_file.name, options)
        cache.set(cache_key, results)

        # Store in session cache for report generation
        session_analysis_cache[sid] = results

        return make_serializable(results)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-async")
async def analyze_async(request: Request, background_tasks: BackgroundTasks, session_id: Optional[str] = Cookie(None)):
    try:
        sid = _get_session_id(request, session_id)
        task_id = str(uuid.uuid4())
        async_tasks[task_id] = {"status": "pending", "progress": 0, "result": None, "error": None}

        session_dir = get_session_dir(sid)
        files = list(session_dir.glob("*"))
        if not files:
            raise HTTPException(status_code=400, detail="请先上传数据文件")

        latest_file = max(files, key=lambda f: f.stat().st_mtime)

        background_tasks.add_task(_run_async_analysis, task_id, str(latest_file), sid)

        return {"task_id": task_id, "message": "分析任务已启动"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Async analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _run_async_analysis(task_id: str, filepath: str, session_id: str):
    try:
        async_tasks[task_id]["status"] = "running"
        async_tasks[task_id]["progress"] = 10

        df = _read_file(filepath)
        async_tasks[task_id]["progress"] = 30

        if df.empty:
            async_tasks[task_id] = {"status": "failed", "error": "数据为空"}
            return

        cleaned_df, cleaning_log = clean_data(df)
        async_tasks[task_id]["progress"] = 50

        if cleaned_df.empty:
            async_tasks[task_id] = {"status": "failed", "error": "清洗后数据为空"}
            return

        results = {
            "filename": Path(filepath).name,
            "cleaning_log": cleaning_log,
            "descriptive_statistics": descriptive_statistics(cleaned_df),
        }
        async_tasks[task_id]["progress"] = 70

        charts = generate_all_charts(cleaned_df, session_id, results)
        results["charts"] = {k: v for k, v in charts.items() if not k.startswith("interactive_") and k != "chart_captions"}
        results["interactive_charts"] = {k: v for k, v in charts.items() if k.startswith("interactive_")}
        async_tasks[task_id]["progress"] = 90

        async_tasks[task_id]["status"] = "completed"
        async_tasks[task_id]["progress"] = 100
        async_tasks[task_id]["result"] = make_serializable(results)

        cache_key = generate_cache_key(Path(filepath).name, {})
        cache.set(cache_key, results)
    except Exception as e:
        logger.error(f"Async analysis task {task_id} failed: {e}")
        async_tasks[task_id] = {"status": "failed", "error": str(e)}

@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in async_tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    return async_tasks[task_id]

@app.get("/api/report")
async def generate_report(request: Request, session_id: Optional[str] = Cookie(None)):
    try:
        sid = _get_session_id(request, session_id)
        session_dir = get_session_dir(sid)

        files = list(session_dir.glob("*"))
        if not files:
            raise HTTPException(status_code=400, detail="没有可用的数据")

        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        df = _read_file(str(latest_file))

        if df.empty:
            raise HTTPException(status_code=400, detail="数据为空")

        missing_strategy = "median"
        outlier_strategy = "detect_only"
        cleaned_df, cleaning_log = clean_data(df, missing_strategy=missing_strategy, outlier_strategy=outlier_strategy)

        analysis_results = session_analysis_cache.get(sid)

        if not analysis_results:
            cache_key_prefix = generate_cache_key(latest_file.name, {})
            analysis_results = cache.get(cache_key_prefix)

        # 如果还是没找到，尝试在所有session中查找最新的分析结果
        if not analysis_results:
            for cached_sid, cached_results in session_analysis_cache.items():
                if cached_results.get("filename") == latest_file.name:
                    analysis_results = cached_results
                    logger.info(f"Found analysis results from session {cached_sid} for file {latest_file.name}")
                    break

        if not analysis_results:
            analysis_results = {
                "descriptive_statistics": descriptive_statistics(cleaned_df),
            }

        if "charts" not in analysis_results:
            charts = generate_all_charts(cleaned_df, sid, analysis_results)
            analysis_results["charts"] = {k: v for k, v in charts.items() if not k.startswith("interactive_")}

        static_charts = analysis_results.get("charts", {})

        serializable_results = make_serializable(analysis_results)
        ai_conclusion = ai_client.generate_conclusion(serializable_results)
        if ai_conclusion and ("失败" in ai_conclusion or "error" in ai_conclusion.lower() or "API Key" in ai_conclusion):
            logger.warning("AI conclusion contains error, skipping")
            ai_conclusion = None

        # 确保issues存在，如果不存在则重新生成
        issues = analysis_results.get("issues", [])
        if not issues:
            issues = issue_advisor.analyze(analysis_results, cleaning_log)
            analysis_results["issues"] = issues

        report_params = request.query_params
        template_type = report_params.get("template_type", "academic")
        language = report_params.get("language", "zh")
        include_raw_data = report_params.get("include_raw_data", "false").lower() == "true"
        include_code_snippet = report_params.get("include_code_snippet", "false").lower() == "true"
        author = report_params.get("author")
        institution = report_params.get("institution")
        project_id = report_params.get("project_id")
        
        author_info = None
        if author or institution or project_id:
            author_info = {
                "author": author,
                "institution": institution,
                "project_id": project_id,
            }
        
        logo_path = report_params.get("logo_path")

        report_path = create_word_report(
            filename=latest_file.name,
            df=cleaned_df,
            cleaning_log=cleaning_log,
            analysis_results=analysis_results,
            charts=static_charts,
            ai_conclusion=ai_conclusion,
            issues=issues,
            template_type=template_type,
            logo_path=logo_path,
            author_info=author_info,
            language=language,
            include_raw_data=include_raw_data,
            include_code_snippet=include_code_snippet,
        )

        return FileResponse(
            path=report_path,
            filename=f"report_{latest_file.name}.docx",
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/latest-analysis")
async def get_latest_analysis(request: Request, session_id: Optional[str] = Cookie(None)):
    """获取最新的分析结果（跨session查找）"""
    try:
        sid = _get_session_id(request, session_id)
        
        # 先查找当前session的结果
        analysis_results = session_analysis_cache.get(sid)
        if analysis_results:
            return make_serializable(analysis_results)
        
        # 如果没有，返回全局最新的自动分析结果
        if latest_auto_analysis and "filename" in latest_auto_analysis:
            logger.info(f"Returning latest auto analysis: {latest_auto_analysis.get('filename')}")
            return make_serializable(latest_auto_analysis)
        
        return {"error": "暂无分析结果"}
    except Exception as e:
        logger.error(f"Get latest analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/normality-test")
async def normality_test(request: Request, session_id: Optional[str] = Cookie(None)):
    try:
        body = await request.json()
        col = body.get("column")
        if not col:
            raise HTTPException(status_code=400, detail="需要指定column参数")

        sid = _get_session_id(request, session_id)
        df = _load_session_df(sid)
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"列'{col}'不存在")

        data = df[col].dropna()
        return make_serializable(check_normality(data))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/homogeneity-test")
async def homogeneity_test(request: Request, session_id: Optional[str] = Cookie(None)):
    try:
        body = await request.json()
        col = body.get("column")
        group_col = body.get("group_column")
        if not col or not group_col:
            raise HTTPException(status_code=400, detail="需要column和group_column参数")

        sid = _get_session_id(request, session_id)
        df = _load_session_df(sid)

        groups = [df[df[group_col] == g][col].dropna() for g in df[group_col].unique()]
        groups = [g for g in groups if len(g) >= 2]

        from scipy.stats import levene
        stat, p_value = levene(*groups)
        return {"test": "Levene", "statistic": float(stat), "p_value": float(p_value), "equal_variance": p_value > 0.05}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/anova")
async def anova(request: Request, session_id: Optional[str] = Cookie(None)):
    try:
        body = await request.json()
        col = body.get("column")
        group_col = body.get("group_column")
        if not col or not group_col:
            raise HTTPException(status_code=400, detail="需要column和group_column参数")

        sid = _get_session_id(request, session_id)
        df = _load_session_df(sid)
        return make_serializable(one_way_anova(df, col, group_col))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/correlation")
async def correlation(request: Request, session_id: Optional[str] = Cookie(None)):
    try:
        body = await request.json()
        method = body.get("method", "pearson")

        sid = _get_session_id(request, session_id)
        df = _load_session_df(sid)
        return make_serializable(correlation_analysis(df, method))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/linear-regression")
async def linear_reg(request: Request, session_id: Optional[str] = Cookie(None)):
    try:
        body = await request.json()
        target = body.get("target")
        features = body.get("features", [])
        if not target or not features:
            raise HTTPException(status_code=400, detail="需要target和features参数")

        sid = _get_session_id(request, session_id)
        df = _load_session_df(sid)
        return make_serializable(linear_regression(df, target, features))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/dose-response")
async def dose_resp(request: Request, session_id: Optional[str] = Cookie(None)):
    try:
        body = await request.json()
        dose_col = body.get("dose_column")
        response_col = body.get("response_column")
        if not dose_col or not response_col:
            raise HTTPException(status_code=400, detail="需要dose_column和response_column参数")

        sid = _get_session_id(request, session_id)
        df = _load_session_df(sid)
        result = dose_response(df, dose_col, response_col)
        result["dose_col"] = dose_col
        result["response_col"] = response_col
        return make_serializable(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/survival-analysis")
async def survival(request: Request, session_id: Optional[str] = Cookie(None)):
    try:
        body = await request.json()
        time_col = body.get("time_column")
        event_col = body.get("event_column")
        group_col = body.get("group_column")
        if not time_col or not event_col:
            raise HTTPException(status_code=400, detail="需要time_column和event_column参数")

        sid = _get_session_id(request, session_id)
        df = _load_session_df(sid)
        return make_serializable(survival_analysis(df, time_col, event_col, group_col))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/recommend-test")
async def recommend(request: Request, session_id: Optional[str] = Cookie(None)):
    try:
        sid = _get_session_id(request, session_id)
        df = _load_session_df(sid)
        return make_serializable(recommend_test(df))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conversation")
async def conversation(request: Request, session_id: Optional[str] = Cookie(None)):
    try:
        body = await request.json()
        message = body.get("message")
        if not message:
            raise HTTPException(status_code=400, detail="需要message参数")

        sid = _get_session_id(request, session_id)
        stats_summary = body.get("stats_summary", {})
        response_text = ai_client.chat(sid, message, stats_summary)
        return {"response": response_text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect-anomalies")
async def detect_anomalies(request: Request, session_id: Optional[str] = Cookie(None)):
    try:
        body = await request.json()
        removed_data_info = body.get("cleaning_log", {})

        sid = _get_session_id(request, session_id)
        analysis = ai_client.analyze_anomalies(removed_data_info)
        return {"analysis": analysis}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/methods-paragraph")
async def methods_paragraph(request: Request, session_id: Optional[str] = Cookie(None)):
    try:
        body = await request.json()
        analysis_context = body.get("analysis_context", {})

        sid = _get_session_id(request, session_id)
        paragraph = ai_client.generate_methods(analysis_context)
        return {"methods_paragraph": paragraph}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _load_session_df(session_id: str) -> pd.DataFrame:
    session_dir = get_session_dir(session_id)
    files = list(session_dir.glob("*"))
    if not files:
        raise HTTPException(status_code=400, detail="没有可用的数据文件")
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    return _read_file(str(latest_file))

@app.post("/api/auto-fetch/config")
async def configure_auto_fetch(request: Request):
    try:
        body = await request.json()
        action = body.get("action", "add")
        if action == "add":
            source = body.get("source", {})
            result = data_fetcher.add_source(source)
        elif action == "remove":
            source_id = body.get("source_id", "")
            result = data_fetcher.remove_source(source_id)
        elif action == "list":
            result = {"sources": data_fetcher.get_sources()}
        else:
            raise HTTPException(status_code=400, detail="不支持的操作")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auto-fetch config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auto-fetch/trigger")
async def trigger_auto_fetch(request: Request):
    try:
        body = await request.json()
        source_id = body.get("source_id", "")
        if not source_id:
            raise HTTPException(status_code=400, detail="需要source_id参数")
        result = data_fetcher.trigger_fetch(source_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auto-fetch trigger error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auto-fetch/history")
async def get_fetch_history(limit: int = 50):
    return {"history": data_fetcher.get_history(limit)}

@app.post("/api/auto-fetch/folder-watch/start")
async def start_folder_watch(request: Request):
    try:
        body = await request.json()
        folder_path = body.get("folder_path", "")
        
        # 修复中文路径编码问题
        try:
            folder_path = folder_path.encode('latin-1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass  # 如果已经是正确的UTF-8编码，则保持原样
        
        interval = body.get("interval", 5)
        
        if not folder_path:
            raise HTTPException(status_code=400, detail="需要folder_path参数")
        
        # 标准化路径（处理Windows路径）
        folder_path = os.path.normpath(folder_path)
        
        logger.info(f"Starting folder watch for: {folder_path}")
        logger.info(f"Path exists: {os.path.isdir(folder_path)}")
        
        source = {
            "type": "folder_watch",
            "folder_path": folder_path,
            "interval": interval,
            "name": f"文件夹监听: {folder_path}",
            "enabled": True,
            "extensions": [".csv", ".xlsx", ".xls"],
        }
        result = data_fetcher.add_source(source)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Start folder watch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auto-fetch/folder-watch/stop")
async def stop_folder_watch(request: Request):
    try:
        body = await request.json()
        source_id = body.get("source_id", "")
        if not source_id:
            sources = data_fetcher.get_sources()
            folder_sources = [s for s in sources if s.get("type") == "folder_watch"]
            if folder_sources:
                source_id = folder_sources[-1]["id"]
            else:
                raise HTTPException(status_code=400, detail="没有正在运行的监听任务")
        
        result = data_fetcher.remove_source(source_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stop folder watch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auto-fetch/folder-watch/scan")
async def scan_folder_now(request: Request):
    try:
        body = await request.json()
        folder_path = body.get("folder_path", "")
        
        if not folder_path:
            raise HTTPException(status_code=400, detail="需要folder_path参数")
        
        source = {
            "type": "folder_watch",
            "folder_path": folder_path,
            "id": "temp_scan",
            "name": "临时扫描",
            "extensions": [".csv", ".xlsx", ".xls"],
        }
        result = data_fetcher.trigger_fetch("temp_scan")
        if not result.get("success"):
            result = data_fetcher._scan_folder(source)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scan folder error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/webhook/data")
async def webhook_data(request: Request):
    try:
        body = await request.body()
        filename = request.query_params.get("filename")
        result = data_fetcher.handle_webhook(body, filename)
        return result
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/bio-database/sources")
async def list_bio_sources():
    """列出所有可用的生物医学数据源"""
    try:
        sources = bio_adapter.list_sources()
        return {"sources": sources}
    except Exception as e:
        logger.error(f"List bio sources error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/bio-database/fetch")
async def fetch_bio_data(request: Request, background_tasks: BackgroundTasks, session_id: Optional[str] = Cookie(None)):
    """从生物医学数据库获取数据"""
    try:
        body = await request.json()
        source = body.get("source", "")
        identifier = body.get("identifier", "")
        
        if not source or not identifier:
            raise HTTPException(status_code=400, detail="需要source和identifier参数")
        
        result = await bio_adapter.fetch_data(source, identifier)
        
        if result.get("success"):
            sid = _get_session_id(request, session_id)
            session_dir = get_session_dir(sid)
            
            import shutil
            dest_path = os.path.join(session_dir, result["filename"])
            shutil.copy2(result["path"], dest_path)
            
            background_tasks.add_task(_auto_analyze_after_bio_fetch, sid, dest_path)
            
            response = JSONResponse(content={
                "success": True,
                "message": f"数据获取成功：{result['rows']}行 x {result['cols']}列",
                "filename": result["filename"],
                "rows": result["rows"],
                "cols": result["cols"],
                "columns": result["columns"],
                "session_id": sid,
            })
            response.set_cookie(key="session_id", value=sid, max_age=86400, httponly=True)
            return response
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "数据获取失败"))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bio database fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _auto_analyze_after_bio_fetch(sid: str, filepath: str):
    """生物数据库获取后自动分析"""
    try:
        import asyncio
        await asyncio.sleep(1)
        
        logger.info(f"Starting auto analysis for: {filepath}")
        
        df = _read_file(filepath)
        if df.empty:
            logger.error("Data is empty after reading")
            return
        
        logger.info(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
        
        cleaned_df, cleaning_log = clean_data(df)
        if cleaned_df.empty:
            logger.error("Data is empty after cleaning")
            return
        
        logger.info(f"Data cleaned: {len(cleaned_df)} rows remaining")
        
        results = auto_engine.analyze(cleaned_df)
        results["cleaning_log"] = cleaning_log
        results["filename"] = os.path.basename(filepath)
        
        # Generate charts with session_id
        try:
            charts = generate_all_charts(cleaned_df, sid, results)
            # Keep chart_captions for report generation
            results["charts"] = {k: v for k, v in charts.items() if not k.startswith("interactive_")}
            results["interactive_charts"] = {k: v for k, v in charts.items() if k.startswith("interactive_")}
            logger.info("Charts generated successfully")
        except Exception as chart_err:
            logger.error(f"Chart generation error: {chart_err}")
            results["charts"] = {}
            results["interactive_charts"] = {}
        
        issues = issue_advisor.analyze(results, cleaning_log)
        results["issues"] = issues
        
        session_analysis_cache[sid] = results
        
        logger.info(f"Auto analysis completed for bio data: {filepath}")
        logger.info(f"Results keys: {list(results.keys())}")
    except Exception as e:
        import traceback
        logger.error(f"Auto analyze after bio fetch error: {e}")
        logger.error(traceback.format_exc())

@app.post("/api/auto-analyze")
async def auto_analyze(request: Request, session_id: Optional[str] = Cookie(None)):
    try:
        sid = _get_session_id(request, session_id)
        session_dir = get_session_dir(sid)

        files = list(session_dir.glob("*"))
        if not files:
            raise HTTPException(status_code=400, detail="请先上传数据文件")

        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        df = _read_file(str(latest_file))

        if df.empty:
            return {"error": "数据为空，无法进行分析"}

        cleaned_df, cleaning_log = clean_data(df)

        if cleaned_df.empty:
            return {"error": "清洗后数据为空"}

        results = auto_engine.analyze(cleaned_df)
        results["cleaning_log"] = cleaning_log
        results["filename"] = latest_file.name

        issues = issue_advisor.analyze(results, cleaning_log)
        results["issues"] = issues

        session_analysis_cache[sid] = results

        return make_serializable(results)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auto analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/issues")
async def get_issues(request: Request, session_id: Optional[str] = Cookie(None)):
    try:
        sid = _get_session_id(request, session_id)
        results = session_analysis_cache.get(sid)
        if not results:
            return {"issues": []}
        issues = results.get("issues", [])
        return {"issues": issues}
    except Exception as e:
        logger.error(f"Get issues error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results")
async def get_results(request: Request, session_id: Optional[str] = Cookie(None)):
    """获取当前会话的分析结果"""
    try:
        sid = _get_session_id(request, session_id)
        results = session_analysis_cache.get(sid)
        if not results:
            return {"results": None, "message": "暂无分析结果"}
        return {"results": make_serializable(results)}
    except Exception as e:
        logger.error(f"Get results error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cleaning-details")
async def get_cleaning_details(request: Request, session_id: Optional[str] = Cookie(None)):
    try:
        sid = _get_session_id(request, session_id)
        results = session_analysis_cache.get(sid)
        if not results:
            session_dir = get_session_dir(sid)
            files = list(session_dir.glob("*"))
            if not files:
                raise HTTPException(status_code=400, detail="没有可用的数据")
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            df = _read_file(str(latest_file))
            _, cleaning_log = clean_data(df)
            return cleaning_log
        cleaning_log = results.get("cleaning_log", {})
        return cleaning_log
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get cleaning details error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生物医药数据智能分析平台")
    parser.add_argument("--port", type=int, default=config.get("server", {}).get("port", 12245), help="服务器端口")
    parser.add_argument("--host", type=str, default=config.get("server", {}).get("host", "0.0.0.0"), help="服务器地址")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

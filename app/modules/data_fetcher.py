import os
import json
import time
import threading
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from app.utils import logger, BASE_DIR

AUTO_FETCH_DIR = BASE_DIR / "data" / "auto_fetched"
AUTO_PROCESSED_DIR = BASE_DIR / "data" / "auto_processed"
METADATA_FILE = BASE_DIR / "data" / "fetch_metadata.json"

AUTO_FETCH_DIR.mkdir(parents=True, exist_ok=True)
AUTO_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

class DataFetcher:
    def __init__(self):
        self._watchers = {}
        self._scheduler_jobs = []
        self._metadata = self._load_metadata()
        self._callbacks = []
        self._running = False
        self._scheduler = None

    def _load_metadata(self) -> Dict[str, Any]:
        if METADATA_FILE.exists():
            try:
                with open(METADATA_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {"sources": [], "history": []}
        return {"sources": [], "history": []}

    def _save_metadata(self):
        try:
            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save fetch metadata: {e}")

    def add_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        source_id = source.get("id") or f"source_{len(self._metadata['sources']) + 1}"
        source["id"] = source_id
        source["created_at"] = datetime.now().isoformat()
        source["enabled"] = source.get("enabled", True)
        self._metadata["sources"].append(source)
        self._save_metadata()

        if source.get("type") == "folder_watch":
            self._start_folder_watcher(source)
        elif source.get("type") == "url_fetch":
            if source.get("schedule"):
                self._schedule_url_fetch(source)

        return {"success": True, "source_id": source_id, "source": source}

    def remove_source(self, source_id: str) -> Dict[str, Any]:
        self._metadata["sources"] = [s for s in self._metadata["sources"] if s["id"] != source_id]
        if source_id in self._watchers:
            self._watchers[source_id]["running"] = False
            del self._watchers[source_id]
        self._save_metadata()
        return {"success": True, "message": f"Source {source_id} removed"}

    def get_sources(self) -> List[Dict[str, Any]]:
        return self._metadata["sources"]

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._metadata["history"][-limit:]

    def trigger_fetch(self, source_id: str) -> Dict[str, Any]:
        source = next((s for s in self._metadata["sources"] if s["id"] == source_id), None)
        if not source:
            return {"success": False, "error": "Source not found"}

        if source["type"] == "url_fetch":
            return self._fetch_from_url(source)
        elif source["type"] == "folder_watch":
            return self._scan_folder(source)
        return {"success": False, "error": "Unsupported source type"}

    def _fetch_from_url(self, source: Dict[str, Any]) -> Dict[str, Any]:
        url = source.get("url", "")
        if not url:
            return {"success": False, "error": "No URL configured"}

        try:
            logger.info(f"Fetching data from URL: {url}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            filename = source.get("filename") or f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = AUTO_FETCH_DIR / filename

            content_type = response.headers.get("content-type", "")
            if "csv" in content_type or filename.endswith(".csv"):
                with open(filepath, "wb") as f:
                    f.write(response.content)
            elif "json" in content_type:
                import pandas as pd
                data = response.json()
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    df = pd.DataFrame({"data": [data]})
                df.to_csv(filepath, index=False, encoding="utf-8-sig")
            else:
                with open(filepath, "wb") as f:
                    f.write(response.content)

            record = {
                "source_id": source["id"],
                "source_name": source.get("name", ""),
                "type": "url_fetch",
                "url": url,
                "filename": filename,
                "filepath": str(filepath),
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            }
            self._metadata["history"].append(record)
            self._save_metadata()

            for callback in self._callbacks:
                callback(str(filepath), record)

            return {"success": True, "filename": filename, "filepath": str(filepath)}
        except Exception as e:
            record = {
                "source_id": source["id"],
                "source_name": source.get("name", ""),
                "type": "url_fetch",
                "url": url,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            self._metadata["history"].append(record)
            self._save_metadata()
            return {"success": False, "error": str(e)}

    def _start_folder_watcher(self, source: Dict[str, Any]):
        folder_path = source.get("folder_path", "")
        if not folder_path or not os.path.isdir(folder_path):
            logger.warning(f"Invalid folder path for watcher: {folder_path}")
            return

        watcher_id = source["id"]
        self._watchers[watcher_id] = {
            "source": source,
            "running": True,
            "thread": None,
            "last_scan": {},
        }

        def watch_loop():
            logger.info(f"Started folder watcher for: {folder_path}")
            while self._watchers.get(watcher_id, {}).get("running", False):
                self._scan_folder(source)
                time.sleep(source.get("interval", 5))

        thread = threading.Thread(target=watch_loop, daemon=True)
        thread.start()
        self._watchers[watcher_id]["thread"] = thread

    def _scan_folder(self, source: Dict[str, Any]) -> Dict[str, Any]:
        folder_path = source.get("folder_path", "")
        if not folder_path or not os.path.isdir(folder_path):
            return {"success": False, "error": "Invalid folder path"}

        extensions = source.get("extensions", [".csv", ".xlsx", ".xls"])
        watcher_id = source["id"]
        last_scan = self._watchers.get(watcher_id, {}).get("last_scan", {})
        new_files = []

        try:
            for filename in os.listdir(folder_path):
                filepath = os.path.join(folder_path, filename)
                if not os.path.isfile(filepath):
                    continue
                ext = os.path.splitext(filename)[1].lower()
                if ext not in extensions:
                    continue

                file_mtime = os.path.getmtime(filepath)
                if filename not in last_scan or last_scan[filename] < file_mtime:
                    dest_path = AUTO_FETCH_DIR / filename
                    import shutil
                    shutil.copy2(filepath, str(dest_path))

                    record = {
                        "source_id": source["id"],
                        "source_name": source.get("name", ""),
                        "type": "folder_watch",
                        "original_path": filepath,
                        "filename": filename,
                        "filepath": str(dest_path),
                        "status": "success",
                        "timestamp": datetime.now().isoformat(),
                    }
                    self._metadata["history"].append(record)
                    new_files.append(str(dest_path))

                    for callback in self._callbacks:
                        callback(str(dest_path), record)

                    last_scan[filename] = file_mtime

            if self._watchers.get(watcher_id):
                self._watchers[watcher_id]["last_scan"] = last_scan
            self._save_metadata()

            return {"success": True, "new_files": new_files}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _schedule_url_fetch(self, source: Dict[str, Any]):
        schedule = source.get("schedule", "")
        if not schedule:
            return

        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger

            if not hasattr(self, "_scheduler") or self._scheduler is None:
                self._scheduler = BackgroundScheduler(timezone="Asia/Shanghai")
                if not self._scheduler.running:
                    self._scheduler.start()

            def scheduled_job():
                if source.get("enabled", True):
                    self._fetch_from_url(source)

            parts = schedule.split()
            if len(parts) == 5:
                trigger = CronTrigger(
                    minute=parts[0], hour=parts[1], day=parts[2],
                    month=parts[3], day_of_week=parts[4],
                    timezone="Asia/Shanghai"
                )
            else:
                trigger = CronTrigger.from_crontab(schedule, timezone="Asia/Shanghai")

            job = self._scheduler.add_job(scheduled_job, trigger, id=source["id"], replace_existing=True)
            self._scheduler_jobs.append(job)
            logger.info(f"Scheduled fetch for source {source['id']}: {schedule}")
        except ImportError:
            logger.warning("APScheduler not installed, scheduled fetch disabled")
        except Exception as e:
            logger.error(f"Failed to schedule fetch: {e}")

    def on_new_data(self, callback: Callable):
        self._callbacks.append(callback)

    def start_all(self):
        for source in self._metadata["sources"]:
            if not source.get("enabled", True):
                continue
            if source["type"] == "folder_watch":
                self._start_folder_watcher(source)
            elif source["type"] == "url_fetch" and source.get("schedule"):
                self._schedule_url_fetch(source)
        self._running = True

    def stop_all(self):
        for watcher_id in list(self._watchers.keys()):
            self._watchers[watcher_id]["running"] = False
        self._watchers.clear()
        if hasattr(self, "_scheduler") and self._scheduler:
            self._scheduler.shutdown()
        self._running = False

    def handle_webhook(self, data: bytes, filename: str = None) -> Dict[str, Any]:
        if not filename:
            filename = f"webhook_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = AUTO_FETCH_DIR / filename

        try:
            import pandas as pd
            if filename.endswith(".json"):
                json_data = json.loads(data)
                if isinstance(json_data, list):
                    df = pd.DataFrame(json_data)
                elif isinstance(json_data, dict):
                    df = pd.DataFrame([json_data])
                else:
                    df = pd.DataFrame({"data": [json_data]})
                df.to_csv(filepath, index=False, encoding="utf-8-sig")
            else:
                with open(filepath, "wb") as f:
                    f.write(data)

            record = {
                "source_id": "webhook",
                "source_name": "Webhook",
                "type": "webhook",
                "filename": filename,
                "filepath": str(filepath),
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            }
            self._metadata["history"].append(record)
            self._save_metadata()

            for callback in self._callbacks:
                callback(str(filepath), record)

            return {"success": True, "filename": filename, "filepath": str(filepath)}
        except Exception as e:
            return {"success": False, "error": str(e)}

data_fetcher = DataFetcher()

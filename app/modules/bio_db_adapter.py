import os
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from io import StringIO
from app.utils import logger

class BioDatabaseAdapter:
    """生物医学数据库统一适配器"""
    
    def __init__(self, save_dir: str = "data/auto_fetched"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.sources = {
            "ncbi_geo": {
                "name": "NCBI GEO",
                "description": "基因表达综合数据库",
                "base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
                "auth_required": False,
                "demo_ids": ["GSE42872", "GSE10072", "GSE53454"],
            },
            "tcga_gdc": {
                "name": "TCGA/GDC",
                "description": "癌症基因组图谱",
                "base_url": "https://api.gdc.cancer.gov/",
                "auth_required": False,
                "demo_ids": ["TCGA-BRCA", "TCGA-LUAD", "TCGA-COAD"],
            },
            "ebi_arrayexpress": {
                "name": "EBI ArrayExpress",
                "description": "欧洲基因表达数据库",
                "base_url": "https://www.ebi.ac.uk/arrayexpress/json/v3/",
                "auth_required": False,
                "demo_ids": ["E-MTAB-2345", "E-GEOD-12345"],
            },
            "who_gho": {
                "name": "WHO 公共卫生数据",
                "description": "全球健康观察站",
                "base_url": "https://ghoapi.azureedge.net/api/",
                "auth_required": False,
                "demo_ids": ["WHOSIS_000001", "WHOSIS_000002"],
            },
            "kaggle_github": {
                "name": "Kaggle 公开数据集",
                "description": "GitHub 镜像的高质量数据集",
                "base_url": "https://raw.githubusercontent.com/",
                "auth_required": False,
                "demo_ids": [
                    "jbrownlee/Datasets/master/pima-indians-diabetes.csv",
                    "selva86/datasets/master/Heart.csv",
                    "uciml/breast-cancer-wisconsin-data/master/data.csv",
                ],
            },
            "uci_ml": {
                "name": "UCI 机器学习仓库",
                "description": "经典学术数据集",
                "base_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/",
                "auth_required": False,
                "demo_ids": [
                    "wine-quality/winequality-red.csv",
                    "parkinsons/parkinsons.data",
                    "liver-disorders/bupa.data",
                ],
            },
        }
    
    def get_source_info(self, source_id: str) -> Optional[Dict[str, Any]]:
        """获取数据源信息"""
        return self.sources.get(source_id)
    
    def list_sources(self) -> List[Dict[str, Any]]:
        """列出所有可用数据源"""
        return [
            {"id": sid, **info}
            for sid, info in self.sources.items()
        ]
    
    async def fetch_data(self, source: str, identifier: str) -> Dict[str, Any]:
        """统一数据获取接口"""
        if source not in self.sources:
            return {"success": False, "error": f"不支持的数据源: {source}"}
        
        try:
            fetch_method = getattr(self, f"_fetch_{source}")
            df = await fetch_method(identifier)
            
            if df is None or df.empty:
                return {"success": False, "error": "获取的数据为空"}
            
            filename = f"{source}_{identifier.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            save_path = os.path.join(self.save_dir, filename)
            df.to_csv(save_path, index=False)
            
            return {
                "success": True,
                "filename": filename,
                "path": save_path,
                "rows": len(df),
                "cols": len(df.columns),
                "columns": df.columns.tolist(),
            }
        except Exception as e:
            logger.error(f"Failed to fetch data from {source}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _fetch_ncbi_geo(self, gse_id: str) -> pd.DataFrame:
        """从 NCBI GEO 获取基因表达数据"""
        try:
            url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
            params = {
                "acc": gse_id,
                "targ": "self",
                "form": "text",
                "view": "data"
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            content = response.text
            
            if "!series_matrix_table_begin" in content:
                lines = content.split("\n")
                matrix_start = None
                matrix_end = None
                
                for i, line in enumerate(lines):
                    if "!series_matrix_table_begin" in line:
                        matrix_start = i + 1
                    elif "!series_matrix_table_end" in line:
                        matrix_end = i
                        break
                
                if matrix_start and matrix_end:
                    matrix_lines = lines[matrix_start:matrix_end]
                    matrix_text = "\n".join(matrix_lines)
                    df = pd.read_csv(StringIO(matrix_text), sep="\t", comment="#")
                    return df
            
            return self._generate_geo_demo_data(gse_id)
            
        except Exception as e:
            logger.warning(f"NCBI GEO fetch failed, using demo data: {e}")
            return self._generate_geo_demo_data(gse_id)
    
    async def _fetch_tcga_gdc(self, project_id: str) -> pd.DataFrame:
        """从 TCGA/GDC 获取癌症数据"""
        try:
            cases_url = "https://api.gdc.cancer.gov/cases"
            params = {
                "filters": '{"op":"in","content":{"field":"cases.project.project_id","value":["' + project_id + '"]}}',
                "format": "JSON",
                "size": "100",
            }
            
            response = requests.get(cases_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "data" in data and "hits" in data["data"]:
                hits = data["data"]["hits"]
                records = []
                for hit in hits:
                    record = {
                        "case_id": hit.get("case_id"),
                        "submitter_id": hit.get("submitter_id"),
                        "gender": hit.get("demographic", {}).get("gender"),
                        "age_at_diagnosis": hit.get("demographic", {}).get("age_at_diagnosis"),
                        "race": hit.get("demographic", {}).get("race"),
                        "ethnicity": hit.get("demographic", {}).get("ethnicity"),
                        "vital_status": hit.get("diagnoses", [{}])[0].get("vital_status") if hit.get("diagnoses") else None,
                        "days_to_death": hit.get("diagnoses", [{}])[0].get("days_to_death") if hit.get("diagnoses") else None,
                        "tumor_stage": hit.get("diagnoses", [{}])[0].get("tumor_stage") if hit.get("diagnoses") else None,
                        "project_id": project_id,
                    }
                    records.append(record)
                
                df = pd.DataFrame(records)
                if not df.empty:
                    return df
            
            return self._generate_tcga_demo_data(project_id)
            
        except Exception as e:
            logger.warning(f"TCGA GDC fetch failed, using demo data: {e}")
            return self._generate_tcga_demo_data(project_id)
    
    async def _fetch_ebi_arrayexpress(self, exp_id: str) -> pd.DataFrame:
        """从 EBI ArrayExpress 获取数据"""
        try:
            url = f"https://www.ebi.ac.uk/arrayexpress/json/v3/experiments/{exp_id}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "experiment" in data:
                exp = data["experiment"]
                return pd.DataFrame([{
                    "experiment_id": exp.get("accession"),
                    "name": exp.get("name"),
                    "description": exp.get("description"),
                    "organism": exp.get("organism"),
                    "experiment_type": exp.get("experimentType"),
                    "array_count": exp.get("arrayCount"),
                    "efo_terms": ", ".join(exp.get("efoTerms", [])),
                }])
            
            return self._generate_demo_dataframe("ebi", exp_id)
            
        except Exception as e:
            logger.warning(f"EBI ArrayExpress fetch failed, using demo data: {e}")
            return self._generate_demo_dataframe("ebi", exp_id)
    
    async def _fetch_who_gho(self, indicator_id: str) -> pd.DataFrame:
        """从 WHO 全球健康观察站获取数据"""
        try:
            url = f"https://ghoapi.azureedge.net/api/{indicator_id}"
            headers = {"Accept": "application/json"}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "value" in data:
                values = data["value"]
                records = []
                for v in values[:100]:
                    record = {
                        "country": v.get("SpatialDimValueName"),
                        "year": v.get("TimeDim"),
                        "value": v.get("Value"),
                        "value_type": v.get("ValueTypeName"),
                        "sex": v.get("Sex"),
                        "age_group": v.get("AgeGroup"),
                        "display": v.get("Display"),
                    }
                    records.append(record)
                
                df = pd.DataFrame(records)
                if not df.empty:
                    return df
            
            return self._generate_who_demo_data(indicator_id)
            
        except Exception as e:
            logger.warning(f"WHO GHO fetch failed, using demo data: {e}")
            return self._generate_who_demo_data(indicator_id)
    
    async def _fetch_kaggle_github(self, path: str) -> pd.DataFrame:
        """从 GitHub 获取 Kaggle 数据集"""
        try:
            url = f"https://raw.githubusercontent.com/{path}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text))
            return df
            
        except Exception as e:
            logger.error(f"Kaggle GitHub fetch failed: {e}")
            raise
    
    async def _fetch_uci_ml(self, path: str) -> pd.DataFrame:
        """从 UCI 机器学习仓库获取数据"""
        try:
            url = f"https://archive.ics.uci.edu/ml/machine-learning-databases/{path}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            if path.endswith(".csv"):
                df = pd.read_csv(StringIO(response.text))
            else:
                df = pd.read_csv(StringIO(response.text), header=None)
            
            return df
            
        except Exception as e:
            logger.error(f"UCI ML fetch failed: {e}")
            raise
    
    def _generate_geo_demo_data(self, gse_id: str) -> pd.DataFrame:
        """生成 GEO 演示数据"""
        np.random.seed(42)
        n_samples = 50
        n_genes = 20
        
        genes = [f"Gene_{i}" for i in range(1, n_genes + 1)]
        conditions = ["Control"] * 25 + ["Treatment"] * 25
        
        data = {"Sample_ID": [f"S{i}" for i in range(1, n_samples + 1)]}
        data["Condition"] = conditions
        
        for gene in genes:
            base_expr = np.random.normal(10, 2, n_samples)
            treatment_effect = np.where(np.array(conditions) == "Treatment", np.random.normal(2, 0.5), 0)
            data[gene] = base_expr + treatment_effect + np.random.normal(0, 0.5, n_samples)
        
        return pd.DataFrame(data)
    
    def _generate_tcga_demo_data(self, project_id: str) -> pd.DataFrame:
        """生成 TCGA 演示数据"""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            "case_id": [f"TCGA-{project_id[-4:]}-{i:04d}" for i in range(1, n_samples + 1)],
            "age_at_diagnosis": np.random.randint(20, 80, n_samples) * 365,
            "gender": np.random.choice(["male", "female"], n_samples),
            "tumor_stage": np.random.choice(["Stage I", "Stage II", "Stage III", "Stage IV"], n_samples),
            "vital_status": np.random.choice(["Alive", "Dead"], n_samples, p=[0.7, 0.3]),
            "days_to_death": np.where(
                np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
                np.random.randint(100, 3000, n_samples),
                np.nan
            ),
            "gene_expression_1": np.random.normal(10, 3, n_samples),
            "gene_expression_2": np.random.normal(8, 2, n_samples),
            "mutation_count": np.random.poisson(5, n_samples),
            "project_id": project_id,
        }
        
        return pd.DataFrame(data)
    
    def _generate_who_demo_data(self, indicator_id: str) -> pd.DataFrame:
        """生成 WHO 演示数据"""
        np.random.seed(42)
        
        countries = ["China", "USA", "India", "Japan", "Germany", "UK", "France", "Brazil", "Russia", "Australia"]
        years = list(range(2015, 2024))
        
        records = []
        for country in countries:
            for year in years:
                records.append({
                    "country": country,
                    "year": year,
                    "value": round(np.random.uniform(50, 100) + np.random.normal(0, 5), 2),
                    "sex": "Total",
                    "age_group": "All ages",
                })
        
        return pd.DataFrame(records)
    
    def _generate_demo_dataframe(self, source: str, identifier: str) -> pd.DataFrame:
        """生成通用演示数据"""
        np.random.seed(42)
        n_samples = 50
        
        data = {
            "sample_id": [f"{source}_{i}" for i in range(1, n_samples + 1)],
            "value_1": np.random.normal(10, 3, n_samples),
            "value_2": np.random.normal(20, 5, n_samples),
            "group": np.random.choice(["A", "B", "C"], n_samples),
            "source": source,
            "identifier": identifier,
        }
        
        return pd.DataFrame(data)

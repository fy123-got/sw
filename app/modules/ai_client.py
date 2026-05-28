import os
import json
from typing import Dict, Any, List, Optional
from openai import OpenAI
from app.utils import logger, load_config, BASE_DIR
from dotenv import load_dotenv

# 确保 .env 文件在模块加载时被加载
load_dotenv(BASE_DIR / ".env")

config = load_config()
ai_config = config.get("ai", {})

class AIClient:
    def __init__(self):
        api_key = os.getenv("ARK_API_KEY", "")
        base_url = ai_config.get("base_url", "https://ark.cn-beijing.volces.com/api/v3")
        self.model = ai_config.get("model", "doubao-seed-1-6-251015")
        self.max_tokens = ai_config.get("max_tokens", 2000)
        self.temperature = ai_config.get("temperature", 0.7)

        if not api_key:
            logger.warning("ARK_API_KEY not set, AI features will return mock responses")
            self.client = None
        else:
            logger.info(f"Using API key (last 8 chars): ...{api_key[-8:]}")
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info("Ark AI client initialized")

        self._conversation_history: Dict[str, List[dict]] = {}

    def _call_api(self, messages: List[dict], max_tokens: Optional[int] = None) -> str:
        if self.client is None:
            return "AI API Key未配置，请在.env文件中设置ARK_API_KEY"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            logger.error(f"AI API call failed: {error_msg}")
            
            if "Insufficient Balance" in error_msg or "402" in error_msg:
                return "AI服务暂时不可用（账户余额不足），请检查API账户状态后重试。"
            elif "401" in error_msg or "Unauthorized" in error_msg:
                return "AI API Key无效或已过期，请检查.env文件中的ARK_API_KEY配置。"
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                return "AI服务请求过于频繁，请稍后再试。"
            else:
                return "AI服务暂时不可用，请稍后重试。"

    def generate_conclusion(self, stats_summary: Dict[str, Any], style: str = "academic") -> str:
        system_prompts = {
            "academic": "你是一位资深的生物统计学专家，擅长用严谨的学术语言解释统计分析结果。请使用规范的科研用语，类似于SCI期刊的风格。【强制要求】：你必须且只能使用中文（简体）进行回答，绝对不要使用任何英文单词、短语或句子。所有统计术语请使用中文表达。",
            "nature": "你是一位Nature期刊的审稿人，请用Nature风格的简洁、有力的语言总结实验结果。重点突出关键发现和科学意义。【强制要求】：你必须且只能使用中文（简体）进行回答，绝对不要使用任何英文。",
            "cell": "你是一位Cell期刊的编辑，请用Cell风格的详尽、系统性语言分析实验数据。注重机制解释和生物学意义。【强制要求】：你必须且只能使用中文（简体）进行回答，绝对不要使用任何英文。",
        }
        system_prompt = system_prompts.get(style, system_prompts["academic"])

        # 提取效应量信息
        effect_sizes_info = self._extract_effect_sizes(stats_summary)
        
        # 提取生物学背景信息
        bio_context = self._extract_bio_context(stats_summary)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请根据以下统计分析结果，生成一段学术风格的实验结论。\n\n【重要】：请完全使用中文撰写，不要包含任何英文内容。\n\n【效应量信息】：\n{effect_sizes_info}\n\n【生物学背景】：\n{bio_context}\n\n【要求】：\n1. 明确引用效应量（如Cohen's d、η²、R²）并给出解读（小/中/大效应）\n2. 如果数据中含有基因名称（如Gene_1, Gene_2），请尝试给出可能的基因名并基于已知通路进行简要解释\n3. 基于显著结果给出具体的实验验证建议\n4. 最后请加上免责声明：\"本结论由大语言模型生成，仅供参考，不构成最终科学结论。\"\n\n统计分析结果：\n{json.dumps(stats_summary, ensure_ascii=False, indent=2)}"},
        ]
        return self._call_api(messages)

    def _extract_effect_sizes(self, stats_summary: Dict[str, Any]) -> str:
        """从统计结果中提取效应量信息"""
        effect_info = []
        
        # T检验效应量
        t_tests = stats_summary.get("t_tests", {})
        for key, result in t_tests.items():
            if isinstance(result, dict) and "effect_size" in result:
                es = result["effect_size"]
                if "cohens_d" in es:
                    d = es["cohens_d"]
                    size = "小" if abs(d) < 0.5 else ("中" if abs(d) < 0.8 else "大")
                    effect_info.append(f"{key}: Cohen's d = {d:.3f} ({size}效应)")
        
        # ANOVA效应量
        anova = stats_summary.get("anova_tests", {})
        for key, result in anova.items():
            if isinstance(result, dict) and "effect_size" in result:
                es = result["effect_size"]
                if "eta_squared" in es and es["eta_squared"] is not None:
                    eta = es["eta_squared"]
                    size = "小" if eta < 0.06 else ("中" if eta < 0.14 else "大")
                    effect_info.append(f"{key}: η² = {eta:.3f} ({size}效应)")
        
        # 回归R²
        regression = stats_summary.get("regression_analysis", {})
        if isinstance(regression, dict) and "r_squared" in regression:
            r2 = regression["r_squared"]
            size = "小" if r2 < 0.13 else ("中" if r2 < 0.26 else "大")
            effect_info.append(f"回归模型: R² = {r2:.3f} ({size}效应)")
        
        return "\n".join(effect_info) if effect_info else "未检测到效应量信息"

    def _extract_bio_context(self, stats_summary: Dict[str, Any]) -> str:
        """提取生物学背景信息"""
        bio_info = []
        
        # 检查是否有基因相关列名
        all_cols = []
        desc = stats_summary.get("descriptive_statistics", {})
        if desc:
            all_cols = list(desc.keys())
        
        gene_cols = [c for c in all_cols if "gene" in c.lower() or "基因" in c]
        if gene_cols:
            bio_info.append(f"检测到可能的基因相关变量：{', '.join(gene_cols)}")
        
        # 检查显著相关性
        corr = stats_summary.get("correlation_analysis", {})
        if isinstance(corr, dict) and "significant_pairs" in corr:
            sig_pairs = [p for p in corr["significant_pairs"] if p.get("significant")]
            if sig_pairs:
                pairs_str = ", ".join([f"{p['var1']}与{p['var2']}" for p in sig_pairs[:5]])
                bio_info.append(f"显著相关的变量对：{pairs_str}")
        
        return "\n".join(bio_info) if bio_info else "未检测到特定生物学背景信息"

    def chat(self, session_id: str, message: str, stats_summary: Optional[Dict[str, Any]] = None) -> str:
        if session_id not in self._conversation_history:
            self._conversation_history[session_id] = []

        context = ""
        if stats_summary:
            context = f"以下是相关的统计分析结果供参考：\n{json.dumps(stats_summary, ensure_ascii=False, indent=2)}\n\n"

        user_message = context + message if context else message

        messages = [
            {"role": "system", "content": "你是一位生物统计学专家，帮助用户理解和解释实验数据与统计分析结果。请用专业但易懂的语言回答。如果涉及统计推断，请同时说明局限性和注意事项。"},
        ]
        messages.extend(self._conversation_history[session_id][-10:])
        messages.append({"role": "user", "content": user_message})

        response = self._call_api(messages)

        self._conversation_history[session_id].append({"role": "user", "content": message})
        self._conversation_history[session_id].append({"role": "assistant", "content": response})

        return response

    def analyze_anomalies(self, removed_data_info: Dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": "你是一位数据质量分析专家，擅长识别和解释实验数据中的异常值和数据质量问题。请根据提供的清洗日志，分析可能的异常原因，并给出专业的推测和建议。"},
            {"role": "user", "content": f"以下是数据清洗过程中发现的异常信息，请分析可能的原因并给出专业建议：\n\n{json.dumps(removed_data_info, ensure_ascii=False, indent=2)}"},
        ]
        return self._call_api(messages)

    def generate_methods(self, analysis_context: Dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": "你是一位经验丰富的科研论文作者，擅长撰写Nature/Cell级别的Methods部分。请根据提供的分析摘要，生成一段规范的方法学描述，包括使用的统计方法、软件工具、显著性标准等。语言要严谨、规范、可重复。**请使用中文回答**。"},
            {"role": "user", "content": f"请根据以下分析摘要，生成一段Methods部分的方法学描述：\n\n{json.dumps(analysis_context, ensure_ascii=False, indent=2)}"},
        ]
        return self._call_api(messages)

    def clear_history(self, session_id: str):
        self._conversation_history.pop(session_id, None)

ai_client = AIClient()

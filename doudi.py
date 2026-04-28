#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import ast
import json
import sys
import logging
import subprocess
from typing import Type, List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# ============================================================
# 1. 配置与工具函数
# ============================================================
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def truncate_log(text: str, max_len: int = 400) -> str:
    if not text or len(text) <= max_len:
        return text
    return f"{text[:max_len]}\n... [已截断，共 {len(text)} 字符] ..."


# ============================================================
# 2. 深度 AST 安全校验
# ============================================================
class CodeValidator:
    ALLOWED_MODULES = {"pandas", "json", "re", "math", "datetime", "typing", "collections", "io", "string"}
    FORBIDDEN_IDENTIFIERS = {
        "eval", "exec", "compile", "open", "__import__", "input", "exit", "quit",
        "__builtins__", "globals", "locals", "getattr", "setattr", "delattr",
        "vars", "dir", "help", "breakpoint", "os", "sys", "subprocess", "socket"
    }

    @classmethod
    def check(cls, code: str) -> tuple[bool, str]:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"语法错误: {e}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] not in cls.ALLOWED_MODULES:
                        return False, f"禁止导入模块: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] not in cls.ALLOWED_MODULES:
                    return False, f"禁止导入模块: {node.module}"
            elif isinstance(node, ast.Name) and node.id in cls.FORBIDDEN_IDENTIFIERS:
                return False, f"检测到危险标识符: {node.id}"
            elif isinstance(node, ast.Attribute) and node.attr in cls.FORBIDDEN_IDENTIFIERS:
                return False, f"检测到危险属性调用: {node.attr}"
            elif isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in ("eval", "exec", "compile", "__import__"):
                    return False, f"禁止调用危险函数: {func_name}"
                if func_name in ("apply", "transform", "map"):
                    for arg in list(node.args) + [kw.value for kw in node.keywords]:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            return False, f"禁止在 {func_name} 中传入字符串逻辑"
        return True, "AST 静态检查通过"

# ============================================================
# 3. 跨平台沙箱执行器
# ============================================================
class PythonRunnerInput(BaseModel):
    code: str = Field(description="待执行的 Python 源代码")

class DataGuardSandbox(BaseTool):
    name: str = "data_guard_sandbox"
    description: str = "在严格隔离的沙箱中执行数据处理代码。输入为代码字符串。"
    args_schema: Type[BaseModel] = PythonRunnerInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    memory_limit_mb: int = Field(default=1024)

    def _extract_code_block(self, text: str) -> str:
        match = re.search(r"```(?:python|py)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        if any(kw in text.lower() for kw in ["import ", "def ", "pd.", "print("]):
            return text.strip()
        return ""

    def _run(self, code: str) -> str:
        clean_code = self._extract_code_block(code)
        if not clean_code:
            return "[SECURITY_BLOCKED] 未提取到有效代码"

        is_safe, msg = CodeValidator.check(clean_code)
        if not is_safe:
            return f"[SECURITY_BLOCKED] {msg}"

        if sys.platform == "win32":
            logger.warning("⚠️ Windows 平台限制：当前仅隔离进程组，未启用 CPU/内存硬限制。生产环境建议迁移至 Docker 或 WSL2。")

        def limit_resources():
            if sys.platform.startswith("linux"):
                import resource
                resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
                resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit_mb * 1024 * 1024, -1))

        env = os.environ.copy()
        env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})

        cmd = [sys.executable, "-c", clean_code]
        kwargs: Dict[str, Any] = {"capture_output": True, "text": True, "timeout": 30, "env": env}

        if sys.platform.startswith("linux"):
            kwargs["preexec_fn"] = limit_resources
            kwargs["start_new_session"] = True
        elif sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            result = subprocess.run(cmd, **kwargs)
            output = (result.stdout + "\n" + result.stderr).strip()
            status = "[EXEC_SUCCESS]" if result.returncode == 0 else "[EXEC_ERROR]"
            return f"{status}\n{output}"
        except subprocess.TimeoutExpired:
            return "[EXEC_ERROR] 执行超时(>30s)"
        except Exception as e:
            return f"[EXEC_ERROR] 沙箱异常: {str(e)}"

# ============================================================
# 4. 业务级 RAG 知识库
# ============================================================
class QueryInput(BaseModel):
    query: str = Field(description="检索查询关键词")

class AuditExpertRAG(BaseTool):
    name: str = "audit_knowledge_base"
    description: str = "检索企业级数据审计标准与防御性编程指南。输入查询词。"
    args_schema: Type[BaseModel] = QueryInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    vector_store: Any = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        docs_text = [
            "【规则】动态列分析：使用 df.select_dtypes(include='number') 和 include='object' 自动区分列。",
            "【Pandas】统计前必须处理脏数据：pd.to_numeric(..., errors='coerce') 过滤非数字。",
            "【空值处理】缺失值统计使用 df[col].isna().sum()，严禁直接参与除法。",
            "【输出协议】必须输出包含 columns, numeric_stats, categorical_stats, data_quality_issues 的标准 JSON。"
        ]
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        documents = splitter.create_documents(docs_text)
        self.vector_store = FAISS.from_documents(documents, embeddings)

    def _run(self, query: str) -> str:
        if self.vector_store is None:
            return "知识库未初始化"
        docs = self.vector_store.similarity_search(query, k=2)
        return "\n".join([d.page_content for d in docs])


# ============================================================
# 5. 主系统（向量化引导 + 深度校验 + 稳定重试）
# ============================================================
class DataGuardCrew:
    def __init__(
            self,
            max_retries: int = 2,
            sandbox_memory_mb: int = 1024,
            log_max_len: int = 400
    ):
        self.max_retries = max_retries or int(os.getenv("MAX_AUDIT_RETRIES", 2))
        self.sandbox_memory_mb = sandbox_memory_mb or int(os.getenv("SANDBOX_MEMORY_MB", 1024))
        self.log_max_len = log_max_len

        self.llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "qwen-max"),
            base_url=os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
            max_tokens=4096,  # ✅ 扩容防 JSON 截断
            request_timeout=120,
        )
        self.sandbox = DataGuardSandbox(memory_limit_mb=self.sandbox_memory_mb)
        self.rag = AuditExpertRAG()

        # ✅ 修复1：修正 prompt 中会导致 TypeError 的 top_freq 写法，对齐 int 类型 Schema
        self.original_code_prompt = """你是一个通用数据审计专家。请编写一个健壮的 Python 脚本，自动读取并分析 CSV 文件。
【核心要求】
1. 读取文件：使用 `pd.read_csv('sales_data.csv')`。禁止硬编码任何列名。
2. 自动分类列：使用 `df.select_dtypes(include='number')` 获取数值列，`include='object'` 获取分类列。
3. 防御性清洗：对疑似数值列使用 `pd.to_numeric(..., errors='coerce')`，将非法字符转为 NaN 再统计。
4. 🚀 标准实现范式（必须遵循）：
   ```python
   # 数值列：向量化统计（自动过滤 NaN）
   num_cols = df.select_dtypes(include='number').columns
   num_stats = df[num_cols].describe().to_dict()
   # 分类列：向量化统计（仅取 Top3 防膨胀）
   cat_cols = df.select_dtypes(include='object').columns
   cat_stats = {
       col: {
           'count': int(df[col].notna().sum()),
           'unique': int(df[col].nunique()),
           'top_value': str(df[col].mode()[0]) if not df[col].mode().empty else None,
           'top_freq': int(df[col].value_counts().iloc[0]) if not df[col].value_counts().empty else 0
       } for col in cat_cols
   }
   # 补全缺失数
   for col in num_cols: num_stats[col]['missing'] = int(df[col].isna().sum())
【强制输出协议】在代码末尾生成并打印以下 JSON，严格包裹在标记之间： ===RESULT_START=== { "row_count": <总行数>, "columns": [<所有列名>], "numeric_stats": {"列名": {"count": int, "mean": float, "min": float, "max": float, "missing": int}}, "categorical_stats": {"列名": {"count": int, "unique": int, "top_value": str, "top_freq": int, "missing": int}}, "data_quality_issues": ["发现的脏数据/异常问题描述1", "问题2"] } ===RESULT_END=== 最后打印 STATUS: DONE。"""

    def _validate_nested_structure(self, data: dict) -> tuple[bool, str]:
        """深度校验 JSON 结构，防止空字典或缺失关键字段"""
        required_outer = {"row_count", "columns", "numeric_stats", "categorical_stats", "data_quality_issues"}
        if not required_outer.issubset(data.keys()):
            return False, f"缺少外层字段: {required_outer - set(data.keys())}"
        if data.get("row_count", 0) <= 0:
            return False, "row_count无效或为0"

        num_inner = {"count", "mean", "min", "max", "missing"}
        cat_inner = {"count", "unique", "top_value", "top_freq", "missing"}

        ns = data.get("numeric_stats", {})
        if not isinstance(ns, dict):
            return False, "numeric_stats非字典"
        if ns:
            sample = next(iter(ns))
            if not num_inner.issubset(ns[sample].keys()):
                return False, f"numeric_stats 内部缺失字段，需包含: {num_inner}"

        cs = data.get("categorical_stats", {})
        if not isinstance(cs, dict):
            return False, "categorical_stats非字典"
        if cs:
            sample = next(iter(cs))
            if not cat_inner.issubset(cs[sample].keys()):
                return False, f"categorical_stats 内部缺失字段，需包含: {cat_inner}"

        return True, "结构校验通过"

    def kickoff(self, user_task: str) -> str:
        # 示例：从 user_task 或环境变量提取路径
        csv_path = user_task.split("分析")[1].strip("'\" ") if "分析" in user_task else "sales_data.csv"
        base_prompt = self.original_code_prompt.replace("sales_data.csv", csv_path)

        logger.info("🚀 启动审计编排流程...")

        planner = Agent(
            role="首席合规架构师",
            goal="制定通用数据审计蓝图，指导如何自动发现列结构并防御脏数据。",
            backstory="20年数据治理专家。强调列类型自动探测、缺失值隔离统计与零崩溃执行。",
            llm=self.llm,
            tools=[self.rag],
            allow_delegation=False,
            verbose=True
        )
        coder = Agent(
            role="自动化数据探查专家",
            goal="编写泛型分析代码，自动适配任意 CSV 结构，输出标准化统计 JSON。",
            backstory="精通 Pandas 动态列操作与防御性编程。绝不硬编码，只依赖数据类型推断。",
            llm=self.llm,
            tools=[self.sandbox],
            allow_delegation=False,
            verbose=True
        )

        plan_task = Task(
            description=f"针对任务 '{user_task}'，设计一套不依赖固定列名的自动探查与清洗策略。查阅审计知识库。",
            expected_output="详细的通用审计路线图，包含列类型识别策略、脏数据隔离方案。",
            agent=planner
        )

        real_data = None
        sandbox_out = ""

        # ✅ 稳定重试循环：每次重建 Task，避免 CrewAI 状态污染
        for attempt in range(self.max_retries):
            # 构建提示词（使用已替换文件路径的 base_prompt）
            prompt = base_prompt
            if attempt > 0:
                prompt += f"\n\n⚠️【第 {attempt} 次失败反馈】: 请严格修复结构校验问题，确保输出完整且字段不缺失。"

            logger.info(f"🔨 执行泛型代码生成与沙箱验证 (第 {attempt + 1} 次)")

            code_task = Task(
                description=prompt,
                expected_output="仅输出可执行的Python代码，包含协议化JSON输出。",
                agent=coder,
                context=[plan_task]
            )

            crew = Crew(agents=[planner, coder], tasks=[plan_task, code_task], process=Process.sequential, verbose=True)
            crew_result = str(crew.kickoff())

            code_block = self.sandbox._extract_code_block(crew_result)
            if not code_block:
                return "⛔ 致命错误：未能从Coder输出中提取代码。"

            sandbox_out = self.sandbox._run(code_block)

            # 提取 JSON 数据并更新 real_data
            json_match = re.search(r"===RESULT_START===\s*(.*?)\s*===RESULT_END===", sandbox_out, re.DOTALL)
            real_data = None  # 重置
            if json_match:
                try:
                    real_data = json.loads(json_match.group(1).strip())
                except json.JSONDecodeError:
                    pass

            is_valid, validate_msg = self._validate_nested_structure(real_data or {})
            exec_success = sandbox_out.strip().startswith("[EXEC_SUCCESS]")

            if exec_success and is_valid:
                logger.info("✅ 深度结构校验通过，准备生成报告。")
                break
            else:
                if not exec_success:
                    error_msg = "沙箱执行失败（代码崩溃/超时/拦截）"
                elif not is_valid:
                    error_msg = validate_msg
                else:
                    error_msg = "未知综合校验失败"

                if attempt < self.max_retries - 1:
                    logger.warning(f"⚠️ 校验失败，触发重试 (第{attempt + 2}次): {error_msg}")
                else:
                    logger.error(
                        f"⛔ 审计拦截：连续 {self.max_retries} 次未通过校验。\n沙箱输出:\n{truncate_log(sandbox_out, self.log_max_len)}")
                    return f"⛔ 审计拦截：连续 {self.max_retries} 次未通过真实性校验。原因: {error_msg}\n沙箱原始输出:\n{truncate_log(sandbox_out, self.log_max_len)}"
                
        if real_data is None:
            return "⛔ 无法生成报告：未获取到有效结构化数据。"

        # ✅ 泛型报告生成器
        logger.info("📝 生成动态数据审计报告...")
        reporter = Agent(
            role="高级数据分析师",
            goal="基于自动发现的列结构与真实统计量，生成专业数据质量报告",
            backstory="你只信任沙箱返回的 JSON。严禁编造列名、数值或业务逻辑。",
            llm=self.llm,
            verbose=True
        )
        report_task = Task(
            description=f"""你收到以下**真实沙箱执行结果**（已自动识别列类型）：
{json.dumps(real_data, ensure_ascii=False, indent=2)}

【报告结构要求】

📊 数据概览：总行数、总列数、文件整体完整性。
🏗️ 字段解析：逐一列出所有列及其数据类型、非空比例。
📈 核心统计：
数值列：均值、极值、分布特征（缺失值单独说明）。
文本列：主流类别、集中度、多样性。
⚠️ 数据质量诊断：基于 data_quality_issues 指出缺失、异常值、格式问题。
💡 业务建议：针对发现的问题给出清洗或后续分析建议。 【铁律】严禁编造任何未在 JSON 中出现的数值或列名。若某列无有效数据，明确标注“数据不足/全缺失”。仅输出纯文本。""",
            expected_output="一份严格基于提供数据的数据分析报告。",
            agent=reporter
        )
        report_crew = Crew(agents=[reporter], tasks=[report_task], process=Process.sequential, verbose=True)
        return str(report_crew.kickoff())

# ============================================================
# 6. 辅助函数与主入口
# ============================================================

def ensure_csv_exists():
    if not os.path.exists("./sales_data.csv"):
        print("⚠️ 未找到 sales_data.csv，正在创建示例混合结构文件...")
        with open("./sales_data.csv", "w", encoding="utf-8") as f:
            f.write("Product,Sales,Profit,Region,Status\n")
            f.write("产品A,1000,200,华东,Active\n产品B,0,-50,华北,Inactive\n")
            f.write("产品C,abc,100,华南,Active\n产品D,500,,华东,\n")
            f.write("产品E,2000,300,华中,Active\n产品F,0,0,华北,Inactive\n")
        print("✅ 已创建 sales_data.csv")
        return True
    return False


if __name__ == "__main__":
    try:
        from crewai.process import Process
    except ImportError:
        from crewai import Process

    print("🛡️ 企业级自动化数据审计系统 (泛型列发现版) 启动...")
    ensure_csv_exists()

    business_task = "自动分析 sales_data.csv 的文件结构，探查各列数据质量与统计特征，并生成综合审计报告。"

    guard_system = DataGuardCrew(
        max_retries=2,
        sandbox_memory_mb=1024,
        log_max_len=500
    )
    try:
        final_report = guard_system.kickoff(business_task)
        print("\n" + "█" * 60)
        print("🏁 审计系统最终产出 (Final Report):")
        print("█" * 60)
        print(final_report)
        print("█" * 60)
    except Exception as e:
        logger.error(f"💥 系统运行异常: {e}")
        print(f"⛔ 运行失败: {e}")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import subprocess
import json
import logging
import sys
import hashlib
import time
from datetime import datetime
from typing import Type, List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    from langchain_openai import ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import CharacterTextSplitter
except ImportError:
    print("❌ 错误：请确保安装了所有依赖: pip install crewai langchain-openai langchain-community faiss-cpu sentence-transformers")
    sys.exit(1)

load_dotenv()

class ReviewResult(BaseModel):
    """审计结果结构化模型：这是 Manager 决定任务是否通过的硬性指标"""
    is_safe: bool = Field(description="代码是否真正通过合规审计（必须有 if 保护且物理执行成功）")
    score: int = Field(description="合规性评分 (0-10)。低于 8 分会被强制打回。")
    compliance_check: Dict[str, bool] = Field(description="合规清单：含 'has_if_check', 'no_try_except_cheat', 'data_validation_passed'")
    defects: List[str] = Field(description="发现的缺陷或业务逻辑违规清单")
    feedback_to_coder: str = Field(description="给 Coder 的具体、可操作的修改指令")

class PythonRunnerInput(BaseModel):
    code: str = Field(description="待执行的 Python 源代码")

class DataGuardSandbox(BaseTool):
    name: str = "data_guard_sandbox"
    description: str = "在严格隔离的 Linux 沙箱中执行数据处理代码。提供资源限制与物理指纹。"
    args_schema: Type[BaseModel] = PythonRunnerInput
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, code: str) -> str:
        match = re.search(r"```[a-zA-Z]*\n(.*?)```", code, re.DOTALL)
        clean_code = match.group(1) if match else code
        clean_code = re.sub(r"```[a-zA-Z]*", "", clean_code).replace("```", "").strip()

        def limit_resources():
            if sys.platform.startswith('linux'):
                import resource
                resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
                resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, -1))

        ts = datetime.now().strftime("%H:%M:%S")
        fingerprint = hash(clean_code) % 10000

        try:
            result = subprocess.run(
                ["python3", "-c", clean_code],
                capture_output=True, text=True, timeout=10,
                preexec_fn=limit_resources if sys.platform.startswith('linux') else None
            )
            output = (result.stdout + "\n" + result.stderr).strip()
            status = "[EXEC_SUCCESS]" if result.returncode == 0 else "[EXEC_ERROR]"
            return f"[{ts}] 执行指纹:{fingerprint}\n--- 物理输出 ---\n{output}\n最终标记: {status}"
        except subprocess.TimeoutExpired:
            return f"[{ts}] 严重错误：代码执行超时（10s限制）\n最终标记: [EXEC_ERROR]"
        except Exception as e:
            return f"[{ts}] 系统级崩溃: {str(e)}\n最终标记: [EXEC_ERROR]"


class AuditExpertRAG(BaseTool):
    name: str = "audit_expert_rag"
    description: str = "检索企业级数据审计标准、Pandas 防御性编程规范及合规性检查准则。"
    
    vector_store: Any = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        documents = [
            "【规则】除法防御：任何计算率、占比的操作，分母前必须加 if denominator != 0，禁止仅用 try-except。",
            "【Pandas】读取 CSV 后，必须使用 pd.to_numeric(..., errors='coerce') 转换数值列，并用 fillna(0) 或 dropna() 处理缺失。",
            "【空值处理】核心维度列（如 ID、日期）缺失必须 dropna()，财务数值列缺失必须 fillna(0) 并记录日志。",
            "【异常值检测】销售额或利润列出现负值，需要标记为警告并在输出中说明，不能直接丢弃。",
            "【代码结构】必须遵循 1.数据加载与清洗 -> 2.异常探测 -> 3.防御性计算 -> 4.结果输出 -> 5.STATUS: DONE。",
            "【利润率计算】利润率 = 利润 / 销售额，必须处理销售额为 0 或空值的情况，返回 'N/A' 或 0。",
            "【分组聚合】groupby 后必须检查分组是否为空，避免空分组导致后续计算错误。",
            "【输出格式】最终结果必须包含：统计摘要、异常记录明细、整体合规性结论。",
            "【硬编码检测】严禁使用 denominator = 1 或类似硬编码绕过除零检查，必须保留原始数据值。",
            "【伪修复检测】try-except 只能用于记录错误，不能替代 if 逻辑判断。",
        ]

        text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        texts = text_splitter.create_documents(documents)
        self.vector_store = FAISS.from_documents(texts, embeddings)

    def _run(self, query: str) -> str:
        docs = self.vector_store.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in docs])


class AgentPrompts:
    PLANNER_ROLE = "首席合规架构师"
    PLANNER_GOAL = "为复杂的业务数据分析任务制定'零崩溃'审计方案。"
    PLANNER_BACKSTORY = """你拥有 20 年财务审计经验。你不仅规划代码逻辑，还要规划'数据脏点'的防御策略。
    你强制要求 Coder 在修复前必须先写出探测崩溃的代码。"""

    CODER_ROLE = "自动化审计开发专家"
    CODER_GOAL = "编写健壮的审计代码，确保能处理任何极端脏数据，并标记 STATUS: DONE。"
    CODER_BACKSTORY = """你精通防御性编程。你会查阅 audit_expert_rag 确保符合财务规范。
    你绝不使用 try-except 来敷衍崩溃，你只相信显式的 if 逻辑判断。"""

    REVIEWER_ROLE = "数据合规主管"
    REVIEWER_GOAL = "通过物理执行验证结果，并对代码进行合规性'一票否决'。"
    REVIEWER_BACKSTORY = """你是最后一道防线。你会对比 RAG 规范，如果代码里没有 if 判断或者存在硬编码作弊，
    你会给出 ReviewResult 中 is_safe=False 的判定，强制要求重做。"""


class DataGuardCrew:
    def __init__(self):
        model_name = os.getenv("MODEL_NAME", "你的模型（如ollama/qwen2.5:7b）")
        base_url = os.getenv("BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        
        self.llm = model_name
        
        self._llm_client = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=0.1
        )
        
        self.sandbox = DataGuardSandbox()
        self.rag = AuditExpertRAG()

    def log_step(self, step_data):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "agent_action",
            "data": str(step_data)
        }
        with open("audit_trace.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def kickoff(self, user_task: str):
        llm_config = os.getenv("MODEL_NAME", "ollama/qwen2.5:7b")

        planner = Agent(
            role=AgentPrompts.PLANNER_ROLE,
            goal=AgentPrompts.PLANNER_GOAL,
            backstory=AgentPrompts.PLANNER_BACKSTORY,
            llm=llm_config,
            verbose=True
        )

        coder = Agent(
            role=AgentPrompts.CODER_ROLE,
            goal=AgentPrompts.CODER_GOAL,
            backstory=AgentPrompts.CODER_BACKSTORY,
            tools=[self.sandbox, self.rag],
            llm=llm_config,
            allow_delegation=True,
            verbose=True
        )

        reviewer = Agent(
            role=AgentPrompts.REVIEWER_ROLE,
            goal=AgentPrompts.REVIEWER_GOAL,
            backstory=AgentPrompts.REVIEWER_BACKSTORY,
            tools=[self.sandbox, self.rag],
            llm=llm_config,
            allow_delegation=True,
            verbose=True
        )

        reporter = Agent(
            role="Report Generator",
            goal="根据分析结果生成数据分析报告",
            backstory="你是一个数据分析师，擅长把数据结果转成清晰报告",
            llm=llm_config,
            verbose=True
        )

        plan_task = Task(
            description=f"针对任务 '{user_task}'，分析潜在的 Pandas/除零崩溃风险，制定合规审计蓝图。",
            expected_output="详细的审计路线图，包含脏数据处理策略。",
            agent=planner
        )

        code_task = Task(
            description="根据蓝图编写代码。要求：1.先探测异常 2.使用 if 显式修复 3.输出 STATUS: DONE。",
            expected_output="物理执行通过且逻辑严密的 Python 源代码。",
            agent=coder,
            context=[plan_task]
        )

        review_task = Task(
            description="执行并审计。判定 is_safe 时，必须查阅 RAG 确认是否符合'禁止 try-except 伪修复'原则。",
            expected_output="结构化的合规报告。",
            output_pydantic=ReviewResult,
            agent=reviewer,
            context=[code_task, plan_task]
        )

        report_task = Task(
            description="""
根据前面的分析结果，生成一份数据分析报告：

要求：
1. 总体数据情况
2. 关键指标（平均值/总和等）
3. 发现的问题
4. 结论

输出为清晰文本报告（不要代码）
""",
            expected_output="一份结构清晰的数据分析报告，包含总体数据情况、关键指标、发现问题和结论。", 
            agent=reporter,
            context=[review_task]
        )

        crew = Crew(
            agents=[planner, coder, reviewer, reporter],
            tasks=[plan_task, code_task, review_task, report_task],
            process=Process.hierarchical,
            manager_llm=self.llm,
            step_callback=self.log_step,
            memory=True,
            verbose=True
        )

        return crew.kickoff()


class ExecutionTracer:
    """
    真实执行追踪器（外挂）
    独立运行，不侵入 CrewAI 系统
    """
    def __init__(self):
        self.last_result = None
        self.execution_history = []

    def extract_code_from_output(self, text: str) -> str:
        """从输出中提取 Python 代码"""
        if not text:
            return ""
        
        match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        lines = text.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if re.match(r'^(import |from |def |class |if __name__|#|print\()', line.strip()):
                in_code = True
            if in_code:
                code_lines.append(line)
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        return text.strip()

    def run_and_capture(self, code: str) -> dict:
        """执行代码并返回带防伪标记的结果"""
        clean_code = self.extract_code_from_output(code)
        
        if not clean_code or len(clean_code) < 10:
            return {
                "status": "error",
                "valid_json": False,
                "data": "代码太短或为空",
                "from": "REAL_EXECUTION",
                "proof": None,
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            result = subprocess.run(
                ["python3", "-c", clean_code],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            output = result.stdout.strip()
            
            try:
                parsed = json.loads(output)
                valid_json = True
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', output, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        valid_json = True
                    except:
                        parsed = {"raw": output}
                        valid_json = False
                else:
                    parsed = {"raw": output}
                    valid_json = False
            
            proof_raw = f"{time.time()}_{os.getpid()}_{hash(output)}_{clean_code[:50]}"
            proof = hashlib.md5(proof_raw.encode()).hexdigest()[:12]
            
            self.last_result = {
                "status": "success" if result.returncode == 0 else "error",
                "valid_json": valid_json,
                "data": parsed,
                "from": "REAL_EXECUTION",
                "proof": proof,
                "timestamp": datetime.now().isoformat(),
                "return_code": result.returncode
            }
            
            self.execution_history.append(self.last_result)
            return self.last_result
            
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "valid_json": False,
                "data": "执行超时（10秒）",
                "from": "REAL_EXECUTION",
                "proof": None,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "valid_json": False,
                "data": str(e),
                "from": "REAL_EXECUTION",
                "proof": None,
                "timestamp": datetime.now().isoformat()
            }
    
    def verify_report_trustworthiness(self, report_text: str) -> dict:
        """验证报告的可信度（基于真实执行）"""
        verification = {
            "has_real_execution_marker": False,
            "has_proof": False,
            "execution_success": False,
            "trust_score": 0,
            "is_trustworthy": False
        }
        
        if self.last_result:
            verification["has_real_execution_marker"] = self.last_result.get("from") == "REAL_EXECUTION"
            verification["has_proof"] = self.last_result.get("proof") is not None
            verification["execution_success"] = self.last_result.get("status") == "success"
        
        score = 0
        if verification["has_real_execution_marker"]:
            score += 30
        if verification["has_proof"]:
            score += 30
        if verification["execution_success"]:
            score += 40
        
        verification["trust_score"] = score
        verification["is_trustworthy"] = score >= 70
        
        return verification


def print_guard_summary(result: dict, verification: dict = None):
    """输出外挂校验结果（不修改原系统输出）"""
    print("\n" + "="*50)
    print("🔍 外挂增强模块 - 真实执行校验结果")
    print("="*50)
    
    if result:
        print(f"   执行状态: {result.get('status', 'unknown')}")
        print(f"   有效 JSON: {result.get('valid_json', False)}")
        print(f"   数据来源: {result.get('from', 'unknown')}")
        print(f"   防伪证明: {result.get('proof', '无')}")
        if result.get('status') == 'success' and result.get('valid_json'):
            data_preview = str(result.get('data', {}))[:200]
            print(f"   数据预览: {data_preview}...")
    
    if verification:
        print(f"\n   信任分数: {verification.get('trust_score', 0)}/100")
        print(f"   可信判定: {'✅ 可信 (基于真实执行)' if verification.get('is_trustworthy') else '⚠️ 不可信 (可能为幻觉)'}")
    
    print("="*50)


def ensure_csv_exists():
    """确保测试 CSV 文件存在（辅助函数）"""
    if not os.path.exists("./sales_data.csv"):
        print("⚠️ 未找到 sales_data.csv，正在创建示例文件...")
        with open("./sales_data.csv", "w") as f:
            f.write("产品,销售额,利润\n")
            f.write("产品A,1000,200\n")
            f.write("产品B,0,-50\n")
            f.write("产品C,abc,100\n")
            f.write("产品D,500,\n")
            f.write("产品E,2000,300\n")
            f.write("产品F,0,0\n")
            f.write("产品G,1500,450\n")
        print("✅ 已创建 sales_data.csv")
        return True
    return False


if __name__ == "__main__":
    print("🛡️ 企业级自动化数据审计系统 (Data Guard Agent) 启动...")
    
    ensure_csv_exists()
    
    business_task = (
        "分析一份名为 'sales_data.csv' 的报表。要求：计算各产品的利润率（利润/销售额）。"
        "注意：原始数据极其混乱，销售额列存在 0、空值和非数字字符，且利润可能为负。 "
        "必须确保脚本在任何情况下都不会崩溃，并给出异常处理日志。"
    )
    
    guard_system = DataGuardCrew()
    final_report = guard_system.kickoff(business_task)
    
    tracer = ExecutionTracer()
    real_result = tracer.run_and_capture(str(final_report))
    verification = tracer.verify_report_trustworthiness(str(final_report))
    
    print("\n" + "█"*60)
    print("🏁 审计系统最终产出 (Final Report):")
    print("█"*60)
    print(final_report)
    print("█"*60)
    
    print_guard_summary(real_result, verification)
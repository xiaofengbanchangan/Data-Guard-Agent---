# Data Guard Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CrewAI](https://img.shields.io/badge/CrewAI-0.203.2-green.svg)](https://www.crewai.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> 🤖 一个基于 **CrewAI** 的多智能体协作系统，用于自动化数据分析、代码生成、安全审计和报告生成。内置 **RAG 知识库**、**物理沙箱**、**防作弊检测** 和 **结构化审计协议**。

## 📌 项目简介

本系统是一个基于 CrewAI 多智能体框架的自动化数据审计平台。它能够接收一个数据审计任务（如“分析 CSV 文件，计算业务指标，生成合规报告”），通过三个专业 Agent 的协作，自动完成以下工作：

1、制定防御性审计计划

2、生成并执行安全的 Python 分析代码

3、根据真实执行结果生成业务报告

所有代码均在 物理沙箱（进程级资源限制 + AST 静态检查） 中运行，确保安全。系统内置 RAG 知识库（FAISS + 向量模型），指导 Agent 遵循企业数据审计规范。

核心理念：零信任执行、完全自动化、泛型列发现、结构化输出校验。

## 🏗️ 架构图
graph TD
    User[用户输入任务] --> Manager[CrewAI 顺序编排器]

    subgraph Crew [多智能体协作]
        Planner[Planner Agent<br/>首席合规架构师]
        Coder[Coder Agent<br/>自动化审计开发专家]
        Reporter[Reporter Agent<br/>报告生成专家]
    end

    subgraph Tools [工具集]
        Sandbox[DataGuardSandbox<br/>沙箱执行器]
        Validator[CodeValidator<br/>AST 静态检查]
        RAG[AuditExpertRAG<br/>知识检索]
    end

    subgraph Knowledge [RAG 知识库]
        Docs[内部审计规范文档]
        Vector[FAISS 向量库]
    end

    subgraph Execution [沙箱执行环境]
        CSV[CSV 数据文件]
        Python[Python 解释器 + 资源限制]
    end

    Manager --> Planner
    Manager --> Coder
    Manager --> Reporter

    Coder --> Sandbox
    Coder --> RAG
    Coder --> Validator
    Sandbox --> Python --> CSV
    RAG --> Vector --> Docs

    Sandbox -->|执行输出| Coder
    Coder -->|输出统计 JSON| Reporter
    Reporter -->|生成报告| Output[Markdown 合规报告]

## 🚀 快速开始

### 环境要求
- Python 3.10+
- [Ollama](https://ollama.com/)（可选，本地免费）或 OpenAI API 密钥

### 安装
git clone https://github.com/xiaofengbanchangan/Data-Guard-Agent---.git
cd Data-Guard-Agent---
python -m venv venv
source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
cp .env.example .env

# 编辑 .env 选择模型

### 运行示例
🛡️ 企业级自动化数据审计系统 (Data Guard Agent) 启动...
✅ 检测到已导入的模型: qwen-gguf
🚀 使用模型: ollama/qwen-gguf

╭──────────────────────────────── Crew Execution Started ─────────────────────────────────╮
│  Crew Execution Started                                                                 │
│  Name: crew                                                                             │
╰─────────────────────────────────────────────────────────────────────────────────────────╯

[Agent: 首席合规架构师] 开始制定审计蓝图...
[Agent: 自动化审计开发专家] 生成代码...
[Agent: 数据合规主管] 执行审计...
[Agent: 报告生成专家] 生成报告...

📊 审计结果:
- is_safe: True
- score: 9
- defects: []
- feedback: "代码使用 if denominator != 0 保护，正确处理空值"

最终报告已生成: report.md
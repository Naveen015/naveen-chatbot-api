# Experience: Quantitative Researcher at Pharvision Advisers

**Role**: Quantitative Researcher  
**Company**: Pharvision Advisers | USA  
**Duration**: Jan 2026 – Present  

---

## Executive Summary
As a Quantitative Researcher at Pharvision Advisers, Naveen Prashanna Gurumurthy develops production-grade machine learning models, multi-factor alpha research pipelines, LLM-based multi-agent systems, and production RAG architectures for quantitative finance and live investment research.

---

## Key Achievements & Projects (STAR Format)

### 1. Multi-Factor ML Alpha Signal Engineering
* **Situation**: Extracting robust predictive signals from high-frequency market data requires processing massive tick-level data feeds while strictly preventing lookahead bias and overfitting.
* **Task**: Engineer multi-factor machine learning models processing over 15 million daily tick-level data points across global equities and macro datasets to generate alpha.
* **Action**: Designed point-in-time data alignment pipelines in Python and C++, developed cross-sectional feature normalization algorithms, and trained non-linear ML models to select optimal signal weights.
* **Result**: Achieved an annualized Sharpe ratio > 1.2 with low portfolio turnover and minimal transaction cost drag.

---

### 2. LLM-Based Multi-Agent Backtesting Automation
* **Situation**: Quantitative researchers previously spent days manually executing backtest runs, configuring parameters, and auditing model validation metrics across dozens of candidate hypotheses.
* **Task**: Orchestrate an automated multi-agent framework capable of managing 10,000+ backtesting runs weekly across 50+ concurrent experiments.
* **Action**: Built an agentic workflow using LangChain, LangGraph, CrewAI, MLflow, and Weights & Biases (W&B). Configured specialized agent roles for hypothesis generation, backtest execution, statistical anomaly auditing, and automated experiment reporting.
* **Result**: Reduced quantitative model validation cycles from days to hours, accelerating the research-to-production deployment pipeline.

---

### 3. Production GraphRAG & Vector Search System for Financial Research
* **Situation**: Financial analysts and researchers required rapid multi-hop research capabilities across 500GB+ of unstructured financial filings, transcripts, and market research documents.
* **Task**: Develop an enterprise production RAG system combining graph traversal and vector search for high precision and zero hallucinations.
* **Action**: Integrated Neo4j GraphRAG for multi-hop entity traversal with FAISS dense vector search and cross-encoder reranking. Built a custom LLM-as-a-judge evaluation harness to measure response faithfulness, context relevance, and retrieval precision continuously.
* **Result**: Enabled instant, verifiable multi-hop financial research over 500GB+ of documentation with high precision.

---

### 4. High-Throughput Quantitative Data Pipelines
* **Situation**: Processing 10+ concurrent vendor market data feeds (handling 50M+ daily rows) introduced latency bottlenecks in alpha generation pipelines.
* **Task**: Build scalable, resilient data ingestion pipelines to expand research coverage without latency degradation.
* **Action**: Engineered parallel data ingestion pipelines using Polars, Apache Airflow, and Delta Lake on Azure cloud storage.
* **Result**: Scaled feed ingestion to 50M+ daily rows seamlessly with zero latency degradation.

# Experience: AI Engineer at Kahana Group Inc

**Role**: AI Engineer  
**Company**: Kahana Group Inc | USA  
**Duration**: Jul 2025 – Jan 2026  

---

## Executive Summary
At Kahana Group Inc, Naveen Prashanna Gurumurthy architected production LLM execution flows, cloud state persistence layers, real-time AI telemetry, and automated evaluation harnesses for an AI-native web browser on Azure cloud infrastructure.

---

## Key Achievements & Projects (STAR Format)

### 1. Reliable AI-Native Browser Action Orchestration
* **Situation**: Complex, multi-step browser automation tasks (e.g. web navigation, data extraction, tab state management) suffered from state drift and non-deterministic LLM behavior.
* **Task**: Program production-ready LLM execution flows to ensure deterministic state transitions and high task completion rates.
* **Action**: Leveraged Azure AI Foundry and Azure AI Services to design state machine workflows, prompt validation guardrails, and automated recovery paths for browser automation routines.
* **Result**: Elevated execution reliability and completion metrics for complex multi-step browser workflows.

---

### 2. High-Speed Azure Cloud Persistence Layer
* **Situation**: Synchronizing browser state, active sessions, bookmarks, and open tabs across user devices introduced latency during rapid state transitions.
* **Task**: Architect an Azure-backed persistence layer with sub-50ms sync overhead under heavy state mutations.
* **Action**: Designed an Azure Blob Storage persistence layer featuring differential delta encoding, asynchronous state flushing, and optimized memory caching.
* **Result**: Achieved sub-50ms state synchronization overhead under intense user activity.

---

### 3. Real-Time AI Telemetry & Cost Optimization
* **Situation**: High LLM throughput (5M+ monthly tokens) during stress testing required fine-grained observability to control cloud infrastructure costs.
* **Task**: Instrument real-time AI observability telemetry to monitor token metering, latency, and cost overhead.
* **Action**: Implemented custom instrumentation using Azure Monitor Application Insights to track token consumption per prompt, latency distributions, and cost metrics across simulated 5M+ throughput stress tests.
* **Result**: Identified token inefficiencies, optimized prompt payload sizes, and significantly reduced cloud compute costs.

---

### 4. Automated Evaluation SDK Integration
* **Situation**: Manual evaluation of LLM prompts and browser action accuracy was slow and unscalable across rapid product iterations.
* **Task**: Establish an automated evaluation harness to measure completion metrics and failure patterns at scale.
* **Action**: Integrated the Azure AI Evaluation SDK into CI/CD pipelines to evaluate user interaction logs, model answers, and action failure modes automatically.
* **Result**: Accelerated prompt optimization cycles and directly increased task completion metrics.

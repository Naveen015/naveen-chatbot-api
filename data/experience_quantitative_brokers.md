# Experience: Software Engineer (ML Systems) at Quantitative Brokers

**Role**: Software Engineer (ML Systems)  
**Company**: Quantitative Brokers | India  
**Duration**: Jul 2022 – Feb 2023  

---

## Executive Summary
At Quantitative Brokers, Naveen Prashanna Gurumurthy developed sub-millisecond C++/Python predictive model execution runtimes, real-time Level-2 order book feature streaming pipelines, and high-throughput Kafka/PostgreSQL telemetry systems for algorithmic trading.

---

## Key Achievements & Projects (STAR Format)

### 1. Sub-800µs Model Dispatch Inference Runtime
* **Situation**: High-frequency algorithmic execution algorithms required model dispatch latencies under 1ms over FIX protocol sessions processing millions of daily messages.
* **Task**: Build low-latency C++/Python inference runtimes executing predictive ML models with sub-millisecond dispatch times.
* **Action**: Implemented zero-copy shared memory queues, C++ bindings for ML models, and event-driven FIX protocol parsers operating over high-throughput network interfaces.
* **Result**: Achieved sub-800µs model dispatch latency across millions of daily messages.

---

### 2. L2 Order Book Feature Streaming for Multi-Leg Execution
* **Situation**: Multi-leg execution strategies experienced market slippage due to order book imbalance tracking latency across 40+ liquid asset pairs.
* **Task**: Dynamically process live Level-2 order book feeds to calculate execution indicators in real time.
* **Action**: Built a C++ feature streaming engine calculating micro-price, order book imbalance, and volume profile features across 40+ asset pairs simultaneously.
* **Result**: Minimized execution slippage under peak institutional trading volume spikes.

---

### 3. High-Throughput Telemetry & Feature Engineering Pipelines
* **Situation**: Offline model development suffered from data leakage and slow feature calculation on massive execution telemetry logs.
* **Task**: Capture 10GB+ of daily streaming fills and model predictions without data loss.
* **Action**: Engineered PostgreSQL and Apache Kafka data pipelines with strict event timestamp alignment to record 10GB+ of daily telemetry.
* **Result**: Completely eliminated data leakage in offline training datasets and accelerated feature engineering iteration by 3.5x.

---

### 4. Core FIX Platform Multi-Leg Trade Extension
* **Situation**: Institutional client demand required extending the core FIX messaging platform to support multi-leg derivative strategies under heavy traffic.
* **Task**: Refactor messaging platform services and data access layers for multi-leg trade handling.
* **Action**: Extended backend services and optimized data-access layers to process multi-leg order messages without increasing latency.
* **Result**: Enabled seamless multi-leg trade execution during peak traffic spikes.

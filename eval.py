#!/usr/bin/env python3
"""
CLI Evaluation Script for Naveen Chatbot Agentic RAG System.
Supports switchable benchmarking between OpenAI API and local 7B vLLM inferencing.
"""
import os
import json
import sys
import argparse
from rag.llm_provider import LLMProviderManager
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import LLMReranker
from rag.agentic_engine import AgenticRAGEngine
from rag.evaluator import RAGEvaluator

TEST_DATASET = [
    {
        "query": "What are Naveen Prashanna's primary technical skills and research interests?",
        "ground_truth": "Naveen Prashanna specializes in Software Engineering, Machine Learning, Data Science, and Computer Science research."
    },
    {
        "query": "What is the topic of Naveen's CS6348 final research paper?",
        "ground_truth": "CS6348 paper focuses on data security, computer security, machine learning, and privacy analysis."
    },
    {
        "query": "Which academic degree or university background does Naveen have?",
        "ground_truth": "Naveen attended UT Dallas (University of Texas at Dallas) pursuing Computer Science."
    }
]

def main():
    parser = argparse.ArgumentParser(description="Agentic RAG Benchmark Evaluator")
    parser.add_argument("--provider", choices=["openai", "vllm", "local"], default="openai", help="LLM Provider: openai or vllm / local")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1", help="Base URL for vLLM local server")
    args = parser.parse_args()

    print(f"🚀 Initializing Agentic RAG Evaluation Suite (Provider: {args.provider.upper()})...")

    docs_dir = "data"
    if not os.path.exists(docs_dir):
        print(f"❌ Error: {docs_dir} directory not found.")
        sys.exit(1)

    provider_manager = LLMProviderManager(provider=args.provider, vllm_base_url=args.vllm_url)

    print("🔍 Building/Loading Hybrid Retriever (Dense + BM25)...")
    retriever = HybridRetriever(docs_dir=docs_dir, provider_manager=provider_manager)

    print(f"🎯 Initializing LLM Reranker & Agentic Engine using {args.provider.upper()}...")
    reranker = LLMReranker(provider_manager=provider_manager)
    engine = AgenticRAGEngine(retriever=retriever, reranker=reranker, provider_manager=provider_manager)
    evaluator = RAGEvaluator(provider_manager=provider_manager)

    print(f"\n📊 Running Benchmark on {len(TEST_DATASET)} Test Queries...\n")
    summary = evaluator.run_benchmark(TEST_DATASET, engine, provider=args.provider)

    print("=" * 60)
    print(f"       AGENTIC RAG EVALUATION REPORT [{args.provider.upper()}]       ")
    print("=" * 60)
    print(f"Provider:                      {summary['provider'].upper()}")
    print(f"Total Samples Evaluated:        {summary['total_samples']}")
    print(f"Avg Faithfulness (Groundedness): {summary['avg_faithfulness'] * 100:.1f}%")
    print(f"Avg Answer Relevance:          {summary['avg_answer_relevance'] * 100:.1f}%")
    print(f"Avg Context Precision:         {summary['avg_context_precision'] * 100:.1f}%")
    print(f"Avg Context Recall:            {summary['avg_context_recall'] * 100:.1f}%")
    print(f"🔥 OVERALL AGENTIC RAG SCORE:   {summary['overall_rag_score'] * 100:.1f}%")
    print("=" * 60)

    report_file = f"evaluation_report_{args.provider}.json"
    with open(report_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Detailed evaluation report saved to {report_file}")

if __name__ == "__main__":
    main()

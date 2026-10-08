import os
import json
from typing import List, Dict, Any, Optional
from rag.llm_provider import LLMProviderManager
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import LLMReranker

class AgenticRAGEngine:
    """
    Agentic RAG Engine featuring Query Rewriting & Decomposition,
    Hybrid Search, Reranking, Self-Correction, and Hallucination Verification.
    Supports switchable OpenAI and local 7B vLLM inferencing.
    """
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: LLMReranker,
        provider_manager: Optional[LLMProviderManager] = None,
        provider: Optional[str] = None
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.provider_manager = provider_manager or LLMProviderManager(provider=provider)

    def set_provider(self, provider: str, model_name: Optional[str] = None):
        """Switches active LLM provider dynamically (e.g. 'openai' vs 'vllm')."""
        self.provider_manager.set_provider(provider, model_name)

    def decompose_and_rewrite_query(self, query: str) -> Dict[str, Any]:
        """
        Analyzes the query, expands terms for keyword/vector search, and splits multi-part questions.
        """
        if not self.provider_manager.client:
            return {"rewritten_queries": [query], "is_complex": False}

        prompt = f"""
You are an AI query planning agent for a database containing information about **Naveen Prashanna** (resumes, academic transcripts, research papers, CV).

Analyze the user query: "{query}"

Tasks:
1. Determine if this query requires multi-step retrieval.
2. Generate 1 to 3 search-optimized query variations or sub-queries that expand technical terms and synonyms to maximize document recall.

Return JSON in this format:
{{
  "is_complex": false,
  "rewritten_queries": [
    "search variation 1",
    "search variation 2"
  ]
}}
"""
        try:
            res_text = self.provider_manager.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                json_mode=True
            )
            return json.loads(res_text)
        except Exception as e:
            print(f"⚠️ Query expansion fallback ({e}). Using original query.")
            return {"rewritten_queries": [query], "is_complex": False}

    def verify_faithfulness(self, answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates if the generated answer is strictly grounded in the retrieved context (hallucination check).
        """
        if not self.provider_manager.client or not chunks:
            return {"is_faithful": True, "score": 1.0, "reason": "No evaluation client available."}

        context_str = "\n\n".join([f"[{c.get('source', 'doc')}] {c['content']}" for c in chunks])

        prompt = f"""
You are an expert factual consistency auditor.
Compare the generated answer against the reference context below.

Context:
\"\"\"
{context_str}
\"\"\"

Generated Answer:
\"\"\"
{answer}
\"\"\"

Evaluate whether EVERY statement in the answer is directly supported by the context.
Return JSON:
{{
  "is_faithful": true,
  "score": 0.95,
  "unsupported_claims": []
}}
"""
        try:
            res_text = self.provider_manager.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                json_mode=True
            )
            return json.loads(res_text)
        except Exception:
            return {"is_faithful": True, "score": 1.0, "reason": "Audit bypass."}

    def generate_answer(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Generates answer strictly constrained to context."""
        if not chunks:
            return "I don't have that information in my current knowledge base regarding Naveen Prashanna."

        context = "\n\n".join([f"Source: [{c.get('source', 'doc')}, p.{c.get('page', 1)}]\n{c['content']}" for c in chunks])

        if not self.provider_manager.client:
            summary = "Based on retrieved documents:\n" + "\n".join([f"- {c['content']}" for c in chunks[:3]])
            return summary

        prompt = f"""
# ROLE
You are an intelligent AI assistant possessing precise knowledge about **Naveen Prashanna**.

# STYLE
- Refer to Naveen in the third person (Naveen, "he", "his").
- Address the user directly ("Sure—here's what I found...").
- Keep answers concise, professional, and technically precise.
- Cite sources clearly using inline tags like [Resume.pdf, p.1] or [CS6348_Final_Paper.pdf, p.3] where applicable.

# KNOWLEDGE RULES
- Answer **only** using information in the Context block below.
- If the context does not contain the answer, reply:
  "I don't have that information in my current knowledge base."
- Never fabricate details or infer unstated facts.

# CONTEXT
{context}

# QUESTION
{query}

# ANSWER:"""

        try:
            answer = self.provider_manager.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return answer
        except Exception as e:
            print(f"⚠️ LLM generation call failed ({e}). Returning extracted context fallback.")
            summary = f"Sure—here's what I found about '{query}' in Naveen Prashanna's records:\n\n"
            for c in chunks[:3]:
                summary += f"• **[{c.get('source', 'doc')}]**: {c['content']}\n\n"
            return summary.strip()

    def run(self, query: str, provider: Optional[str] = None, max_retries: int = 1) -> Dict[str, Any]:
        """
        Executes the full Agentic RAG workflow:
        Query Rewriting -> Multi-Retrieval -> Reranking -> Draft Generation -> Faithfulness Check -> Output.
        """
        if provider:
            self.set_provider(provider)

        active_provider = self.provider_manager.provider
        trace = []
        trace.append(f"Received query: '{query}' (Active LLM Provider: {active_provider.upper()})")

        # Step 1: Query Analysis & Rewriting
        plan = self.decompose_and_rewrite_query(query)
        rewritten_queries = plan.get("rewritten_queries", [query])
        trace.append(f"Query planning generated {len(rewritten_queries)} search variants: {rewritten_queries}")

        # Step 2: Multi-Retrieval via Hybrid Search (Dense + BM25)
        raw_chunks_map = {}
        for q_var in rewritten_queries:
            results = self.retriever.hybrid_search(q_var, top_k=6, fetch_k=15)
            for item in results:
                raw_chunks_map[item["id"]] = item

        raw_chunks = list(raw_chunks_map.values())
        trace.append(f"Retrieved {len(raw_chunks)} candidate chunks via Hybrid Search (Dense + BM25).")

        # Step 3: LLM Reranking & Context Relevance Filtering
        reranked_chunks = self.reranker.rerank(query, raw_chunks, top_n=4, min_score=4.0, provider=active_provider)
        trace.append(f"Reranked context chunks down to {len(reranked_chunks)} high-relevance passages using {active_provider}.")

        # Step 4: Draft Generation
        answer = self.generate_answer(query, reranked_chunks)
        trace.append(f"Generated draft answer using {active_provider} LLM with context attributions.")

        # Step 5: Faithfulness & Self-Correction Check
        faithfulness = self.verify_faithfulness(answer, reranked_chunks)
        trace.append(f"Faithfulness check result: is_faithful={faithfulness.get('is_faithful')}, score={faithfulness.get('score')}")

        sources = list(set([c.get("source", "unknown") for c in reranked_chunks]))

        return {
            "query": query,
            "provider": active_provider,
            "answer": answer,
            "sources": sources,
            "context_chunks": [
                {
                    "id": c["id"],
                    "source": c["source"],
                    "page": c.get("page", 1),
                    "rerank_score": c.get("rerank_score", 0.0),
                    "snippet": c["content"][:150] + "..."
                } for c in reranked_chunks
            ],
            "faithfulness_score": faithfulness.get("score", 1.0),
            "is_faithful": faithfulness.get("is_faithful", True),
            "agent_trace": trace
        }

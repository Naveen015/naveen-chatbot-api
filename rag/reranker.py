import os
import json
import re
from typing import List, Dict, Any, Optional
from rag.llm_provider import LLMProviderManager

def tokenize_text(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())

STOP_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'that', 'this', 'these', 'those', 'so', 'why', 'what', 'how', 'who',
    'where', 'when', 'it', 'its', 'in', 'on', 'at', 'to', 'for', 'of',
    'and', 'or', 'if', 'because', 'as', 'until', 'while', 'by', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'from', 'up', 'down', 'out', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'did', 'does'
}

class LLMReranker:
    """
    Reranks candidate document chunks using LLM relevance scoring to maximize context precision,
    supporting OpenAI and local 7B vLLM inferencing with local keyword density fallback.
    """
    def __init__(self, provider_manager: Optional[LLMProviderManager] = None, provider: Optional[str] = None):
        self.provider_manager = provider_manager or LLMProviderManager(provider=provider)

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: int = 4,
        min_score: float = 2.0,
        provider: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Scores each candidate chunk on a 0-10 relevance scale relative to the query.
        Returns top_n highest scoring chunks.
        """
        if not chunks:
            return []

        if provider:
            self.provider_manager.set_provider(provider)

        if self.provider_manager.client:
            prompt = f"""
You are an expert RAG search reranker and relevance evaluator.
Evaluate how relevant each candidate context chunk is to answering the user query.

User Query: "{query}"

Candidate Chunks:
"""
            for idx, chunk in enumerate(chunks):
                prompt += f"\n--- Chunk ID: {chunk['id']} (Index {idx}) ---\nContent: {chunk['content']}\n"

            prompt += """
For each candidate chunk, assign a relevance score from 0.0 to 10.0:
- 0.0: Completely irrelevant or off-topic.
- 5.0: Partially relevant, touches on query concepts.
- 10.0: Directly answers or contains crucial evidence for the query.

Respond ONLY with a JSON object in this exact format:
{
  "scores": [
    {"index": 0, "score": 8.5, "reasoning": "Directly mentions Naveen's degree and GPA."}
  ]
}
"""
            try:
                res_text = self.provider_manager.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    json_mode=True
                )
                data = json.loads(res_text)

                scores_list = data.get("scores", [])
                idx_score_map = {item["index"]: item.get("score", 5.0) for item in scores_list}

                scored_chunks = []
                for idx, chunk in enumerate(chunks):
                    score = idx_score_map.get(idx, 5.0)
                    chunk_copy = dict(chunk)
                    chunk_copy["rerank_score"] = float(score)
                    if score >= min_score:
                        scored_chunks.append(chunk_copy)

                scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
                return scored_chunks[:top_n]

            except Exception as e:
                print(f"⚠️ Reranking LLM call bypassed ({e}). Utilizing local relevance scoring fallback.")

        # Local keyword overlap relevance fallback filtering out stop words
        query_tokens = set(t for t in tokenize_text(query) if t not in STOP_WORDS)
        if not query_tokens:
            return []

        scored_chunks = []
        for chunk in chunks:
            chunk_copy = dict(chunk)
            chunk_tokens = tokenize_text(chunk["content"])
            matches = sum(1 for t in chunk_tokens if t in query_tokens)
            score = min(10.0, float(matches * 2.5))
            chunk_copy["rerank_score"] = score
            if score >= min_score:
                scored_chunks.append(chunk_copy)

        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_chunks[:top_n]

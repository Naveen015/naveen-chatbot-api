import os
import re
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from pinecone import Pinecone
from rag.llm_provider import LLMProviderManager

def tokenize_text(text: str) -> List[str]:
    """Basic lowercased word tokenization for BM25 search."""
    return re.findall(r'\w+', text.lower())

class HybridRetriever:
    """
    Hybrid Retriever combining Sparse (BM25) search and Dense (Pinecone / Vector) search
    with Reciprocal Rank Fusion (RRF). Supports switchable OpenAI or local vLLM provider.
    """
    def __init__(
        self,
        docs_dir: str = "data",
        index_name: str = "naveen-chatbot",
        embed_model: str = "text-embedding-ada-002",
        bm25_pickle_path: str = "data/bm25_index.pkl",
        provider_manager: Optional[LLMProviderManager] = None
    ):
        self.docs_dir = docs_dir
        self.index_name = index_name
        self.embed_model = embed_model
        self.bm25_pickle_path = bm25_pickle_path
        self.provider_manager = provider_manager or LLMProviderManager()

        # Initialize Pinecone Client
        pinecone_key = os.getenv("PINECONE_API_KEY")
        self.pc_index = None
        if pinecone_key:
            try:
                pc = Pinecone(api_key=pinecone_key)
                self.pc_index = pc.Index(index_name)
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize Pinecone index: {e}")

        # In-memory storage for BM25 and fallback dense storage
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.in_memory_embeddings: Dict[str, List[float]] = {}

        # Load or initialize BM25 index
        self.load_or_build_bm25()

    def get_embedding(self, text: str) -> List[float]:
        """Generates embedding vector safely via provider manager."""
        try:
            return self.provider_manager.get_embedding(text, embed_model=self.embed_model)
        except Exception as e:
            print(f"⚠️ Embedding creation bypassed ({e}). Returning empty vector.")
            return []

    def load_or_build_bm25(self, force_rebuild: bool = False):
        """Loads BM25 index from pickle, or builds it from document chunks if missing."""
        if not force_rebuild and os.path.exists(self.bm25_pickle_path):
            try:
                with open(self.bm25_pickle_path, "rb") as f:
                    data = pickle.load(f)
                    self.chunks = data.get("chunks", [])
                    tokenized_corpus = data.get("tokenized_corpus", [])
                    if tokenized_corpus:
                        self.bm25 = BM25Okapi(tokenized_corpus)
                        print(f"✅ Loaded BM25 index with {len(self.chunks)} chunks from {self.bm25_pickle_path}")
                        return
            except Exception as e:
                print(f"⚠️ Warning: Failed to load BM25 pickle: {e}. Rebuilding index...")

        from rag.chunker import DocumentChunker
        chunker = DocumentChunker()
        self.chunks = chunker.process_directory(self.docs_dir)

        if not self.chunks:
            print("⚠️ Warning: No document chunks found in data directory.")
            return

        tokenized_corpus = [tokenize_text(c["content"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        try:
            os.makedirs(os.path.dirname(self.bm25_pickle_path), exist_ok=True)
            with open(self.bm25_pickle_path, "wb") as f:
                pickle.dump({
                    "chunks": self.chunks,
                    "tokenized_corpus": tokenized_corpus
                }, f)
            print(f"✅ Built and saved BM25 index with {len(self.chunks)} chunks to {self.bm25_pickle_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save BM25 index pickle: {e}")

    def dense_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Performs Vector (Dense) Search using Pinecone or in-memory cosine fallback."""
        query_vec = self.get_embedding(query)
        if not query_vec:
            return []

        if self.pc_index:
            try:
                res = self.pc_index.query(vector=query_vec, top_k=top_k, include_metadata=True)
                results = []
                for match in res.get("matches", []):
                    meta = match.get("metadata", {})
                    results.append({
                        "id": match["id"],
                        "source": meta.get("source", "unknown"),
                        "content": meta.get("content", ""),
                        "score": match.get("score", 0.0),
                        "retrieval_type": "dense"
                    })
                return results
            except Exception as e:
                print(f"⚠️ Pinecone query failed ({e}), falling back to local search.")

        if not self.chunks:
            return []

        results = []
        for chunk in self.chunks:
            chunk_id = chunk["id"]
            if chunk_id not in self.in_memory_embeddings:
                emb = self.get_embedding(chunk["content"])
                if not emb:
                    continue
                self.in_memory_embeddings[chunk_id] = emb

            chunk_vec = self.in_memory_embeddings[chunk_id]
            sim = float(np.dot(query_vec, chunk_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)))
            results.append({
                "id": chunk_id,
                "source": chunk.get("source", "unknown"),
                "content": chunk.get("content", ""),
                "score": sim,
                "retrieval_type": "dense"
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def sparse_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Performs BM25 (Sparse) Search."""
        if not self.bm25 or not self.chunks:
            return []

        tokens = tokenize_text(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "id": chunk["id"],
                "source": chunk.get("source", "unknown"),
                "content": chunk.get("content", ""),
                "score": score,
                "retrieval_type": "sparse"
            })

        return results

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_k: int = 5,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """Combines Dense and Sparse rankings using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        for rank, doc in enumerate(dense_results):
            doc_id = doc["id"]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, doc in enumerate(sparse_results):
            doc_id = doc["id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        fused_results = []
        for doc_id in sorted_ids[:top_k]:
            item = dict(doc_map[doc_id])
            item["rrf_score"] = rrf_scores[doc_id]
            fused_results.append(item)

        return fused_results

    def hybrid_search(self, query: str, top_k: int = 5, fetch_k: int = 15) -> List[Dict[str, Any]]:
        """Runs dense and sparse retrieval in parallel and merges using RRF."""
        dense_res = self.dense_search(query, top_k=fetch_k)
        sparse_res = self.sparse_search(query, top_k=fetch_k)
        return self.reciprocal_rank_fusion(dense_res, sparse_res, top_k=top_k)

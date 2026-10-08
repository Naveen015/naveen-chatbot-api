import pytest
import os
from fastapi.testclient import TestClient
from rag.chunker import DocumentChunker
from rag.hybrid_retriever import HybridRetriever, tokenize_text
from rag.llm_provider import LLMProviderManager
from app import app

client = TestClient(app)

def test_document_chunker():
    chunker = DocumentChunker(chunk_size=300, overlap=50)
    chunks = chunker.process_directory("data")
    assert len(chunks) > 0, "Chunker should extract chunks from data directory"
    first = chunks[0]
    assert "id" in first
    assert "content" in first
    assert "source" in first
    assert "page" in first

def test_tokenize_text():
    tokens = tokenize_text("Naveen Prashanna CS6348 Paper!")
    assert "naveen" in tokens
    assert "cs6348" in tokens

def test_llm_provider_manager():
    # Test OpenAI provider init
    mgr_openai = LLMProviderManager(provider="openai")
    assert mgr_openai.provider == "openai"

    # Test vLLM provider init & switching
    mgr_vllm = LLMProviderManager(provider="vllm", vllm_base_url="http://localhost:8000/v1")
    assert mgr_vllm.provider == "vllm"
    assert mgr_vllm.vllm_base_url == "http://localhost:8000/v1"

    mgr_vllm.set_provider("openai")
    assert mgr_vllm.provider == "openai"

def test_hybrid_retriever_bm25():
    retriever = HybridRetriever(docs_dir="data")
    results = retriever.sparse_search("Naveen", top_k=3)
    assert len(results) > 0, "BM25 search should return results for Naveen"
    assert "content" in results[0]
    assert "id" in results[0]

def test_reciprocal_rank_fusion():
    retriever = HybridRetriever(docs_dir="data")
    dense_dummy = [
        {"id": "doc1", "content": "Naveen software engineer", "source": "A.pdf", "score": 0.9},
        {"id": "doc2", "content": "UT Dallas student", "source": "B.pdf", "score": 0.8}
    ]
    sparse_dummy = [
        {"id": "doc2", "content": "UT Dallas student", "source": "B.pdf", "score": 5.2},
        {"id": "doc3", "content": "CS6348 paper", "source": "C.pdf", "score": 4.1}
    ]
    fused = retriever.reciprocal_rank_fusion(dense_dummy, sparse_dummy, top_k=3)
    assert len(fused) == 3
    assert fused[0]["id"] == "doc2"

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "active_provider" in data
    assert "vllm_base_url" in data
    assert data["total_bm25_chunks"] > 0

def test_greeting_and_history_query():
    # Test casual greeting intent
    resp_greeting = client.post("/query", json={"query": "Hi", "provider": "openai"})
    assert resp_greeting.status_code == 200
    data_g = resp_greeting.json()
    assert "NaviBot" in data_g["answer"]
    assert "Handled via fast casual intent rule." in data_g["agent_trace"]

    # Test vague query with stop-words returns clean message (no raw chunk blurting)
    resp_vague = client.post("/query", json={"query": "Why is that so?", "provider": "openai"})
    assert resp_vague.status_code == 200
    data_v = resp_vague.json()
    assert "CS6348_Final_Paper.pdf" not in data_v["answer"] or "couldn't find" in data_v["answer"].lower() or "records" in data_v["answer"].lower()


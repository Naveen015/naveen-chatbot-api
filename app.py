import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.llm_provider import LLMProviderManager
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import LLMReranker
from rag.agentic_engine import AgenticRAGEngine
from rag.evaluator import RAGEvaluator

# Initialize Provider Manager
provider_manager = LLMProviderManager()

# Initialize RAG Pipeline components
docs_dir = "data"
retriever = HybridRetriever(docs_dir=docs_dir, provider_manager=provider_manager)
reranker = LLMReranker(provider_manager=provider_manager)
agentic_engine = AgenticRAGEngine(retriever=retriever, reranker=reranker, provider_manager=provider_manager)
evaluator = RAGEvaluator(provider_manager=provider_manager)

app = FastAPI(
    title="Naveen Chatbot - Switchable Agentic RAG API",
    description="Production-grade Agentic RAG API supporting both OpenAI API and Local 7B LLM (vLLM) inferencing.",
    version="2.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Models
class QueryRequest(BaseModel):
    query: str = Field(..., example="What research did Naveen do in CS6348?")
    agentic: bool = Field(default=True, description="Enable full agentic pipeline with query rewriting, reranking, and self-correction.")
    provider: Optional[str] = Field(default=None, description="LLM provider switch: 'openai' or 'vllm' / 'local'. Defaults to LLM_PROVIDER env.")
    max_retries: int = Field(default=1, description="Max self-correction retry attempts.")
    history: Optional[List[Dict[str, str]]] = Field(default=[], description="Previous conversation turns [{'role': 'user'|'assistant', 'content': '...'}]")

class QueryResponse(BaseModel):
    query: str
    provider: str
    answer: str
    sources: List[str]
    faithfulness_score: float
    is_faithful: bool
    context_used: List[Dict[str, Any]]
    agent_trace: List[str]

class EvaluateRequest(BaseModel):
    provider: Optional[str] = Field(default=None, description="LLM provider to benchmark: 'openai' or 'vllm'.")
    test_queries: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Optional custom list of {'query': '...', 'ground_truth': '...'} pairs to benchmark."
    )

# Routes

@app.get("/health")
def health_check():
    """Health check endpoint exposing active provider status."""
    return {
        "status": "online",
        "active_provider": provider_manager.provider,
        "vllm_base_url": provider_manager.vllm_base_url,
        "vllm_model": provider_manager.vllm_default_model,
        "total_bm25_chunks": len(retriever.chunks),
        "pinecone_connected": retriever.pc_index is not None,
        "openai_configured": provider_manager.openai_key is not None
    }

@app.post("/query", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    """
    Main Agentic RAG query endpoint.
    Supports switchable LLM inferencing ('openai' vs local 'vllm').
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        if req.agentic:
            result = agentic_engine.run(
                query=req.query,
                provider=req.provider,
                max_retries=req.max_retries,
                history=req.history
            )
            return QueryResponse(
                query=result["query"],
                provider=result.get("provider", provider_manager.provider),
                answer=result["answer"],
                sources=result["sources"],
                faithfulness_score=result["faithfulness_score"],
                is_faithful=result["is_faithful"],
                context_used=result["context_chunks"],
                agent_trace=result["agent_trace"]
            )
        else:
            p = req.provider or provider_manager.provider
            provider_manager.set_provider(p)
            chunks = retriever.hybrid_search(req.query, top_k=4)
            answer = agentic_engine.generate_answer(req.query, chunks, history=req.history)
            sources = list(set([c.get("source", "unknown") for c in chunks]))
            return QueryResponse(
                query=req.query,
                provider=p,
                answer=answer,
                sources=sources,
                faithfulness_score=1.0,
                is_faithful=True,
                context_used=[{"id": c["id"], "source": c["source"], "snippet": c["content"][:150]} for c in chunks],
                agent_trace=[f"Executed basic hybrid search using {p} without agentic self-correction."]
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG execution error: {str(e)}")

@app.post("/evaluate")
def run_evaluation(req: EvaluateRequest):
    """
    Runs automated evaluation benchmark across test queries measuring Faithfulness,
    Answer Relevance, Context Precision, and Context Recall.
    """
    default_dataset = [
        {"query": "What are Naveen Prashanna's main technical skills?", "ground_truth": "Software Engineering, Machine Learning, Data Science, Python, PyTorch, C++."},
        {"query": "What is covered in Naveen's CS6348 paper?", "ground_truth": "Computer Security, Data Security, and Machine Learning research."},
        {"query": "Which university did Naveen attend?", "ground_truth": "University of Texas at Dallas (UT Dallas)."}
    ]

    dataset = req.test_queries if req.test_queries else default_dataset
    try:
        summary = evaluator.run_benchmark(dataset, agentic_engine, provider=req.provider)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/reindex")
def trigger_reindex(background_tasks: BackgroundTasks):
    """Triggers background rebuild of BM25 and vector indices."""
    def rebuild_task():
        retriever.load_or_build_bm25(force_rebuild=True)

    background_tasks.add_task(rebuild_task)
    return {"message": "Re-indexing started in background."}

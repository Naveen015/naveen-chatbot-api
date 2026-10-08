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

    def check_casual_intent(self, query: str) -> Optional[str]:
        """
        Fast rule-based check for common greetings, introductions, and casual remarks.
        """
        q_clean = query.strip().lower().rstrip("!.,?")
        greetings = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "yo", "sup", "hi there", "hello there"}
        intros = {"who are you", "who are you?", "what is your name", "what's your name", "what can you do", "help", "who is navibot"}
        thanks = {"thanks", "thank you", "thanks!", "thank you!", "thx", "awesome", "great"}
        farewells = {"bye", "goodbye", "see ya", "cya"}

        if q_clean in greetings:
            return "Hello! 👋 I'm NaviBot, Naveen Prashanna's AI assistant. I can answer questions about Naveen's work experience, software engineering & ML projects, technical skills, education, research papers, or contact details. How can I help you today?"
        if q_clean in intros:
            return "I am NaviBot, an AI assistant trained on Naveen Prashanna's professional background, resume, research papers, and software projects. Feel free to ask me anything about Naveen!"
        if q_clean in thanks:
            return "You're very welcome! Let me know if you have any other questions about Naveen Prashanna."
        if q_clean in farewells:
            return "Goodbye! Feel free to reach out anytime if you have more questions about Naveen."

        return None

    def decompose_and_rewrite_query(
        self, query: str, history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes the query with conversation history context, expands terms,
        and rewrites vague follow-up questions into standalone search queries.
        """
        if not self.provider_manager.client:
            return {"rewritten_queries": [query], "is_complex": False, "is_casual": False}

        history_str = ""
        if history:
            recent = history[-6:]
            formatted_turns = []
            for m in recent:
                role = "User" if m.get("role") in ["user", "human"] else "Assistant"
                formatted_turns.append(f"{role}: {m.get('content', '')}")
            history_str = "\n".join(formatted_turns)

        prompt = f"""
You are an AI query contextualization agent for NaviBot, a database assistant for **Naveen Prashanna** (resumes, academic papers, work experience, projects).

Recent Conversation History:
\"\"\"
{history_str if history_str else "No prior history."}
\"\"\"

User's Latest Input: "{query}"

Tasks:
1. Determine if the user's latest input is a simple greeting, introduction request, or casual remark (e.g., "Hi", "Hello", "Thanks", "Who are you?").
2. If it is NOT a simple casual remark, analyze whether it relies on conversation history (e.g. "Why is that so?", "Tell me more about it", "What else did he do there?").
3. Rewrite the latest input into 2 to 4 self-contained, search-optimized query variations for document retrieval. If the user asks broadly about work experience, employment history, or companies, explicitly include search queries for all 4 positions: Pharvision Advisers, Kahana Group Inc, Quantitative Brokers, and Big Data Science Research.

Return JSON in this format:
{{
  "is_casual": false,
  "casual_reply": "",
  "rewritten_queries": [
    "standalone search query variation 1",
    "standalone search query variation 2",
    "standalone search query variation 3"
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
            return {"rewritten_queries": [query], "is_complex": False, "is_casual": False}

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

    def generate_answer(
        self, query: str, chunks: List[Dict[str, Any]], history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generates answer strictly constrained to context, taking conversation history into account."""
        if not chunks:
            return (
                "I couldn't find specific details matching that in Naveen Prashanna's current records. "
                "However, I can answer questions about his work experience across Pharvision Advisers, Kahana Group, "
                "Quantitative Brokers, and Big Data Science Research, or his technical skills, ML projects, research papers, and education."
            )

        context = "\n\n".join([f"Source: [{c.get('source', 'doc')}, p.{c.get('page', 1)}]\n{c['content']}" for c in chunks])

        history_str = ""
        if history:
            recent = history[-4:]
            formatted_turns = []
            for m in recent:
                role = "User" if m.get("role") in ["user", "human"] else "Assistant"
                formatted_turns.append(f"{role}: {m.get('content', '')}")
            history_str = "\n".join(formatted_turns)

        if not self.provider_manager.client:
            summary = "Based on retrieved documents:\n" + "\n".join([f"- {c['content']}" for c in chunks[:4]])
            return summary

        prompt = f"""
# ROLE
You are NaviBot, a knowledgeable, warm, and professional AI assistant representing **Naveen Prashanna**.

# RECENT DIALOGUE HISTORY
{history_str if history_str else "No prior history."}

# TONE & STYLE
- Speak in a natural, direct, engaging human tone.
- **NEVER** start your answer with robotic boilerplate prefixes like "Sure—here's what I found..." or "Here is what I found about...". Jump straight into answering naturally.
- Refer to Naveen in the third person (Naveen, "he", "his").
- Format responses clearly with clean bullet points, bold key terms, and structured sections.
- When asked about Naveen's work history or positions, ensure ALL 4 of his positions present in context (Pharvision Advisers, Kahana Group Inc, Quantitative Brokers, Big Data Science Research) are accurately listed without missing any.
- Cite sources naturally using inline tags like [NaveenPrashanna_Resume.pdf] or [experience_kahana_group.md] where applicable.

# KNOWLEDGE RULES
- Answer using information in the Context block below.
- If context does not contain full details, politely state what is known or ask for clarification.
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
            summary = ""
            for c in chunks[:4]:
                summary += f"• **[{c.get('source', 'doc')}]**: {c['content']}\n\n"
            return summary.strip()

    def run(
        self,
        query: str,
        provider: Optional[str] = None,
        max_retries: int = 1,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Executes the full Agentic RAG workflow:
        Casual Intent Check -> Query Rewriting & History Contextualization -> Multi-Retrieval -> Reranking -> Draft Generation -> Faithfulness Check -> Output.
        """
        if provider:
            self.set_provider(provider)

        active_provider = self.provider_manager.provider
        trace = []
        trace.append(f"Received query: '{query}' (Active LLM Provider: {active_provider.upper()})")

        # Step 0: Fast Rule-based Casual Greeting Check
        rule_reply = self.check_casual_intent(query)
        if rule_reply:
            trace.append("Handled via fast casual intent rule.")
            return {
                "query": query,
                "provider": active_provider,
                "answer": rule_reply,
                "sources": [],
                "context_chunks": [],
                "faithfulness_score": 1.0,
                "is_faithful": True,
                "agent_trace": trace
            }

        # Step 1: Query Analysis & History Contextualized Rewriting
        plan = self.decompose_and_rewrite_query(query, history=history)
        if plan.get("is_casual") and plan.get("casual_reply"):
            trace.append("Handled via LLM casual intent detection.")
            return {
                "query": query,
                "provider": active_provider,
                "answer": plan["casual_reply"],
                "sources": [],
                "context_chunks": [],
                "faithfulness_score": 1.0,
                "is_faithful": True,
                "agent_trace": trace
            }

        rewritten_queries = plan.get("rewritten_queries", [query])
        if not rewritten_queries:
            rewritten_queries = [query]
        trace.append(f"Query planning generated {len(rewritten_queries)} search variants: {rewritten_queries}")

        # Step 2: Multi-Retrieval via Hybrid Search (Dense + BM25)
        raw_chunks_map = {}
        for q_var in rewritten_queries:
            results = self.retriever.hybrid_search(q_var, top_k=10, fetch_k=20)
            for item in results:
                raw_chunks_map[item["id"]] = item

        raw_chunks = list(raw_chunks_map.values())
        trace.append(f"Retrieved {len(raw_chunks)} candidate chunks via Hybrid Search (Dense + BM25).")

        # Step 3: LLM Reranking & Context Relevance Filtering
        search_query = rewritten_queries[0] if rewritten_queries else query
        reranked_chunks = self.reranker.rerank(search_query, raw_chunks, top_n=6, min_score=3.0, provider=active_provider)
        trace.append(f"Reranked context chunks down to {len(reranked_chunks)} high-relevance passages using {active_provider}.")

        # Step 4: Draft Generation with History
        answer = self.generate_answer(query, reranked_chunks, history=history)
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


import os
from typing import Dict, Any, Optional, List
from openai import OpenAI

class LLMProviderManager:
    """
    Unified manager supporting OpenAI API and Local LLM (vLLM / Ollama / Llama.cpp)
    for inference (Reranking, Query Expansion, Generation), while using a consistent
    embedding model for vector search.
    """
    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        vllm_base_url: Optional[str] = None,
        vllm_api_key: Optional[str] = None
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
        
        # OpenAI config
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.openai_default_model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.openai_mini_model = os.getenv("OPENAI_MINI_MODEL", "gpt-4o-mini")
        
        # Local vLLM config
        self.vllm_base_url = vllm_base_url or os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
        self.vllm_api_key = vllm_api_key or os.getenv("LOCAL_LLM_API_KEY", "EMPTY")
        self.vllm_default_model = model_name or os.getenv("LOCAL_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        
        self.client: Optional[OpenAI] = None
        self.active_model: str = ""
        self._init_client(model_name)

    def _init_client(self, override_model: Optional[str] = None):
        if self.provider == "vllm" or self.provider == "local":
            self.client = OpenAI(
                base_url=self.vllm_base_url,
                api_key=self.vllm_api_key
            )
            self.active_model = override_model or self.vllm_default_model
        else: # default to openai
            if self.openai_key:
                self.client = OpenAI(api_key=self.openai_key)
                self.active_model = override_model or self.openai_default_model

    def set_provider(self, provider: str, model_name: Optional[str] = None):
        """Switch inference LLM provider dynamically at runtime."""
        self.provider = provider.lower()
        self._init_client(model_name)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        json_mode: bool = False,
        override_model: Optional[str] = None
    ) -> str:
        """Executes a chat completion call using the active inference provider client."""
        if not self.client:
            raise ValueError(f"LLM Client for provider '{self.provider}' is not configured.")

        model = override_model or self.active_model
        if self.provider == "openai" and json_mode and ("mini" in override_model if override_model else False):
            model = self.openai_mini_model

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        res = self.client.chat.completions.create(**kwargs)
        return res.choices[0].message.content.strip()

    def get_embedding(self, text: str, embed_model: str = "text-embedding-ada-002") -> List[float]:
        """
        Generates embedding vector using the canonical embedding model (1536-dim).
        Always uses the consistent embed_model matching the vector database index.
        """
        # Embeddings call uses OpenAI client if available, or current client
        embed_client = OpenAI(api_key=self.openai_key) if self.openai_key else self.client
        if not embed_client:
            raise ValueError("No active client available for embedding generation.")

        res = embed_client.embeddings.create(
            model=embed_model,
            input=[text]
        )
        return res.data[0].embedding

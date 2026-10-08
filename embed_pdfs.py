import os
import fitz
from openai import OpenAI
from pinecone import Pinecone
from rag.chunker import DocumentChunker
from rag.hybrid_retriever import HybridRetriever

DOCS_DIR = "data"
INDEX_NAME = "naveen-chatbot"
MODEL = "text-embedding-ada-002"

def main():
    print("🚀 Starting PDF Document Ingestion & Hybrid Index Building...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")

    if not openai_key:
        print("❌ OPENAI_API_KEY environment variable is missing.")
        return

    client = OpenAI(api_key=openai_key)
    
    # 1. Chunk documents using DocumentChunker
    chunker = DocumentChunker(chunk_size=600, overlap=150)
    chunks = chunker.process_directory(DOCS_DIR)
    print(f"📄 Extracted {len(chunks)} metadata-enriched text chunks from PDFs in '{DOCS_DIR}'.")

    if not chunks:
        print("⚠️ No chunks extracted.")
        return

    # 2. Upload to Pinecone (Dense Index)
    if pinecone_key:
        try:
            pc = Pinecone(api_key=pinecone_key)
            index = pc.Index(INDEX_NAME)

            print("⚡ Generating OpenAI Embeddings & Upserting to Pinecone...")
            vectors = []
            batch_size = 50

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                texts = [c["content"] for c in batch]
                res = client.embeddings.create(model=MODEL, input=texts)
                embeddings = [rec.embedding for rec in res.data]

                for chunk, emb in zip(batch, embeddings):
                    meta = {
                        "source": chunk["source"],
                        "page": chunk["page"],
                        "content": chunk["content"]
                    }
                    vectors.append((chunk["id"], emb, meta))

            index.upsert(vectors=vectors)
            print(f"✅ Upserted {len(vectors)} vectors into Pinecone index '{INDEX_NAME}'.")
        except Exception as e:
            print(f"⚠️ Warning: Could not upsert to Pinecone: {e}")
    else:
        print("⚠️ PINECONE_API_KEY missing. Skipping Pinecone vector upload.")

    # 3. Build & Persist BM25 Sparse Index
    print("🔍 Building & Persisting BM25 Index...")
    retriever = HybridRetriever(docs_dir=DOCS_DIR, index_name=INDEX_NAME, embed_model=MODEL)
    retriever.load_or_build_bm25(force_rebuild=True)
    print("🎉 Hybrid indexing complete! Both Sparse (BM25) and Dense vector indexes are ready.")

if __name__ == "__main__":
    main()

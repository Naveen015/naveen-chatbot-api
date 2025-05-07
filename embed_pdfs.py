import os
import openai
import fitz  # PyMuPDF
from pinecone import Pinecone
# --- 1. Embed function ---
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("naveen-chatbot")

MODEL = "text-embedding-ada-002"
CHUNK_SIZE = 800  # chars (not tokens)

DOCS_DIR = "data"  # your folder of PDFs


def extract_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    
    # Split into overlapping chunks
    chunks = []
    text = full_text.strip().replace("\n", " ")
    for i in range(0, len(text), CHUNK_SIZE):
        chunk = text[i:i+CHUNK_SIZE]
        if len(chunk) > 200:  # skip tiny junk
            chunks.append(chunk)
    return chunks



def embed_texts(texts):
    res = client.embeddings.create(
        model=MODEL,
        input=texts
    )
    return [record.embedding for record in res.data]



def process_pdf_file(filepath, doc_id_start):
    filename = os.path.basename(filepath)
    chunks = extract_chunks(filepath)
    embeddings = embed_texts(chunks)
    vectors = []
    
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        uid = f"{filename}_{i + doc_id_start}"
        meta = {
            "source": filename,
            "content": chunk
        }
        vectors.append((uid, emb, meta))
    
    index.upsert(vectors=vectors)
    print(f"✅ Uploaded {len(vectors)} chunks from {filename}")
    return len(vectors)


if __name__ == "__main__":
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    doc_id = 10000  # avoid clashing with earlier upserts

    for pdf_file in files:
        full_path = os.path.join(DOCS_DIR, pdf_file)
        num_uploaded = process_pdf_file(full_path, doc_id)
        doc_id += num_uploaded

    print("🎉 All PDF docs processed and uploaded.")

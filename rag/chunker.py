import os
import re
import fitz  # PyMuPDF
from typing import List, Dict, Any

class DocumentChunker:
    """
    Metadata-aware PDF text chunker with sliding window sentence/character boundaries.
    """
    def __init__(self, chunk_size: int = 600, overlap: int = 150, min_chunk_len: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_len = min_chunk_len

    def clean_text(self, text: str) -> str:
        """Normalizes whitespace and removes unwanted linebreaks/artifacts."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_chunks_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Parses a PDF file page-by-page and creates overlapping chunks with rich metadata.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

        filename = os.path.basename(pdf_path)
        doc = fitz.open(pdf_path)
        chunks = []
        global_chunk_idx = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            raw_text = page.get_text()
            cleaned = self.clean_text(raw_text)

            if len(cleaned) < self.min_chunk_len:
                continue

            # Sliding window chunking
            start = 0
            while start < len(cleaned):
                end = start + self.chunk_size
                chunk_text = cleaned[start:end]

                # Adjust boundary to last space or period if available
                if end < len(cleaned):
                    last_punct = max(chunk_text.rfind('. '), chunk_text.rfind('? '), chunk_text.rfind('! '))
                    if last_punct != -1 and last_punct > (self.chunk_size // 2):
                        end = start + last_punct + 1
                        chunk_text = cleaned[start:end]

                chunk_text = chunk_text.strip()
                if len(chunk_text) >= self.min_chunk_len:
                    chunk_id = f"{filename}_p{page_num + 1}_c{global_chunk_idx}"
                    chunks.append({
                        "id": chunk_id,
                        "source": filename,
                        "page": page_num + 1,
                        "content": chunk_text,
                        "char_count": len(chunk_text)
                    })
                    global_chunk_idx += 1

                start += (self.chunk_size - self.overlap)

        doc.close()
        return chunks

    def process_directory(self, docs_dir: str) -> List[Dict[str, Any]]:
        """Processes all PDF files in a given directory."""
        all_chunks = []
        if not os.path.exists(docs_dir):
            return all_chunks

        pdf_files = [f for f in os.listdir(docs_dir) if f.endswith('.pdf')]
        for pdf_file in pdf_files:
            full_path = os.path.join(docs_dir, pdf_file)
            chunks = self.extract_chunks_from_pdf(full_path)
            all_chunks.extend(chunks)

        return all_chunks

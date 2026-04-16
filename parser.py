import fitz  # PyMuPDF
import re
from typing import List, Optional
from pydantic import BaseModel

class DocumentChunk(BaseModel):
    page_start: int
    page_end: int
    text: str
    section: str = ""
    paper_id: str = ""

class PDFParser:
    def __init__(self, chunk_size: int = 2000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_text_with_pages(self, pdf_path: str) -> List[dict]:
        """Extracts text block by block, associating each with its page number and finding potential headers."""
        doc = fitz.open(pdf_path)
        
        # Heuristic for body text size by sampling first few pages
        font_sizes = {}
        for page_num in range(min(3, len(doc))):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict").get("blocks", [])
            for b in blocks:
                if b['type'] == 0:  # text block
                    for l in b["lines"]:
                        for s in l["spans"]:
                            size = round(s["size"], 1)
                            font_sizes[size] = font_sizes.get(size, 0) + len(s["text"])
        
        if not font_sizes:
            # Fallback if dictionary extraction fails
            return self._fallback_extraction(doc)

        body_font_size = max(font_sizes, key=font_sizes.get)
        
        extracted_data = []
        current_section = "Main"

        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("dict").get("blocks", [])
            for b in blocks:
                if b['type'] == 0:  # text
                    block_text = ""
                    is_header = False
                    
                    for l in b["lines"]:
                        for s in l["spans"]:
                            text = s["text"].strip()
                            if not text: continue
                            size = round(s["size"], 1)
                            
                            # Simple heuristic: if font is notably larger or bold, it might be a header
                            if size > body_font_size + 0.5 or (size >= body_font_size and "bold" in s["font"].lower()):
                                if len(text) < 100: # Headers are usually short
                                    is_header = True
                            
                            block_text += text + " "
                    
                    block_text = block_text.strip()
                    if not block_text: continue
                    
                    if is_header and len(block_text.split()) < 15:
                        current_section = block_text
                        
                    extracted_data.append({
                        "page": page_num,
                        "text": block_text,
                        "section": current_section
                    })
                    
        return extracted_data
        
    def _fallback_extraction(self, doc) -> List[dict]:
        extracted_data = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                extracted_data.append({
                    "page": page_num,
                    "text": text,
                    "section": ""
                })
        return extracted_data

    def chunk_document(self, extracted_data: List[dict]) -> List[DocumentChunk]:
        """Groups extracted text blocks into evenly sized chunks while preserving page numbers and sections."""
        if not extracted_data:
            return []

        chunks = []
        current_text = ""
        current_pages = []
        current_section = extracted_data[0].get("section", "")
        
        for block in extracted_data:
            section = block.get("section", "")
            
            # Start a new chunk if section changes significantly or chunk size exceeded
            if len(current_text) > self.chunk_size or (section != current_section and len(current_text) > self.chunk_size // 2):
                if current_text.strip():
                    page_start = min(current_pages) if current_pages else 1
                    page_end = max(current_pages) if current_pages else 1
                    chunks.append(DocumentChunk(
                        page_start=page_start,
                        page_end=page_end,
                        text=current_text.strip(),
                        section=current_section
                    ))
                
                # Setup next chunk with overlap (if within same section)
                if section == current_section:
                    overlap_text = current_text[-self.overlap:] if len(current_text) > self.overlap else ""
                    current_text = overlap_text + " " + block["text"]
                else:
                    current_text = block["text"]
                
                current_pages = [block["page"]]
                current_section = section
            else:
                current_text += "\n" + block["text"] if current_text else block["text"]
                current_pages.append(block["page"])

        # Add the last chunk
        if current_text.strip():
            page_start = min(current_pages) if current_pages else 1
            page_end = max(current_pages) if current_pages else 1
            chunks.append(DocumentChunk(
                page_start=page_start,
                page_end=page_end,
                text=current_text.strip(),
                section=current_section
            ))

        return chunks

    def parse(self, pdf_path: str) -> List[DocumentChunk]:
        extracted_data = self.extract_text_with_pages(pdf_path)
        chunks = self.chunk_document(extracted_data)
        return chunks

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        parser = PDFParser()
        chunks = parser.parse(sys.argv[1])
        for i, c in enumerate(chunks[:3]):
            print(f"--- Chunk {i+1} (Pages {c.page_start}-{c.page_end}) [{c.section}] ---")
            print(c.text[:200] + "...\n")
        print(f"Total chunks: {len(chunks)}")

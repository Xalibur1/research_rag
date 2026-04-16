from fastapi import FastAPI, Request, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os

# Bypass SSL verify for local HuggingFace downloads
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

import json
import shutil
from pathlib import Path
import time
from fastapi.middleware.cors import CORSMiddleware

from parser import PDFParser
from multi_extractor import MultiBackendExtractor
from graph_builder import GraphBuilder
from graph_index import HybridIndex
from multi_generator import MultiPaperRAGGenerator

app = FastAPI(title="Research RAG Engine")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals for RAG engine
graph = None
index = None
rag_generator = None
extractions = {}
paper_titles = {}

pdf_dir = Path("./papers")
artifacts_dir = Path("./artifacts")
pdf_dir.mkdir(exist_ok=True)
artifacts_dir.mkdir(exist_ok=True)

@app.on_event("startup")
def startup_event():
    global graph, index, rag_generator, extractions, paper_titles
    
    print("--- RAG Engine Startup ---")
    pdf_paths = list(pdf_dir.glob("*.pdf"))
    
    doc_parser = PDFParser(chunk_size=1500, overlap=200)
    extractor = MultiBackendExtractor(backend="groq")
    gb = GraphBuilder()
    
    all_chunks = []
    
    for pdf in pdf_paths:
        paper_id = pdf.stem
        artifact_path = artifacts_dir / f"{paper_id}.json"
        
        print(f"Checking {paper_id}...")
        
        if artifact_path.exists():
            print(f"Loading cached extraction for {paper_id}")
            with open(artifact_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                from schemas import PaperExtraction
                extraction = PaperExtraction(**data)
                extractions[paper_id] = extraction
                paper_titles[paper_id] = extraction.title or paper_id
                
            # We still need to parse chunks to build the chunk_index if we want text retrieval
            # Caching chunks in artifacts would be better, but for now we re-parse text
            chunks = doc_parser.parse(str(pdf))
            for c in chunks: 
                c.paper_id = paper_id
            all_chunks.extend(chunks)
        else:
            print(f"Parsing and extracting {paper_id} (New PDF)...")
            try:
                chunks = doc_parser.parse(str(pdf))
                for c in chunks: 
                    c.paper_id = paper_id
                all_chunks.extend(chunks)
                extraction = extractor.extract_from_chunks(chunks, paper_id)
                extractions[paper_id] = extraction
                paper_titles[paper_id] = extraction.title or paper_id
                
                with open(artifact_path, "w", encoding="utf-8") as f:
                    json.dump(extraction.model_dump(), f, indent=2, ensure_ascii=False)
                    
                time.sleep(5) # rate limit mitigation for new extraction
            except Exception as e:
                print(f"Error processing {pdf.name}: {e}")
                
    for paper_id, extraction in extractions.items():
        gb.build_from_extraction(extraction)
        
    graph = gb.graph
    
    print("Building Hybrid Index...")
    index = HybridIndex()
    index.build_chunk_index(all_chunks)
    index.build_graph_index(graph)
    
    rag_generator = MultiPaperRAGGenerator(
        index=index,
        paper_titles=paper_titles,
        backend="groq",
    )
    print("Startup Complete. Engine Ready.")

# API Models
class ChatQuery(BaseModel):
    query: str
    paper_ids: list[str] | None = None  # if set, restrict context to these papers

class VerifyQuery(BaseModel):
    answer: str
    query: str
    paper_ids: list[str] | None = None

@app.get("/api/dashboard/stats")
def get_stats():
    total_papers = len(extractions)
    graph_nodes = graph.number_of_nodes() if graph else 0
    limitations_found = sum([len(e.method.limitations) if e.method else 0 for e in extractions.values()]) if extractions else 0

    recent_verifications = []
    for pid, ext in extractions.items():
        inputs_count = sum([len(ext.method.inputs), len(ext.method.outputs)]) if ext.method else 0
        recent_verifications.append({
            "id": pid,
            "title": ext.title or pid,
            "inputs": inputs_count,
            "status": "Verified"
        })

    return {
        "metrics": {
            "total_papers": total_papers,
            "graph_nodes": graph_nodes,
            "limitations_found": limitations_found,
        },
        "recent_verifications": recent_verifications
    }


@app.get("/api/papers")
def get_papers():
    return [{"id": k, "title": v} for k,v in paper_titles.items()]

@app.get("/api/papers/{paper_id}")
def get_paper_details(paper_id: str):
    if paper_id not in extractions:
        return {"error": "Paper not found"}
    
    ext = extractions[paper_id].model_dump()
    
    # Adapt PaperExtraction to front-end expected model
    method = ext.get("method") or {}
    inputs_count = len(method.get("inputs", [])) + len(method.get("outputs", []))
    abstract_text = method.get("purpose_one_sentence", "No abstract available.")
    core_idea = method.get("core_idea", "No details.")
    
    ext["entities"] = [g.get("term") for g in ext.get("glossary", [])]
    ext["chunks"] = [core_idea]
    ext["inputs"] = inputs_count
    ext["status"] = "Verified"
    ext["abstract"] = abstract_text
    
    return ext

@app.post("/api/chat")
def chat_with_rag(query: ChatQuery):
    if not rag_generator:
        return {"error": "Engine not initialized"}
    answer = rag_generator.generate_answer(
        query.query,
        paper_ids=query.paper_ids or None
    )
    return answer

@app.post("/api/verify")
def verify_grounding(payload: VerifyQuery):
    if not rag_generator:
        return {"error": "Engine not initialized"}
    
    # Re-retrieve context to check against
    text_results = rag_generator.index.search_chunks(payload.query, top_k=5)
    if payload.paper_ids:
        text_results = [(c, s) for c, s in text_results if c.paper_id in payload.paper_ids]
        
    context_str = "--- TEXT EVIDENCE ---\n"
    for chunk, score in text_results:
        context_str += f"[Pages {chunk.page_start}-{chunk.page_end}] {chunk.text[:800]}\n\n"
        
    return rag_generator.check_grounding(payload.answer, context_str)

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global graph, index, rag_generator, extractions, paper_titles
    if not file.filename.endswith('.pdf'):
        return {"error": "File must be a PDF."}
    
    file_path = pdf_dir / file.filename
    paper_id = file_path.stem
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    doc_parser = PDFParser(chunk_size=1500, overlap=200)
    extractor = MultiBackendExtractor(backend="groq")
    
    try:
        chunks = doc_parser.parse(str(file_path))
        for c in chunks: 
            c.paper_id = paper_id
        extraction = extractor.extract_from_chunks(chunks, paper_id)
        extractions[paper_id] = extraction
        paper_titles[paper_id] = extraction.title or paper_id
        
        artifact_path = artifacts_dir / f"{paper_id}.json"
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(extraction.model_dump(), f, indent=2, ensure_ascii=False)
            
        gb = GraphBuilder()
        for pid, ext in extractions.items():
            gb.build_from_extraction(ext)
        graph = gb.graph
        
        all_chunks = []
        for p_file in pdf_dir.glob("*.pdf"):
            p_id = p_file.stem
            c_list = doc_parser.parse(str(p_file))
            for c in c_list:
                c.paper_id = p_id
            all_chunks.extend(c_list)
            
        index = HybridIndex()
        index.build_chunk_index(all_chunks)
        index.build_graph_index(graph)
        
        rag_generator.index = index
        rag_generator.paper_titles = paper_titles
        
    except Exception as e:
        return {"error": str(e)}

    return {"status": "success", "paper_id": paper_id, "title": paper_titles[paper_id]}


# ── Collections (stored as JSON on disk) ───────────────────────────────
import uuid as _uuid
collections_path = Path("./collections.json")

def _load_collections() -> list:
    if collections_path.exists():
        with open(collections_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def _save_collections(data: list):
    with open(collections_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

class CollectionCreate(BaseModel):
    name: str
    paper_ids: list[str] = []

class CollectionUpdate(BaseModel):
    name: str | None = None
    paper_ids: list[str] | None = None

@app.get("/api/collections")
def get_collections():
    return _load_collections()

@app.post("/api/collections")
def create_collection(body: CollectionCreate):
    cols = _load_collections()
    new_col = {
        "id": str(_uuid.uuid4())[:8],
        "name": body.name,
        "paper_ids": body.paper_ids,
    }
    cols.append(new_col)
    _save_collections(cols)
    return new_col

@app.put("/api/collections/{col_id}")
def update_collection(col_id: str, body: CollectionUpdate):
    cols = _load_collections()
    for col in cols:
        if col["id"] == col_id:
            if body.name is not None: col["name"] = body.name
            if body.paper_ids is not None: col["paper_ids"] = body.paper_ids
            _save_collections(cols)
            return col
    return {"error": "Collection not found"}

@app.delete("/api/collections/{col_id}")
def delete_collection(col_id: str):
    cols = _load_collections()
    cols = [c for c in cols if c["id"] != col_id]
    _save_collections(cols)
    return {"status": "deleted"}

# Serve Frontend static files
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

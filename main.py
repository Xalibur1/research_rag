import sys
import argparse
import json
import os
from parser import PDFParser
from extractor import Extractor
from graph_builder import GraphBuilder
from graph_index import HybridIndex
from generator import RAGGenerator

def main():
    parser = argparse.ArgumentParser(description="Research-Graph RAG: Answer questions on research PDFs.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF research paper.")
    args = parser.parse_args()
    
    pdf_path = args.pdf_path
    paper_id = os.path.basename(pdf_path).replace(".pdf", "")

    if not os.path.exists(pdf_path):
        print(f"Error: Could not find '{pdf_path}'.")
        sys.exit(1)

    print(f"--- 1/4: Parsing PDF '{pdf_path}' ---")
    doc_parser = PDFParser(chunk_size=1500, overlap=200)
    chunks = doc_parser.parse(pdf_path)
    print(f"Extracted {len(chunks)} chunks.")

    print(f"\n--- 2/4: Extracting Entities with LLM ---")
    extractor = Extractor()
    try:
        extraction = extractor.extract_from_chunks(chunks, paper_id)
        print(f"Extracted paper: {extraction.title}")
    except Exception as e:
        print(f"Failed to extract entities: {e}")
        sys.exit(1)

    print(f"\n--- 3/4: Building Knowledge Graph ---")
    gb = GraphBuilder()
    graph = gb.build_from_extraction(extraction)
    print(f"Graph created with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

    print(f"\n--- 4/4: Building Hybrid Index ---")
    index = HybridIndex()
    index.build_chunk_index(chunks)
    index.build_graph_index(graph)
    print("Indexing complete.")

    print("\n" + "="*50)
    print("Welcome to Research-Graph RAG!")
    print("Ask questions about the paper. Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")

    rag = RAGGenerator(index)
    
    while True:
        query = input("\nYour Question: ")
        if query.lower() in ["quit", "exit"]:
            break
        if not query.strip():
            continue
            
        try:
            print("Thinking...")
            answer = rag.generate_answer(query, top_k_chunks=3, top_k_graph=5)
            
            print("\n----- REASONING SUMMARY -----")
            print(answer.get("reasoning_summary", "None"))
            print("\n----- NARRATIVE ANSWER -----")
            print(answer.get("narrative_answer", "None"))
            
            insuff = answer.get("insufficiency", {})
            if insuff.get("flag"):
                print("\n[!] WARNING: Insufficient Evidence.")
                print("Missing fields: ", ", ".join(insuff.get("missing_fields", [])))
            
            print("\n----- STRUCTURED FACTS -----")
            import pprint
            pprint.pprint(answer.get("structured_summary", {}))
            
            print("\n----- EVIDENCE CITED -----")
            evidence = answer.get("evidence", [])
            for ev in evidence:
                print(f" - Pages {ev.get('page_numbers')}: {ev.get('snippet')} [{ev.get('paper_title')}]")
                
        except Exception as e:
            print(f"Error generating answer: {e}")

if __name__ == "__main__":
    main()

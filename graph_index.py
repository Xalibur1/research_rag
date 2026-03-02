import faiss
import numpy as np
import networkx as nx
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
from parser import DocumentChunk

class HybridIndex:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embed_model = SentenceTransformer(model_name)
        
        # Text Chunks
        self.chunks: List[DocumentChunk] = []
        self.chunk_index = None 
        self.bm25 = None
        
        # Graph Nodes
        self.graph = nx.DiGraph()
        self.node_ids: List[str] = []
        self.node_index = None 

    def build_chunk_index(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        if not chunks:
            return
            
        texts = [c.text for c in chunks]
        
        # Dense
        embeddings = self.embed_model.encode(texts, convert_to_numpy=True)
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        dim = embeddings.shape[1]
        self.chunk_index = faiss.IndexFlatIP(dim) # Inner product = cosine sim if normalized
        self.chunk_index.add(embeddings)
        
        # Sparse
        tokenized_corpus = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def build_graph_index(self, graph: nx.DiGraph):
        self.graph = graph
        if not graph.nodes:
            return
            
        node_texts = []
        self.node_ids = []
        for node_id, data in graph.nodes(data=True):
            self.node_ids.append(node_id)
            desc = f"{node_id}: " + " ".join([f"{k}={v}" for k, v in data.items()])
            node_texts.append(desc)
            
        embeddings = self.embed_model.encode(node_texts, convert_to_numpy=True)
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.node_index = faiss.IndexFlatIP(dim)
        self.node_index.add(embeddings)

    def search_chunks(self, query: str, top_k: int = 3, dense_weight: float = 0.5) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks: return []
        
        # Dense
        q_emb = self.embed_model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(q_emb)
        D, I = self.chunk_index.search(q_emb, k=top_k * 3)
        dense_scores = {I[0][j]: D[0][j] for j in range(len(I[0])) if I[0][j] >= 0}
        
        # Sparse
        tokenized_query = query.lower().split()
        bm25_scores_raw = self.bm25.get_scores(tokenized_query)
        # Min-max scale bm25 scores roughly based on top
        max_bm25 = max(bm25_scores_raw) if len(bm25_scores_raw) > 0 and max(bm25_scores_raw) > 0 else 1.0
        
        top_bm25_idx = np.argsort(bm25_scores_raw)[::-1][:top_k * 3]
        sparse_scores = {idx: bm25_scores_raw[idx] / max_bm25 for idx in top_bm25_idx}
        
        # Combine
        combined_scores = {}
        all_idx = set(dense_scores.keys()).union(set(sparse_scores.keys()))
        for idx in all_idx:
            ds = dense_scores.get(idx, 0.0)
            ss = sparse_scores.get(idx, 0.0)
            combined_scores[idx] = (ds * dense_weight) + (ss * (1.0 - dense_weight))
            
        sorted_idx = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.chunks[idx], score) for idx, score in sorted_idx[:top_k]]

    def search_graph(self, query: str, top_k: int = 3) -> Dict[str, Dict[str, Any]]:
        """Returns subgraph node link data for top matching graph nodes."""
        if not self.node_ids or self.node_index is None: return {}
        
        q_emb = self.embed_model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(q_emb)
        D, I = self.node_index.search(q_emb, k=top_k)
        
        neighborhoods = {}
        for idx in I[0]:
            if idx < 0 or idx >= len(self.node_ids): continue
            node_id = self.node_ids[idx]
            # Use undirected graph view to get incoming and outgoing edges
            submap = nx.ego_graph(self.graph.to_undirected(), node_id, radius=1)
            # Filter graph down to a directional view but containing all connections 
            # or just use undirected for context summary
            neighborhoods[node_id] = nx.node_link_data(submap)
            
        return neighborhoods

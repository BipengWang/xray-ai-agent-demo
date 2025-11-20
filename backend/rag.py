from typing import List, Tuple, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    SentenceTransformer = None
    faiss = None


class SimpleRAGStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None or faiss is None:
            print("RAG disabled: sentence-transformers/faiss not installed.")
            self.enabled = False
            return

        self.enabled = True
        self.model = SentenceTransformer(model_name)

        # Toy knowledge base; you can replace with real X-ray / materials docs.
        self.docs = [
            "X-ray absorption spectroscopy (XAS) probes unoccupied electronic states and local structure.",
            "Near-edge features in XAS can be related to oxidation states and coordination geometry.",
            "Extended X-ray absorption fine structure (EXAFS) oscillations encode radial distribution of neighboring atoms.",
            "Synchrotron X-ray sources provide high brightness and tunable energy for advanced spectroscopy.",
        ]
        self.doc_ids = list(range(len(self.docs)))

        embeddings = self.model.encode(self.docs, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, int]]:
        if not self.enabled:
            return []

        q_emb = self.model.encode([query], show_progress_bar=False).astype("float32")
        D, I = self.index.search(q_emb, k)
        results: List[Tuple[str, int]] = []
        for idx in I[0]:
            doc = self.docs[idx]
            doc_id = self.doc_ids[idx]
            results.append((doc, doc_id))
        return results


rag_store: Optional[SimpleRAGStore] = None


def get_rag_store() -> Optional[SimpleRAGStore]:
    global rag_store
    if rag_store is None:
        rag_store = SimpleRAGStore()
    return rag_store if rag_store and rag_store.enabled else None

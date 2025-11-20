from typing import List, Tuple, Optional
import os
import uuid

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from .config import settings

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "xray-rag")

# --- Initialize OpenAI embedding client ---
oai_client = OpenAI(api_key=settings.OPENAI_API_KEY)


def embed_text(texts: List[str]) -> List[List[float]]:
  """
  Use OpenAI embedding model to encode a list of texts.
  """
  resp = oai_client.embeddings.create(
      model=settings.EMBEDDING_MODEL,
      input=texts,
  )
  # resp.data is a list with .embedding attribute
  return [item.embedding for item in resp.data]


class PineconeRAG:
  def __init__(self):
    if not PINECONE_API_KEY:
      print("Pinecone API key not set. RAG disabled.")
      self.enabled = False
      return

    self.pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if not exists (serverless example)
    existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
      print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
      # You need to choose a region supported by your account, e.g. "us-west1-gcp"
      self.pc.create_index(
          name=PINECONE_INDEX_NAME,
          dimension=1536,  # text-embedding-3-small dimension
          metric="cosine",
          spec=ServerlessSpec(cloud="aws", region="us-east-1"),
      )

    self.index = self.pc.Index(PINECONE_INDEX_NAME)
    self.enabled = True

  # ----------------------------------------------
  # Upsert docs
  # ----------------------------------------------
   # ----------------------------------------------
  # Upsert docs (with dedup check)
  # ----------------------------------------------
  def upsert_docs(self, docs: List[str], namespace: str = "default") -> None:
    """
    Upsert docs into Pinecone with deduplication:
    - compute embedding for each doc
    - check if a very similar embedding already exists
    - only insert if NOT found (score < 0.99)
    """
    if not self.enabled or not docs:
      return

    # 1. Create embeddings for all docs
    embeddings = embed_text(docs)

    vectors_to_insert = []

    for text, emb in zip(docs, embeddings):
      # 2. Search top-1 to check similarity
      try:
        existing = self.index.query(
            namespace=namespace,
            vector=emb,
            top_k=1,
            include_metadata=True,
        )
      except Exception as e:
        print("Pinecone query failed during dedup check:", e)
        continue

      # 3. Dedup logic: if extremely similar (cosine score > 0.99), skip
      if existing and existing.matches:
        top_match = existing.matches[0]
        if top_match.score > 0.99:
          print(f"[Dedup] Skipping existing doc: {text[:60]}...")
          continue

      # 4. Insert if new
      vid = str(uuid.uuid4())
      vectors_to_insert.append({
          "id": vid,
          "values": emb,
          "metadata": {"text": text},
      })

    # 5. Actually insert all new docs
    if vectors_to_insert:
      print(f"Pinecone: upserting {len(vectors_to_insert)} new docs...")
      self.index.upsert(
          vectors=vectors_to_insert,
          namespace=namespace,
      )
    else:
      print("Pinecone: no new docs to insert (all duplicates).")

  def retrieve(self, query: str, k: int = 3, namespace: str = "default") -> List[Tuple[str, str]]:
    """
    Returns a list of (text, id)
    """
    if not self.enabled:
        return []

    q_emb = embed_text([query])[0]

    resp = self.index.query(
        namespace=namespace,
        vector=q_emb,
        top_k=k,
        include_metadata=True,
    )

    results: List[Tuple[str, str, float]] = []
    for match in resp.matches:
        text = match.metadata.get("text", "")
        score = match.score  # cosine similarity
        results.append((text, match.id, score))
    return results



# Global RAG instance
_rag: Optional[PineconeRAG] = None


def get_rag_store() -> Optional[PineconeRAG]:
  global _rag
  if _rag is None:
    _rag = PineconeRAG()
    if _rag.enabled:
      # Toy docs: you can replace with X-ray PDFs chunked content later
      toy_docs = [
          "X-ray absorption spectroscopy (XAS) probes unoccupied electronic states and local structure.",
          "Near-edge features in XAS can be related to oxidation states and coordination geometry.",
          "Extended X-ray absorption fine structure (EXAFS) oscillations encode radial distribution of neighboring atoms.",
          "Synchrotron X-ray sources provide high brightness and tunable energy for advanced spectroscopy.",
      ]
      # For simplicity, we upsert on first init. In a real system, you might
      # have a separate offline ingestion script.
      _rag.upsert_docs(toy_docs)
  return _rag if _rag and _rag.enabled else None

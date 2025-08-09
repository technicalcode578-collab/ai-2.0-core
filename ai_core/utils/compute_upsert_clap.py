# ai_core/utils/compute_upsert_clap.py
import json
from pathlib import Path
from ai_core.utils.clap_embedder import CLAPEmbedder

# If you have a chroma client module, import it; otherwise create inline client.
try:
    # try to reuse your existing chroma client if present
    from ai_core.embeddings import collection
    print("[compute_upsert_clap] Using existing chroma collection from ai_core.embeddings")
except Exception:
    # fallback: minimal chorma client initialization
    import chromadb
    from chromadb.config import Settings
    DB_DIR = Path(__file__).resolve().parent.parent / "data" / "vector_db"
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection("ai_core_vectors")

INDEX_FILE = Path(__file__).resolve().parent.parent / "data" / "audio_index.json"

def run_upsert():
    if not INDEX_FILE.exists():
        print("audio_index.json not found. Run audio ingest first.")
        return

    with open(INDEX_FILE, "r") as f:
        index = json.load(f)

    embedder = CLAPEmbedder()
    ids, metadatas, docs, embeddings = [], [], [], []
    for i, entry in enumerate(index):
        path = entry["path"]
        file_name = entry["file_name"]
        print(f"[upsert] embedding {file_name}")
        vec = embedder.embed_audio_path(path).tolist()
        doc_text = entry.get("metadata", {}).get("title", file_name)
        ids.append(f"audio_{i}")
        metadatas.append({"file_name": file_name, "path": path, **entry.get("metadata", {})})
        docs.append(doc_text)
        embeddings.append(vec)

    # Upsert to chroma
    collection.add(
        documents=docs,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )
    # persist if using persistent client
    try:
        collection._client.persist()
    except Exception:
        pass

    print(f"Upserted {len(ids)} audio embeddings into collection.")

if __name__ == "__main__":
    run_upsert()

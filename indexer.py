import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datetime import datetime

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts):
    """Convert a list of texts into vector embeddings."""
    return model.encode(texts, show_progress_bar=False)

def build_index(articles):
    """
    Build a FAISS index from news articles.
    Returns the index, and a list of article metadata (with timestamps).
    """
    texts = []
    for a in articles:
        combined = f"{a['title']}. {a['description']}"
        texts.append(combined)

    print(f"  Embedding {len(texts)} articles...")
    embeddings = embed_texts(texts)
    embeddings = np.array(embeddings).astype('float32')

    # Normalize so cosine similarity = dot product
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity after normalization
    index.add(embeddings)

    # Store metadata separately (FAISS only stores vectors, not text)
    metadata = []
    for i, a in enumerate(articles):
        metadata.append({
            "index": i,
            "title": a["title"],
            "description": a["description"],
            "published_at": a["published_at"],
            "source": a["source"]
        })

    return index, metadata, embeddings

def time_decay(published_at_str, reference_time=None, lam=0.1):
    """
    Compute temporal decay score: e^(-lambda * delta_t)
    delta_t is in days between article date and reference time (now).
    """
    if reference_time is None:
        reference_time = datetime.utcnow()

    try:
        pub_time = datetime.strptime(published_at_str[:19], "%Y-%m-%dT%H:%M:%S")
    except:
        pub_time = reference_time

    delta_days = max((reference_time - pub_time).days, 0)
    return np.exp(-lam * delta_days)

def trag_retrieve(query, index, metadata, embeddings, top_k=5, alpha=0.7, lam=0.05):
    """
    T-RAG retrieval: Score = alpha * semantic_similarity + (1 - alpha) * time_decay
    Returns top_k most relevant + recent articles.
    """
    query_embedding = embed_texts([query])
    query_embedding = np.array(query_embedding).astype('float32')
    faiss.normalize_L2(query_embedding)

    # Get semantic similarity scores for all documents
    all_scores, all_indices = index.search(query_embedding, len(metadata))
    all_scores = all_scores[0]
    all_indices = all_indices[0]

    # Compute T-RAG score for each document
    scored = []
    for rank, idx in enumerate(all_indices):
        if idx == -1:
            continue
        sem_score = float(all_scores[rank])
        temp_score = time_decay(metadata[idx]["published_at"], lam=lam)
        trag_score = alpha * sem_score + (1 - alpha) * temp_score
        scored.append((trag_score, idx))

    # Sort by T-RAG score descending
    scored.sort(reverse=True, key=lambda x: x[0])

    results = []
    for score, idx in scored[:top_k]:
        results.append({
            "score": round(score, 4),
            "title": metadata[idx]["title"],
            "description": metadata[idx]["description"],
            "published_at": metadata[idx]["published_at"],
            "source": metadata[idx]["source"]
        })

    return results
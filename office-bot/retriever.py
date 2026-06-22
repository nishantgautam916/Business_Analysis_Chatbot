from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma

embeddings = VoyageAIEmbeddings(
    model="voyage-3",
    voyage_api_key = "pa-mJkECKqEwxls4lRjlEOfwyxzO8mcVkDjLODyHBkamDR"
)

vectorstore = Chroma(
    persist_directory="./vector_db",
    embedding_function=embeddings
)

def retrieve_context(query: str, score_threshold: float = 1.0):
    """
    Retrieve relevant KPI context for a given query.

    Chroma returns cosine *distance* scores, not similarity scores:
        0.0  = identical
        1.0  = orthogonal (unrelated)
        2.0  = opposite

    So we KEEP results where score < threshold (close matches),
    and DISCARD results where score >= threshold (weak matches).

    A threshold of 1.0 is a safe default — it accepts any document
    that is at least loosely related to the query.
    """

    docs = vectorstore.similarity_search_with_score(
        query,
        k=3
    )

    context = ""
    metrics = []

    for doc, score in docs:

        # FIX: was `score > 0.7` which kept weak matches and dropped good ones.
        # Correct logic: keep only close matches (low distance score).
        if score >= score_threshold:
            continue

        context += doc.page_content + "\n"

        metric = doc.metadata.get("metric")

        if metric and metric not in metrics:
            metrics.append(metric)

    return context, metrics

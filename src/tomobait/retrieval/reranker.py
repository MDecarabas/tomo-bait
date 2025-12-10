import logging
from typing import List, Optional, Tuple

from langchain_core.schema import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class DocumentReranker:
    """Reranks documents using a cross-encoder model"""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        score_threshold: Optional[float] = None,
    ):
        """
        Initialize reranker.

        Args:
            model_name: HuggingFace cross-encoder model name
            score_threshold: Minimum score to keep (0-1)
        """
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.score_threshold = score_threshold
        logger.info("Cross-encoder model loaded")

    def rerank(
        self, query: str, documents: List[Document], top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Return top k results (returns all if None)

        Returns:
            List of (document, score) tuples sorted by score
        """
        if not documents:
            return []

        # Prepare query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]

        # Score with cross-encoder
        scores = self.model.predict(pairs)

        # Combine documents with scores
        doc_scores = list(zip(documents, scores))

        # Filter by threshold
        if self.score_threshold is not None:
            doc_scores = [
                (doc, score)
                for doc, score in doc_scores
                if score >= self.score_threshold
            ]

        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        if top_k:
            return doc_scores[:top_k]
        return doc_scores

    def rerank_and_format(
        self, query: str, documents: List[Document], top_k: int = 3
    ) -> str:
        """
        Rerank documents and format as string.

        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of results to return

        Returns:
            Formatted string with reranked results
        """
        reranked = self.rerank(query, documents, top_k)

        if not reranked:
            return "No relevant documents found."

        formatted_results = []
        for i, (doc, score) in enumerate(reranked, 1):
            content = doc.page_content
            metadata = doc.metadata
            source = metadata.get("source", "Unknown source")

            formatted_results.append(
                f"Result {i} (relevance: {score:.3f}):\n"
                f"Source: {source}\n"
                f"Content:\n{content}\n"
            )

        return "\n---\n".join(formatted_results)

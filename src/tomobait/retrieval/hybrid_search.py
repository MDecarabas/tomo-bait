import logging
from typing import Dict, List, Tuple

import numpy as np
from langchain_core.schema import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines semantic and keyword search using RRF (Reciprocal Rank Fusion)"""

    def __init__(self, vector_store, documents: List[Document], k: int = 3):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: ChromaDB vector store
            documents: List of all documents for BM25 indexing
            k: Number of results to return
        """
        self.vector_store = vector_store
        self.k = k

        # Build BM25 index
        self._build_bm25_index(documents)

    def _build_bm25_index(self, documents: List[Document]):
        """Build BM25 index from documents"""
        logger.info(f"Building BM25 index from {len(documents)} documents")

        # Tokenize documents
        self.documents = documents
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]

        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)

        logger.info("BM25 index built successfully")

    def _semantic_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Perform semantic vector search"""
        results = self.vector_store.similarity_search_with_score(query, k=k)
        # Convert ChromaDB scores to similarity scores (higher is better)
        return [(doc, 1.0 / (1.0 + score)) for doc, score in results]

    def _keyword_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Perform BM25 keyword search"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top k indices
        top_indices = np.argsort(scores)[-k:][::-1]

        # Return documents with scores
        return [(self.documents[i], scores[i]) for i in top_indices]

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[Document, float]],
        keyword_results: List[Tuple[Document, float]],
        k: int = 60,
    ) -> List[Document]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).

        RRF formula: RRF(d) = Σ 1 / (k + rank(d))
        where k is typically 60 (constant to reduce impact of high ranks)

        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            k: RRF constant (default: 60)

        Returns:
            Merged and ranked documents
        """
        # Build RRF scores
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # Add semantic results
        for rank, (doc, score) in enumerate(semantic_results, 1):
            doc_id = doc.page_content[:100]  # Use content prefix as ID
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
            doc_map[doc_id] = doc

        # Add keyword results
        for rank, (doc, score) in enumerate(keyword_results, 1):
            doc_id = doc.page_content[:100]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
            doc_map[doc_id] = doc

        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top k documents
        return [doc_map[doc_id] for doc_id, score in sorted_docs[: self.k]]

    def search(
        self, query: str, semantic_weight: float = 0.7, k: int = None
    ) -> List[Document]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            semantic_weight: Weight for semantic search (0-1)
            k: Number of results (uses self.k if None)

        Returns:
            List of documents ranked by RRF
        """
        k = k or self.k

        # Retrieve more candidates for fusion
        candidate_k = k * 3

        # Perform both searches
        semantic_results = self._semantic_search(query, candidate_k)
        keyword_results = self._keyword_search(query, candidate_k)

        # Apply weights
        semantic_results = [
            (doc, score * semantic_weight) for doc, score in semantic_results
        ]
        keyword_results = [
            (doc, score * (1 - semantic_weight)) for doc, score in keyword_results
        ]

        # Merge using RRF
        merged_results = self._reciprocal_rank_fusion(semantic_results, keyword_results)

        return merged_results[:k]

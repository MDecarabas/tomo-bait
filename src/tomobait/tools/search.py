import logging

from ..retriever import query_vector_db
from .base import ToolMetadata
from .registry import tool_registry

logger = logging.getLogger(__name__)


def semantic_search(query: str, k: int = 3) -> str:
    """
    Perform semantic vector search on documentation.

    Args:
        query: Search query string
        k: Number of results to return (default: 3)

    Returns:
        Formatted string with search results and metadata
    """
    try:
        results = query_vector_db(query, k=k)

        if not results or len(results) == 0:
            return "No relevant documentation found for this query."

        formatted_results = []
        for i, doc in enumerate(results, 1):
            content = doc.page_content
            metadata = doc.metadata

            # Extract source information
            source = metadata.get("source", "Unknown source")

            # Filter out internal config sources for display
            if "config_resource" in source:
                continue

            # Extract URLs from metadata
            urls = []
            for url_key in [
                "documentation",
                "official_page",
                "github",
                "pypi",
                "source",
            ]:
                if url_key in metadata and metadata[url_key]:
                    urls.append(f"{url_key}: {metadata[url_key]}")

            url_section = f"\nURLs: {', '.join(urls)}" if urls else ""

            formatted_results.append(
                f"Result {i}:\nSource: {source}{url_section}\nContent:\n{content}\n"
            )

        if not formatted_results:
            return "No relevant documentation found for this query."

        return "\n---\n".join(formatted_results)

    except Exception as e:
        logger.error(f"Error in semantic_search: {e}")
        return f"Error performing search: {str(e)}"


def keyword_search(query: str, k: int = 3) -> str:
    """
    Perform keyword-based search on documentation using BM25.

    Args:
        query: Search query string
        k: Number of results to return (default: 3)

    Returns:
        Formatted string with search results
    """
    # TODO: Implement BM25 keyword search
    # For now, fallback to semantic search
    return semantic_search(query, k)


def hybrid_search(query: str, k: int = 3, semantic_weight: float = 0.7) -> str:
    """
    Perform hybrid search combining semantic and keyword search.
    Uses Reciprocal Rank Fusion (RRF) to merge results.

    Args:
        query: Search query string
        k: Number of results to return (default: 3)
        semantic_weight: Weight for semantic results (0-1, default: 0.7)

    Returns:
        Formatted string with merged search results
    """
    # TODO: Implement true hybrid search with RRF
    # For now, use semantic search with higher k
    return semantic_search(query, k=k * 2)


# Register search tools
tool_registry.register(
    ToolMetadata(
        name="semantic_search",
        description="Search documentation using semantic vector similarity. Best for conceptual questions and finding related topics.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 3)",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
        category="search",
    ),
    semantic_search,
)

tool_registry.register(
    ToolMetadata(
        name="keyword_search",
        description="Search documentation using keyword matching (BM25). Best for exact terms and specific technical keywords.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query with keywords",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 3)",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
        category="search",
    ),
    keyword_search,
)

tool_registry.register(
    ToolMetadata(
        name="hybrid_search",
        description="Search documentation using both semantic and keyword methods, merging results. Best for comprehensive search coverage.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 3)",
                    "default": 3,
                },
                "semantic_weight": {
                    "type": "number",
                    "description": "Weight for semantic results (0-1, default: 0.7)",
                    "default": 0.7,
                },
            },
            "required": ["query"],
        },
        category="search",
    ),
    hybrid_search,
)

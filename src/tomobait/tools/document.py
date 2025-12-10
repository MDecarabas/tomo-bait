from .base import ToolMetadata
from .registry import tool_registry


def get_document_outline(doc_path: str) -> str:
    """
    Get the table of contents / outline for a documentation file.

    Args:
        doc_path: Path to the documentation file

    Returns:
        Formatted outline with sections and subsections
    """
    # TODO: Implement document outline extraction
    return f"Outline for {doc_path}: (Not yet implemented)"


def get_section(doc_path: str, section_name: str) -> str:
    """
    Retrieve a specific section from a documentation file.

    Args:
        doc_path: Path to the documentation file
        section_name: Name or heading of the section to retrieve

    Returns:
        Content of the specified section
    """
    # TODO: Implement section extraction
    return f"Section '{section_name}' from {doc_path}: (Not yet implemented)"


# Register document tools
tool_registry.register(
    ToolMetadata(
        name="get_document_outline",
        description="Get the table of contents or outline for a documentation file. Useful for understanding document structure.",
        parameters={
            "type": "object",
            "properties": {
                "doc_path": {
                    "type": "string",
                    "description": "Path to the documentation file",
                }
            },
            "required": ["doc_path"],
        },
        category="document",
    ),
    get_document_outline,
)

tool_registry.register(
    ToolMetadata(
        name="get_section",
        description="Retrieve a specific section from a documentation file by section name or heading.",
        parameters={
            "type": "object",
            "properties": {
                "doc_path": {
                    "type": "string",
                    "description": "Path to the documentation file",
                },
                "section_name": {
                    "type": "string",
                    "description": "Name or heading of the section",
                },
            },
            "required": ["doc_path", "section_name"],
        },
        category="document",
    ),
    get_section,
)

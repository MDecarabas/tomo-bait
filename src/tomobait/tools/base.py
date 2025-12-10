from typing import Any, Dict

from pydantic import BaseModel


class ToolMetadata(BaseModel):
    """Metadata for a tool"""

    name: str
    description: str
    parameters: Dict[str, Any]
    category: str = "general"  # search, document, code, comparison
    enabled: bool = True

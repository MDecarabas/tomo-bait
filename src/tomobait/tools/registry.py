from typing import Any, Callable, Dict, List, Optional

from .base import ToolMetadata


class ToolRegistry:
    """Registry for managing tools"""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._metadata: Dict[str, ToolMetadata] = {}

    def register(self, metadata: ToolMetadata, func: Callable):
        """Register a tool"""
        self._tools[metadata.name] = func
        self._metadata[metadata.name] = metadata

    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool by name"""
        return self._tools.get(name)

    def get_metadata(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata"""
        return self._metadata.get(name)

    def list_tools(
        self, category: Optional[str] = None, enabled_only: bool = True
    ) -> List[str]:
        """List available tools"""
        tools = []
        for name, metadata in self._metadata.items():
            if enabled_only and not metadata.enabled:
                continue
            if category and metadata.category != category:
                continue
            tools.append(name)
        return tools

    def get_tool_schemas(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """Get Autogen-compatible tool schemas"""
        schemas = []
        for name, metadata in self._metadata.items():
            if enabled_only and not metadata.enabled:
                continue
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": metadata.name,
                        "description": metadata.description,
                        "parameters": metadata.parameters,
                    },
                }
            )
        return schemas


# Global tool registry
tool_registry = ToolRegistry()

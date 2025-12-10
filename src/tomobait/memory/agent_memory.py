import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MemoryEntry(BaseModel):
    """Single memory entry"""

    timestamp: datetime
    agent_name: str
    content: str
    metadata: Dict[str, Any] = {}
    importance: float = 0.5  # 0-1 scale


class AgentMemory:
    """Shared memory system for agents"""

    def __init__(self, max_entries: int = 100):
        """
        Initialize agent memory.

        Args:
            max_entries: Maximum number of entries to keep
        """
        self.max_entries = max_entries
        self.entries: List[MemoryEntry] = []
        self.knowledge_graph: Dict[str, Any] = {}

    def add_entry(
        self,
        agent_name: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ):
        """Add a memory entry"""
        entry = MemoryEntry(
            timestamp=datetime.now(),
            agent_name=agent_name,
            content=content,
            metadata=metadata or {},
            importance=importance,
        )

        self.entries.append(entry)

        # Prune old entries if exceeding max
        if len(self.entries) > self.max_entries:
            # Keep most important entries
            self.entries.sort(key=lambda e: e.importance, reverse=True)
            self.entries = self.entries[: self.max_entries]
            # Re-sort by timestamp
            self.entries.sort(key=lambda e: e.timestamp)

    def get_recent(
        self, n: int = 10, agent_name: Optional[str] = None
    ) -> List[MemoryEntry]:
        """Get recent memory entries"""
        entries = self.entries

        if agent_name:
            entries = [e for e in entries if e.agent_name == agent_name]

        return entries[-n:]

    def search(self, query: str, n: int = 5) -> List[MemoryEntry]:
        """Search memory entries"""
        # Simple keyword search for now
        # TODO: Implement semantic search over memory
        query_lower = query.lower()

        matching = [e for e in self.entries if query_lower in e.content.lower()]

        # Sort by importance
        matching.sort(key=lambda e: e.importance, reverse=True)

        return matching[:n]

    def update_knowledge_graph(self, entity: str, properties: Dict[str, Any]):
        """Update knowledge graph with entity information"""
        if entity not in self.knowledge_graph:
            self.knowledge_graph[entity] = {}

        self.knowledge_graph[entity].update(properties)

    def get_entity_info(self, entity: str) -> Optional[Dict[str, Any]]:
        """Get information about an entity"""
        return self.knowledge_graph.get(entity)

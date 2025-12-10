from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ConversationTurn(BaseModel):
    """Single conversation turn"""

    role: str  # 'user' or 'assistant' or agent name
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}


class ConversationMemory:
    """Manages conversation context and summarization"""

    def __init__(self, max_turns: int = 50):
        """
        Initialize conversation memory.

        Args:
            max_turns: Maximum conversation turns to keep in memory
        """
        self.max_turns = max_turns
        self.turns: List[ConversationTurn] = []
        self.summary: Optional[str] = None

    def add_turn(
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a conversation turn"""
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )

        self.turns.append(turn)

        # Prune old turns if exceeding max
        if len(self.turns) > self.max_turns:
            # Keep recent turns and update summary
            old_turns = self.turns[: -self.max_turns]
            self.turns = self.turns[-self.max_turns :]

            # TODO: Generate summary of old turns using LLM

    def get_recent_context(self, n: int = 10) -> str:
        """Get recent conversation context as formatted string"""
        recent = self.turns[-n:]

        formatted = []
        for turn in recent:
            formatted.append(f"{turn.role}: {turn.content}")

        return "\n".join(formatted)

    def get_full_context(self) -> str:
        """Get full conversation context including summary"""
        parts = []

        if self.summary:
            parts.append(f"Previous conversation summary:\n{self.summary}\n")

        parts.append(self.get_recent_context(len(self.turns)))

        return "\n".join(parts)

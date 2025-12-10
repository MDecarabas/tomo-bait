import logging

import tiktoken

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages context window for LLM calls"""

    def __init__(
        self,
        model_name: str = "gpt-4",
        max_tokens: int = 8000,
        reserve_tokens: int = 2000,
    ):
        """
        Initialize context manager.

        Args:
            model_name: Name of the model for tokenization
            max_tokens: Total token limit for the context
            reserve_tokens: Tokens to reserve for the answer
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        try:
            self.encoder = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            logger.warning(
                f"Model '{self.model_name}' not found. Using 'cl100k_base' encoding."
            )
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in a string"""
        return len(self.encoder.encode(text))

    def build_context(
        self,
        system_message: str,
        conversation_history: str,
        retrieved_docs: str,
    ) -> str:
        """
        Build a context string that fits within the token limit.

        Prioritizes:
        1. System message
        2. Retrieved documents
        3. Most recent conversation history

        Args:
            system_message: The agent's system message
            conversation_history: Formatted string of recent conversation
            retrieved_docs: Formatted string of retrieved documents

        Returns:
            A context string ready for the LLM prompt.
        """
        context_parts = []
        available_tokens = self.max_tokens - self.reserve_tokens

        # 1. Add system message
        system_tokens = self._count_tokens(system_message)
        if system_tokens < available_tokens:
            context_parts.append(system_message)
            available_tokens -= system_tokens
        else:
            # Truncate system message if it's too long
            # (should not happen with reasonable system messages)
            truncated_sys = self.encoder.decode(
                self.encoder.encode(system_message)[:available_tokens]
            )
            context_parts.append(truncated_sys)
            return "".join(context_parts)

        # 2. Add retrieved documents
        docs_tokens = self._count_tokens(retrieved_docs)
        if docs_tokens < available_tokens:
            context_parts.append(retrieved_docs)
            available_tokens -= docs_tokens
        else:
            # Truncate documents
            truncated_docs = self.encoder.decode(
                self.encoder.encode(retrieved_docs)[:available_tokens]
            )
            context_parts.append(truncated_docs)
            available_tokens = 0

        # 3. Add conversation history (most recent first)
        if available_tokens > 0:
            history_tokens = self._count_tokens(conversation_history)
            if history_tokens < available_tokens:
                context_parts.insert(
                    1, conversation_history
                )  # Insert after system message
            else:
                # Truncate from the beginning (oldest messages)
                encoded_history = self.encoder.encode(conversation_history)
                truncated_history_encoded = encoded_history[-available_tokens:]
                truncated_history = self.encoder.decode(truncated_history_encoded)
                context_parts.insert(1, truncated_history)

        return "\n\n".join(context_parts)

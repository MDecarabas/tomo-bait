import os
import time
from typing import Annotated, Optional

import autogen
from autogen import LLMConfig
from dotenv import load_dotenv

from .config import get_config, reload_config
from .retriever import get_documentation_retriever

load_dotenv()


class LLMNotConfiguredError(Exception):
    """Raised when LLM is not properly configured (missing API key)."""
    pass


# --- Tool Definition (static, doesn't need API key) ---
query_documentation_tool_dict = {
    "type": "function",
    "function": {
        "name": "query_documentation",
        "description": "Search the project documentation for a given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for the documentation"
                }
            },
            "required": ["query"]
        }
    }
}


class AgentManager:
    """
    Manages LLM agents with lazy initialization.
    Agents are only created when first needed, allowing the backend to start
    even if API keys are not configured.
    """

    def __init__(self):
        self._technician_agent: Optional[autogen.AssistantAgent] = None
        self._worker_agent: Optional[autogen.UserProxyAgent] = None
        self._retriever = None
        self._initialized = False
        self._initialization_error: Optional[str] = None

    def _check_api_key(self) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Check if the configured API key is available.
        Returns: (is_available, api_key, error_message)

        Checks in order:
        1. Direct api_key in config (for ANL Argo username or testing)
        2. Environment variable from api_key_env
        """
        config = get_config()

        # First check for direct api_key in config
        if config.llm.api_key:
            return True, config.llm.api_key, None

        # Then check environment variable
        api_key_env = config.llm.api_key_env
        if not api_key_env:
            return False, None, "No api_key or api_key_env configured in config.yaml"

        api_key = os.getenv(api_key_env)
        if not api_key:
            return False, None, f"{api_key_env} environment variable not set"

        return True, api_key, None

    def get_llm_status(self) -> dict:
        """
        Get the current LLM configuration status.
        Returns a dict with provider info and availability.
        """
        config = get_config()
        is_available, _, error = self._check_api_key()

        return {
            "provider": config.llm.api_type,
            "model": config.llm.model,
            "api_key_env": config.llm.api_key_env,
            "available": is_available,
            "error": error,
            "initialized": self._initialized,
        }

    def _initialize(self) -> None:
        """
        Initialize the agents. Called lazily on first use.
        Raises LLMNotConfiguredError if API key is missing.
        """
        if self._initialized:
            return

        # Reload config to get latest settings
        config = reload_config()

        # Check API key
        is_available, api_key, error = self._check_api_key()
        if not is_available:
            self._initialization_error = error
            raise LLMNotConfiguredError(error)

        # Build LLM config
        llm_config_dict = {
            "api_type": config.llm.api_type,
            "model": config.llm.model,
            "api_key": api_key,
            "tools": [query_documentation_tool_dict],
        }

        # Add base_url if configured (e.g., for ANL Argo)
        if config.llm.base_url:
            llm_config_dict["base_url"] = config.llm.base_url

        llm_config = LLMConfig(config_list=[llm_config_dict])

        # Initialize retriever
        self._retriever = get_documentation_retriever()

        # Create agents
        self._technician_agent = autogen.AssistantAgent(
            "doc_expert",
            llm_config=llm_config,
            system_message=config.llm.system_message
        )

        self._worker_agent = autogen.UserProxyAgent(
            "tool_worker",
            llm_config=False,
            human_input_mode="NEVER",
            is_termination_msg=lambda msg: not msg.get("tool_calls"),
            code_execution_config=False,
        )

        # Register the tool
        @self._worker_agent.register_for_execution(name="query_documentation")
        def query_documentation(
            query: Annotated[str, "The search query for the documentation"]
        ) -> str:
            return self._query_documentation_impl(query)

        self._initialized = True
        self._initialization_error = None
        print(f"✅ Agents initialized with {config.llm.api_type}/{config.llm.model}")

    def _query_documentation_impl(self, query: str) -> str:
        """
        Implementation of the query_documentation tool.
        """
        print(f"\n--- TOOL: Querying for '{query}' ---")

        results = self._retriever.invoke(query)

        # Format each chunk with its metadata (including source URLs)
        formatted_chunks = []
        for doc in results:
            chunk = doc.page_content

            # Collect source information and URLs from metadata
            sources = []

            # Add file source if available
            if 'source' in doc.metadata:
                source_path = doc.metadata['source']
                # Only show source path for non-config resources
                if 'config_resources' not in str(source_path):
                    sources.append(f"Source: {source_path}")

            # Add relevant URLs from metadata (prioritize web-accessible links)
            url_fields = [
                'documentation', 'docs', 'official_page',
                'website', 'github', 'pypi', 'url'
            ]
            for field in url_fields:
                if field in doc.metadata and doc.metadata[field]:
                    url = doc.metadata[field]
                    field_name = field.replace('_', ' ').title()
                    sources.append(f"{field_name}: {url}")

            # Combine content with source information
            if sources:
                chunk_with_source = f"{chunk}\n\n[Sources: {' | '.join(sources)}]"
            else:
                chunk_with_source = chunk

            formatted_chunks.append(chunk_with_source)

        context_str = "\n\n---\n\n".join(formatted_chunks)

        print(f"--- TOOL: Found {len(results)} chunks. ---")
        return context_str

    def reset(self) -> None:
        """
        Reset the agent manager to allow re-initialization.
        Call this after config changes to pick up new settings.
        """
        self._technician_agent = None
        self._worker_agent = None
        self._retriever = None
        self._initialized = False
        self._initialization_error = None
        print("🔄 Agent manager reset")

    def run_chat(self, user_question: str, max_retries: int = 3) -> str:
        """
        Run a chat between agents to answer a user's question.
        Includes retry logic with exponential backoff for API errors.

        Args:
            user_question: The user's query
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            The agent's final answer

        Raises:
            LLMNotConfiguredError: If API key is not configured
        """
        # Lazy initialization
        self._initialize()

        print("Starting agent chat...")

        for attempt in range(max_retries):
            try:
                chat_result = self._worker_agent.initiate_chat(
                    recipient=self._technician_agent,
                    message=(
                        f"Please answer this question: '{user_question}'. "
                        "You *must* use the 'query_documentation' tool to find the "
                        "relevant context first. "
                        "Provide a concise but complete answer (2-3 paragraphs). "
                        "If the question asks 'how to' do something, provide "
                        "step-by-step instructions as a numbered list. "
                        "Include relevant source links from the context at the end "
                        "of your response."
                    ),
                )

                final_answer = chat_result.summary
                if final_answer:
                    print("\n--- FINAL ANSWER ---")
                    print(final_answer)
                    return final_answer
                return "Sorry, I couldn't find an answer."

            except Exception as e:
                error_msg = str(e)
                is_retryable = (
                    "503" in error_msg or
                    "overloaded" in error_msg.lower() or
                    "429" in error_msg or
                    "rate limit" in error_msg.lower()
                )

                if is_retryable and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(
                        f"⚠️  API error (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Error after {attempt + 1} attempts: {error_msg}")
                    raise

        return "Sorry, I couldn't process your request after multiple attempts."


# Global agent manager instance
_agent_manager = AgentManager()


def get_agent_manager() -> AgentManager:
    """Get the global agent manager instance."""
    return _agent_manager


def get_llm_status() -> dict:
    """Get the current LLM configuration status."""
    return _agent_manager.get_llm_status()


def run_agent_chat(user_question: str, max_retries: int = 3) -> str:
    """
    Convenience function to run agent chat.
    Raises LLMNotConfiguredError if API key is not configured.
    """
    return _agent_manager.run_chat(user_question, max_retries)


def reset_agents() -> None:
    """Reset agents to pick up new configuration."""
    _agent_manager.reset()

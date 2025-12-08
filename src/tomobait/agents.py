import os
import time
from typing import Annotated

import autogen
from autogen import LLMConfig
from dotenv import load_dotenv

from .config import get_config
from .retriever import get_documentation_retriever

load_dotenv()

# Load configuration
config = get_config()

# --- 1. Interchangeable LLM Config ---
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

# Handle LLM provider configuration
# All providers now use a unified OpenAI-compatible approach
api_key = os.getenv(config.llm.api_key_env) if config.llm.api_key_env else None
if not api_key:
    print(f"❌ ERROR: {config.llm.api_key_env} environment variable not set.")
    exit()

# Build LLM config - works for all OpenAI-compatible providers
# (Gemini, OpenAI, Anthropic, Azure, and ANL Argo with base_url)
llm_config_dict = {
    "api_type": config.llm.api_type,
    "model": config.llm.model,
    "api_key": api_key,
    "tools": [query_documentation_tool_dict],
}

# Add base_url if configured (e.g., for ANL Argo: https://apps-dev.inside.anl.gov/argoapi/v1/)
if config.llm.base_url:
    llm_config_dict["base_url"] = config.llm.base_url

llm_config = LLMConfig(config_list=[llm_config_dict])

# --- 2. Load Retriever (from Phase 1) ---
retriever = get_documentation_retriever()


# --- 3. Define Agents ---
technician_agent = autogen.AssistantAgent(
    "doc_expert",
    llm_config=llm_config,
    system_message=config.llm.system_message
)

worker_agent = autogen.UserProxyAgent(
    "tool_worker",
    llm_config=False,
    human_input_mode="NEVER",
    # Terminate the conversation when the other agent sends a message
    # that doesn't contain a tool call.
    is_termination_msg=lambda msg: not msg.get("tool_calls"),
    code_execution_config=False,
)

@worker_agent.register_for_execution(name="query_documentation")
def query_documentation(
    query: Annotated[str, "The search query for the documentation"]
) -> str:
    """
    A tool that takes a user's query, retrieves relevant
    document chunks, and returns them as a single string with source links.
    """
    print(f"\n--- TOOL: Querying for '{query}' ---")

    results = retriever.invoke(query)

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
                # Format the field name nicely
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

def run_agent_chat(user_question: str, max_retries: int = 3) -> str:
    """
    Initializes and runs a chat between agents to answer a user's question.
    Includes retry logic with exponential backoff for API errors.

    Args:
        user_question: The user's query
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        The agent's final answer
    """
    print("Starting agent chat...")

    for attempt in range(max_retries):
        try:
            chat_result = worker_agent.initiate_chat(
                recipient=technician_agent,
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

            # The summary is the last message that was sent in the chat.
            # In our case, this is the final answer from the technician agent.
            final_answer = chat_result.summary
            if final_answer:
                print("\n--- FINAL ANSWER ---")
                print(final_answer)
                return final_answer
            return "Sorry, I couldn't find an answer."

        except Exception as e:
            error_msg = str(e)
            is_retryable = ("503" in error_msg or "overloaded" in error_msg.lower() or
                          "429" in error_msg or "rate limit" in error_msg.lower())

            if is_retryable and attempt < max_retries - 1:
                # Exponential backoff: 2^attempt seconds
                wait_time = 2 ** attempt
                print(
                    f"⚠️  API error (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
                continue
            else:
                # Not retryable or out of retries
                print(f"❌ Error after {attempt + 1} attempts: {error_msg}")
                raise

    return "Sorry, I couldn't process your request after multiple attempts."

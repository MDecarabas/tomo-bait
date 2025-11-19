import os
from typing import Annotated

import autogen
from autogen import LLMConfig
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import backup_config, get_config, register_reload_callback, reload_config, save_config, TomoBaitConfig
from .config_generator import generate_config_from_prompt, config_dict_to_yaml
from .config_watcher import start_config_watcher
from .retriever import get_documentation_retriever

load_dotenv()

# Load configuration
config = get_config()

# --- FastAPI App ---
api = FastAPI()

class ChatQuery(BaseModel):
    query: str

class ConfigResponse(BaseModel):
    config: dict

class GenerateConfigRequest(BaseModel):
    prompt: str

class GenerateConfigResponse(BaseModel):
    yaml_config: str
    config_dict: dict

@api.post("/chat")
async def chat_endpoint(chat_query: ChatQuery):
    """
    Endpoint to receive a query and return the agent's response.
    """
    answer = run_agent_chat(chat_query.query)
    return {"response": answer}

@api.get("/config")
async def get_config_endpoint():
    """
    Get current configuration.
    """
    return {"config": config.model_dump()}

@api.post("/config")
async def update_config_endpoint(new_config: dict):
    """
    Update configuration (requires restart to apply).
    """
    # This is a placeholder - in production you'd want to validate and save
    return {"message": "Configuration updated. Restart backend to apply changes."}

@api.post("/generate-config")
async def generate_config_endpoint(request: GenerateConfigRequest):
    """
    Generate a configuration from natural language prompt using Gemini.
    """
    try:
        # Generate config using Gemini
        config_dict = generate_config_from_prompt(request.prompt)

        # Convert to YAML
        yaml_config = config_dict_to_yaml(config_dict)

        return GenerateConfigResponse(
            yaml_config=yaml_config,
            config_dict=config_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate config: {str(e)}")

@api.post("/apply-config")
async def apply_config_endpoint(new_config: dict):
    """
    Apply a new configuration after backing up the old one.
    """
    try:
        # Backup current config
        backup_path = backup_config()

        # Validate and save new config
        validated_config = TomoBaitConfig(**new_config)
        save_config(validated_config)

        return {
            "message": "Configuration applied successfully!",
            "backup_path": backup_path
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")

# --- 1. Interchangeable LLM Config ---
api_key = os.getenv(config.llm.api_key_env)
if not api_key:
    print(f"❌ ERROR: {config.llm.api_key_env} environment variable not set.")
    exit()

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

llm_config = LLMConfig(
    config_list=[
        {
            "api_type": config.llm.api_type,
            "model": config.llm.model,
            "api_key": api_key,
            "tools": [query_documentation_tool_dict],
        }
    ]
)

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
    document chunks, and returns them as a single string.
    """
    print(f"\n--- TOOL: Querying for '{query}' ---")
    
    results = retriever.invoke(query)
    
    context_str = "\n\n---\n\n".join(
        [doc.page_content for doc in results]
    )
    
    print(f"--- TOOL: Found {len(results)} chunks. ---")
    return context_str

def run_agent_chat(user_question: str) -> str:
    """
    Initializes and runs a chat between agents to answer a user's question.
    """
    print("Starting agent chat...")
    chat_result = worker_agent.initiate_chat(
        recipient=technician_agent,
        message=(
            f"Please answer this question: '{user_question}'. "
            "You *must* use the 'query_documentation' tool to "
            "find the relevant context first."
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


# --- Startup Event: Initialize Config Watcher ---
@api.on_event("startup")
async def startup_event():
    """Initialize config file watcher on startup."""
    def on_config_reload():
        """Callback for config reload."""
        print("🔄 Config reloaded in backend")
        # Note: We reload config but don't recreate agents/retriever
        # A full restart would be needed for those changes
        reload_config()

    # Start watching config file
    start_config_watcher(callback=on_config_reload)
    print("✅ Config watcher started")


if __name__ == '__main__':
    USER_QUESTION = "What epics devices are connected to the beamline"
    run_agent_chat(USER_QUESTION)

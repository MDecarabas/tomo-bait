from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agents.base import AssistantAgentFactory, ToolWorkerFactory
from .agents.registry import agent_registry
from .config import (
    TomoBaitConfig,
    backup_config,
    get_config,
    reload_config,
    save_config,
)
from .config_watcher import start_config_watcher
from .llm_config import LLMNotConfiguredError, get_llm_config, get_llm_status
from .orchestration.orchestrator import AgentOrchestrator
from .orchestration.workflows import (
    beamline_expert_workflow,
    iterative_refinement_workflow,
    standard_qa_workflow,
)
from .tools import document, search  # noqa: F401
from .tools.registry import tool_registry

load_dotenv()

# Load configuration
config = get_config()

# --- Agent and Tool Initialization ---


def initialize_agents_and_tools(config: TomoBaitConfig):
    """Initialize all agents and tools from configuration."""
    agent_registry.clear_instances()
    llm_config = get_llm_config(config)

    # Register factories
    agent_registry.register_factory("assistant", AssistantAgentFactory())
    agent_registry.register_factory("user_proxy", ToolWorkerFactory())

    # Register agent configs
    if config.agents:
        for agent_config_model in config.agents.agents:
            agent_registry.register_config(agent_config_model.name, agent_config_model)

    # Create all agents
    for agent_name in agent_registry.list_agents():
        agent_registry.create_agent(agent_name, llm_config)

    # Register tools to tool worker
    tool_worker = agent_registry.get_agent("tool_worker")
    if tool_worker:
        for tool_name in tool_registry.list_tools():
            tool_func = tool_registry.get_tool(tool_name)
            tool_worker.register_function(tool_func, name=tool_name)

    print("✅ Agents and tools initialized.")


# Initialize orchestrator
orchestrator = AgentOrchestrator(agent_registry, tool_registry)
orchestrator.register_workflow("standard_qa", standard_qa_workflow)
orchestrator.register_workflow("beamline_expert", beamline_expert_workflow)
orchestrator.register_workflow("iterative_refinement", iterative_refinement_workflow)

# Initialize agents and tools on startup
initialize_agents_and_tools(config)


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


@api.get("/health")
async def health_check():
    """
    Health check endpoint that reports LLM availability.
    Returns 200 even if LLM is not configured, but indicates status.
    """
    llm_status = get_llm_status()
    return {
        "status": "healthy",
        "llm": llm_status,
    }


@api.post("/chat")
async def chat_endpoint(chat_query: ChatQuery):
    """
    Endpoint to receive a query and return the agent's response.
    """
    try:
        # TODO: Add workflow routing based on query analysis
        workflow_name = (
            config.agents.default_workflow if config.agents else "standard_qa"
        )

        result = await orchestrator.execute_workflow(
            workflow_name=workflow_name,
            user_message=chat_query.query,
        )
        return {"response": result.get("answer", "No answer found.")}
    except LLMNotConfiguredError as e:
        llm_status = get_llm_status()
        api_key_env = llm_status.get("api_key_env")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "llm_not_configured",
                "message": f"LLM not available: {str(e)}",
                "hint": f"Set {api_key_env} or configure a different provider",
                "provider": llm_status.get("provider"),
                "model": llm_status.get("model"),
            },
        )
    except Exception as e:
        error_msg = str(e)
        if "503" in error_msg or "overloaded" in error_msg.lower():
            raise HTTPException(
                status_code=503,
                detail=(
                    "The AI service is currently overloaded. "
                    "Please try again in a moment."
                ),
            )
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please wait a moment before trying again.",
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while processing your request: {error_msg}",
            )


@api.get("/config")
async def get_config_endpoint():
    """
    Get current configuration.
    """
    return {"config": get_config().model_dump()}


@api.post("/apply-config")
async def apply_config_endpoint(new_config: dict):
    """
    Apply a new configuration after backing up the old one.
    """
    try:
        backup_path = backup_config()
        validated_config = TomoBaitConfig(**new_config)
        save_config(validated_config)
        reset_agents()
        return {
            "message": "Configuration applied successfully!",
            "backup_path": backup_path,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")


@api.post("/reset-agents")
async def reset_agents_endpoint():
    """
    Reset agents to pick up new configuration.
    """
    reset_agents()
    llm_status = get_llm_status()
    return {
        "message": "Agents reset. Will reinitialize on next chat request.",
        "llm": llm_status,
    }


def reset_agents():
    """Reset agents to pick up new configuration."""
    print("🔄 Resetting agents...")
    config = reload_config()
    initialize_agents_and_tools(config)


# --- Startup Event: Initialize Config Watcher ---
@api.on_event("startup")
async def startup_event():
    """Initialize config file watcher on startup."""

    def on_config_reload(config):
        """Callback for config reload."""
        print("🔄 Config reloaded in backend")
        reset_agents()

    start_config_watcher(callback=lambda: on_config_reload(reload_config()))
    print("✅ Config watcher started")

    llm_status = get_llm_status()
    if llm_status["available"]:
        print(f"✅ LLM configured: {llm_status['provider']}/{llm_status['model']}")
    else:
        print(f"⚠️  LLM not available: {llm_status['error']}")
        print(
            f"   Set {llm_status.get('api_key_env')} or configure a different provider"
        )

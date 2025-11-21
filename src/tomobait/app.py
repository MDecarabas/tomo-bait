import os
from typing import Annotated

import autogen
from autogen import LLMConfig
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import backup_config, get_config, save_config, TomoBaitConfig
from .config_generator import generate_config_from_prompt, config_dict_to_yaml
from .agents import run_agent_chat

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

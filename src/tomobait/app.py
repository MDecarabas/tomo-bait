
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from .agents import run_agent_chat
from .config import get_config

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

def main():
    """
    Main function to run the FastAPI application using uvicorn.
    """
    uvicorn.run(api, host="127.0.0.1", port=8001)


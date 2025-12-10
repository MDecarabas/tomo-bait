from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from autogen import AssistantAgent, UserProxyAgent
from pydantic import BaseModel


class AgentConfig(BaseModel):
    """Configuration for a single agent"""

    name: str
    type: str  # 'assistant' or 'user_proxy'
    system_message: str
    llm_config: Optional[Dict[str, Any]] = None
    max_consecutive_auto_reply: int = 10
    human_input_mode: str = "NEVER"
    code_execution_config: bool = False
    description: str = ""
    enabled: bool = True


class BaseAgentFactory(ABC):
    """Base class for agent factories"""

    @abstractmethod
    def create_agent(self, config: AgentConfig, llm_config: Dict[str, Any]) -> Any:
        """Create an agent instance"""
        pass

    @abstractmethod
    def get_default_config(self) -> AgentConfig:
        """Get default configuration for this agent type"""
        pass


class AssistantAgentFactory(BaseAgentFactory):
    """Factory for creating AssistantAgent instances"""

    def create_agent(
        self, config: AgentConfig, llm_config: Dict[str, Any]
    ) -> AssistantAgent:
        return AssistantAgent(
            name=config.name,
            system_message=config.system_message,
            llm_config=llm_config,
            max_consecutive_auto_reply=config.max_consecutive_auto_reply,
        )


class ToolWorkerFactory(BaseAgentFactory):
    """Factory for creating UserProxyAgent tool workers"""

    def create_agent(
        self, config: AgentConfig, llm_config: Dict[str, Any]
    ) -> UserProxyAgent:
        return UserProxyAgent(
            name=config.name,
            system_message=config.system_message,
            llm_config=False,
            human_input_mode=config.human_input_mode,
            max_consecutive_auto_reply=config.max_consecutive_auto_reply,
            code_execution_config=config.code_execution_config,
        )

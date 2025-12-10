from typing import Any, Dict, List, Optional

from .base import AgentConfig, BaseAgentFactory


class AgentRegistry:
    """Registry for managing agent factories and instances"""

    def __init__(self):
        self._factories: Dict[str, BaseAgentFactory] = {}
        self._configs: Dict[str, AgentConfig] = {}
        self._instances: Dict[str, Any] = {}

    def register_factory(self, agent_type: str, factory: BaseAgentFactory):
        """Register an agent factory"""
        self._factories[agent_type] = factory

    def register_config(self, agent_name: str, config: AgentConfig):
        """Register agent configuration"""
        self._configs[agent_name] = config

    def create_agent(self, agent_name: str, llm_config: Dict[str, Any]) -> Any:
        """Create an agent instance"""
        if agent_name in self._instances:
            return self._instances[agent_name]

        config = self._configs.get(agent_name)
        if not config or not config.enabled:
            raise ValueError(f"No enabled config found for agent: {agent_name}")

        factory = self._factories.get(config.type)
        if not factory:
            raise ValueError(f"No factory registered for agent type: {config.type}")

        agent = factory.create_agent(config, llm_config)
        self._instances[agent_name] = agent
        return agent

    def list_agents(self) -> List[str]:
        """List all registered agent names"""
        return [name for name, config in self._configs.items() if config.enabled]

    def get_agent(self, agent_name: str) -> Optional[Any]:
        """Get an existing agent instance"""
        return self._instances.get(agent_name)

    def clear_instances(self):
        """Clear all agent instances (useful for config reload)"""
        self._instances.clear()


# Global registry instance
agent_registry = AgentRegistry()

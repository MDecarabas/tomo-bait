import os
from typing import Any, Dict, Optional

from .config import TomoBaitConfig, get_config


class LLMNotConfiguredError(Exception):
    """Raised when LLM is not properly configured (missing API key)."""

    pass


def _check_api_key(config: TomoBaitConfig) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Check if the configured API key is available.
    Returns: (is_available, api_key, error_message)

    Checks in order:
    1. Direct api_key in config
    2. Environment variable from api_key_env
    """
    if config.llm.api_key:
        return True, config.llm.api_key, None

    api_key_env = config.llm.api_key_env
    if not api_key_env:
        return False, None, "No api_key or api_key_env configured"

    api_key = os.getenv(api_key_env)
    if not api_key:
        return False, None, f"{api_key_env} environment variable not set"

    return True, api_key, None


def get_llm_status() -> dict:
    """
    Get the current LLM configuration status.
    Returns a dict with provider info and availability.
    """
    config = get_config()
    is_available, _, error = _check_api_key(config)

    return {
        "provider": config.llm.api_type,
        "model": config.llm.model,
        "api_key_env": config.llm.api_key_env,
        "available": is_available,
        "error": error,
    }


def get_llm_config(config: TomoBaitConfig) -> Dict[str, Any]:
    """
    Get the LLM configuration for autogen.
    Raises LLMNotConfiguredError if API key is missing.
    """
    is_available, api_key, error = _check_api_key(config)
    if not is_available:
        raise LLMNotConfiguredError(error)

    llm_config_dict = {
        "api_type": config.llm.api_type,
        "model": config.llm.model,
        "api_key": api_key,
    }

    if config.llm.base_url:
        llm_config_dict["base_url"] = config.llm.base_url

    return llm_config_dict

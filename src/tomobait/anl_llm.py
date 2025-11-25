"""
ANL Argo LLM integration for Autogen.

This module provides a custom model client that translates between Autogen's
OpenAI-compatible interface and ANL Argo's custom API format.
"""

from typing import Any, Dict, List, Optional

import requests
from autogen import ModelClient


class ANLArgoClient(ModelClient):
    """
    Custom model client for ANL Argo API.

    Autogen expects an OpenAI-compatible interface, but ANL Argo has a custom
    request/response format. This client bridges the gap.
    """

    def __init__(
        self,
        api_url: str,
        user: str,
        model: str,
        temperature: float = 0.1,
        top_p: float = 0.1,
        **kwargs,
    ):
        """
        Initialize the ANL Argo client.

        Args:
            api_url: The ANL Argo API endpoint URL
            user: The ANL username for API requests
            model: The model name to use
            temperature: Sampling temperature (0.0-1.0)
            top_p: Top-p sampling parameter (0.0-1.0)
        """
        self.api_url = api_url
        self.user = user
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    def create(self, params: Dict[str, Any]) -> Any:
        """
        Create a completion using the ANL Argo API.

        This method is called by Autogen and expects OpenAI-compatible parameters.
        We translate them to ANL's format.

        Args:
            params: Parameters from Autogen (OpenAI-compatible format)

        Returns:
            Response in OpenAI-compatible format
        """
        # Extract messages from params
        messages = params.get("messages", [])

        # Convert messages to ANL format
        # ANL expects a single prompt string, so we concatenate messages
        prompt = self._messages_to_prompt(messages)

        # Get other parameters
        temperature = params.get("temperature", self.temperature)
        stop = params.get("stop", [])

        # Make request to ANL API
        anl_request = {
            "user": self.user,
            "model": self.model,
            "prompt": [prompt],  # ANL expects a list of prompts
            "system": "",
            "stop": stop if stop else [],
            "temperature": temperature,
        }

        try:
            response = requests.post(self.api_url, json=anl_request, timeout=60)
            response.raise_for_status()

            anl_response = response.json()
            response_text = anl_response.get("response", "")

            # Convert to OpenAI-compatible format for Autogen
            return self._create_openai_response(response_text, params)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"ANL Argo API request failed: {str(e)}")
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Invalid ANL Argo API response: {str(e)}")

    def message_retrieval(self, response: Any) -> List[str]:
        """
        Retrieve messages from the response.

        Args:
            response: The response object from create()

        Returns:
            List of message strings
        """
        # Extract the content from our OpenAI-compatible response
        choices = response.get("choices", [])
        if not choices:
            return []

        message = choices[0].get("message", {})
        content = message.get("content", "")

        return [content] if content else []

    def cost(self, response: Any) -> float:
        """
        Calculate the cost of the API call.

        Args:
            response: The response object from create()

        Returns:
            Cost in dollars (0.0 for ANL Argo since it's internal)
        """
        # ANL Argo is an internal service, so no cost
        return 0.0

    @staticmethod
    def get_usage(response: Any) -> Dict[str, int]:
        """
        Get token usage from the response.

        Args:
            response: The response object from create()

        Returns:
            Dictionary with token usage statistics
        """
        # ANL API doesn't provide token usage, so we estimate
        usage = response.get("usage", {})
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI-style messages to a single prompt string.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Concatenated prompt string
        """
        prompt_parts = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "tool":
                # Handle tool responses
                prompt_parts.append(f"Tool: {content}")

        return "\n\n".join(prompt_parts)

    def _create_openai_response(
        self, response_text: str, original_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create an OpenAI-compatible response structure.

        Args:
            response_text: The response text from ANL API
            original_params: The original parameters passed to create()

        Returns:
            OpenAI-compatible response dictionary
        """
        # Estimate token counts (rough approximation)
        prompt_tokens = sum(len(m.get("content", "").split()) for m in original_params.get("messages", []))
        completion_tokens = len(response_text.split())

        return {
            "id": "anl-argo-response",
            "object": "chat.completion",
            "created": 0,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


def create_anl_llm_config(
    api_url: str,
    user: str,
    model: str,
    temperature: float = 0.1,
    top_p: float = 0.1,
) -> Dict[str, Any]:
    """
    Create an Autogen LLM config for ANL Argo.

    Args:
        api_url: The ANL Argo API endpoint URL
        user: The ANL username for API requests
        model: The model name to use
        temperature: Sampling temperature
        top_p: Top-p sampling parameter

    Returns:
        Autogen-compatible LLM config dictionary
    """
    client = ANLArgoClient(
        api_url=api_url,
        user=user,
        model=model,
        temperature=temperature,
        top_p=top_p,
    )

    return {
        "model": model,
        "model_client_cls": lambda config: client,
        "temperature": temperature,
    }

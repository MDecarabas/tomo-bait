"""
AI-powered configuration generator using Gemini.
"""

import os
from typing import Dict, Optional

import yaml
from google import genai
from google.genai import types

from .config import TomoBaitConfig

# System prompt for config generation
SYSTEM_PROMPT = """
You are a configuration generator for TomoBait, a RAG system for 
tomography beamline documentation.

Your task is to generate a valid YAML configuration based on the user's 
natural language description.

The configuration structure is:
```yaml
documentation:
  git_repos: [list of git repository URLs]
  local_folders: [list of local folder paths]
  docs_output_dir: string (where to store cloned documentation)
  sphinx_build_html_path: string (path to built Sphinx HTML)

retriever:
  db_path: string (ChromaDB storage path)
  embedding_model: string (
    HuggingFace model name, default: sentence-transformers/all-MiniLM-L6-v2
  )
  k: integer (number of documents to retrieve, 1-20)
  search_type: string (similarity, mmr, or similarity_score_threshold)
  score_threshold: float or null (0.0-1.0, for similarity_score_threshold)

llm:
  api_key: string or null (direct API key, for ANL Argo use your ANL username)
  api_key_env: string or null (
    environment variable name, e.g., GEMINI_API_KEY, OPENAI_API_KEY
  )
  model: string (model name: gemini-2.5-flash, gpt-4, claude-3-opus, gpt4o, etc.)
  api_type: string (google, openai, azure, anthropic)
  base_url: string or null (custom base URL for OpenAI-compatible APIs like ANL Argo)
  system_message: string (system prompt for the agent)

text_processing:
  chunk_size: integer (100-5000, default: 1000)
  chunk_overlap: integer (0-1000, default: 200)

server:
  backend_host: string (default: 127.0.0.1)
  backend_port: integer (default: 8001)
  frontend_host: string (default: 0.0.0.0)
  frontend_port: integer (default: 8000)
```

IMPORTANT RULES:
1. Output ONLY valid YAML, no explanations or markdown code blocks
2. Use appropriate defaults when not specified
3. For local_folders, use absolute paths
4. For git_repos, use full HTTPS URLs
5. embedding_model should be a valid HuggingFace model
6. Validate that k is between 1 and 20
7. If user mentions "high accuracy", increase k and use smaller chunks
8. If user mentions "fast", use lower k and larger chunks
9. Always include all sections, even if using defaults
10. For OpenAI: use api_type="openai", api_key_env="OPENAI_API_KEY"
11. For Anthropic: use api_type="anthropic", api_key_env="ANTHROPIC_API_KEY"
12. For Azure: use api_type="azure", api_key_env="AZURE_OPENAI_API_KEY"
13. For ANL Argo: use api_type="openai", api_key="username", 
    base_url="https://apps-dev.inside.anl.gov/argoapi/v1/"

Example user request: 
"I want to index local documentation in /data/tomo with high accuracy"
Expected output:
```yaml
documentation:
  git_repos: []
  local_folders:
  - /data/tomo
  docs_output_dir: tomo_documentation
  sphinx_build_html_path: tomo_documentation/docs/_build/html

retriever:
  db_path: ./chroma_db
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  k: 5
  search_type: similarity
  score_threshold: null

llm:
  api_key_env: GEMINI_API_KEY
  model: gemini-2.5-flash
  api_type: google
  system_message: 'You are an expert on this project''s documentation. 
    A user will ask a question. 
    Your ''query_documentation'' tool will provide you with the *only* 
    relevant context. **You must answer the user''s question based *only* on 
    that context.** If the context is not sufficient, say so. 
    Do not make up answers.'

text_processing:
  chunk_size: 800
  chunk_overlap: 200

server:
  backend_host: 127.0.0.1
  backend_port: 8001
  frontend_host: 0.0.0.0
  frontend_port: 8000
```

Example user request: 
"Use OpenAI GPT-4 with fast retrieval from GitHub repo 
https://github.com/xray-imaging/2bm-docs"
Expected output:
```yaml
documentation:
  git_repos:
  - https://github.com/xray-imaging/2bm-docs
  local_folders: []
  docs_output_dir: tomo_documentation
  sphinx_build_html_path: tomo_documentation/docs/_build/html

retriever:
  db_path: ./chroma_db
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  k: 2
  search_type: similarity
  score_threshold: null

llm:
  api_key_env: OPENAI_API_KEY
  model: gpt-4
  api_type: openai
  system_message: 'You are an expert on this project''s documentation. 
    A user will ask a question. 
    Your ''query_documentation'' tool will provide you with the *only* 
    relevant context. **You must answer the user''s question based *only* on 
    that context.** If the context is not sufficient, say so. 
    Do not make up answers.'

text_processing:
  chunk_size: 1200
  chunk_overlap: 150

server:
  backend_host: 127.0.0.1
  backend_port: 8001
  frontend_host: 0.0.0.0
  frontend_port: 8000
```
"""


def generate_config_from_prompt(
    user_prompt: str, api_key: Optional[str] = None
) -> Dict:
    """
    Generate a configuration from a natural language prompt using Gemini.

    Args:
        user_prompt: User's description of their configuration needs
        api_key: Optional Gemini API key (will use GEMINI_API_KEY env var if
            not provided)

    Returns:
        Dictionary representing the generated config

    Raises:
        ValueError: If API key is not provided or config generation fails
        yaml.YAMLError: If generated YAML is invalid
    """
    # Get API key
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "Gemini API key not found. Set GEMINI_API_KEY environment variable."
        )

    # Initialize Gemini client
    client = genai.Client(api_key=api_key)

    # Prepare the prompt
    full_prompt = (
        f"{SYSTEM_PROMPT}\n\nUser request: {user_prompt}\n\n"
        "Generate the YAML configuration:"
    )

    try:
        # Call Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,  # Low temperature for consistent, deterministic output
                top_p=0.95,
                max_output_tokens=2048,
            ),
        )

        # Extract generated text
        generated_text = response.text.strip()

        # Remove markdown code blocks if present
        if generated_text.startswith("```yaml"):
            generated_text = generated_text[7:]  # Remove ```yaml
        elif generated_text.startswith("```"):
            generated_text = generated_text[3:]  # Remove ```

        if generated_text.endswith("```"):
            generated_text = generated_text[:-3]  # Remove ```

        generated_text = generated_text.strip()

        # Parse YAML
        config_dict = yaml.safe_load(generated_text)

        if not config_dict or not isinstance(config_dict, dict):
            raise ValueError("Generated config is empty or invalid")

        # Validate against Pydantic model
        validated_config = TomoBaitConfig(**config_dict)

        # Return as dict
        return validated_config.model_dump()

    except Exception as e:
        raise ValueError(f"Failed to generate config: {str(e)}")


def validate_generated_config(config_dict: Dict) -> tuple[bool, Optional[str]]:
    """
    Validate a generated configuration dictionary.

    Args:
        config_dict: Configuration dictionary to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        TomoBaitConfig(**config_dict)
        return True, None
    except Exception as e:
        return False, str(e)


def config_dict_to_yaml(config_dict: Dict) -> str:
    """
    Convert configuration dictionary to YAML string.

    Args:
        config_dict: Configuration dictionary

    Returns:
        YAML-formatted string
    """
    return yaml.safe_dump(
        config_dict, default_flow_style=False, sort_keys=False, indent=2
    )

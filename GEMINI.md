# GEMINI.md

## Project Overview

This project, `tomo-bait`, is a RAG (Retrieval-Augmented Generation) system designed to answer questions about tomography beamline documentation at the Advanced Photon Source (APS). It uses AI-powered agents to provide answers based on ingested documentation.

For a comprehensive architectural overview, please refer to the "Project Overview" and "Architecture" sections in [CLAUDE.md](CLAUDE.md).

## Building and Running

This project uses `uv` for environment and task management. For detailed steps on initial setup, installation, environment variables, and running the application, please refer to the "Development Environment" and "Common Commands" sections in [CLAUDE.md](CLAUDE.md).

**Key Commands (run from the project root):**

*   **Install dependencies:**
    ```bash
    uv venv
    uv pip install -e .
    ```
*   **Ingest documentation:**
    *This must be run before starting the application for the first time.*
    ```bash
    uv run python -m tomobait.data_ingestion
    ```
*   **Run the application (backend and frontend):**
    1.  Start the backend:
        ```bash
        uv run start-backend
        ```
        The backend will be available at `http://127.0.0.1:8001`.
    2.  Start the frontend:
        ```bash
        uv run start-frontend
        ```
        The frontend will be available at `http://127.0.0.1:8000`.

*   **Run the CLI:**
    ```bash
    uv run python -m tomobait.cli "Your question here"
    ```

## Configuration

TomoBait uses a centralized `config.yaml` file. For a detailed breakdown of configuration sections, please refer to the "Key Configuration" section in [CLAUDE.md](CLAUDE.md).

**Gemini-Specific LLM Configuration:**

When configuring the Language Model (LLM) provider for Gemini, ensure your `config.yaml` includes:

```yaml
llm:
provider: GEMINI_API_KEY
  model: gemini-2.5-flash # or other supported Gemini models
  api_type: google
```

Also, set your `GEMINI_API_KEY` environment variable in your `.env` file:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your Gemini API key from: https://aistudio.google.com/app/apikey

## Directory Structure

When you run the ingestion process, TomoBait creates a project-specific directory:

```
.bait-tomo/
├── chroma_db/          # Vector database (embeddings)
├── conversations/      # Saved chat history
└── documentation/      # Cloned repositories and built Sphinx docs
```

For more details on the project-based data isolation, refer to [CLAUDE.md](CLAUDE.md).

## Development Conventions

*   **Linting:** The project uses `ruff` for linting.
    ```bash
    ruff check .
    ```
*   **Formatting:** The project uses `ruff` for formatting.
    ```bash
    ruff format .
    ```
*   **Coding Style:**
    *   Line length: 88 characters
    *   Indent width: 4 spaces
    *   Quote style: double quotes

For further details on code style and development workflow, please refer to [CLAUDE.md](CLAUDE.md).

## Important Implementation Details for Gemini

The backend uses Autogen (AG2) multi-agent framework with Gemini 2.5 Flash as the default LLM. The system employs a two-agent system:
- `doc_expert` (AssistantAgent): An LLM-powered agent (using Gemini) that answers questions.
- `tool_worker` (UserProxyAgent): Executes the `query_documentation` tool to retrieve context for the `doc_expert`.

This setup requires the `GEMINI_API_KEY` environment variable to be set.

For more general "Important Implementation Details" that apply across LLM providers, consult [CLAUDE.md](CLAUDE.md).
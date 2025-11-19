# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TomoBait is a RAG (Retrieval-Augmented Generation) system for tomography beamline documentation. It ingests Sphinx documentation from the 2-BM beamline, stores it in a vector database (ChromaDB), and provides a conversational interface for querying the documentation using AI agents.

## Development Environment

This project uses **Pixi** (not pip) for dependency management and task running. All commands should be run through Pixi tasks.

### Initial Setup

```bash
pixi install  # Install dependencies
pixi run install  # Install package in editable mode
```

## Common Commands

### Running the Application

```bash
# Start the FastAPI backend (port 8001)
pixi run start-backend

# Start the Gradio frontend (port 8000)
pixi run start-frontend

# Run CLI interface
pixi run run-cli "Your question here"
```

### Code Quality

```bash
# Check code style
pixi run lint

# Format code
pixi run format
```

### Data Ingestion

```bash
# Ingest documentation (clones repo, builds Sphinx docs, creates vector DB)
pixi run ingest
```

## Architecture

### Three-Layer System

1. **Data Ingestion Layer** (`data_ingestion.py`)
   - Clones/updates the 2-BM docs repository from GitHub
   - Builds Sphinx documentation to HTML
   - Uses `ReadTheDocsLoader` to load HTML documentation
   - Chunks documents using `RecursiveCharacterTextSplitter` (1000 char chunks, 200 overlap)
   - Embeds using HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, no API)
   - Stores in ChromaDB at `./chroma_db`

2. **Backend/Agent Layer** (`app.py`)
   - FastAPI server exposing `/chat` endpoint
   - Uses Autogen (AG2) multi-agent framework with Gemini 2.5 Flash
   - **Two-agent system**:
     - `doc_expert` (AssistantAgent): LLM-powered agent that answers questions
     - `tool_worker` (UserProxyAgent): Executes the `query_documentation` tool
   - Agent workflow: User question → doc_expert calls tool → tool_worker retrieves from ChromaDB → doc_expert synthesizes answer
   - Requires `GEMINI_API_KEY` environment variable

3. **Frontend Layer** (`frontend.py`)
   - Gradio chatbot interface
   - Makes HTTP requests to FastAPI backend
   - Handles image rendering from documentation (parses markdown image paths)
   - Serves static files from `tomo_documentation/2bm-docs/docs/_build/html`

### Retriever Module (`retriever.py`)

- Shared utility for accessing ChromaDB
- Returns top 3 most relevant document chunks (k=3)
- Can be tested standalone: `python src/tomobait/retriever.py "test query"`

### CLI Interface (`cli.py`)

- Simple argparse wrapper around `run_agent_chat()`
- Provides command-line access to the agent system

## Key Configuration

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (must match between ingestion and retrieval)
- **ChromaDB Path**: `./chroma_db` (relative to project root)
- **LLM**: Google Gemini 2.5 Flash via Autogen
- **Ports**: Backend on 8001, Frontend on 8000

## Environment Variables

Required:
- `GEMINI_API_KEY`: Google Gemini API key for the LLM

## Code Style

- Ruff for linting and formatting
- Line length: 88 characters
- Linting rules: Pyflakes (F), pycodestyle (E), isort (I)
- Double quotes, space indentation

## Important Implementation Details

### Agent Termination Logic

The `tool_worker` agent terminates when it receives a message WITHOUT tool calls. This means the conversation flow is:
1. User question sent to doc_expert
2. doc_expert generates tool call
3. tool_worker executes tool, returns results
4. doc_expert generates final answer (no tool calls)
5. Conversation terminates

### Documentation Source

The system is designed specifically for the 2-BM tomography beamline documentation at `https://github.com/xray-imaging/2bm-docs.git`. The ingestion process expects a Sphinx documentation structure with a `docs/` directory.

### Image Handling in Frontend

The Gradio frontend has custom logic to:
- Parse image paths from agent responses
- Resolve relative paths to absolute paths in the built HTML directory
- Serve images through Gradio's `allowed_paths` mechanism

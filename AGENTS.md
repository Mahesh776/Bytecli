# ByteCli — AI Coding Agent Framework

## Overview
ByteCli is a production-quality AI coding agent framework that runs locally.  
Default backend: LM Studio (OpenAI-compatible API at `localhost:1234/v1`).  
Built with Python 3.12+, clean architecture, and full async design.

## Architecture
Domains (innermost to outermost):  
- `core/` — Error hierarchy, event bus, agent loop  
- `providers/` — LLM backends (LM Studio, Ollama, OpenAI, Anthropic, etc.)  
- `tools/` — File system, terminal, git, search, code execution  
- `cli/` — prompt_toolkit REPL, Rich rendering, keybindings, autocomplete  
- `commands/` — Slash command system (30+ commands)  
- `plugins/` — Hook-based plugin system (30+ hooks)  
- `skills/` — File-based skill engine (priority-scored injection)  
- `memory/` — Multi-level memory (conversation, session, project, workspace)  
- `config/` — Pydantic-v2, cascading from YAML/TOML/JSON5 sources  
- `workspace/` — Auto-detect language, framework, dependencies, git  
- `logging/` — Loguru structured logging (JSON file + colored console)  
- `cache/` — SQLite-backed persistent + in-memory cache

## Conventions
- **No comments in source code** unless the code does something non-obvious
- **Async-first**: all I/O is async, all providers stream, all events async
- **Type hints everywhere**: strict mypy mode
- **Import style**: absolute imports only
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants
- **Error handling**: use the typed error hierarchy; never `raise Exception("msg")`
- **Testing**: pytest + pytest-asyncio; every module has unit + integration tests
- **No placeholder implementations**: every feature must work

## Testing
```bash
pytest tests/ -v --cov=src/bytecli --cov-report=term-missing
mypy src/bytecli/ --strict
ruff check src/bytecli/
```

## Key Files
- `pyproject.toml` — Build config, dependencies, entry points
- `src/bytecli/main.py` — CLI entry point, REPL launcher
- `src/bytecli/core/errors.py` — Complete error hierarchy
- `src/bytecli/core/events.py` — Async event bus with auto-logging
- `src/bytecli/config/schema.py` — All Pydantic config models
- `src/bytecli/config/loader.py` — YAML/TOML/JSON5 config loader

## Rules for AI Agents
1. Always read the full file before editing it.
2. Always run `pytest` after making changes to verify nothing is broken.
3. Add tests for any new functionality.
4. Do not add unnecessary comments.
5. Keep functions small and focused (single responsibility).
6. Use the error hierarchy instead of generic exceptions.
7. Ensure all async functions have proper type annotations.
8. Run `mypy --strict` before considering code complete.

# ByteCli — AI Coding Agent Framework

Production-quality local AI coding agent. Async-first, Python 3.12+, pluggable LLM backends. Default: LM Studio (`localhost:1234/v1`).

## Features

- Interactive REPL (`prompt_toolkit` + `rich`)
- Multi-provider: LM Studio, Ollama, OpenAI, Anthropic, Gemini, DeepSeek, Mistral, OpenRouter, vLLM
- Streaming, tool-calling agent loop with 25-turn guard
- Tools: filesystem, terminal, git, search, code execution, web
- 30+ slash commands, 30+ plugin hooks, skill engine with priority injection
- Multi-level memory (conversation / session / project / workspace)
- Pydantic-v2 config from YAML / TOML / JSON5 + env override
- Workspace auto-detect, Loguru logging, SQLite + in-memory cache

## Install

```bash
pip install -e .
# dev
pip install -e ".[dev]"
```

Requires Python >=3.12.4.

## Usage

```bash
bytecli
bytecli --config config.yaml
bytecli --verbose
bytecli --version
bytecli --repl
bytecli --eval "print(config)"
```

## Config

```yaml
provider:
  default: lm_studio
  lm_studio:
    base_url: http://localhost:1234/v1
    timeout: 60.0
model:
  default: local-model
  temperature: 0.7
  max_tokens: 4096
```

Env override supported. Never commit `.env` or keys — see `.gitignore`.

Providers via env:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

## Architecture

```
src/bytecli/
  core/ config/ providers/ tools/
  cli/ commands/ plugins/ skills/
  memory/ workspace/ logging/ cache/
```

See `AGENTS.md` for conventions.

## Dev

```bash
pytest tests/ -v --cov=src/bytecli --cov-report=term-missing
mypy src/bytecli/ --strict
ruff check src/bytecli/
```

## License

MIT — see `LICENSE`.

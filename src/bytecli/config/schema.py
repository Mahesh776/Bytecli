import re
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class ProviderType(StrEnum):
    LM_STUDIO = "lm_studio"
    OLLAMA = "ollama"
    VLLM = "vllm"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"


class ThemeType(StrEnum):
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


class ProviderEndpointConfig(BaseModel):
    base_url: str = "http://localhost:1234/v1"
    api_key: str | None = None
    timeout: float = 60.0
    max_retries: int = 3
    organization: str | None = None


class ProviderConfig(BaseModel):
    default: ProviderType = ProviderType.LM_STUDIO
    lm_studio: ProviderEndpointConfig = ProviderEndpointConfig(timeout=300.0)
    ollama: ProviderEndpointConfig = ProviderEndpointConfig(base_url="http://localhost:11434/v1", timeout=300.0)
    vllm: ProviderEndpointConfig = ProviderEndpointConfig(base_url="http://localhost:8000/v1", timeout=300.0)
    openai: ProviderEndpointConfig = ProviderEndpointConfig(
        base_url="https://api.openai.com/v1",
    )
    openrouter: ProviderEndpointConfig = ProviderEndpointConfig(
        base_url="https://openrouter.ai/api/v1",
    )
    anthropic: ProviderEndpointConfig = ProviderEndpointConfig(
        base_url="https://api.anthropic.com/v1",
    )
    gemini: ProviderEndpointConfig = ProviderEndpointConfig(
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )
    mistral: ProviderEndpointConfig = ProviderEndpointConfig(
        base_url="https://api.mistral.ai/v1",
    )
    deepseek: ProviderEndpointConfig = ProviderEndpointConfig(
        base_url="https://api.deepseek.com/v1",
    )


BYTECLI_SYSTEM_PROMPT = """You are ByteCli, an AI coding agent working directly in the user's project.

Rules:
1. Do what the user asks. Use tools (read, write, edit, append_file, glob, grep, bash, debug,
   run_python, run_tests, git, web_fetch, web_search, open_file, copy_file, file_tree, download)
   whenever you need to inspect or change files.
2. Never introduce yourself and never reply with filler like "Okay", "Sure", "I understand",
   or "I am ByteCli". Answer or act directly.
3. If a request is unclear or missing details, ask exactly one short question.
4. Be concise and concrete. Prefer doing over explaining.
5. Never put tool calls inside code blocks.
6. If the user gives a folder to save a file into, ALWAYS save the file there with the full path
   (e.g. Tool: write with path: E:\\folder\\game.html). Never just print the code when the user
   asked you to save it.

Call a tool on its own lines like this:
Tool: read
  path: src/main.py

To write a file, indent the full file text under content::
Tool: write
  path: src/main.py
  content:
    def greet():
        print("hello")

The example paths above (like src/main.py) are only to show the FORMAT. Always use the REAL path from the user's request. Never output a placeholder or example path.

Or inline: Tool: grep(pattern="def main", path="src")

After a tool result comes back, look at it and continue: call another tool or give the final answer.

When the user asks for a program, game, or file, write the COMPLETE real code for exactly what was asked.
Never use placeholders like "(full code here)", "...", or "your code here", and never substitute a
simpler or example program.

When the system message includes skill instructions for the user's request, ALWAYS follow them exactly."""

BYTECLI_SMALL_MODEL_PROMPT = """You are ByteCli, an AI coding agent.
Reply to the user directly and briefly. Never reply with "I understand" or "I am ByteCli".
Only call a tool when the user asks to look at or change files. Use this exact format:

To read a file:
Tool: read
  path: src/main.py

To write a file, put the FULL file text indented after content: (required):
Tool: write
  path: src/main.py
  content:
    def greet():
        print("hello")

The example paths above are only to show the FORMAT. Always use the REAL path from the user's request. Never output a placeholder or example path.

To run a shell command:
Tool: bash
  command: python main.py

To run Python code:
Tool: run_python
  file: /path/to/script.py

When the system message includes skill instructions for the user's request, ALWAYS follow them exactly.

If the user asks to save a file to a folder (like "save it to E:\\Downloads\\"), ALWAYS call Tool: write
with the FULL path inside that folder, for example:
Tool: write
  path: E:\\Downloads\\snake.html
  content:
    <!DOCTYPE html>
    ...full file here...

Never just print the code — you MUST call Tool: write and then tell the user the saved path."""

SMALL_MODEL_MAX_PARAMS_BILLIONS = 4.0

_SMALL_MODEL_KEYWORDS = ("gemma-2", "gemma2", "phi-2", "phi2", "phi-3", "phi3", "tiny", "nano", "lite", "mini")


def parse_model_params(model: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*([mb])\b", model.lower())
    if not match:
        return None
    value = float(match.group(1))
    if match.group(2) == "b":
        return value
    return value / 1000.0


def is_small_model(model: str) -> bool:
    params = parse_model_params(model)
    if params is not None:
        return params <= SMALL_MODEL_MAX_PARAMS_BILLIONS
    name = model.lower()
    return any(keyword in name for keyword in _SMALL_MODEL_KEYWORDS)


def select_system_prompt(model: str, custom_prompt: str | None = None) -> str:
    if custom_prompt:
        return custom_prompt
    if is_small_model(model):
        return BYTECLI_SMALL_MODEL_PROMPT
    return BYTECLI_SYSTEM_PROMPT


class ModelConfig(BaseModel):
    default: str = "local-model"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=131072)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stop_sequences: list[str] = Field(default_factory=list)
    system_prompt: str | None = None


class CLIConfig(BaseModel):
    theme: ThemeType = ThemeType.AUTO
    syntax_theme: str = "monokai"
    multi_line: bool = True
    history_size: int = Field(default=1000, ge=10, le=100000)
    show_status_bar: bool = True
    show_token_count: bool = False
    auto_suggest: bool = True


class MemoryConfig(BaseModel):
    max_context_tokens: int = Field(default=32000, ge=1000, le=200000)
    compaction_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    session_persistence: bool = True
    max_sessions: int = Field(default=50, ge=1, le=1000)


class SecurityConfig(BaseModel):
    ask_before_tool: bool = False
    allowed_commands: list[str] = Field(default_factory=list)
    blocked_commands: list[str] = Field(default_factory=lambda: ["rm -rf /", "sudo rm -rf /", "del /f /s /q"])
    sandbox_enabled: bool = False


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_format: bool = False
    verbose: bool = False
    log_file: str | None = None
    rotation: str = "10 MB"
    retention: int = 5
    event_filter: list[str] | None = None


class CacheConfig(BaseModel):
    enabled: bool = True
    persist_dir: str | None = None
    default_ttl_provider: int = 300
    default_ttl_workspace: int = 60


class SkillConfig(BaseModel):
    enabled: bool = True
    skill_dirs: list[str] = Field(
        default_factory=lambda: [
            "~/.config/bytecli/skills",
            ".bytecli/skills",
        ]
    )
    auto_load: bool = True
    max_instructions_length: int = Field(default=20000, ge=1000, le=100000)
    priority_threshold: int = Field(default=0, ge=0, le=100)


class WorkspaceConfig(BaseModel):
    auto_detect: bool = True
    index_depth: int = Field(default=3, ge=1, le=10)
    ignored_dirs: list[str] = Field(
        default_factory=lambda: [
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            ".tox",
            ".egg-info",
            "dist",
            "build",
            ".idea",
            ".vscode",
        ]
    )


class PluginConfig(BaseModel):
    enabled: bool = True
    safe_mode: bool = True
    plugin_dirs: list[str] = Field(
        default_factory=lambda: [
            "~/.config/bytecli/plugins",
            ".bytecli/plugins",
        ]
    )
    enabled_plugins: list[str] = Field(default_factory=lambda: ["*"])
    auto_load: bool = True


class ByteCliConfig(BaseModel):
    provider: ProviderConfig = ProviderConfig()
    model: ModelConfig = ModelConfig()
    cli: CLIConfig = CLIConfig()
    memory: MemoryConfig = MemoryConfig()
    security: SecurityConfig = SecurityConfig()
    logging: LoggingConfig = LoggingConfig()
    cache: CacheConfig = CacheConfig()
    workspace: WorkspaceConfig = WorkspaceConfig()
    plugins: PluginConfig = PluginConfig()
    skills: SkillConfig = SkillConfig()

    @field_validator("logging")
    @classmethod
    def _validate_logging_level(cls, v: LoggingConfig) -> LoggingConfig:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v.level}. Must be one of {valid_levels}")
        return v

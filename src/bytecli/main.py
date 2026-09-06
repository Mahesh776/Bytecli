import argparse
import sys
from pathlib import Path
from typing import Protocol

from loguru import logger

from bytecli import __version__
from bytecli.config.manager import ConfigManager
from bytecli.config.schema import ByteCliConfig
from bytecli.core.errors import ByteCliError, ConfigError
from bytecli.logging.logger import setup_logging
from bytecli.memory.manager import MemoryManager
from bytecli.plugins.manager import PluginManager
from bytecli.providers.factory import ProviderFactory
from bytecli.skills.manager import SkillManager
from bytecli.tools.registry import ToolRegistry


class ReplLike(Protocol):
    async def run(self) -> int: ...


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bytecli",
        description="ByteCli - AI Coding Agent Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bytecli                        Start interactive REPL
  bytecli --config config.yaml   Start with custom config
  bytecli --verbose              Start with debug logging
  bytecli --version              Show version and exit
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
        default=None,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file",
        default=None,
    )
    parser.add_argument(
        "--json-log",
        action="store_true",
        help="Output logs in JSON format",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )
    parser.add_argument(
        "--eval",
        type=str,
        help="Evaluate Python code in bytecli context and exit",
        default=None,
    )
    parser.add_argument(
        "--repl",
        action="store_true",
        help="Use the classic line-based REPL instead of the full-screen terminal",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"ByteCli v{__version__}")
        return 0

    config_path = Path(args.config) if args.config else None

    console_logging = args.repl

    try:
        config_manager = ConfigManager()
        config = config_manager.load(config_path=config_path, env_override=True)

        log_level = "DEBUG" if args.verbose else config.logging.level
        log_file = (
            Path(args.log_file)
            if args.log_file
            else (Path(config.logging.log_file) if config.logging.log_file else None)
        )

        if not console_logging and log_file is None:
            import platformdirs

            log_file = Path(platformdirs.user_log_dir("bytecli", ensure_exists=True)) / "bytecli.log"

        setup_logging(
            level=log_level,
            log_file=log_file,
            json_format=args.json_log or config.logging.json_format,
            verbose=args.verbose or config.logging.verbose,
            console=console_logging,
        )

        logger.info("ByteCli v{} starting", __version__)
        logger.debug("Config loaded: provider={}, model={}", config.provider.default, config.model.default)

        if args.eval:
            _run_eval(args.eval, config_manager)
            return 0

        return asyncio_run(_start_repl(config, config_manager, classic=args.repl))

    except ConfigError as e:
        _report_error(f"Configuration error: {e.message}", console_logging)
        return 1
    except ByteCliError as e:
        _report_error(f"ByteCli error: {e.message}", console_logging)
        return 1
    except Exception as e:
        if console_logging:
            logger.exception("Unexpected error: {}", e)
        else:
            import traceback

            traceback.print_exc(file=sys.stderr)
        return 1


def _report_error(message: str, console_logging: bool) -> None:
    if console_logging:
        logger.error(message)
    else:
        print(message, file=sys.stderr)


def asyncio_run(coro: object) -> int:
    import asyncio

    try:
        return asyncio.run(coro)  # type: ignore[arg-type]
    except KeyboardInterrupt:
        return 0


async def _start_repl(config: ByteCliConfig, config_manager: ConfigManager, classic: bool = False) -> int:
    provider_endpoint = getattr(config.provider, config.provider.default.value)
    provider = ProviderFactory.create(config.provider.default, provider_endpoint)

    workspace_dir = Path.cwd() if config.workspace.auto_detect else None

    memory = MemoryManager(
        session_id="default",
        workspace_dir=workspace_dir,
        max_context_tokens=config.memory.max_context_tokens,
        compaction_threshold=config.memory.compaction_threshold,
    )

    plugin_manager: PluginManager | None = None
    if config.plugins.enabled:
        resolved_dirs: list[Path] = []
        for d in config.plugins.plugin_dirs:
            p = Path(d).expanduser()
            if p.is_dir() or not p.exists():
                resolved_dirs.append(p)
        plugin_manager = PluginManager(
            plugin_dirs=resolved_dirs,
            safe_mode=config.plugins.safe_mode,
            enabled_plugins=config.plugins.enabled_plugins,
        )
        if config.plugins.auto_load:
            loaded = await plugin_manager.initialize()
            if loaded:
                logger.info("Loaded plugins: {}", ", ".join(loaded))

    skill_manager: SkillManager | None = None
    if config.skills.enabled:
        resolved_skill_dirs: list[Path] = []
        for d in config.skills.skill_dirs:
            p = Path(d).expanduser()
            if p.is_dir() or not p.exists():
                resolved_skill_dirs.append(p)
        skill_manager = SkillManager(
            skill_dirs=resolved_skill_dirs,
            auto_load=config.skills.auto_load,
            max_instructions_length=config.skills.max_instructions_length,
        )
        if config.skills.auto_load:
            loaded = await skill_manager.initialize()
            if loaded:
                logger.info("Loaded skills: {}", ", ".join(loaded))

    from bytecli.core.agent import AgentLoop
    from bytecli.core.turn import AgentConfig

    agent_config = AgentConfig(
        model=config.model.default,
        temperature=config.model.temperature,
        max_tokens=config.model.max_tokens,
        max_turns=25,
        system_prompt=config.model.system_prompt,
    )

    agent = AgentLoop(
        provider=provider,
        tool_registry=ToolRegistry if ToolRegistry.list_tools() else None,  # type: ignore[arg-type]
        memory=memory,
        config=agent_config,
        plugin_manager=plugin_manager,
        skill_manager=skill_manager,
    )

    import platformdirs

    app_dir = Path(platformdirs.user_data_dir("bytecli", ensure_exists=True))
    history_file = app_dir / "history.txt"

    provider_name = config.provider.default.value

    if classic:
        from bytecli.cli.repl import ReplSession

        session: ReplLike = ReplSession(
            agent=agent,
            memory=memory,
            history_file=history_file,
            plugin_manager=plugin_manager,
            skill_manager=skill_manager,
        )
    else:
        from bytecli.cli.terminal import ByteCliTerminal

        session = ByteCliTerminal(
            agent=agent,
            memory=memory,
            history_file=history_file,
            plugin_manager=plugin_manager,
            skill_manager=skill_manager,
            provider_name=provider_name,
        )

    return await session.run()


def _run_eval(code: str, config_manager: ConfigManager) -> None:
    namespace = {
        "config_manager": config_manager,
        "config": config_manager.config,
        "__builtins__": __builtins__,
    }
    try:
        exec(code, namespace)
    except Exception as e:
        logger.error("Eval error: {}", e)
        sys.exit(1)

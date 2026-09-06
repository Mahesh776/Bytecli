import platform

from bytecli import __version__
from bytecli.commands.base import Command, CommandContext
from bytecli.tools.registry import ToolRegistry


class LogCommand(Command):
    name = "log"
    description = "Show or change log level"
    category = "Diagnostics"
    usage = "/log [level]"

    async def execute(self, args: str, context: CommandContext) -> bool:
        if args:
            level = args.upper()
            valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            if level in valid:
                from loguru import logger

                logger.remove()
                logger.add(lambda msg: print(msg, end=""), level=level)
                context.renderer.info(f"Log level changed to: {level}")
            else:
                context.renderer.error(f"Invalid level: {args}. Valid: {', '.join(valid)}")
        else:
            context.renderer.info("Current log level: use /debug to toggle.")
        return True


class DebugCommand(Command):
    name = "debug"
    description = "Toggle debug logging on/off"
    category = "Diagnostics"
    usage = "/debug"

    async def execute(self, args: str, context: CommandContext) -> bool:
        from loguru import logger

        logger.remove()
        import sys

        logger.add(sys.stderr, level="DEBUG")
        context.renderer.info("Debug logging enabled.")
        return True


class VersionCommand(Command):
    name = "version"
    aliases = ["v"]  # noqa: RUF012
    description = "Show ByteCli version"
    category = "Diagnostics"
    usage = "/version"

    async def execute(self, args: str, context: CommandContext) -> bool:
        context.renderer.info(f"ByteCli v{__version__}")
        return True


class EnvCommand(Command):
    name = "env"
    description = "Show environment information"
    category = "Diagnostics"
    usage = "/env"

    async def execute(self, args: str, context: CommandContext) -> bool:
        import os

        lines = [
            f"Python: {platform.python_version()}",
            f"Platform: {platform.system()} {platform.release()}",
            f"CWD: {os.getcwd()}",
            f"ByteCli: v{__version__}",
            f"Tools registered: {len(ToolRegistry.list_tools())}",
        ]
        context.renderer.info("Environment:\n" + "\n".join(f"  {line}" for line in lines))
        return True


class PluginsCommand(Command):
    name = "plugins"
    description = "List loaded plugins"
    category = "Diagnostics"
    usage = "/plugins"

    async def execute(self, args: str, context: CommandContext) -> bool:
        mgr = context.plugin_manager
        if mgr is None:
            context.renderer.info("Plugin system is disabled.")
            return True
        plugins = mgr.list_plugins()
        if not plugins:
            context.renderer.info("No plugins loaded.")
            return True
        lines: list[str] = []
        for p in plugins:
            info = f"  {p.name} v{p.version}"
            if p.description:
                info += f" - {p.description}"
            lines.append(info)
        context.renderer.info(f"Loaded plugins ({len(plugins)}):\n" + "\n".join(lines))
        return True


class PluginCommand(Command):
    name = "plugin"
    description = "Load or unload a plugin"
    category = "Diagnostics"
    usage = "/plugin <load|unload|reload> <name>"
    aliases = []  # noqa: RUF012

    async def execute(self, args: str, context: CommandContext) -> bool:
        from pathlib import Path

        mgr = context.plugin_manager
        if mgr is None:
            context.renderer.info("Plugin system is disabled.")
            return True
        parts = args.split(maxsplit=1)
        if not parts:
            context.renderer.error("Usage: /plugin <load|unload|reload> <name>")
            return True
        action = parts[0].lower()
        if len(parts) < 2:
            context.renderer.error(f"Missing plugin name for action '{action}'")
            return True
        name = parts[1]

        if action == "load":
            plugin_path = Path(name)
            if not plugin_path.is_file():
                context.renderer.error(f"Plugin file not found: {name}")
                return True
            try:
                await mgr.load(plugin_path)
                context.renderer.info(f"Plugin loaded: {name}")
            except Exception as e:
                context.renderer.error(f"Failed to load plugin: {e}")
        elif action == "unload":
            success = await mgr.unload(name)
            if success:
                context.renderer.info(f"Plugin unloaded: {name}")
            else:
                context.renderer.error(f"Plugin not found: {name}")
        elif action == "reload":
            await mgr.unload(name)
            plugin_path_candidate = Path(name)
            if plugin_path_candidate.is_file():
                try:
                    await mgr.load(plugin_path_candidate)
                    context.renderer.info(f"Plugin reloaded: {name}")
                except Exception as e:
                    context.renderer.error(f"Failed to reload plugin: {e}")
            else:
                context.renderer.error(f"Plugin file not found for reload: {name}")
        else:
            context.renderer.error(f"Unknown action: {action}. Use load, unload, or reload.")
        return True


class SkillsCommand(Command):
    name = "skills"
    description = "List loaded skills"
    category = "Diagnostics"
    usage = "/skills"

    async def execute(self, args: str, context: CommandContext) -> bool:
        mgr = context.skill_manager
        if mgr is None:
            context.renderer.info("Skill system is disabled.")
            return True
        skills = mgr.list_skills()
        if not skills:
            context.renderer.info("No skills loaded.")
            return True
        lines: list[str] = []
        for s in skills:
            status = "[x]" if s.enabled else "[ ]"
            info = f"  {status} {s.name} v{s.version} (priority: {s.priority})"
            if s.description:
                info += f" - {s.description}"
            lines.append(info)
        context.renderer.info(f"Loaded skills ({len(skills)}):\n" + "\n".join(lines))
        return True


class SkillCommand(Command):
    name = "skill"
    description = "Load or unload a skill"
    category = "Diagnostics"
    usage = "/skill <load|unload|list> [name]"

    async def execute(self, args: str, context: CommandContext) -> bool:
        from pathlib import Path

        mgr = context.skill_manager
        if mgr is None:
            context.renderer.info("Skill system is disabled.")
            return True
        parts = args.split(maxsplit=1)
        if not parts:
            context.renderer.error("Usage: /skill <load|unload|list> [name]")
            return True
        action = parts[0].lower()

        if action == "list":
            name = parts[1] if len(parts) > 1 else ""
            if name:
                skill = mgr.get_skill(name)
                if skill:
                    context.renderer.info(
                        f"Skill: {skill.name} v{skill.version}\n"
                        f"  Description: {skill.description}\n"
                        f"  Priority: {skill.priority}\n"
                        f"  Patterns: {', '.join(skill.patterns) if skill.patterns else '(none)'}\n"
                        f"  Enabled: {skill.enabled}\n"
                        f"  Instructions ({len(skill.instructions)} chars)"
                    )
                else:
                    context.renderer.error(f"Skill not found: {name}")
            else:
                skills = mgr.list_skills()
                if not skills:
                    context.renderer.info("No skills loaded.")
                else:
                    lines = [f"  {s.name} v{s.version} (p:{s.priority}) - {s.description}" for s in skills]
                    context.renderer.info(f"Skills ({len(skills)}):\n" + "\n".join(lines))
            return True

        if len(parts) < 2:
            context.renderer.error(f"Missing skill name for action '{action}'")
            return True
        name = parts[1]

        if action == "load":
            skill_path = Path(name)
            if not skill_path.is_file():
                context.renderer.error(f"Skill file not found: {name}")
                return True
            try:
                skill = mgr.load_skill_file(skill_path)
                context.renderer.info(f"Skill loaded: {skill.name}")
            except Exception as e:
                context.renderer.error(f"Failed to load skill: {e}")
        elif action == "unload":
            success = mgr.unload_skill(name)
            if success:
                context.renderer.info(f"Skill unloaded: {name}")
            else:
                context.renderer.error(f"Skill not found: {name}")
        else:
            context.renderer.error(f"Unknown action: {action}. Use load, unload, or list.")
        return True

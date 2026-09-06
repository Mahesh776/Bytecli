from pathlib import Path

from bytecli.commands.base import Command, CommandContext


class WorkspaceCommand(Command):
    name = "workspace"
    description = "Show workspace information"
    category = "Workspace"
    usage = "/workspace"

    async def execute(self, args: str, context: CommandContext) -> bool:
        try:
            from bytecli.workspace.manager import WorkspaceManager

            wd = Path.cwd()
            mgr = WorkspaceManager(wd)
            mgr.analyze()
            summary = mgr.summary()
            context.renderer.info(summary)
        except Exception as e:
            context.renderer.error(f"Failed to analyze workspace: {e}")
        return True


class TreeCommand(Command):
    name = "tree"
    description = "Show file tree of the workspace"
    category = "Workspace"
    usage = "/tree [depth]"

    async def execute(self, args: str, context: CommandContext) -> bool:
        try:
            from bytecli.workspace.tree import FileTreeBuilder

            depth = int(args) if args else 2
            wd = Path.cwd()
            builder = FileTreeBuilder(root=wd, max_depth=depth)
            tree = builder.build()
            context.renderer.info(f"File tree (depth={depth}):\n{tree}")
        except ValueError:
            context.renderer.error("Depth must be an integer.")
        except Exception as e:
            context.renderer.error(f"Failed to get file tree: {e}")
        return True


class GitCommand(Command):
    name = "git"
    description = "Show git repository status"
    category = "Workspace"
    usage = "/git"

    async def execute(self, args: str, context: CommandContext) -> bool:
        try:
            from bytecli.workspace.manager import WorkspaceManager

            wd = Path.cwd()
            mgr = WorkspaceManager(wd)
            info = mgr.analyze()
            if info.git:
                lines = [
                    f"Branch: {info.git.current_branch or 'unknown'}",
                    f"Commit: {info.git.commit_hash or 'unknown'}",
                    f"Remote: {info.git.remote_url or 'none'}",
                    f"Status: {len(info.git.modified_files)} modified, {len(info.git.untracked_files)} untracked",
                    f"Ahead: {info.git.ahead}, Behind: {info.git.behind}",
                ]
                context.renderer.info("Git Status:\n" + "\n".join(f"  {line}" for line in lines))
            else:
                context.renderer.info("Not a git repository or no git info available.")
        except Exception as e:
            context.renderer.error(f"Failed to get git info: {e}")
        return True


class LanguageCommand(Command):
    name = "language"
    aliases = ["lang"]  # noqa: RUF012
    description = "Show detected programming languages"
    category = "Workspace"
    usage = "/language"

    async def execute(self, args: str, context: CommandContext) -> bool:
        try:
            from bytecli.workspace.manager import WorkspaceManager

            wd = Path.cwd()
            mgr = WorkspaceManager(wd)
            info = mgr.analyze()
            if info.languages:
                items = [(lang.name, f"v{lang.version}" if lang.version else "detected") for lang in info.languages]
                context.renderer.help_table(items, title="Languages")
            else:
                context.renderer.info("No languages detected.")
        except Exception as e:
            context.renderer.error(f"Failed to detect languages: {e}")
        return True


class FrameworksCommand(Command):
    name = "frameworks"
    aliases = ["fw"]  # noqa: RUF012
    description = "Show detected frameworks"
    category = "Workspace"
    usage = "/frameworks"

    async def execute(self, args: str, context: CommandContext) -> bool:
        try:
            from bytecli.workspace.manager import WorkspaceManager

            wd = Path.cwd()
            mgr = WorkspaceManager(wd)
            info = mgr.analyze()
            if info.frameworks:
                items = [(f.name, f.version or "detected") for f in info.frameworks]
                context.renderer.help_table(items, title="Frameworks")
            else:
                context.renderer.info("No frameworks detected.")
        except Exception as e:
            context.renderer.error(f"Failed to detect frameworks: {e}")
        return True

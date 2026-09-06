# ruff: noqa: RUF012
# mypy: disable-error-code="override"
import asyncio
from pathlib import Path

from bytecli.tools.base import Tool, ToolParameter, ToolResult


class GitTool(Tool):
    name = "git"
    description = "Execute git operations. Use 'git status', 'git diff', 'git log', etc."
    name_aliases = {"git_run", "run_git", "git_command", "git_execute"}
    parameters = [
        ToolParameter(name="command", type="string", description="Git command and args (e.g. 'status', 'diff')"),
        ToolParameter(name="repo_path", type="string", description="Path to the git repository", required=False),
    ]

    async def _execute(self, command: str, repo_path: str | None = None) -> ToolResult:
        workdir = Path(repo_path) if repo_path else Path.cwd()
        if not (workdir / ".git").exists():
            return ToolResult(success=False, error=f"Not a git repository: {workdir}", tool_name=self.name)
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", *command.split(),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                cwd=str(workdir),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        except TimeoutError:
            return ToolResult(success=False, error="Git command timed out", tool_name=self.name)
        except FileNotFoundError:
            return ToolResult(success=False, error="Git not found on system", tool_name=self.name)
        except Exception as e:
            return ToolResult(success=False, error=f"Git execution failed: {e}", tool_name=self.name)
        stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""
        retcode = proc.returncode or 0
        return ToolResult(
            success=retcode == 0,
            data={"stdout": stdout_str, "stderr": stderr_str, "exit_code": retcode, "command": f"git {command}"},
            error=stderr_str if retcode != 0 else None,
            tool_name=self.name,
        )

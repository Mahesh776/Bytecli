# ruff: noqa: RUF012
# mypy: disable-error-code="override"
import shutil
from pathlib import Path
from typing import Any

from bytecli.tools.base import Tool, ToolParameter, ToolResult

_FILE_PATH_ALIASES: dict[str, str] = {
    "path": "file_path",
    "file": "file_path",
    "filename": "file_path",
    "file_name": "file_path",
    "filepath": "file_path",
}


class ReadTool(Tool):
    name = "read"
    description = "Read a file from the filesystem. Returns the file contents with line numbers."
    arg_aliases = _FILE_PATH_ALIASES
    name_aliases = {"read_file", "readfile", "open_file", "cat", "view_file", "read_text_file"}
    parameters = [
        ToolParameter(name="file_path", type="string", description="Absolute path to the file to read"),
        ToolParameter(name="offset", type="integer", description="Line number to start from (1-indexed)", required=False, default=1),  # noqa: E501
        ToolParameter(name="limit", type="integer", description="Maximum number of lines to read", required=False, default=2000),  # noqa: E501
    ]

    async def _execute(self, file_path: str, offset: int = 1, limit: int = 2000) -> ToolResult:
        path = Path(file_path)
        if not path.exists():
            return ToolResult(success=False, error=f"File not found: {file_path}", tool_name=self.name)
        if not path.is_file():
            return ToolResult(success=False, error=f"Not a file: {file_path}", tool_name=self.name)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to read file: {e}", tool_name=self.name)
        start = max(0, offset - 1)
        end = start + limit
        selected = lines[start:end]
        total = len(lines)
        content = "\n".join(f"{i + start + 1}: {line}" for i, line in enumerate(selected))
        return ToolResult(
            success=True,
            data={
                "file_path": file_path,
                "content": content,
                "total_lines": total,
                "start_line": start + 1,
                "end_line": min(end, total),
                "truncated": end < total,
            },
            tool_name=self.name,
        )


class WriteTool(Tool):
    name = "write"
    description = "Write content to a file. Creates parent directories if needed. Overwrites existing files."
    arg_aliases = _FILE_PATH_ALIASES
    name_aliases = {
        "write_file",
        "writefile",
        "create_file",
        "createfile",
        "create",
        "make_file",
        "save_file",
        "save",
        "new_file",
        "overwrite_file",
    }
    parameters = [
        ToolParameter(name="file_path", type="string", description="Absolute path to the file to write"),
        ToolParameter(name="content", type="string", description="Content to write to the file"),
    ]

    async def _execute(self, file_path: str, content: str) -> ToolResult:
        path = Path(file_path)
        if path.exists() and path.is_dir():
            return ToolResult(
                success=False,
                error=(
                    f"'{file_path}' is a directory. Provide a full file path including a filename, "
                    f"e.g. '{path / 'example.html'!s}'."
                ),
                tool_name=self.name,
            )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to write file: {e}", tool_name=self.name)
        return ToolResult(
            success=True,
            data={"file_path": file_path, "bytes_written": len(content.encode("utf-8"))},
            tool_name=self.name,
        )


class EditTool(Tool):
    name = "edit"
    description = "Replace exact text in a file. Fails if old_string is not found or matches multiple times."
    name_aliases = {"edit_file", "editfile", "update_file", "modify", "replace_text", "replace_in_file", "patch", "update"}  # noqa: E501
    arg_aliases = {
        **_FILE_PATH_ALIASES,
        "old": "old_string",
        "old_str": "old_string",
        "new": "new_string",
        "new_str": "new_string",
        "replacement": "new_string",
        "replace_with": "new_string",
    }
    parameters = [
        ToolParameter(name="file_path", type="string", description="Absolute path to the file to edit"),
        ToolParameter(name="old_string", type="string", description="The exact text to replace"),
        ToolParameter(name="new_string", type="string", description="The replacement text"),
    ]

    async def _execute(self, file_path: str, old_string: str, new_string: str) -> ToolResult:
        path = Path(file_path)
        if not path.exists():
            return ToolResult(success=False, error=f"File not found: {file_path}", tool_name=self.name)
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to read file: {e}", tool_name=self.name)
        count = content.count(old_string)
        if count == 0:
            return ToolResult(success=False, error=f"old_string not found in {file_path}", tool_name=self.name)
        if count > 1:
            return ToolResult(
                success=False,
                error=f"Found {count} matches for old_string in {file_path}. Provide more surrounding context.",
                tool_name=self.name,
            )
        new_content = content.replace(old_string, new_string, 1)
        try:
            path.write_text(new_content, encoding="utf-8")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to write file: {e}", tool_name=self.name)
        return ToolResult(
            success=True,
            data={"file_path": file_path, "old_length": len(old_string), "new_length": len(new_string)},
            tool_name=self.name,
        )


class MkdirTool(Tool):
    name = "mkdir"
    description = "Create a directory and any missing parents. Succeeds even if it already exists."
    name_aliases = {
        "make_dir",
        "make_directory",
        "create_dir",
        "create_directory",
        "create_folder",
        "make_folder",
        "new_folder",
        "new_dir",
        "new_directory",
        "makedir",
        "mkdirs",
    }
    arg_aliases = {
        "dir": "path",
        "directory": "path",
        "folder": "path",
        "folder_name": "path",
    }
    parameters = [
        ToolParameter(name="path", type="string", description="Absolute path of the directory to create"),
    ]

    async def _execute(self, path: str) -> ToolResult:
        directory = Path(path)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to create directory: {e}", tool_name=self.name)
        return ToolResult(
            success=True,
            data={"path": str(directory), "created": directory.is_dir()},
            tool_name=self.name,
        )


class GlobTool(Tool):
    name = "glob"
    description = "Find files matching a glob pattern. Returns paths sorted by modification time."
    name_aliases = {
        "list_files",
        "find_files",
        "find",
        "file_search",
        "search_files",
        "locate",
        "glob_files",
        "walk_files",
    }
    arg_aliases = {
        "root": "path",
        "directory": "path",
        "folder": "path",
    }
    parameters = [
        ToolParameter(name="pattern", type="string", description="Glob pattern (e.g. '**/*.py', 'src/**/*.ts')"),
        ToolParameter(name="path", type="string", description="Root directory to search in (defaults to CWD)", required=False),  # noqa: E501
    ]

    async def _execute(self, pattern: str, path: str | None = None) -> ToolResult:
        search_path = Path(path) if path else Path.cwd()
        if not search_path.exists():
            return ToolResult(success=False, error=f"Path not found: {path}", tool_name=self.name)
        try:
            matches = sorted(search_path.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            files = [str(m) for m in matches if m.is_file()]
        except Exception as e:
            return ToolResult(success=False, error=f"Glob failed: {e}", tool_name=self.name)
        return ToolResult(
            success=True,
            data={"files": files, "count": len(files), "pattern": pattern, "root": str(search_path)},
            tool_name=self.name,
        )


class GrepTool(Tool):
    name = "grep"
    description = "Search file contents using a regular expression. Returns matching file paths with line numbers."
    name_aliases = {"search", "search_in_files", "text_search", "find_text", "regex_search", "search_code", "rg"}
    arg_aliases = {
        "regex": "pattern",
        "root": "path",
        "directory": "path",
        "folder": "path",
    }
    parameters = [
        ToolParameter(name="pattern", type="string", description="Regex pattern to search for"),
        ToolParameter(name="include", type="string", description="File glob to filter (e.g. '*.py')", required=False),
        ToolParameter(name="path", type="string", description="Root directory to search in", required=False),
    ]

    async def _execute(self, pattern: str, include: str | None = None, path: str | None = None) -> ToolResult:
        search_path = Path(path) if path else Path.cwd()
        if not search_path.exists():
            return ToolResult(success=False, error=f"Path not found: {path}", tool_name=self.name)
        import re
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            return ToolResult(success=False, error=f"Invalid regex: {e}", tool_name=self.name)
        results: list[dict[str, Any]] = []
        files_to_search: list[Path] = []
        if include:
            files_to_search = sorted(search_path.rglob(include), key=lambda p: p.stat().st_mtime, reverse=True)
        else:
            files_to_search = [p for p in search_path.rglob("*") if p.is_file()]
        for file_path in files_to_search:
            try:
                if file_path.stat().st_size > 1_000_000:
                    continue
                text = file_path.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(text.splitlines(), 1):
                    if compiled.search(line):
                        results.append({
                            "file": str(file_path),
                            "line": i,
                            "content": line.strip()[:200],
                        })
            except (OSError, UnicodeDecodeError):
                continue
        return ToolResult(
            success=True,
            data={"matches": results, "count": len(results), "pattern": pattern},
            tool_name=self.name,
        )


class AppendFileTool(Tool):
    name = "append_file"
    description = "Append content to the end of a file. Creates the file and parent directories if missing."
    name_aliases = {"append", "append_to_file", "add_to_file"}
    arg_aliases = _FILE_PATH_ALIASES
    parameters = [
        ToolParameter(name="file_path", type="string", description="Absolute path to the file to append to"),
        ToolParameter(name="content", type="string", description="Content to append to the file"),
    ]

    async def _execute(self, file_path: str, content: str) -> ToolResult:
        path = Path(file_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to append to file: {e}", tool_name=self.name)
        return ToolResult(
            success=True,
            data={"file_path": file_path, "appended_chars": len(content)},
            tool_name=self.name,
        )


class DeleteFileTool(Tool):
    name = "delete_file"
    description = "Delete a file from the filesystem. Fails if the file does not exist."
    name_aliases = {"remove_file", "rm", "delete", "unlink", "remove"}
    arg_aliases = _FILE_PATH_ALIASES
    parameters = [
        ToolParameter(name="file_path", type="string", description="Absolute path to the file to delete"),
    ]

    async def _execute(self, file_path: str) -> ToolResult:
        path = Path(file_path)
        if not path.exists():
            return ToolResult(success=False, error=f"File not found: {file_path}", tool_name=self.name)
        if path.is_dir():
            return ToolResult(success=False, error=f"Not a file: {file_path}", tool_name=self.name)
        try:
            path.unlink()
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to delete file: {e}", tool_name=self.name)
        return ToolResult(success=True, data={"file_path": file_path}, tool_name=self.name)


class MoveFileTool(Tool):
    name = "move_file"
    description = "Move or rename a file or directory."
    name_aliases = {"rename_file", "rename", "move", "mv"}
    arg_aliases = {
        "path": "source",
        "source_path": "source",
        "src": "source",
        "from": "source",
        "target": "destination",
        "target_path": "destination",
        "dest": "destination",
        "to": "destination",
        "new_path": "destination",
        "new_name": "destination",
    }
    parameters = [
        ToolParameter(name="source", type="string", description="Current path of the file or directory"),
        ToolParameter(name="destination", type="string", description="New path after moving or renaming"),
    ]

    async def _execute(self, source: str, destination: str) -> ToolResult:
        src = Path(source)
        dst = Path(destination)
        if not src.exists():
            return ToolResult(success=False, error=f"Source not found: {source}", tool_name=self.name)
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to move: {e}", tool_name=self.name)
        return ToolResult(
            success=True,
            data={"from": source, "to": destination},
            tool_name=self.name,
        )


class ListDirectoryTool(Tool):
    name = "list_directory"
    description = "List the contents of a directory with sizes and modification times."
    name_aliases = {"ls", "list_dir", "directory_listing", "list_folder", "dir_listing"}
    arg_aliases = {
        "root": "path",
        "directory": "path",
        "folder": "path",
        "dir": "path",
    }
    parameters = [
        ToolParameter(name="path", type="string", description="Directory to list (defaults to CWD)", required=False),
    ]

    async def _execute(self, path: str | None = None) -> ToolResult:
        directory = Path(path) if path else Path.cwd()
        if not directory.exists():
            return ToolResult(success=False, error=f"Path not found: {path}", tool_name=self.name)
        if not directory.is_dir():
            return ToolResult(success=False, error=f"Not a directory: {path}", tool_name=self.name)
        entries: list[dict[str, Any]] = []
        for entry in sorted(directory.iterdir(), key=lambda e: (e.is_file(), e.name.lower())):
            try:
                stat = entry.stat()
                entries.append({
                    "name": entry.name,
                    "is_dir": entry.is_dir(),
                    "size": stat.st_size if entry.is_file() else None,
                    "modified": stat.st_mtime,
                })
            except OSError:
                entries.append({"name": entry.name, "is_dir": entry.is_dir(), "size": None, "modified": None})
        return ToolResult(
            success=True,
            data={"path": str(directory), "entries": entries, "count": len(entries)},
            tool_name=self.name,
        )


class FileInfoTool(Tool):
    name = "file_info"
    description = "Get metadata about a file or directory (size, type, last modified)."
    name_aliases = {"stat", "file_metadata", "get_file_info", "metadata", "file_stat"}
    arg_aliases = {
        "file": "path",
        "file_path": "path",
        "filepath": "path",
        "filename": "path",
    }
    parameters = [
        ToolParameter(name="path", type="string", description="Path of the file or directory to inspect"),
    ]

    async def _execute(self, path: str) -> ToolResult:
        target = Path(path)
        if not target.exists():
            return ToolResult(success=False, error=f"Path not found: {path}", tool_name=self.name)
        try:
            stat = target.stat()
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to stat: {e}", tool_name=self.name)
        return ToolResult(
            success=True,
            data={
                "path": str(target),
                "is_dir": target.is_dir(),
                "is_file": target.is_file(),
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "extension": target.suffix or None,
            },
            tool_name=self.name,
        )


class OpenFileTool(Tool):
    name = "open_file"
    description = "Open a file or URL in the default application/browser (e.g. an HTML file in the browser)."
    name_aliases = {"open", "open_in_browser", "launch", "start", "open_in_default_app", "reveal"}
    arg_aliases = _FILE_PATH_ALIASES
    parameters = [
        ToolParameter(name="file_path", type="string", description="Absolute path to the file, or a URL, to open"),
    ]

    async def _execute(self, file_path: str) -> ToolResult:
        import asyncio
        import os

        target = Path(file_path)
        if target.exists() and target.is_dir():
            return ToolResult(success=False, error=f"'{file_path}' is a directory, not a file", tool_name=self.name)
        if not target.exists() and not file_path.lower().startswith(("http://", "https://")):
            return ToolResult(success=False, error=f"File not found: {file_path}", tool_name=self.name)
        try:
            await asyncio.to_thread(os.startfile, str(target) if target.exists() else file_path)
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to open: {e}", tool_name=self.name)
        return ToolResult(
            success=True,
            data={"opened": file_path},
            tool_name=self.name,
        )


class CopyFileTool(Tool):
    name = "copy_file"
    description = "Copy a file or directory (recursively) to a destination path."
    name_aliases = {"copy", "cp", "duplicate", "clone", "copy_dir", "copy_directory"}
    arg_aliases = {
        "path": "source",
        "source_path": "source",
        "src": "source",
        "from": "source",
        "target": "destination",
        "target_path": "destination",
        "dest": "destination",
        "to": "destination",
        "new_path": "destination",
    }
    parameters = [
        ToolParameter(name="source", type="string", description="Path of the file or directory to copy"),
        ToolParameter(name="destination", type="string", description="Destination path (file or directory)"),
    ]

    async def _execute(self, source: str, destination: str) -> ToolResult:
        src = Path(source)
        dst = Path(destination)
        if not src.exists():
            return ToolResult(success=False, error=f"Source not found: {source}", tool_name=self.name)
        try:
            if src.is_dir():
                if dst.exists() and not dst.is_dir():
                    return ToolResult(success=False, error=f"Destination exists and is not a directory: {destination}", tool_name=self.name)  # noqa: E501
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dst / src.name, dirs_exist_ok=True)
                copied = dst / src.name
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied = dst
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to copy: {e}", tool_name=self.name)
        return ToolResult(
            success=True,
            data={"from": source, "to": str(copied)},
            tool_name=self.name,
        )


class FileTreeTool(Tool):
    name = "file_tree"
    description = "Recursively list a directory tree with indentation. Great for exploring project structure."
    name_aliases = {"tree", "list_tree", "show_tree", "directory_tree", "walk"}
    arg_aliases = {
        "root": "path",
        "directory": "path",
        "folder": "path",
        "dir": "path",
    }
    parameters = [
        ToolParameter(name="path", type="string", description="Root directory to list (defaults to CWD)", required=False),  # noqa: E501
        ToolParameter(name="max_depth", type="integer", description="Maximum recursion depth", required=False, default=3),  # noqa: E501
    ]

    async def _execute(self, path: str | None = None, max_depth: int = 3) -> ToolResult:
        root = Path(path) if path else Path.cwd()
        if not root.exists():
            return ToolResult(success=False, error=f"Path not found: {path}", tool_name=self.name)
        if not root.is_dir():
            return ToolResult(success=False, error=f"Not a directory: {path}", tool_name=self.name)

        def render(current: Path, prefix: str = "", depth: int = 0) -> list[str]:
            if depth > max_depth:
                return [f"{prefix}..."]
            lines: list[str] = []
            try:
                entries = sorted(current.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
            except OSError:
                return lines
            for index, entry in enumerate(entries):
                is_last = index == len(entries) - 1
                connector = "└── " if is_last else "├── "
                lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
                if entry.is_dir() and depth < max_depth:
                    child_prefix = prefix + ("    " if is_last else "│   ")
                    lines.extend(render(entry, child_prefix, depth + 1))
            return lines

        tree = render(root)
        return ToolResult(
            success=True,
            data={"path": str(root), "tree": "\n".join(tree), "entries": len(tree)},
            tool_name=self.name,
        )

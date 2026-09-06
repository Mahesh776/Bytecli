import tempfile
from pathlib import Path

import pytest

from bytecli.tools.base import Tool, ToolParameter, ToolResult
from bytecli.tools.code import DebugTool, RunPythonTool, RunTestsTool
from bytecli.tools.filesystem import (
    AppendFileTool,
    CopyFileTool,
    DeleteFileTool,
    EditTool,
    FileInfoTool,
    FileTreeTool,
    GlobTool,
    GrepTool,
    ListDirectoryTool,
    MkdirTool,
    MoveFileTool,
    OpenFileTool,
    ReadTool,
    WriteTool,
)
from bytecli.tools.git import GitTool
from bytecli.tools.registry import ToolRegistry
from bytecli.tools.terminal import BashTool
from bytecli.tools.web import DownloadTool, WebFetchTool, WebSearchTool  # noqa: F401


class TestToolBase:
    def test_tool_result_defaults(self) -> None:
        r = ToolResult()
        assert r.success is True
        assert r.data is None
        assert r.error is None

    def test_tool_parameter_defaults(self) -> None:
        p = ToolParameter(name="test", type="string")
        assert p.name == "test"
        assert p.type == "string"
        assert p.required is True

    def test_tool_parameter_to_json_schema(self) -> None:
        p = ToolParameter(name="path", type="string", description="A file path")
        schema = p.to_json_schema()
        assert schema["type"] == "string"
        assert schema["description"] == "A file path"

    def test_tool_parameter_optional_default(self) -> None:
        p = ToolParameter(name="count", type="integer", required=False, default=5)
        schema = p.to_json_schema()
        assert schema["default"] == 5

    def test_tool_get_json_schema(self) -> None:
        class TestTool(Tool):
            name = "test"
            description = "A test tool"
            parameters = [  # noqa: RUF012
                ToolParameter(name="input", type="string", description="An input"),
                ToolParameter(name="flag", type="boolean", required=False, default=False),
            ]

        tool = TestTool()
        schema = tool.get_json_schema()
        assert schema["type"] == "object"
        assert "input" in schema["properties"]
        assert "flag" in schema["properties"]
        assert schema["required"] == ["input"]

    @pytest.mark.asyncio
    async def test_tool_execute_catches_exceptions(self) -> None:
        class BrokenTool(Tool):
            name = "broken"

            async def _execute(self, **kwargs: ToolParameter) -> ToolResult:
                raise ValueError("something broke")

        tool = BrokenTool()
        result = await tool.execute()
        assert result.success is False
        assert "ValueError" in (result.error or "")

    @pytest.mark.asyncio
    async def test_tool_execute_raises_tool_execution_error(self) -> None:
        from bytecli.core.errors import ToolExecutionError

        class FailingTool(Tool):
            name = "failing"

            async def _execute(self, **kwargs: ToolParameter) -> ToolResult:
                raise ToolExecutionError("failing", "deliberate")

        tool = FailingTool()
        with pytest.raises(ToolExecutionError):
            await tool.execute()


class TestReadTool:
    @pytest.fixture
    def tool(self) -> ReadTool:
        return ReadTool()

    @pytest.mark.asyncio
    async def test_read_existing_file(self, tool: ReadTool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            path.write_text("line1\nline2\nline3\n")
            result = await tool.execute(file_path=str(path))
            assert result.success is True
            assert result.data is not None
            assert "line1" in result.data["content"]
            assert result.data["total_lines"] == 3

    @pytest.mark.asyncio
    async def test_read_missing_file(self, tool: ReadTool) -> None:
        result = await tool.execute(file_path="/nonexistent/file.txt")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_read_with_offset(self, tool: ReadTool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            path.write_text("a\nb\nc\nd\ne\n")
            result = await tool.execute(file_path=str(path), offset=3, limit=2)
            assert result.success is True
            assert result.data is not None
            assert result.data["start_line"] == 3
            assert result.data["end_line"] == 4

    @pytest.mark.asyncio
    async def test_read_accepts_path_alias(self, tool: ReadTool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            path.write_text("hello\n")
            result = await tool.execute(path=str(path))
            assert result.success is True
            assert result.data is not None
            assert "hello" in result.data["content"]


class TestWriteTool:
    @pytest.fixture
    def tool(self) -> WriteTool:
        return WriteTool()

    @pytest.mark.asyncio
    async def test_write_file(self, tool: WriteTool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            result = await tool.execute(file_path=str(path), content="hello world")
            assert result.success is True
            assert path.read_text() == "hello world"

    @pytest.mark.asyncio
    async def test_write_creates_parent_dirs(self, tool: WriteTool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "nested" / "test.txt"
            result = await tool.execute(file_path=str(path), content="nested")
            assert result.success is True
            assert path.exists()

    @pytest.mark.asyncio
    async def test_write_missing_content_reports_missing_params(self, tool: WriteTool) -> None:
        result = await tool.execute(file_path="/tmp/somefile.txt")
        assert result.success is False
        assert "Missing required parameter" in (result.error or "")
        assert "content" in (result.error or "")
        assert "file_path" in (result.error or "")
        assert "TypeError" not in (result.error or "")

    @pytest.mark.asyncio
    async def test_write_unknown_kwarg_reports_valid_params(self, tool: WriteTool) -> None:
        result = await tool.execute(file_path="/tmp/somefile.txt", content="x", data="extra")
        assert result.success is False
        assert "Valid parameters: file_path, content" in (result.error or "")

    @pytest.mark.asyncio
    async def test_write_to_directory_path_errors(self, tool: WriteTool, tmp_path: Path) -> None:
        result = await tool.execute(file_path=str(tmp_path), content="x")
        assert result.success is False
        assert "is a directory" in (result.error or "")
        assert "example.html" in (result.error or "")


class TestOpenFileTool:
    @pytest.fixture
    def tool(self) -> OpenFileTool:
        return OpenFileTool()

    @pytest.mark.asyncio
    async def test_open_missing_file_errors(self, tool: OpenFileTool) -> None:
        result = await tool.execute(file_path=r"C:\definitely\missing\file.html")
        assert result.success is False
        assert "File not found" in (result.error or "")

    @pytest.mark.asyncio
    async def test_open_url_ok(self, tool: OpenFileTool) -> None:
        result = await tool.execute(file_path="https://example.com/")
        assert result.success is True


class TestCopyFileTool:
    @pytest.fixture
    def tool(self) -> CopyFileTool:
        return CopyFileTool()

    @pytest.mark.asyncio
    async def test_copy_file(self, tool: CopyFileTool, tmp_path: Path) -> None:
        src = tmp_path / "a.txt"
        src.write_text("data")
        dst = tmp_path / "sub" / "b.txt"
        result = await tool.execute(source=str(src), destination=str(dst))
        assert result.success is True
        assert dst.read_text() == "data"

    @pytest.mark.asyncio
    async def test_copy_missing_source_errors(self, tool: CopyFileTool, tmp_path: Path) -> None:
        result = await tool.execute(source=str(tmp_path / "nope.txt"), destination=str(tmp_path / "x.txt"))
        assert result.success is False
        assert "Source not found" in (result.error or "")

    @pytest.mark.asyncio
    async def test_copy_directory_recursive(self, tool: CopyFileTool, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        (src_dir / "inner").mkdir(parents=True)
        (src_dir / "inner" / "f.txt").write_text("hi")
        dst_dir = tmp_path / "dst"
        result = await tool.execute(source=str(src_dir), destination=str(dst_dir))
        assert result.success is True
        assert (dst_dir / "src" / "inner" / "f.txt").read_text() == "hi"


class TestFileTreeTool:
    @pytest.fixture
    def tool(self) -> FileTreeTool:
        return FileTreeTool()

    @pytest.mark.asyncio
    async def test_file_tree(self, tool: FileTreeTool, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "one.py").write_text("x")
        (tmp_path / "b.txt").write_text("y")
        result = await tool.execute(path=str(tmp_path), max_depth=5)
        assert result.success is True
        assert "one.py" in result.data["tree"]
        assert "b.txt" in result.data["tree"]

    @pytest.mark.asyncio
    async def test_file_tree_missing_path_errors(self, tool: FileTreeTool) -> None:
        result = await tool.execute(path=str(Path(r"C:\does_not_exist_xyz")))
        assert result.success is False
        assert "Path not found" in (result.error or "")


class TestDownloadTool:
    @pytest.fixture
    def tool(self) -> DownloadTool:
        return DownloadTool()

    @pytest.mark.asyncio
    async def test_download_missing_params(self, tool: DownloadTool) -> None:
        result = await tool.execute(url="https://example.com")
        assert result.success is False
        assert "Missing required parameter" in (result.error or "")


class TestEditTool:
    @pytest.fixture
    def tool(self) -> EditTool:
        return EditTool()

    @pytest.mark.asyncio
    async def test_edit_success(self, tool: EditTool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.py"
            path.write_text("old_function()\n")
            result = await tool.execute(file_path=str(path), old_string="old_function()", new_string="new_function()")
            assert result.success is True
            content = path.read_text()
            assert "new_function()" in content

    @pytest.mark.asyncio
    async def test_edit_not_found(self, tool: EditTool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.py"
            path.write_text("content\n")
            result = await tool.execute(file_path=str(path), old_string="missing", new_string="replacement")
            assert result.success is False
            assert "not found" in (result.error or "")

    @pytest.mark.asyncio
    async def test_edit_missing_file(self, tool: EditTool) -> None:
        result = await tool.execute(file_path="/nonexistent/file", old_string="a", new_string="b")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_edit_accepts_path_alias(self, tool: EditTool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.py"
            path.write_text("old_function()\n")
            result = await tool.execute(path=str(path), old_string="old_function()", new_string="new_function()")
            assert result.success is True
            assert "new_function()" in path.read_text()

    @pytest.mark.asyncio
    async def test_edit_accepts_old_new_aliases(self, tool: EditTool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.py"
            path.write_text("foo bar\n")
            result = await tool.execute(path=str(path), old="foo", new="baz")
            assert result.success is True
            assert "baz bar" in path.read_text()


class TestMkdirTool:
    @pytest.fixture
    def tool(self) -> MkdirTool:
        return MkdirTool()

    @pytest.mark.asyncio
    async def test_create_directory(self, tool: MkdirTool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "hello"
            result = await tool.execute(path=str(target))
            assert result.success is True
            assert target.is_dir()

    @pytest.mark.asyncio
    async def test_create_nested_directories(self, tool: MkdirTool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "a" / "b" / "hello"
            result = await tool.execute(path=str(target))
            assert result.success is True
            assert target.is_dir()

    @pytest.mark.asyncio
    async def test_create_existing_is_ok(self, tool: MkdirTool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "hello"
            target.mkdir()
            result = await tool.execute(path=str(target))
            assert result.success is True

    @pytest.mark.asyncio
    async def test_create_over_file_fails(self, tool: MkdirTool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "hello"
            target.write_text("not a directory")
            result = await tool.execute(path=str(target))
            assert result.success is False

    @pytest.mark.asyncio
    async def test_create_with_dir_alias(self, tool: MkdirTool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "hello"
            result = await tool.execute(directory=str(target))
            assert result.success is True
            assert target.is_dir()


class TestGlobTool:
    @pytest.fixture
    def tool(self) -> GlobTool:
        return GlobTool()

    @pytest.mark.asyncio
    async def test_glob_finds_files(self, tool: GlobTool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "a.py").touch()
            Path(tmpdir, "b.py").touch()
            Path(tmpdir, "c.md").touch()
            result = await tool.execute(pattern="*.py", path=tmpdir)
            assert result.success is True
            assert result.data is not None
            assert result.data["count"] == 2


class TestGrepTool:
    @pytest.fixture
    def tool(self) -> GrepTool:
        return GrepTool()

    @pytest.mark.asyncio
    async def test_grep_finds_pattern(self, tool: GrepTool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("def hello():\n    pass\n")
            result = await tool.execute(pattern="def hello", path=tmpdir)
            assert result.success is True
            assert result.data is not None
            assert result.data["count"] == 1

    @pytest.mark.asyncio
    async def test_grep_no_match(self, tool: GrepTool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("nothing here\n")
            result = await tool.execute(pattern="missing", path=tmpdir)
            assert result.success is True
            assert result.data is not None
            assert result.data["count"] == 0

    @pytest.mark.asyncio
    async def test_grep_invalid_regex(self, tool: GrepTool) -> None:
        result = await tool.execute(pattern="[invalid")
        assert result.success is False


class TestBashTool:
    @pytest.fixture
    def tool(self) -> BashTool:
        return BashTool()

    @pytest.mark.asyncio
    async def test_echo(self, tool: BashTool) -> None:
        result = await tool.execute(command="echo hello")
        assert result.success is True
        assert result.data is not None
        assert "hello" in result.data["stdout"]

    @pytest.mark.asyncio
    async def test_failing_command(self, tool: BashTool) -> None:
        result = await tool.execute(command="exit 1")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_timeout(self, tool: BashTool) -> None:
        result = await tool.execute(command="sleep 10", timeout=100)
        assert result.success is False
        assert "timed out" in (result.error or "")


class TestGitTool:
    @pytest.fixture
    def tool(self) -> GitTool:
        return GitTool()

    @pytest.mark.asyncio
    async def test_not_a_repo(self, tool: GitTool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await tool.execute(command="status", repo_path=tmpdir)
            assert result.success is False
            assert "Not a git repository" in (result.error or "")


class TestToolRegistry:
    def test_register_and_get(self) -> None:
        ToolRegistry._tools.clear()
        tool = ReadTool()
        ToolRegistry.register(tool)
        assert ToolRegistry.get("read") is tool

    def test_get_unknown(self) -> None:
        ToolRegistry._tools.clear()
        assert ToolRegistry.get("nonexistent") is None

    def test_get_by_alias(self) -> None:
        ToolRegistry._tools.clear()
        tool = ReadTool()
        ToolRegistry.register(tool)
        assert ToolRegistry.get("read_file") is tool
        assert ToolRegistry.get("open_file") is tool

    def test_get_case_insensitive(self) -> None:
        ToolRegistry._tools.clear()
        tool = WriteTool()
        ToolRegistry.register(tool)
        assert ToolRegistry.get("WRITE") is tool
        assert ToolRegistry.get("Create_File") is tool

    def test_alias_does_not_leak_into_listing(self) -> None:
        ToolRegistry._tools.clear()
        ToolRegistry.register(ReadTool())
        names = [t.name for t in ToolRegistry.list_tools()]
        assert names == ["read"]
        assert "read_file" not in names

    def test_new_tool_aliases_resolve(self) -> None:
        ToolRegistry._tools.clear()
        append = AppendFileTool()
        run_python = RunPythonTool()
        debug = DebugTool()
        ToolRegistry.register(append)
        ToolRegistry.register(run_python)
        ToolRegistry.register(debug)
        ToolRegistry.register(ListDirectoryTool())
        assert ToolRegistry.get("append") is append
        assert ToolRegistry.get("append_to_file") is append
        assert ToolRegistry.get("execute_python") is run_python
        assert ToolRegistry.get("debug_code") is debug
        assert ToolRegistry.get("ls") is not None

    def test_list_tools(self) -> None:
        ToolRegistry._tools.clear()
        ToolRegistry.register(ReadTool())
        ToolRegistry.register(WriteTool())
        tools = ToolRegistry.list_tools()
        assert len(tools) == 2

    def test_get_json_schemas(self) -> None:
        ToolRegistry._tools.clear()
        ToolRegistry.register(ReadTool())
        schemas = ToolRegistry.get_json_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "read"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self) -> None:
        ToolRegistry._tools.clear()
        with pytest.raises(ValueError, match="Unknown tool"):
            await ToolRegistry.execute_tool("nonexistent")

    @pytest.mark.asyncio
    async def test_execute_via_alias(self, tmp_path) -> None:
        ToolRegistry._tools.clear()
        ToolRegistry.register(WriteTool())
        target = tmp_path / "a" / "b.txt"
        result = await ToolRegistry.execute_tool(
            "create_file", path=str(target), content="hello",
        )
        assert result.success
        assert result.tool_name == "write"
        assert target.read_text(encoding="utf-8") == "hello"


class TestAppendFileTool:
    @pytest.fixture
    def tool(self) -> AppendFileTool:
        return AppendFileTool()

    @pytest.mark.asyncio
    async def test_append_to_existing_file(self, tool: AppendFileTool, tmp_path: Path) -> None:
        target = tmp_path / "log.txt"
        target.write_text("first\n")
        result = await tool.execute(file_path=str(target), content="second\n")
        assert result.success is True
        assert target.read_text(encoding="utf-8") == "first\nsecond\n"

    @pytest.mark.asyncio
    async def test_append_creates_file(self, tool: AppendFileTool, tmp_path: Path) -> None:
        target = tmp_path / "sub" / "new.txt"
        result = await tool.execute(file_path=str(target), content="hello")
        assert result.success is True
        assert target.read_text(encoding="utf-8") == "hello"


class TestDeleteFileTool:
    @pytest.fixture
    def tool(self) -> DeleteFileTool:
        return DeleteFileTool()

    @pytest.mark.asyncio
    async def test_delete_existing_file(self, tool: DeleteFileTool, tmp_path: Path) -> None:
        target = tmp_path / "gone.txt"
        target.write_text("bye")
        result = await tool.execute(file_path=str(target))
        assert result.success is True
        assert not target.exists()

    @pytest.mark.asyncio
    async def test_delete_missing_file(self, tool: DeleteFileTool, tmp_path: Path) -> None:
        result = await tool.execute(file_path=str(tmp_path / "nope.txt"))
        assert result.success is False
        assert "not found" in (result.error or "")


class TestMoveFileTool:
    @pytest.fixture
    def tool(self) -> MoveFileTool:
        return MoveFileTool()

    @pytest.mark.asyncio
    async def test_move_file(self, tool: MoveFileTool, tmp_path: Path) -> None:
        src = tmp_path / "a.txt"
        dst = tmp_path / "b.txt"
        src.write_text("data")
        result = await tool.execute(source=str(src), destination=str(dst))
        assert result.success is True
        assert not src.exists()
        assert dst.read_text(encoding="utf-8") == "data"

    @pytest.mark.asyncio
    async def test_move_accepts_aliases(self, tool: MoveFileTool, tmp_path: Path) -> None:
        src = tmp_path / "a.txt"
        dst = tmp_path / "c.txt"
        src.write_text("data")
        result = await tool.execute(path=str(src), to=str(dst))
        assert result.success is True
        assert dst.exists()

    @pytest.mark.asyncio
    async def test_move_missing_source(self, tool: MoveFileTool, tmp_path: Path) -> None:
        result = await tool.execute(source=str(tmp_path / "nope.txt"), destination=str(tmp_path / "b.txt"))
        assert result.success is False


class TestListDirectoryTool:
    @pytest.fixture
    def tool(self) -> ListDirectoryTool:
        return ListDirectoryTool()

    @pytest.mark.asyncio
    async def test_list_directory(self, tool: ListDirectoryTool, tmp_path: Path) -> None:
        (tmp_path / "file_a.txt").write_text("x")
        (tmp_path / "file_b.txt").write_text("yy")
        (tmp_path / "folder").mkdir()
        result = await tool.execute(path=str(tmp_path))
        assert result.success is True
        assert result.data is not None
        names = {entry["name"] for entry in result.data["entries"]}
        assert names == {"file_a.txt", "file_b.txt", "folder"}
        assert result.data["count"] == 3

    @pytest.mark.asyncio
    async def test_list_missing_directory(self, tool: ListDirectoryTool) -> None:
        result = await tool.execute(path="/nonexistent/dir")
        assert result.success is False


class TestFileInfoTool:
    @pytest.fixture
    def tool(self) -> FileInfoTool:
        return FileInfoTool()

    @pytest.mark.asyncio
    async def test_file_info(self, tool: FileInfoTool, tmp_path: Path) -> None:
        target = tmp_path / "data.py"
        target.write_text("x" * 42)
        result = await tool.execute(path=str(target))
        assert result.success is True
        assert result.data is not None
        assert result.data["size"] == 42
        assert result.data["is_file"] is True
        assert result.data["extension"] == ".py"

    @pytest.mark.asyncio
    async def test_file_info_missing(self, tool: FileInfoTool) -> None:
        result = await tool.execute(path="/nonexistent/file")
        assert result.success is False


class TestRunPythonTool:
    @pytest.fixture
    def tool(self) -> RunPythonTool:
        return RunPythonTool()

    @pytest.mark.asyncio
    async def test_run_code_snippet(self, tool: RunPythonTool) -> None:
        result = await tool.execute(code="print('hello-bytecli')")
        assert result.success is True
        assert result.data is not None
        assert "hello-bytecli" in result.data["stdout"]

    @pytest.mark.asyncio
    async def test_run_script(self, tool: RunPythonTool, tmp_path: Path) -> None:
        script = tmp_path / "script.py"
        script.write_text("import sys\nprint('ran-from-file:' + sys.argv[0])")
        result = await tool.execute(file=str(script))
        assert result.success is True
        assert result.data is not None
        assert "ran-from-file:" in result.data["stdout"]

    @pytest.mark.asyncio
    async def test_run_error_captures_traceback(self, tool: RunPythonTool) -> None:
        result = await tool.execute(code="raise ValueError('boom')")
        assert result.success is False
        assert "Traceback" in (result.error or "")
        assert "boom" in (result.error or "")

    @pytest.mark.asyncio
    async def test_run_requires_input(self, tool: RunPythonTool) -> None:
        result = await tool.execute()
        assert result.success is False

    @pytest.mark.asyncio
    async def test_run_with_args_list_of_path(self, tool: RunPythonTool, tmp_path: Path) -> None:
        script = tmp_path / "hello.py"
        script.write_text("import sys\nprint('from-script:' + sys.argv[0])")
        result = await tool.execute(args=f"[{script}]")
        assert result.success is True
        assert result.data is not None
        assert "from-script:" in result.data["stdout"]

    @pytest.mark.asyncio
    async def test_run_with_args_passes_script_args(self, tool: RunPythonTool, tmp_path: Path) -> None:
        script = tmp_path / "echo_args.py"
        script.write_text("import sys\nprint(','.join(sys.argv[1:]))")
        result = await tool.execute(file=str(script), args=["one", "two"])
        assert result.success is True
        assert result.data is not None
        assert "one,two" in result.data["stdout"]


class TestRunTestsTool:
    @pytest.fixture
    def tool(self) -> RunTestsTool:
        return RunTestsTool()

    @pytest.mark.asyncio
    async def test_run_tests_pass(self, tool: RunTestsTool, tmp_path: Path) -> None:
        (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n")
        result = await tool.execute(path=str(tmp_path))
        assert result.success is True
        assert result.data is not None
        assert "passed" in result.data["stdout"]

    @pytest.mark.asyncio
    async def test_run_tests_missing_path(self, tool: RunTestsTool, tmp_path: Path) -> None:
        result = await tool.execute(path=str(tmp_path / "does_not_exist"))
        assert result.success is False


class TestDebugTool:
    @pytest.fixture
    def tool(self) -> DebugTool:
        return DebugTool()

    @pytest.mark.asyncio
    async def test_debug_command_output(self, tool: DebugTool) -> None:
        result = await tool.execute(command="echo debug-test-output")
        assert result.success is True
        assert result.data is not None
        assert "debug-test-output" in result.data["stdout"]

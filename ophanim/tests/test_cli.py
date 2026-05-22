"""Tests for CLI."""
import pytest
from typer.testing import CliRunner
from ophanim.cli.app import app

runner = CliRunner()


class TestCliSmoke:
    def test_help_shows(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Ophanim" in result.output or "visual perception" in result.output
    
    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # Typer may exit 0 (help shown) or 2 (error) depending on version
        assert result.exit_code in (0, 2)
        assert "Usage" in result.output
    
    def test_probe_nonexistent(self):
        result = runner.invoke(app, ["probe", "nonexistent.mp4"])
        assert result.exit_code != 0  # Should error
    
    def test_probe_with_test_video(self):
        """If test video exists, probe it."""
        import os
        test_video = "test_data/sample.mp4"
        if not os.path.exists(test_video):
            # Check for any mp4 in test_data
            from pathlib import Path
            videos = list(Path("test_data").glob("*.mp4")) if Path("test_data").exists() else []
            if not videos:
                pytest.skip("No test video available")
            test_video = str(videos[0])
        
        result = runner.invoke(app, ["probe", test_video])
        assert result.exit_code == 0
        assert "Duration" in result.output
        assert "Resolution" in result.output
    
    def test_probe_json_output(self):
        """Test JSON output format."""
        from pathlib import Path
        videos = list(Path("test_data").glob("*.mp4")) if Path("test_data").exists() else []
        if not videos:
            pytest.skip("No test video available")
        
        result = runner.invoke(app, ["probe", str(videos[0]), "--json"])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert "duration_seconds" in data
        assert "width" in data
    
    def test_status_stub(self):
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
    
    def test_observe_nonexistent(self):
        result = runner.invoke(app, ["observe", "test.mp4"])
        assert result.exit_code != 0  # File not found
    
    def test_ask_nonexistent(self):
        result = runner.invoke(app, ["ask", "test.mp4", "what is this?"])
        assert result.exit_code != 0  # File not found

    def test_memory_help(self):
        result = runner.invoke(app, ["memory", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "view" in result.output
        assert "delete" in result.output

    def test_memory_list_empty(self):
        result = runner.invoke(app, ["memory", "list"])
        assert result.exit_code == 0
        assert "No saved memories" in result.output

    def test_memory_view_no_name(self):
        result = runner.invoke(app, ["memory", "view"])
        assert result.exit_code != 0

    def test_memory_delete_no_name(self):
        result = runner.invoke(app, ["memory", "delete"])
        assert result.exit_code != 0

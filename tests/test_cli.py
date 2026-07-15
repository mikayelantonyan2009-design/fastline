from workspace import __version__
from workspace.cli import main


def test_version():
    assert __version__ == "0.1.0"


def test_main_runs(capsys):
    main()
    assert "Hello from workspace!" in capsys.readouterr().out

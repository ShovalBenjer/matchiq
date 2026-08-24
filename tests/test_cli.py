from wc2026.cli import main


def test_cli_info(capsys):
    rc = main(["info"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "betting" in out


def test_cli_ingest(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("WC2026_CONFIG", "")
    rc = main(["ingest"])
    assert rc == 0
    assert "Ingested" in capsys.readouterr().out

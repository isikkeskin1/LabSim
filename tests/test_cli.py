from labsim.cli import main


def test_cli_runs_built_in_model(capsys):
    assert main(["oscillator", "--duration", "0.1", "--dt", "0.1"]) == 0
    output = capsys.readouterr().out
    assert "model=oscillator" in output
    assert "samples=2" in output


def test_cli_can_export_csv(tmp_path, capsys):
    output_path = tmp_path / "run.csv"
    assert main(["damped", "--duration", "0.2", "--dt", "0.1", "--output", str(output_path)]) == 0
    capsys.readouterr()
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").startswith("time,state_0,state_1")

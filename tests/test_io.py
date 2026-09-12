from labsim import integrate_ode
from labsim.io import write_csv


def test_write_csv_exports_time_and_state_columns(tmp_path):
    solution = integrate_ode(lambda _time, state: (1.0,), (0.0,), dt=0.5, steps=2)
    output = tmp_path / "trajectory.csv"

    write_csv(solution, output)

    assert output.read_text(encoding="utf-8").splitlines() == [
        "time,state_0",
        "0.0,0.0",
        "0.5,0.5",
        "1.0,1.0",
    ]

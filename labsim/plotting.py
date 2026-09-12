"""Optional Matplotlib visualization helpers for simulation results."""

from __future__ import annotations

from .solvers import ODESolution


def plot_solution(
    solution: ODESolution,
    *,
    labels: tuple[str, ...] | None = None,
    ax=None,
):
    """Plot every state component against time and return the Matplotlib axes.

    Matplotlib is imported lazily so the numerical core remains dependency-free.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "plot_solution requires matplotlib; install it with 'pip install matplotlib'"
        ) from exc

    if labels is not None and len(labels) != solution.state_dimension:
        raise ValueError("labels must match the solution state dimension")

    if ax is None:
        _, ax = plt.subplots()

    for index in range(solution.state_dimension):
        values = [state[index] for state in solution.states]
        label = labels[index] if labels is not None else f"state_{index}"
        ax.plot(solution.times, values, label=label)

    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return ax

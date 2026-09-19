"""Compare forward and centered numerical derivatives as the step shrinks."""

import math
from pathlib import Path


def _check_step(step):
    if isinstance(step, bool) or not isinstance(step, (int, float)):
        raise ValueError("step needs to be a positive finite number")
    if not math.isfinite(step) or step <= 0:
        raise ValueError("step needs to be a positive finite number")


def forward_difference(function, x, step):
    """Estimate a derivative from the point and one point to its right."""
    _check_step(step)
    return (function(x + step) - function(x)) / step


def centered_difference(function, x, step):
    """Estimate a derivative using one point on either side."""
    _check_step(step)
    return (function(x + step) - function(x - step)) / (2 * step)


def error_table(function, derivative, x, steps):
    """Return the absolute errors of both formulas for each step."""
    exact = derivative(x)
    rows = []

    for step in steps:
        forward_error = abs(forward_difference(function, x, step) - exact)
        centered_error = abs(centered_difference(function, x, step) - exact)
        rows.append((step, forward_error, centered_error))

    return rows


def save_error_plot(rows, filename=None):
    """Save the two error curves on logarithmic axes."""
    from matplotlib import pyplot as plt

    if filename is None:
        filename = Path(__file__).with_name("finite_difference_accuracy.png")

    steps = [row[0] for row in rows]
    forward_errors = [row[1] for row in rows]
    centered_errors = [row[2] for row in rows]

    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.loglog(steps, forward_errors, marker="o", label="forward difference")
    axis.loglog(steps, centered_errors, marker="o", label="centered difference")
    axis.invert_xaxis()
    axis.set(
        xlabel="step size",
        ylabel="absolute error",
        title="Smaller steps help, until rounding gets in the way",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def run_example():
    x = 1.0
    steps = [10.0**-power for power in range(1, 17)]
    rows = error_table(math.sin, math.cos, x, steps)

    print("Approximating the derivative of sin(x) at x = 1")
    print(f"Exact value: cos(1) = {math.cos(x):.12f}\n")
    print(f"{'step':>10}  {'forward error':>15}  {'centered error':>15}")
    for step, forward_error, centered_error in rows:
        print(f"{step:10.0e}  {forward_error:15.3e}  {centered_error:15.3e}")

    best = min(rows, key=lambda row: row[2])
    print(f"\nSmallest centered error here used a step of {best[0]:.0e}.")
    # At the far end, x + h can be stored as the same number as x.
    print("The very smallest steps get worse again because floats have limited precision.")

    save_error_plot(rows)
    print("Saved finite_difference_accuracy.png")


if __name__ == "__main__":
    run_example()

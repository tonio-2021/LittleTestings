"""Trying a few gradient descent step sizes on a simple quadratic."""

from pathlib import Path


def quadratic_value(x, centre=1.0, curvature=2.0):
    """Value of a quadratic whose minimum is at centre."""
    return 0.5 * curvature * (x - centre) ** 2


def gradient_descent(start, step_size, iterations, centre=1.0, curvature=2.0):
    """Return the visited points and function values."""
    if step_size <= 0:
        raise ValueError("step_size has to be positive")
    if iterations < 0:
        raise ValueError("iterations cannot be negative")
    if curvature <= 0:
        raise ValueError("curvature has to be positive")

    x = start
    positions = [x]
    values = [quadratic_value(x, centre, curvature)]

    for _ in range(iterations):
        gradient = curvature * (x - centre)
        x = x - step_size * gradient

        # I keep every step because the jumping is the interesting part here.
        positions.append(x)
        values.append(quadratic_value(x, centre, curvature))

    return positions, values


def save_comparison_plot(filename=None):
    """Run three step sizes and save their paths in one figure."""
    from matplotlib import pyplot as plt

    if filename is None:
        filename = Path(__file__).with_name("quadratic_gradient_descent.png")

    start = -4.0
    centre = 1.0
    curvature = 2.0
    iterations = 18
    experiments = [
        ("small step", 0.2),
        ("close to the limit", 0.9),
        ("too large", 1.05),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for label, step_size in experiments:
        positions, values = gradient_descent(
            start,
            step_size,
            iterations,
            centre,
            curvature,
        )
        iteration_numbers = range(len(positions))
        axes[0].plot(iteration_numbers, positions, marker="o", markersize=3, label=label)
        axes[1].semilogy(iteration_numbers, values, marker="o", markersize=3, label=label)

    axes[0].axhline(centre, color="black", linewidth=1, linestyle="--")
    axes[0].set(
        xlabel="iteration",
        ylabel="x",
        title="Where the updates move",
    )
    axes[1].set(
        xlabel="iteration",
        ylabel="quadratic value",
        title="What happens to the objective",
    )
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("The step-size limit is 1 for this quadratic")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def run_example():
    curvature = 2.0
    limit = 2 / curvature

    print(f"The stability limit for the step size is {limit:.2f}.\n")

    for label, step_size in [
        ("small", 0.2),
        ("near limit", 0.9),
        ("too large", 1.05),
    ]:
        positions, values = gradient_descent(-4.0, step_size, 18)
        print(
            f"{label:10} step={step_size:>4}: "
            f"final x={positions[-1]:>8.4f}, final value={values[-1]:.6f}"
        )

    save_comparison_plot()
    print("\nSaved the comparison as optimization/quadratic_gradient_descent.png")


if __name__ == "__main__":
    run_example()

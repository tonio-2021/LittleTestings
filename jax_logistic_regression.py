"""A small binary classifier trained with JAX gradients."""

import jax
import jax.numpy as jnp


def predict_probabilities(params, x):
    scores = x @ params[:2] + params[2]
    return jax.nn.sigmoid(scores)


def logistic_loss(params, x, y):
    scores = x @ params[:2] + params[2]
    # This version also behaves nicely when a score gets quite large.
    return jnp.mean(jnp.logaddexp(0.0, scores) - y * scores)


def check_data(x, y):
    x = jnp.asarray(x, dtype=float)
    y = jnp.asarray(y, dtype=float)

    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] == 0:
        raise ValueError("x needs to contain rows with two features")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("y needs one label for every row in x")
    if not bool(jnp.all((y == 0) | (y == 1))):
        raise ValueError("the labels need to be zero or one")

    return x, y


def train_classifier(x, y, step_size=0.2, steps=200):
    """Learn two weights and an intercept with gradient descent."""
    x, y = check_data(x, y)
    if step_size <= 0 or steps < 0:
        raise ValueError("step_size must be positive and steps cannot be negative")

    # The last value is the intercept, so I do not add a column of ones to x.
    params = jnp.zeros(3)
    losses = [float(logistic_loss(params, x, y))]
    gradient = jax.grad(logistic_loss)

    for _ in range(steps):
        params = params - step_size * gradient(params, x, y)
        losses.append(float(logistic_loss(params, x, y)))

    return params, losses


def classify(params, x, threshold=0.5):
    if not 0 < threshold < 1:
        raise ValueError("threshold needs to be between zero and one")
    x = jnp.asarray(x, dtype=float)
    return (predict_probabilities(params, x) >= threshold).astype(int)


def save_plot(params, x, y, filename="jax_logistic_regression.png"):
    import matplotlib.pyplot as plt

    x, y = check_data(x, y)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x[y == 0, 0], x[y == 0, 1], label="group 0", color="tab:blue")
    ax.scatter(x[y == 1, 0], x[y == 1, 1], label="group 1", color="tab:orange")

    weight_1, weight_2, intercept = [float(value) for value in params]
    if abs(weight_2) > 1e-10:
        left = float(jnp.min(x[:, 0])) - 0.4
        right = float(jnp.max(x[:, 0])) + 0.4
        line_x = jnp.array([left, right])
        line_y = -(weight_1 * line_x + intercept) / weight_2
        ax.plot(line_x, line_y, "--", color="black", label="learned boundary")

    ax.set_xlabel("first feature")
    ax.set_ylabel("second feature")
    ax.set_title("A tiny classifier learned with JAX")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def run_example():
    x = jnp.array(
        [
            [-2.0, -1.0],
            [-1.5, -2.0],
            [-1.0, -0.4],
            [0.1, -1.2],
            [0.3, 1.2],
            [1.0, 0.4],
            [1.5, 2.0],
            [2.0, 1.0],
        ]
    )
    y = jnp.array([0, 0, 0, 0, 1, 1, 1, 1])

    params, losses = train_classifier(x, y)
    predictions = classify(params, x)
    accuracy = float(jnp.mean(predictions == y))

    print("Separating two small groups of points")
    print(f"Starting loss:     {losses[0]:.4f}")
    print(f"After 20 updates:  {losses[20]:.4f}")
    print(f"After 200 updates: {losses[-1]:.4f}")
    print(f"Training accuracy: {accuracy:.0%}")
    print(f"Learned values:    {params}")

    save_plot(params, x, y)
    print("Saved jax_logistic_regression.png")


if __name__ == "__main__":
    run_example()

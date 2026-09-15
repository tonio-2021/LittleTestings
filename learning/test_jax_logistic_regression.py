import math
import unittest

import jax
import jax.numpy as jnp

from learning.jax_logistic_regression import (
    classify,
    logistic_loss,
    predict_probabilities,
    train_classifier,
)


class JaxLogisticRegressionTests(unittest.TestCase):
    def setUp(self):
        self.x = jnp.array(
            [
                [-2.0, -1.0],
                [-1.0, -0.4],
                [0.1, -1.2],
                [0.3, 1.2],
                [1.0, 0.4],
                [2.0, 1.0],
            ]
        )
        self.y = jnp.array([0, 0, 0, 1, 1, 1])

    def test_zero_scores_give_half_probability(self):
        probabilities = predict_probabilities(jnp.zeros(3), self.x)

        self.assertTrue(bool(jnp.allclose(probabilities, 0.5)))

    def test_loss_stays_finite_for_large_scores(self):
        params = jnp.array([1000.0, -1000.0, 500.0])
        loss = logistic_loss(params, self.x, self.y)

        self.assertTrue(math.isfinite(float(loss)))

    def test_jax_gradient_matches_a_small_finite_difference(self):
        params = jnp.array([0.3, -0.2, 0.1])
        jax_gradient = jax.grad(logistic_loss)(params, self.x, self.y)
        epsilon = 0.001
        finite_difference = []

        for index in range(3):
            movement = jnp.zeros(3).at[index].set(epsilon)
            upper = logistic_loss(params + movement, self.x, self.y)
            lower = logistic_loss(params - movement, self.x, self.y)
            finite_difference.append((upper - lower) / (2 * epsilon))

        self.assertTrue(
            bool(jnp.allclose(jax_gradient, jnp.array(finite_difference), atol=0.001))
        )

    def test_training_lowers_loss_and_separates_the_points(self):
        params, losses = train_classifier(self.x, self.y)
        predictions = classify(params, self.x)

        self.assertEqual(len(losses), 201)
        self.assertLess(losses[-1], losses[0])
        self.assertTrue(bool(jnp.all(predictions == self.y)))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            train_classifier([[0, 1]], [0, 1])
        with self.assertRaises(ValueError):
            train_classifier([[0, 1]], [2])
        with self.assertRaises(ValueError):
            train_classifier([[0, 1]], [0], step_size=0)
        with self.assertRaises(ValueError):
            classify(jnp.zeros(3), [[0, 1]], threshold=1)


if __name__ == "__main__":
    unittest.main()

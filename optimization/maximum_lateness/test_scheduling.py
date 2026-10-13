import math
import random
import unittest

from optimization.maximum_lateness.scheduling import (
    Job,
    brute_force_schedule,
    earliest_deadline_schedule,
    evaluate_order,
)


class MaximumLatenessTests(unittest.TestCase):
    def setUp(self):
        self.jobs = [
            Job("write", 4, 4),
            Job("email", 1, 2),
            Job("review", 2, 7),
            Job("upload", 3, 9),
        ]

    def test_evaluate_order_calculates_each_lateness(self):
        schedule = evaluate_order(self.jobs)

        self.assertEqual(
            [item.completion_time for item in schedule.jobs],
            [4, 5, 7, 10],
        )
        self.assertEqual([item.lateness for item in schedule.jobs], [0, 3, 0, 1])
        self.assertEqual(schedule.maximum_lateness, 3)

    def test_earliest_deadline_uses_expected_order(self):
        schedule = earliest_deadline_schedule(self.jobs)

        self.assertEqual(
            [item.job.name for item in schedule.jobs],
            ["email", "write", "review", "upload"],
        )
        self.assertEqual(schedule.maximum_lateness, 1)

    def test_greedy_matches_brute_force(self):
        greedy = earliest_deadline_schedule(self.jobs)
        exact = brute_force_schedule(self.jobs)

        self.assertEqual(greedy.maximum_lateness, exact.maximum_lateness)

    def test_random_small_inputs_match_brute_force(self):
        rng = random.Random(9)

        for trial in range(12):
            jobs = [
                Job(
                    f"job-{trial}-{index}",
                    processing_time=rng.randint(1, 5),
                    deadline=rng.randint(2, 15),
                )
                for index in range(5)
            ]
            with self.subTest(trial=trial):
                greedy = earliest_deadline_schedule(jobs)
                exact = brute_force_schedule(jobs)
                self.assertEqual(
                    greedy.maximum_lateness,
                    exact.maximum_lateness,
                )

    def test_empty_schedule(self):
        schedule = earliest_deadline_schedule([])

        self.assertEqual(schedule.jobs, ())
        self.assertEqual(schedule.maximum_lateness, 0)

    def test_bad_jobs_are_rejected(self):
        bad_lists = [
            [Job("zero", 0, 2)],
            [Job("late", 1, math.inf)],
            [Job("same", 1, 2), Job("same", 2, 4)],
            [("not", "a", "job")],
        ]

        for jobs in bad_lists:
            with self.subTest(jobs=jobs):
                with self.assertRaises(ValueError):
                    earliest_deadline_schedule(jobs)


if __name__ == "__main__":
    unittest.main()

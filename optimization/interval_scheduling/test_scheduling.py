import unittest

from optimization.interval_scheduling.scheduling import (
    Interval,
    are_compatible,
    greedy_schedule,
    maximum_schedule,
)


class IntervalSchedulingTests(unittest.TestCase):
    def test_touching_intervals_are_compatible(self):
        intervals = [
            Interval("first", 0, 2),
            Interval("second", 2, 4),
            Interval("third", 4, 5),
        ]

        self.assertTrue(are_compatible(intervals))

    def test_earliest_finish_avoids_long_early_interval(self):
        intervals = [
            Interval("long", 0, 10),
            Interval("a", 1, 2),
            Interval("b", 2, 3),
            Interval("c", 3, 4),
            Interval("d", 4, 5),
        ]

        earliest_finish = greedy_schedule(intervals)
        earliest_start = greedy_schedule(intervals, "earliest_start")

        self.assertEqual([item.name for item in earliest_finish], ["a", "b", "c", "d"])
        self.assertEqual([item.name for item in earliest_start], ["long"])

    def test_shortest_duration_can_block_two_intervals(self):
        intervals = [
            Interval("left", 0, 3),
            Interval("middle", 2, 4),
            Interval("right", 3, 6),
        ]

        earliest_finish = greedy_schedule(intervals)
        shortest = greedy_schedule(intervals, "shortest_duration")

        self.assertEqual([item.name for item in earliest_finish], ["left", "right"])
        self.assertEqual([item.name for item in shortest], ["middle"])

    def test_duration_rule_can_still_take_an_earlier_interval(self):
        intervals = [
            Interval("early", 0, 2),
            Interval("short late", 10, 11),
        ]

        chosen = greedy_schedule(intervals, "shortest_duration")

        self.assertEqual([item.name for item in chosen], ["early", "short late"])

    def test_greedy_matches_brute_force_on_small_example(self):
        intervals = [
            Interval("a", 0, 3),
            Interval("b", 1, 2),
            Interval("c", 2, 5),
            Interval("d", 4, 7),
            Interval("e", 5, 6),
            Interval("f", 6, 8),
        ]

        greedy = greedy_schedule(intervals)
        optimum = maximum_schedule(intervals)

        self.assertTrue(are_compatible(greedy))
        self.assertEqual(len(greedy), len(optimum))

    def test_empty_schedule(self):
        self.assertEqual(greedy_schedule([]), [])
        self.assertEqual(maximum_schedule([]), [])

    def test_bad_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            greedy_schedule([Interval("backwards", 3, 2)])
        with self.assertRaises(ValueError):
            greedy_schedule([Interval("same", 0, 1), Interval("same", 2, 3)])
        with self.assertRaises(ValueError):
            greedy_schedule([Interval("one", 0, 1)], "latest_finish")


if __name__ == "__main__":
    unittest.main()

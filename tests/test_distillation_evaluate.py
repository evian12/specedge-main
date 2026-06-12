import unittest

from distillation.evaluate_alignment import _accept_depths


class AcceptanceDepthTest(unittest.TestCase):
    def test_counts_each_position_as_a_cycle_start(self):
        self.assertEqual(
            _accept_depths(
                [True, True, False, True, True, True],
                max_depth=2,
            ),
            [2, 1, 0, 2, 2, 1],
        )


if __name__ == "__main__":
    unittest.main()

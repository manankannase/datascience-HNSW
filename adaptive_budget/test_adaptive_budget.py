import unittest

from adaptive_budget import AdaptiveClusterBudget


class AdaptiveClusterBudgetTests(unittest.TestCase):
    def setUp(self):
        self.policy = AdaptiveClusterBudget(
            budgets=(1, 2, 4, 8),
            thresholds=(0.05, 0.20, 0.50),
        )

    def test_large_margin_uses_small_budget(self):
        self.assertEqual(self.policy.choose([1.0, 2.0, 3.0]), 1)

    def test_small_margin_uses_largest_budget(self):
        self.assertEqual(self.policy.choose([1.0, 1.01, 2.0]), 8)

    def test_medium_margins_use_intermediate_budgets(self):
        self.assertEqual(self.policy.choose([1.0, 1.10, 2.0]), 4)
        self.assertEqual(self.policy.choose([1.0, 1.35, 2.0]), 2)
        self.assertEqual(self.policy.choose([1.0, 1.75, 2.0]), 1)

    def test_invalid_distances_are_rejected(self):
        with self.assertRaises(ValueError):
            self.policy.choose([1.0])
        with self.assertRaises(ValueError):
            self.policy.choose([1.0, float("nan")])


if __name__ == "__main__":
    unittest.main()

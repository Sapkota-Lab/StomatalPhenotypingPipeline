import unittest

from build_external_multidomain_dataset import split_and_balance


class MultidomainSplitTests(unittest.TestCase):
    def test_validation_is_disjoint_and_training_is_balanced(self):
        training, validation = split_and_balance(
            list(range(10)),
            validation_images=3,
            training_records=20,
            seed=7,
        )

        self.assertEqual(len(training), 20)
        self.assertEqual(len(validation), 3)
        self.assertTrue(set(training).isdisjoint(validation))
        self.assertEqual(len(set(training)), 7)

    def test_split_is_deterministic(self):
        arguments = {
            "validation_images": 2,
            "training_records": 8,
            "seed": 42,
        }
        first = split_and_balance(list(range(6)), **arguments)
        second = split_and_balance(list(range(6)), **arguments)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

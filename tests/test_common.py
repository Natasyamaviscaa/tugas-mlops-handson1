import unittest

from src.common import dataset_md5, load_split, set_seed


class SplitTest(unittest.TestCase):
    def test_split_is_reproducible_and_stratified(self):
        first = load_split()
        second = load_split()
        self.assertTrue(first[0].equals(second[0]))
        self.assertEqual(len(first[0]), 455)
        self.assertEqual(len(first[1]), 114)
        self.assertAlmostEqual(first[2].mean(), first[3].mean(), delta=0.01)

    def test_hash_and_seed_are_stable(self):
        set_seed()
        first_hash = dataset_md5()
        set_seed()
        self.assertEqual(first_hash, dataset_md5())
        self.assertEqual(len(first_hash), 32)


if __name__ == "__main__":
    unittest.main()


import unittest

from src.deep_learning import train_model


class DeepLearningTest(unittest.TestCase):
    def test_one_epoch_training_produces_valid_metrics(self):
        _, history, _ = train_model(epochs=1, batch_size=128)
        self.assertEqual(len(history), 1)
        self.assertGreater(history[0]["train_loss"], 0)
        self.assertGreaterEqual(history[0]["val_f1"], 0)
        self.assertLessEqual(history[0]["val_f1"], 1)
        self.assertGreaterEqual(history[0]["val_auc"], 0)
        self.assertLessEqual(history[0]["val_auc"], 1)


if __name__ == "__main__":
    unittest.main()

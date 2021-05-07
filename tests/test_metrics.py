from utils import roc_auc_grouped, precision_at_k_grouped
from sklearn.metrics import roc_auc_score
import numpy as np
import unittest


def get_random_data():
    scores = np.random.normal(size=(500,))
    truth = np.random.binomial(n=1, p=0.1, size=(500,))
    return scores, truth


def precision_at_10_naive(y_true, y_pred):
    order = np.argsort(-y_pred)
    return y_true[order[:10]].mean()


class TestMetrics(unittest.TestCase):
    def test_precision(self):
        # generate two users
        scores_1, truth_1 = get_random_data()
        scores_2, truth_2 = get_random_data()

        # calculate p@10 naively
        precision_1 = precision_at_10_naive(truth_1, scores_1)
        precision_2 = precision_at_10_naive(truth_2, scores_2)

        # calculate p@k with optimized code
        precisions_fast = precision_at_k_grouped(np.hstack((truth_1, truth_2)),
                                                 np.hstack((scores_1, scores_2)),
                                                 np.array([0] * len(scores_1) + [1] * len(scores_2)),
                                                 k=10,
                                                 return_precision_list=True)

        diff_1 = np.absolute(precision_1 - precisions_fast[0])
        diff_2 = np.absolute(precision_2 - precisions_fast[1])
        self.assertTrue(diff_1 < 1e-4 and diff_2 < 1e-4)

    def test_auc(self):
        # generate two users
        scores_1, truth_1 = get_random_data()
        scores_2, truth_2 = get_random_data()

        # calculate auc-roc with sklearn
        auc_1 = roc_auc_score(truth_1, scores_1)
        auc_2 = roc_auc_score(truth_2, scores_2)

        # calculate auc-roc with optimized code
        aucs_fast = roc_auc_grouped(np.hstack((truth_1, truth_2)),
                                    np.hstack((scores_1, scores_2)),
                                    np.array([0] * len(scores_1) + [1] * len(scores_2)),
                                    return_aucs_list=True)
        diff_1 = np.absolute(auc_1 - aucs_fast[0])
        diff_2 = np.absolute(auc_2 - aucs_fast[1])
        self.assertTrue(diff_1 < 1e-4 and diff_2 < 1e-4)

if __name__ == '__main__':
    unittest.main()
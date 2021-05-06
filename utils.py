import numpy as np
from scipy import sparse
from scipy.stats import ttest_rel
from typing import Union, Tuple


def load_data(data_path, delimiter):
    raw_data = np.loadtxt(data_path, dtype=np.float, delimiter=delimiter, usecols=[0, 1, 2])
    users = list(set(raw_data[:, 0].astype(np.int)))
    users.sort()
    user_dict = {k: i for i, k in enumerate(users)}
    items = list(set(raw_data[:, 1].astype(np.int)))
    items.sort()
    item_dict = {k: i for i, k in enumerate(items)}
    for i in range(len(raw_data)):
        raw_data[i, 0] = user_dict[raw_data[i, 0]]
        raw_data[i, 1] = item_dict[raw_data[i, 1]]
    return raw_data


def build_user_item_matrix(ratings, n_user, n_item):
    data = ratings[:, 2]
    row_index = ratings[:, 0]
    col_index = ratings[:, 1]
    shape = (n_user, n_item)
    return sparse.csr_matrix((data, (row_index, col_index)), shape=shape)


def RMSE(estimation, truth):
    truth_coo = truth.tocoo()
    row_idx = truth_coo.row
    col_idx = truth_coo.col
    data = truth.data
    pred = np.zeros(shape=data.shape)
    for i in range(len(data)):
        pred[i] = estimation[row_idx[i], col_idx[i]]
    sse = np.sum(np.square(data - pred))
    return np.sqrt(np.divide(sse, len(data)))


def RMSE_with_ttest(estimation, old_estimation, truth):
    truth_coo = truth.tocoo()
    row_idx = truth_coo.row
    col_idx = truth_coo.col
    data = truth_coo.data
    pred_dis = np.zeros(shape=data.shape)
    old_pred_dis = np.zeros(shape=data.shape)
    for i in range(len(data)):
        pred_dis[i] = abs(estimation[row_idx[i], col_idx[i]] - data[i])
        old_pred_dis[i] = abs(old_estimation[row_idx[i], col_idx[i]] - data[i])
    _, p_value = ttest_rel(pred_dis, old_pred_dis)
    sse = np.sum(np.square(pred_dis))
    sse_old = np.sum(np.square(old_pred_dis))
    return np.sqrt(np.divide(sse, len(data))), np.sqrt(np.divide(sse_old, len(data))), p_value


def RMSE_weighted_with_t_test(estimation, old_estimation, val_confidence):
    val_confidence_dense = val_confidence.toarray()
    val_preference_dense = val_confidence_dense.copy()
    val_preference_dense[val_preference_dense > 0] = 1
    val_confidence_dense[val_confidence_dense == 0] = 1
    old_error = val_confidence_dense * np.power(old_estimation - val_preference_dense, 2)
    new_error = val_confidence_dense * np.power(estimation - val_preference_dense, 2)
    _, p_val = ttest_rel(new_error.flatten(), old_error.flatten())
    return np.sqrt(np.mean(new_error)), np.sqrt(np.mean(old_error)), p_val


def roc_auc_grouped(labels: np.ndarray,
                    predictions: np.ndarray,
                    group_ids: np.ndarray,
                    return_aucs_list=False) -> Union[Tuple[float, float, int], np.ndarray]:
    # l_max = labels.max()
    # l_min = labels.min()
    # logging.info(str(l_max) + ' ' + str(l_min))
    # labels = (labels > l_max * 0.8).astype(int)
    # sort group_ids, predictions and labels jointly by (group_id, prediction) key
    indices = np.lexsort((predictions, group_ids))
    group_ids = group_ids[indices]
    labels = labels[indices]

    # unique monotonic group_id
    _, group_ids2 = np.unique(group_ids, return_inverse=True)
    _, unique_counts = np.unique(group_ids, return_counts=True)

    offsets = np.cumsum(unique_counts)
    offsets = np.insert(offsets, 0, 0)

    # number of negatives up to current element
    nneg_thru = np.cumsum(1 - labels)

    # number of negatives at the beginning of each group
    group_starts = nneg_thru[offsets - 1]
    group_starts[0] = 0

    # number of negatives up to current element, restarting at each group
    nneg = nneg_thru - group_starts[group_ids2]

    # number of ordered pairs with the current element
    inversions = (nneg * labels)

    # number of negatives in each group
    nneg_counts = nneg[offsets[1:] - 1]
    npos_counts = unique_counts - nneg_counts

    total_pairs = nneg_counts * npos_counts

    # Number of ordered pairs in each group
    ordered_pairs = np.bincount(group_ids2, weights=inversions)

    aucs = ordered_pairs[total_pairs > 0] / total_pairs[total_pairs > 0]

    if return_aucs_list:
        return aucs
    else:
        return float(np.mean(aucs)), float(np.std(aucs)), int(np.sum(total_pairs > 0))


def roc_auc_with_t_test(estimation, old_estimation, truth):
    user_ids = np.repeat(np.array(range(estimation.shape[1])), estimation.shape[0])
    aucs_old = roc_auc_grouped(truth.toarray().flatten(), old_estimation.flatten(), user_ids, True)
    aucs_new = roc_auc_grouped(truth.toarray().flatten(), estimation.flatten(), user_ids, True)
    _, p_value = ttest_rel(aucs_new, aucs_old)
    return np.mean(aucs_new), np.mean(aucs_old), p_value


def precision_at_k_grouped(labels: np.ndarray,
                           predictions: np.ndarray,
                           group_ids: np.ndarray,
                           k: int = 10,
                           return_precision_list: bool = False) -> Union[Tuple[float, float, int], np.ndarray]:
    # l_max = labels.max()
    # l_min = labels.min()
    # logging.info(str(l_max) + ' ' + str(l_min))
    # labels = (labels > l_max * 0.8).astype(int)
    # sort group_ids, predictions and labels jointly by (group_id, prediction) key
    indices = np.lexsort((-predictions, group_ids))
    group_ids = group_ids[indices]
    labels = labels[indices]

    # 0000, 1111, 222, 3, 555555

    # unique monotonic group_id
    _, group_ids2 = np.unique(group_ids, return_inverse=True)
    _, unique_counts = np.unique(group_ids, return_counts=True)

    offsets = np.cumsum(unique_counts)
    offsets = np.insert(offsets, 0, 0)

    # independent indexing in each group. e.g., [0, 1, 2, 0, 0, 0, 1, 2, 3, 4, 0, 1, 2]
    group_indices = np.arange(group_ids.shape[0]) - offsets[group_ids2]

    # number of points in each group or k
    denominator = np.minimum(unique_counts[group_ids2], np.repeat(k, group_ids.shape[0]))
    pr_at_k_vals = labels / denominator

    pr_at_k_vals[group_indices >= k] = 0

    group_pr_at_k = np.zeros(unique_counts.shape[0])
    np.add.at(group_pr_at_k, group_ids2, pr_at_k_vals)

    if return_precision_list:
        return group_pr_at_k
    else:
        return float(np.mean(group_pr_at_k)), float(np.std(group_pr_at_k)), group_pr_at_k.shape[0]


def precision_at_10_with_t_test(estimation, old_estimation, truth):
    user_ids = np.repeat(np.array(range(estimation.shape[1])), estimation.shape[0])
    precisions_old = precision_at_k_grouped(truth.toarray().flatten(), old_estimation.flatten(), user_ids, 10, True)
    precisions_new = precision_at_k_grouped(truth.toarray().flatten(), estimation.flatten(), user_ids, 10, True)
    _, p_value = ttest_rel(precisions_new, precisions_old)
    return np.mean(precisions_new), np.mean(precisions_old), p_value


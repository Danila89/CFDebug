import argparse
import os
from utils import *
from copy import deepcopy
from multiprocessing import Process
from numpy.linalg import inv
from scipy import sparse
import logging
import implicit

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# hyperparameters
parser = argparse.ArgumentParser(description="CFDebug")
parser.add_argument("--dataset", type=str, default="movielens", help="dataset")
parser.add_argument("--delim", type=str, default="::", help="delimiter of each line in the dataset file")
parser.add_argument("--fold", type=int, default=4, help="# of fold to split the data")
parser.add_argument("--factor", type=int, default=10, help="# of dimension parameter of the CF model")
parser.add_argument("--lambda_u", type=float, default=0.1, help="regularization parameter lambda_u of the CF model")
parser.add_argument("--lambda_v", type=float, default=0.1, help="regularization parameter lambda_v of the CF model")
parser.add_argument("--als_iter", type=int, default=15, help="# of iterations for ALS training")
parser.add_argument("--debug_iter", type=int, default=20, help="# of iterations in the debugging stage")
parser.add_argument("--debug_lr", type=float, default=0.05, help="learning rate in the debugging stage")
parser.add_argument("--retrain", type=str, default="full", help="the retraining mode in the debugging stage: full/inc")
parser.add_argument("--process", type=int, default=4, help="# of processes in the debugging stage")
parser.add_argument("--mode", type=str, default="debug", help="debug/test")
parser.add_argument("--implicit", action='store_true', help="use implicit ALS")
parser.add_argument("--alpha", type=int, default=1, help="confidence scaling for implicit feedback dataset")
parser.add_argument("--als_threads", type=int, default=6, help="num threads during implicit ALS fit")

args = parser.parse_args()


def partition(ratings, seed, fold):
    np.random.RandomState(seed).shuffle(ratings)
    test_size = int(0.2 * ratings.shape[0])

    test_data = ratings[:test_size]
    val_data = dict()
    fold_size = (ratings.shape[0] - test_size) // fold
    for i in range(fold - 1):
        val_data[i + 1] = ratings[test_size + i * fold_size: test_size + (i + 1) * fold_size]
    val_data[fold] = ratings[test_size + (fold - 1) * fold_size:]
    return val_data, test_data


def data_split(data_path, args):
    fold = args.fold
    ratings = load_data(data_path, args)
    n_users = int(max(ratings[:, 0]) + 1)
    n_items = int(max(ratings[:, 1]) + 1)
    max_rating = max(ratings[:, 2])
    min_rating = min(ratings[:, 2])

    val_data, test_data = partition(ratings, 0, fold)

    lambda_dict = dict()
    for i in range(fold):
        if i == 0:
            lambda_dict[1] = deepcopy(val_data[2])
            for j in range(3, fold + 1):
                lambda_dict[1] = np.vstack((lambda_dict[1], val_data[j]))
        else:
            lambda_dict[i + 1] = deepcopy(val_data[1])
            for j in range(2, i + 1):
                lambda_dict[i + 1] = np.vstack((lambda_dict[i + 1], val_data[j]))
            for j in range(i + 2, fold + 1):
                lambda_dict[i + 1] = np.vstack((lambda_dict[i + 1], val_data[j]))

    zipped_index_dict = dict()
    for i in range(fold):
        zipped_index_dict[i + 1] = [(int(_[0]), int(_[1])) for _ in lambda_dict[i + 1]]

    train_csr_dict = dict()
    for i in range(fold):
        train_csr_dict[i + 1] = build_user_item_matrix(lambda_dict[i + 1], n_users, n_items)
    val_csr_dict = dict()
    for i in range(fold):
        val_csr_dict[i + 1] = build_user_item_matrix(val_data[i + 1], n_users, n_items)
    test_csr = build_user_item_matrix(test_data, n_users, n_items)
    return train_csr_dict, val_csr_dict, test_csr, zipped_index_dict, max_rating, min_rating


def cf_ridge_regression(csr_matrix, reg_lambda, fixed_feature, update_feature):
    n_feature = fixed_feature.shape[1]
    for i in range(csr_matrix.shape[0]):
        _, idx = csr_matrix[i, :].nonzero()
        valid_feature = fixed_feature.take(idx, axis=0)
        ratings = csr_matrix[i, idx].todense()
        A_i = np.dot(valid_feature.T, valid_feature) + reg_lambda * np.eye(n_feature)
        V_i = np.dot(valid_feature.T, ratings.T)
        update_feature[i, :] = np.squeeze(np.dot(inv(A_i), V_i))


def ALS(train_csr, args, n_iters, init_user_features=None, init_item_features=None):
    if args.implicit:
        logging.info('ALS, alpha {} max rating {}'.format(args.alpha, train_csr.data.max()))
        model = implicit.als.AlternatingLeastSquares(factors=args.factor, iterations=n_iters, num_threads=args.als_threads,
                                                     regularization=max(args.lambda_u, args.lambda_v),
                                                     random_state=0)
        model.fit(train_csr.T, show_progress=False)
        return model.user_factors, model.item_factors
    else:
        user_features = 0.1 * np.random.RandomState(seed=0).rand(train_csr.shape[0], args.factor)
        item_features = 0.1 * np.random.RandomState(seed=0).rand(train_csr.shape[1], args.factor)
        if init_user_features is not None:
            user_features = init_user_features
        if init_item_features is not None:
            item_features = init_item_features
        train_csr_transpose = train_csr.T.tocsr()
        for iteration in range(n_iters):
            logging.info('ALS iteration {}'.format(iteration))
            cf_ridge_regression(train_csr, args.lambda_u, item_features, user_features)
            cf_ridge_regression(train_csr_transpose, args.lambda_v, user_features, item_features)
        return user_features, item_features


def grad_calc(train_csr, val_csr, zipped_index, user_features, item_features, args):
    n_users = train_csr.shape[0]
    n_items = train_csr.shape[1]

    grad_r_user = np.zeros(shape=user_features.shape, dtype=np.float)
    grad_r_item = np.zeros(shape=item_features.shape, dtype=np.float)

    val_coo = val_csr.tocoo()
    pred_val = np.dot(user_features, item_features.T)
    for i, j, v in zip(val_coo.row, val_coo.col, val_coo.data):
        loss = 2 * (pred_val[i, j] - v)
        grad_r_user[i] += loss * item_features[j]
        grad_r_item[j] += loss * user_features[i]

    grad_user_m = np.zeros(shape=(train_csr.nnz, args.factor))
    grad_user_dict = {}
    cnt = 0
    for i in range(n_users):
        _, item_idx = train_csr[i, :].nonzero()
        item_feat = item_features.take(item_idx, axis=0)
        A = np.eye(args.factor, dtype=np.float) * args.lambda_u + np.dot(item_feat.T, item_feat)
        grad_user_m_i = np.dot(item_features, inv(A))
        for i_idx in item_idx:
            tup = (i, i_idx)
            grad_user_dict[tup] = cnt
            grad_user_m[cnt] = grad_user_m_i[i_idx][:]
            cnt += 1

    train_csc = train_csr.tocsc()
    grad_item_m = np.zeros(shape=(train_csc.nnz, args.factor))
    grad_item_dict = {}
    cnt = 0
    for i in range(n_items):
        user_idx, _ = train_csc[:, i].nonzero()
        user_feat = user_features.take(user_idx, axis=0)
        A = np.eye(args.factor, dtype=np.float) * args.lambda_v + np.dot(user_feat.T, user_feat)
        grad_item_m_i = np.dot(user_features, inv(A))
        for u_idx in user_idx:
            tup = (i, u_idx)
            grad_item_dict[tup] = cnt
            grad_item_m[cnt] = grad_item_m_i[u_idx][:]
            cnt += 1

    row = [i for i, j in zipped_index]
    col = [j for i, j in zipped_index]
    data = [(np.dot(grad_r_user[i], grad_user_m[grad_user_dict[(i, j)]].T)
             + np.dot(grad_r_item[j], grad_item_m[grad_item_dict[(j, i)]].T))
            for i, j in zipped_index]
    return sparse.coo_matrix((data, (row, col)), shape=(n_users, n_items)).tocsr()


def grad_calc_implicit(train_csr, val_csr, zipped_index, user_features, item_features, args):
    n_users = train_csr.shape[0]
    n_items = train_csr.shape[1]

    pred_val = np.dot(user_features, item_features.T)
    confidence_val_dense = val_csr.toarray()
    confidence_train_dense = train_csr.toarray()
    preference_val_dense = confidence_val_dense.copy()
    preference_val_dense[preference_val_dense > 0] = 1
    preference_train_dense = confidence_train_dense.copy()
    preference_train_dense[preference_train_dense > 0] = 1
    grad_r_user, grad_r_item = loss_d_emb(confidence_val_dense,
                                          preference_val_dense,
                                          pred_val,
                                          user_features,
                                          item_features)

    grad_user_m = np.zeros(shape=(train_csr.nnz, args.factor))
    grad_user_dict = {}
    cnt = 0
    VVT = np.dot(item_features.T, item_features)
    UUT = np.dot(user_features.T, user_features)

    for i in range(n_users):
        _, item_idx = train_csr[i, :].nonzero()
        grad_user_m_i = u_emb_d_c(args.lambda_u, confidence_train_dense, preference_train_dense, item_features, i, VVT)
        for item_num, i_idx in enumerate(item_idx):
            tup = (i, i_idx)
            grad_user_dict[tup] = cnt
            grad_user_m[cnt] = grad_user_m_i[item_num][:]
            cnt += 1

    train_csc = train_csr.tocsc()
    grad_item_m = np.zeros(shape=(train_csc.nnz, args.factor))
    grad_item_dict = {}
    cnt = 0
    for i in range(n_items):
        user_idx, _ = train_csc[:, i].nonzero()
        grad_item_m_i = i_emb_d_c(args.lambda_v, confidence_train_dense, preference_train_dense, user_features, i, UUT)
        for user_num, u_idx in enumerate(user_idx):
            tup = (i, u_idx)
            grad_item_dict[tup] = cnt
            grad_item_m[cnt] = grad_item_m_i[user_num][:]
            cnt += 1

    row = [i for i, j in zipped_index]
    col = [j for i, j in zipped_index]
    data = [(np.dot(grad_r_user[i], grad_user_m[grad_user_dict[(i, j)]].T)
             + np.dot(grad_r_item[j], grad_item_m[grad_item_dict[(j, i)]].T))
            for i, j in zipped_index]
    return sparse.coo_matrix((data, (row, col)), shape=(n_users, n_items)).tocsr()


def grad_update_loop(train_csr, val_csr, zipped_index, max_rating, min_rating, args):
    user_feature, item_feature = ALS(train_csr, args, args.als_iter)

    A_i = train_csr
    C_i = sparse.csr_matrix(train_csr.shape)
    user_feature_i = user_feature
    item_feature_i = item_feature
    for i in range(args.debug_iter):
        if args.implicit:
            gradients = grad_calc_implicit(A_i, val_csr, zipped_index, user_feature_i, item_feature_i, args)
        else:
            gradients = grad_calc(A_i, val_csr, zipped_index, user_feature_i, item_feature_i, args)
        A_i = A_i - gradients * args.debug_lr
        for _ in range(A_i.nnz):
            A_i.data[_] = min(max_rating, max(min_rating, A_i.data[_]))
        logging.info("A_i mean {}, min {}, max {}".format(A_i.data.mean(), A_i.data.min(), A_i.data.max()))
        if args.retrain == "full":
            user_feature_i, item_feature_i = ALS(A_i, args, args.als_iter)
        if args.retrain == "inc":
            user_feature_i, item_feature_i = ALS(A_i, args, 1, user_feature_i, item_feature_i)
        C_i = A_i - train_csr
    return C_i


def get_path(args, part_id):
    path = f"./save/{args.dataset}/f{args.fold}_m{args.debug_iter}_lr{args.debug_lr}_part{part_id}_{args.retrain}"
    if args.implicit:
        path += '_implicit'
    return path + '.txt'


def debug_process(train_csr, val_csr, zipped_index, max_rating, min_rating, id, args):
    if args.implicit:
        alpha = args.alpha
    else:
        alpha = 1
    change_csr = grad_update_loop(alpha * train_csr, alpha * val_csr, zipped_index, max_rating, min_rating, args)
    change_arr = change_csr.toarray()
    path = get_path(args, id)
    with open(path, "w+") as f:
        for i, j in zipped_index:
            print(i, j, change_arr[i, j], file=f, sep=',')


def aggregate_process(edit, sorted_edges, train_csr, test_csr, args, old_pred, max_rating, min_rating, percent):
    if args.implicit:
        alpha = args.alpha
    else:
        alpha = 1
    cut_pos = int(len(sorted_edges) * percent * 0.01)
    base_arr = train_csr.todense()
    for i, j, v in sorted_edges[:cut_pos]:
        if edit == "del":
            base_arr[i, j] = 0
        elif edit == "mod":
            base_arr[i, j] += v
            base_arr[i, j] = min(max_rating, max(min_rating, base_arr[i, j]))
    user_feature, item_feature = ALS(alpha * sparse.csr_matrix(base_arr), args, args.als_iter)
    new_pred = np.dot(user_feature, item_feature.T)
    if args.implicit:
        aucs = roc_auc_with_t_test(new_pred, old_pred, test_csr)
        mse = RMSE_weighted_with_t_test(new_pred, old_pred, alpha * test_csr)
        precisions = precision_at_10_with_t_test(new_pred, old_pred, test_csr)
    else:
        test_csr_binarized = test_csr.copy()
        test_csr_binarized[test_csr_binarized <= 3] = 0
        test_csr_binarized[test_csr_binarized > 3] = 1
        aucs = roc_auc_with_t_test(new_pred, old_pred, test_csr_binarized)
        mse = RMSE_with_ttest(new_pred, old_pred, test_csr)
        precisions = precision_at_10_with_t_test(new_pred, old_pred, test_csr_binarized)
    return aucs, mse, precisions


# main process
if __name__ == "__main__":
    file_path = "./data/" + args.dataset + ".txt"
    if not os.path.exists(f"./save/{args.dataset}"):
        os.mkdir(f"./save/{args.dataset}")

    if args.mode == "debug":
        train_csr, val_csr, test_csr, zipped_index, max_rating, min_rating = data_split(file_path, args)
        if args.implicit:
            max_rating *= args.alpha
            min_rating = 0
        fold_id = 0
        for rnd in range((args.fold + args.process - 1) // args.process):
            processes = []
            for i in range(fold_id, fold_id + args.process):
                process = Process(target=debug_process, args=(train_csr[i + 1], val_csr[i + 1], zipped_index[i + 1],
                                                              max_rating, min_rating, i + 1, args))
                processes.append(process)
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            fold_id += args.process
    elif args.mode == "test":
        train_csr, val_csr, test_csr, zipped_index, max_rating, min_rating = data_split(file_path, args)
        if args.implicit:
            max_rating *= args.alpha
            min_rating = 0
            alpha = args.alpha
        else:
            alpha = 1
        test_train_csr = sparse.csr_matrix(test_csr.shape)
        for i in range(args.fold):
            test_train_csr = test_train_csr + val_csr[i + 1]
        user_feature, item_feature = ALS(alpha * test_train_csr, args, args.als_iter)
        old_pred = np.dot(user_feature, item_feature.T)

        edge_dict = dict()
        for i in range(1, args.fold + 1):
            path = get_path(args, i)
            for line in open(path):
                l = line.strip().split(',')
                x = int(l[0])
                y = int(l[1])
                r = float(l[2])
                if (x, y) not in edge_dict.keys():
                    edge_dict[(x, y)] = r / (args.fold - 1)
                else:
                    if edge_dict[(x, y)] * r > 0:
                        edge_dict[(x, y)] += r / (args.fold - 1)
                    else:
                        edge_dict[(x, y)] = 0
        edges = [(key[0], key[1], values) for key, values in edge_dict.items()]
        if args.implicit:
            sorted_edges = sorted(edges, key=lambda _: _[2], reverse=False)
        else:
            sorted_edges = sorted(edges, key=lambda _: abs(_[2]), reverse=True)
        for edit in ["del", "mod"]:
            for percent in [0.1, 0.2, 0.5, 1, 2, 5, 10]:
                aucs, mse, precisions = aggregate_process(edit, sorted_edges, test_train_csr,
                                                          test_csr, args, old_pred,
                                                          max_rating, min_rating, percent)
                print(f"{edit} {percent}% training data, ALS objective on test: {mse[1]} -> {mse[0]}, p_value: {mse[2]}")
                print(f"{edit} {percent}% training data, aucs on test: {aucs[1]} -> {aucs[0]}, p_value: {aucs[2]}")
                print(f"{edit} {percent}% training data, p@10 on test: {precisions[1]} -> {precisions[0]}, p_value: {precisions[2]}")

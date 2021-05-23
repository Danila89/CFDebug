from utils import loss_d_emb, u_emb_d_c, i_emb_d_c
import numpy as np
from scipy import sparse
import unittest
from implicit._als import least_squares


def calc_u_emb(C, i_emb, reg_lambda):
    u_emb = np.zeros(shape=(C.shape[0], i_emb.shape[1]))
    # makes 'half' of ALS iteration: calculates user embeddings given confidence and item embeddings
    c_sparse = sparse.csr_matrix(C)
    least_squares(c_sparse, u_emb, i_emb, reg_lambda, 0)
    return u_emb


def calc_i_emb(C, u_emb, reg_lambda):
    i_emb = np.zeros(shape=(C.shape[1], u_emb.shape[1]))
    # makes 'half' of ALS iteration: calculates item embeddings given confidence and user embeddings
    c_sparse = sparse.csr_matrix(C.T)
    least_squares(c_sparse, i_emb, u_emb, reg_lambda, 0)
    return i_emb


def u_emb_d_c_i(lamb, C, R, v, user_ind, item_ind, VTV):
    # calculates derivative of each component of the embedding of user 'user_ind'
    # wrt confidence value of the user to item 'item_ind'
    c_u = C[user_ind] * np.eye(M=C[user_ind].shape[0], N=C[user_ind].shape[0])
    c_u_m_one = (C[user_ind] - R[user_ind]) * np.eye(M=C[user_ind].shape[0], N=C[user_ind].shape[0])
    m_inv = np.linalg.inv(lamb * np.eye(v.shape[1], v.shape[1]) + VTV + v.T.dot(c_u_m_one).dot(v))
    # returns vector with size equal to the embedding size
    return -m_inv.dot(np.outer(v[item_ind], v[item_ind])).dot(m_inv).dot(v.T).dot(c_u).dot(R[user_ind]) \
           + m_inv.dot(v[item_ind].T)


def i_emb_d_c_j(lamb, C, R, u, item_ind, user_ind, UTU):
    # calculates derivative of each component of the embedding of item 'item_ind'
    # wrt confidence value of the user 'user_ind' to the current item
    c_i = C[:, item_ind] * np.eye(M=C[:, item_ind].shape[0], N=C[:, item_ind].shape[0])
    c_i_m_one = (C[:, item_ind] - R[:, item_ind]) * np.eye(M=C[:, item_ind].shape[0], N=C[:, item_ind].shape[0])
    m_inv = np.linalg.inv(lamb * np.eye(u.shape[1], u.shape[1]) + UTU + u.T.dot(c_i_m_one).dot(u))
    # returns vector with size equal to the embedding size
    return -m_inv.dot(np.outer(u[user_ind], u[user_ind])).dot(m_inv).dot(u.T).dot(c_i).dot(R[:, item_ind]) \
           + m_inv.dot(u[user_ind].T)


def loss_gamma(C, P, U, V):
    # calculates loss on validation
    # (note that following the original paper we ignore regularization here)
    pred = U.dot(V.T)
    c_full = C.copy()
    c_full[c_full == 0] = 1
    return c_full * np.power(P - pred, 2)


class TestGradients(unittest.TestCase):
    def test_grad_loss_wrt_embeddings(self):
        alpha = 5
        confidence = np.random.binomial(1, 0.01, size=(100, 1000))
        preference = confidence.copy()
        confidence *= alpha
        user_vec = np.random.normal(size=(100, 10))
        item_vec = np.random.normal(size=(1000, 10))

        dx = 0.0001
        idx_u = 3
        user_num = 52
        idx_i = 4
        item_num = 6
        # increment user_vec[user_num, idx_u] by dx
        user_vec_1 = user_vec.copy()
        user_vec_1[user_num, idx_u] += dx
        # increment item_vec[item_num, idx_i] by dx
        item_vec_1 = item_vec.copy()
        item_vec_1[item_num, idx_i] += dx

        # calc differentials of loss numerically
        d_loss_u_numeric = np.sum(loss_gamma(confidence, preference,
                                             user_vec_1, item_vec)) - \
                           np.sum(loss_gamma(confidence, preference, user_vec, item_vec))
        d_loss_i_numeric = np.sum(loss_gamma(confidence, preference,
                                             user_vec, item_vec_1)) - \
                           np.sum(loss_gamma(confidence, preference, user_vec, item_vec))

        # calc differentials analytically
        grad_r_user, grad_r_item = loss_d_emb(confidence,
                                              preference,
                                              user_vec.dot(item_vec.T),
                                              user_vec,
                                              item_vec)
        d_loss_u_analytic = grad_r_user[user_num, idx_u] * dx
        d_loss_i_analytic = grad_r_item[item_num, idx_i] * dx
        diff_1 = np.absolute(d_loss_u_numeric - d_loss_u_analytic)
        diff_2 = np.absolute(d_loss_i_numeric - d_loss_i_analytic)
        self.assertTrue(diff_1 < 1e-4 and diff_2 < 1e-4)

    def test_grad_embeddings_wrt_confidence(self):
        alpha = 5
        reg_lambda = 0.1
        confidence = np.random.binomial(1, 0.01, size=(100, 1000))
        preference = confidence.copy()
        confidence *= alpha
        user_vec = np.random.normal(size=(100, 10))
        item_vec = np.random.normal(size=(1000, 10))

        dx = 0.0001
        idx_u = 1
        idx_item_in_user = 5
        idx_i = np.argwhere(confidence[idx_u]).flatten()[idx_item_in_user]
        idx_user_in_item = np.argwhere(np.argwhere(confidence[:, idx_i]).flatten() == idx_u).flatten()[0]
        # increase [idx_u, idx_i] element in confidence by dx
        confidence_1 = confidence.copy().astype(float)
        confidence_1[idx_u, idx_i] += dx
        VTV = item_vec.T.dot(item_vec)
        UTU = user_vec.T.dot(user_vec)

        d_user_vec_numeric = calc_u_emb(confidence_1, item_vec, reg_lambda) - calc_u_emb(confidence, item_vec, reg_lambda)
        d_user_vec_numeric = d_user_vec_numeric[idx_u]
        d_item_vec_numeric = calc_i_emb(confidence_1, user_vec, reg_lambda) - calc_i_emb(confidence, user_vec, reg_lambda)
        d_item_vec_numeric = d_item_vec_numeric[idx_i]

        # simple implementation
        d_user_vec_analytic_0 = u_emb_d_c_i(reg_lambda, confidence, preference, item_vec, idx_u, idx_i, VTV) * dx
        d_item_vec_analytic_0 = i_emb_d_c_j(reg_lambda, confidence, preference, user_vec, idx_i, idx_u, UTU) * dx

        # efficient implementation
        d_user_vec_analytic_1 = u_emb_d_c(reg_lambda, confidence, preference, item_vec, idx_u, VTV)[idx_item_in_user] * dx
        d_item_vec_analytic_1 = i_emb_d_c(reg_lambda, confidence, preference, user_vec, idx_i, UTU)[idx_user_in_item] * dx

        diffs_1 = np.absolute(d_user_vec_numeric - d_user_vec_analytic_0).sum()
        diffs_2 = np.absolute(d_user_vec_numeric - d_user_vec_analytic_1).sum()
        diffs_3 = np.absolute(d_item_vec_numeric - d_item_vec_analytic_0).sum()
        diffs_4 = np.absolute(d_item_vec_numeric - d_item_vec_analytic_1).sum()
        self.assertTrue(diffs_1 < 1e-4 and diffs_2 < 1e-4 and diffs_3 < 1e-4 and diffs_4 < 1e-4)


if __name__ == '__main__':
    unittest.main()

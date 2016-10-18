import numpy as np
from cvxopt import matrix
from cvxopt import solvers

class SVM_classifier:
    def __init__(self, dataset):
        self.dataset = dataset
        self.alpha = None
        self.b = None

    def optimize(self, hard_margin_weight=1.0, eps=1e-5):
        n = len(self.dataset)
        W = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                Wi = self.dataset[i][1] * self.dataset[j][1] \
                     * self.kernel_func(self.dataset[i][0], self.dataset[j][0])
                if i == j:
                    W[i][i] = Wi
                else:
                    W[i][j] = W[j][i] = 0.5 * Wi
        h = -1 * np.ones(n)
        A = np.zeros(n)
        for i in range(n):
            A[i] = self.dataset[i][1]
        A = A.reshape(1, n)
        b = np.array([0.0])
        C = np.vstack((-1 * np.identity(n), np.identity(n)))
        d = np.hstack((np.zeros(n), np.array([hard_margin_weight] * n)))

        print('QP setting: W:{}, h:{}, A:{}, b:{}, C:{}, d:{}'.format(
            W.shape, h.shape, A.shape, b.shape, C.shape, d.shape))

        sol = solvers.qp(matrix(W), matrix(h), matrix(C), matrix(d), matrix(A), matrix(b))
        self.alpha = np.array(sol['x']).reshape(n)

        index_for_b = filter((lambda i:
                              self.alpha[i] > eps and
                              self.alpha[i] < hard_margin_weight - eps),
                             range(n))

        print('num of sv: out: {}, bordear: {}, in: {}'.format(
            len(filter((lambda i: self.alpha[i] <= eps), range(n))),
            len(index_for_b),
            len(filter((lambda i: self.alpha[i] >= hard_margin_weight - eps), range(n)))))
        self.b = 0
        if len(index_for_b) > 0:
            for i in index_for_b:
                self.b += self.dataset[i][1]
                self.b += - sum([self.alpha[j] * self.dataset[j][1] * self.kernel_func(self.dataset[j][0], self.dataset[i][0]) for j in range(n)])
            self.b /= len(index_for_b)

    def kernel_func(self, x1, x2, gamma = -1.0):
        return np.exp(gamma * np.linalg.norm(np.array(x1) - np.array(x2))**2)
        # return sum(np.array(x1) * np.array(x2))

    def test(self, x):
        if self.alpha is None:
            print('optimize first')
            return
        n = len(self.dataset)
        ret = 0
        for i in range(n):
            ret += self.alpha[i] * self.dataset[i][1] \
                   * self.kernel_func(self.dataset[i][0], x)
        ret += self.b
        return ret

    def validate_dataset(self, ds):
        succ_num = 0
        for (i, d) in enumerate(ds):
            test_ret = self.test(d[0])
            truth_ret = d[1]
            succ = (self.test(d[0]) * d[1]) >= 0
            if succ:
                succ_num += 1
            print('[{}/{}: {}]  result:{:.2}, ground truth:{:.2}, succ rate: {:.2}={}/{}'.format(
                i, len(ds), ('Success' if succ else 'Fail!'),
                test_ret, truth_ret, float(succ_num)/(i+1), succ_num, i+1))

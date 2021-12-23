import numpy as np
import pandas as pd
from collections import deque
from math import log, ceil
from .base import Agent


class CLA:
    """
    Implementation of the CLA algorithm for solving quadratic optimization
    problems with constraints (equalities and inequalities). Based on the
    implementation by M. Lopez de Prado (2013).

    Parameters
    ----------

    mu: np.ndarray
        Vector of expected returns.
    sigma: np.ndarray
        Covariance matrix.
    lb: np.ndarray
        Lower bound conditions for each weight.
    ub: np.ndarray
        Upper bound conditions for each weight.

    Arguments
    ---------
    w: List | np.ndarray
        Optimal weights.
    f: List
        Free weights.
    g: List
        Gammas (see Lopez de Prado).
    l: List
        Lambdas (see Lopez de Prado).
    """

    def __init__(self, mean: np.ndarray, covar: np.ndarray, lb: np.ndarray, ub: np.ndarray):

        self.mean = mean
        self.covar = covar
        self.lb = lb
        self.ub = ub
        self.w = []  # solution
        self.l = []  # lambdas
        self.g = []  # gammas
        self.f = []  # free weights

    def solve(self):
        """
        Computes the turning points, free sets and weights.
        """
        f, w = self.init_algo()
        self.w.append(np.copy(w))
        self.l.append(None)
        self.g.append(None)
        self.f.append(f[:])

        while True:
            l_in = None
            if len(f) > 1:
                covar_f, covar_fb, mean_f, w_b = self.get_matrices(f)
                covar_f_inv = np.linalg.inv(covar_f)
                j = 0
                for i in f:
                    l, bi = self.compute_lambda(
                        covar_f_inv, covar_fb, mean_f, w_b, j,
                        [self.lb[i], self.ub[i]]
                    )
                    if (l is not None and l_in is not None and l > l_in) or (l is not None and l_in is None):
                        l_in, i_in, bi_in = l, i, bi
                    j += 1
            l_out = None
            if len(f) < self.mean.shape[0]:
                b = self.get_b(f)
                for i in b:
                    covar_f, covar_fb, mean_f, w_b = self.get_matrices(f+[i])
                    covar_f_inv = np.linalg.inv(covar_f)
                    l, bi = self.compute_lambda(
                        covar_f_inv, covar_fb, mean_f, w_b,
                        mean_f.shape[0] - 1, self.w[-1][i]
                    )
                    if (self.l[-1] == None or l is None or l < self.l[-1]) and (l is not None and (l_out is None or l > l_out)):
                        l_out, i_out = l, i
            if (l_in == None or l_in < 0) and (l_out == None or l_out < 0):

                self.l.append(0)
                covar_f, covar_fb, mean_f, w_b = self.get_matrices(f)
                covar_f_inv = np.linalg.inv(covar_f)
                mean_f = np.zeros(mean_f.shape)

            else:

                if l_in is not None and (l_out is None or l_in > l_out):
                    self.l.append(l_in)
                    f.remove(i_in)
                    w[i_in] = bi_in
                else:
                    self.l.append(l_out)
                    f.append(i_out)
                covar_f, covar_fb, mean_f, w_b = self.get_matrices(f)
                covar_f_inv = np.linalg.inv(covar_f)

            w_f, g = self.compute_w(covar_f_inv, covar_fb, mean_f, w_b)

            for i in range(len(f)):
                w[f[i]] = w_f[i]

            self.w.append(np.copy(w))
            self.g.append(g)
            self.f.append(f[:])
            if self.l[-1] == 0:
                break

        self.purge_num_err(10e-10)
        self.purge_excess()

    def init_algo(self):

        a = np.zeros((self.mean.shape[0]), dtype=[("id", int), ("mu", float)])
        b = [self.mean[i][0] for i in range(self.mean.shape[0])]
        a[:] = list(zip(range(self.mean.shape[0]), b))
        b = np.sort(a, order="mu")
        i, w = b.shape[0], np.copy(self.lb)
        while sum(w) < 1:
            i -= 1
            w[b[i][0]] = self.ub[b[i][0]]
        w[b[i][0]] += 1-sum(w)
        return [b[i][0]], w

    def compute_bi(self, c, bi):

        if c > 0:
            bi = bi[1][0]
        if c < 0:
            bi = bi[0][0]
        return bi

    def compute_w(self, covar_f_inv, covar_fb, mean_f, w_b):

        ones_f = np.ones(mean_f.shape)
        g1 = np.dot(np.dot(ones_f.T, covar_f_inv), mean_f)
        g2 = np.dot(np.dot(ones_f.T, covar_f_inv), ones_f)
        if w_b is None:
            g, w1 = float(-self.l[-1]*g1/g2 + 1/g2), 0
        else:
            ones_b = np.ones(w_b.shape)
            g3 = np.dot(ones_b.T, w_b)
            g4 = np.dot(covar_f_inv, covar_fb)
            w1 = np.dot(g4, w_b)
            g4 = np.dot(ones_f.T, w1)
            g = float(-self.l[-1]*g1/g2+(1.-g3*g4)/g2)
        w2 = np.dot(covar_f_inv, ones_f)
        w3 = np.dot(covar_f_inv, mean_f)
        return -w1+g*w2+self.l[-1]*w3, g

    def compute_lambda(self, covar_f_inv, covar_fb, mean_f, w_b, i, bi):

        ones_f = np.ones(mean_f.shape)
        c1 = np.dot(np.dot(ones_f.T, covar_f_inv), ones_f)
        c2 = np.dot(covar_f_inv, mean_f)
        c3 = np.dot(np.dot(ones_f.T, covar_f_inv), mean_f)
        c4 = np.dot(covar_f_inv, ones_f)
        c = -c1*c2[i]+c3*c4[i]
        if c == 0:
            return None, None
        if type(bi) == list:
            bi = self.compute_bi(c, bi)
        if w_b is None:
            return float((c4[i]-c1*bi)/c), bi
        else:
            ones_b = np.ones(w_b.shape)
            l1 = np.dot(ones_b.T, w_b)
            l2 = np.dot(covar_f_inv, covar_fb)
            l3 = np.dot(l2, w_b)
            l2 = np.dot(ones_f.T, l3)
            return float(((1-l1+l2)*c4[i]-c1*(bi+l3[i]))/c), bi

    def get_matrices(self, f):

        covar_f = self.reduce_matrix(self.covar, f, f)
        mean_f = self.reduce_matrix(self.mean, f, [0])
        b = self.get_b(f)
        covar_fb = self.reduce_matrix(self.covar, f, b)
        w_b = self.reduce_matrix(self.w[-1], b, [0])
        return covar_f, covar_fb, mean_f, w_b

    def get_b(self, f):

        return self.diff_lists(range(self.mean.shape[0]), f)

    def diff_lists(self, list_1, list_2):

        return list(set(list_1)-set(list_2))

    def reduce_matrix(self, matrix, list_x, list_y):

        if len(list_x) == 0 or len(list_y) == 0:
            return

        matrix_ = matrix[:, list_y[0]:list_y[0]+1]

        for i in list_y[1:]:
            a = matrix[:, i:i+1]
            matrix_ = np.append(matrix_, a, 1)

        matrix__ = matrix_[list_x[0]:list_x[0]+1, :]

        for i in list_x[1:]:
            a = matrix_[i:i+1, :]
            matrix__ = np.append(matrix__, a, 0)

        return matrix__

    def purge_num_err(self, tol):

        i = 0
        while True:
            if i == len(self.w):
                break

            w = self.w[i]

            for j in range(w.shape[0]):

                if w[j] - self.lb[j] < -tol or w[j] - self.ub[j] > tol:

                    del self.w[i]

                    del self.l[i]

                    del self.g[i]

                    del self.f[i]

                    break
            i += 1

    def purge_excess(self):

        i, repeat = 0, False
        while True:
            if repeat == False:

                i += 1

                if i == len(self.w)-1:
                    break

                w = self.w[i]
                mu = np.dot(w.T, self.mean)[0, 0]
                j, repeat = i+1, False
                while True:
                    if j == len(self.w):
                        break
                    w = self.w[j]
                    mu_ = np.dot(w.T, self.mean)[0, 0]
                    if mu < mu_:
                        del self.w[i]
                        del self.l[i]
                        del self.g[i]
                        del self.f[i]
                        repeat = True
                        break
                    else:
                        j += 1

    def get_min_var(self):

        var = []
        for w in self.w:
            a = np.dot(np.dot(w.T, self.covar), w)
            var.append(a)
        return min(var)**0.5, self.w[var.index(min(var))]

    def get_max_sr(self):

        w_sr, sr = [], []

        for i in range(len(self.w)-1):

            w0 = np.copy(self.w[i])

            w1 = np.copy(self.w[i+1])

            kargs = {"minimum": False, "args": (w0, w1)}

            a, b = self.golde_section(self.eval_sr, 0, 1, **kargs)

            w_sr.append(a*w0+(1-a)*w1)

            sr.append(b)

        return max(sr), w_sr[sr.index(max(sr))]

    def eval_sr(self, a, w0, w1):

        w = a*w0+(1-a)*w1

        b = np.dot(w.T, self.mean)[0, 0]

        c = np.dot(np.dot(w.T, self.covar), w)[0, 0]**0.5

        return b/c

    def golden_section(self, obj, a, b, **kargs):

        tol, sign, args = 1.e-9, 1, None

        if "minimum" in kargs and kargs["minimum"] == False:

            sign = -1

        if "args" in kargs:
            args = kargs["args"]

        num_iter = int(ceil(-2.078087*log(tol/abs(b-a))))

        r = 0.618033989

        c = 1.-r

        x1 = r*a+c*b

        x2 = c*a+r*b

        f1 = sign*obj(x1, *args)

        f2 = sign*obj(x2, *args)

        for i in range(num_iter):
            if f1 > f2:
                a = x1
                x1 = x2
                f1 = f2
                x2 = c*a+r*b
                f2 = sign*obj(x2, *args)
            else:
                b = x2
                x2 = x1
                f1 = f2
                x1 = r*a + c*b
                f1 = sign*obj(x1, *args)
        if f1 < f2:
            return x1, sign*f1
        else:
            return x2, sign*f2

    def ef_frontier(self, points):
        mu, sigma, weights = [], [], []
        a = np.linspace(0, 1, points/len(self.w))[:-1]
        b = range(len(self.w)-1)
        for i in b:
            w0, w1 = self.w[i], self.w[i+1]
            if i == b[-1]:
                a = np.linspace(0, 1, points/len(self.w))
            for j in a:
                w = w1*j + (1. - j)*w0
                weights.append(np.copy(w))
                mu.append(np.dot(w.T, self.mean)[0, 0])
                sigma.append(np.dot(np.dot(w.T, self.covar), w)[0, 0]**0.5)
        return mu, sigma, weights


class CLAAgent(Agent):

    def __init__(self, action_space, J, window, *args, **kwargs):

        self.action_space = action_space
        self.memory = deque(maxlen=window)
        self.w = self.action_space.sample()
        self.J = J
        self._id = "cla" + J

    def observe(self, observation, action, reward, done, next_observation):

        self.memory.append(observation.values)

    def act(self, observation):

        memory = np.array(self.memory)

        if len(self.memory) != self.memory.maxlen:
            return self.action_space.sample()

        mu = np.mean(memory, axis=0).reshape(self.action_space.shape[0], 1)

        sigma = np.cov(memory.T)

        lb = np.zeros(mu.shape)
        ub = np.ones(mu.shape)

        cla_algo = CLA(mu, sigma, lb=lb, ub=ub)

        cla_algo.solve()

        if self.J == "min_variance":
            self.w = cla_algo.get_min_var()[1]
            if np.any(self.w < 0):
                self.w += np.abs(self.w.min())
            self.w /= self.w.sum()
            return self.w.ravel()
        else:
            self.w = cla_algo.get_max_sr()[1]
            if np.any(self.w < 0):
                self.w += np.abs(self.w.min())
            self.w /= self.w.sum()
            return self.w.ravel()

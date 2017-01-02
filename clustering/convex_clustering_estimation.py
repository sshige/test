#! /usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

from generate_data_2d import GenerateData2D


class CC_estimation:
    # parameter estimation of Gaussian Mixture Model
    def __init__(self, dataset, sigma2):
        self.dimension = 2
        self.dataset = dataset
        self.data_num = len(self.dataset)
        self.sigma2 = sigma2
        self.initialize_random_parameter()
        self.initialize_f()

    def initialize_random_parameter(self):
        self.cluster_prob_list = np.array([1.0 / self.data_num] * self.data_num)

    def initialize_f(self):
        self.f = np.empty([self.data_num, self.data_num])
        for i in range(self.data_num):
            for k in range(self.data_num):
                self.f[i][k] = self.get_gaussian_distribution(i, k)

    def update_estimation(self):
        updated_cluster_prob_list = np.empty_like(self.cluster_prob_list)
        for i in range(self.data_num):
            updated_cluster_prob_list[i] = 0
            for k in range(self.data_num):
                tmp_numerator = self.cluster_prob_list[i] * self.f[i][k]
                tmp_denominator = 0
                for j in range(self.data_num):
                    tmp_denominator += self.cluster_prob_list[j] * self.f[j][k]
                updated_cluster_prob_list[i] += tmp_numerator / tmp_denominator
            updated_cluster_prob_list[i] /= self.data_num
        # overwrite with zero if probability is small enough
        prob_lower_thre = (1.0 / np.count_nonzero(self.cluster_prob_list)) * 0.1
        for i in range(self.data_num):
            if updated_cluster_prob_list[i] < prob_lower_thre:
                updated_cluster_prob_list[i] = 0
        updated_cluster_prob_list /= np.sum(updated_cluster_prob_list)
        self.cluster_prob_list = updated_cluster_prob_list

    def get_log_likelihood(self):
        log_likelihood = 0
        for k in range(self.data_num):
            tmp_likelihood = 0
            for i in range(self.data_num):
                tmp_likelihood += self.cluster_prob_list[i] * self.f[i][k]
            log_likelihood += np.log(tmp_likelihood)
        return log_likelihood

    def get_gaussian_distribution(self, i, k):
        return 1.0 / ((2.0 * np.pi * self.sigma2) ** (self.dimension * 0.5) ** 0.5) \
            * np.exp(- 1.0 / (2.0 * self.sigma2) * (np.linalg.norm(self.dataset[i] - self.dataset[k])) ** 2.0)

    def print_parameter(self):
        print(self.cluster_prob_list)

if __name__ == '__main__':
    sigma2 = 0.5
    gd2d = GenerateData2D(data_num=100,
                          mu_list=np.array([np.array([-2.0, 2.0]), np.array([-1.0, -1.0]), np.array([1.0, 2.0])]),
                          sigma_list=np.array([np.diag([0.1, 0.2]), np.diag([0.4, 0.2]), np.array([[0.3, 0.2], [0.2, 0.3]])]),
                          cluster_prob_list=np.array([1.0, 1.0, 2.0]))
    cce = CC_estimation(dataset=[d[0] for d in gd2d.dataset], sigma2=sigma2)

    mu_list = [d[0] for d in gd2d.dataset]
    sigma_list = [sigma2 * np.identity(2)] * len(gd2d.dataset)
    last_log_likelihood = - np.inf
    while True:
        plt.gca().cla()
        gd2d.plot_dataset()
        gd2d.draw_gaussian_distribution_ellipse_from_dataset()
        gd2d.draw_gaussian_distribution_ellipse(mu_list, sigma_list, cce.cluster_prob_list,
                                                facecolor='none', edgecolor='black', alpha=1.0, linestyle='dashed')
        plt.pause(0.1)
        cce.update_estimation()
        # check convergence
        log_likelihood = cce.get_log_likelihood()
        print('log likelihood: {},  nonzero: {}'.format(log_likelihood, np.count_nonzero(cce.cluster_prob_list)))
        if log_likelihood - last_log_likelihood < 1e-8:
            break
        last_log_likelihood = log_likelihood

    print('=== parameter estimation result ===')
    cce.print_parameter()
    plt.show()

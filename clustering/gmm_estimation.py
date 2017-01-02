#! /usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

from generate_data_2d import GenerateData2D


class GMM_estimation:
    # parameter estimation of Gaussian Mixture Model
    def __init__(self, dataset, cluster_num):
        self.dimension = 2
        self.dataset = dataset
        self.data_num = len(self.dataset)
        self.cluster_num = cluster_num
        self.p_omega = np.zeros([self.data_num, self.cluster_num])
        self.initialize_random_parameter()

    def initialize_random_parameter(self):
        ds_max = np.amax(np.array(self.dataset), axis=0)
        ds_min = np.amin(np.array(self.dataset), axis=0)
        self.mu_list = np.stack([np.random.uniform(low=ds_min[0], high=ds_max[0], size=self.cluster_num),
                                 np.random.uniform(low=ds_min[1], high=ds_max[1], size=self.cluster_num)], axis=1)
        self.sigma_list = np.array([0.1*np.identity(2)] * self.cluster_num,)
        self.cluster_prob_list = np.array([1.0 / self.cluster_num] * self.cluster_num)

    def update_with_em_loop(self):
        for i in range(20):
            self.print_parameter()
            self.update_e_step()
            self.update_m_step()

    def update_e_step(self):
        for k in range(self.data_num):
            for omega_i in range(self.cluster_num):
                self.p_omega[k][omega_i] = self.cluster_prob_list[omega_i] * self.get_gaussian_distribution(self.dataset[k], omega_i)
                tmp_denominator = 0
                for omega_j in range(self.cluster_num):
                    tmp_denominator += self.cluster_prob_list[omega_j] * self.get_gaussian_distribution(self.dataset[k], omega_j)
                self.p_omega[k][omega_i] /= tmp_denominator

    def update_m_step(self):
        # pi
        updated_cluster_prob_list = np.sum(self.p_omega, axis=0) / self.data_num
        # mu
        updated_mu_list = np.zeros_like(self.mu_list)
        for omega_i in range(self.cluster_num):
            for k in range(self.data_num):
                updated_mu_list[omega_i] += self.p_omega[k][omega_i] * self.dataset[k]
            updated_mu_list[omega_i] /= (updated_cluster_prob_list[omega_i] * self.data_num)
        # sigma
        updated_sigma_list = np.zeros_like(self.sigma_list)
        for omega_i in range(self.cluster_num):
            for k in range(self.data_num):
                updated_sigma_list[omega_i] += \
                    self.p_omega[k][omega_i] * np.matrix(self.dataset[k] - updated_mu_list[omega_i]).transpose().dot(np.matrix(self.dataset[k] - updated_mu_list[omega_i]))
            updated_sigma_list[omega_i] /= (updated_cluster_prob_list[omega_i] * self.data_num)
        self.cluster_prob_list = updated_cluster_prob_list
        self.mu_list = updated_mu_list
        self.sigma_list = updated_sigma_list

    def get_log_likelihood(self):
        log_likelihood = 0
        for k in range(self.data_num):
            tmp_likelihood = 0
            for omega_i in range(self.cluster_num):
                tmp_likelihood += self.cluster_prob_list[omega_i] * self.get_gaussian_distribution(self.dataset[k], omega_i)
            log_likelihood += np.log(tmp_likelihood)
        return log_likelihood

    def get_gaussian_distribution(self, x, omega_i):
        return 1.0 / ((2.0 * np.pi) ** (self.dimension * 0.5) * np.linalg.det(self.sigma_list[omega_i]) ** 0.5) \
            * np.exp(-0.5 * (x - self.mu_list[omega_i]).dot(np.linalg.inv(self.sigma_list[omega_i]).dot(x - self.mu_list[omega_i])))

    def print_parameter(self):
        print(self.mu_list)
        print(self.sigma_list)
        print(self.cluster_prob_list)

if __name__ == '__main__':
    gd2d = GenerateData2D(mu_list=np.array([np.array([-2.0, 2.0]), np.array([-1.0, -1.0]), np.array([1.0, 2.0])]),
                                   sigma_list=np.array([np.diag([0.1, 0.2]), np.diag([0.4, 0.2]), np.array([[0.3, 0.2], [0.2, 0.3]])]),
                                   cluster_prob_list=np.array([1.0, 1.0, 2.0]))
    gmme = GMM_estimation(dataset=[d[0] for d in gd2d.dataset], cluster_num=3)

    last_log_likelihood = - np.inf
    while True:
        plt.gca().cla()
        gd2d.plot_dataset()
        gd2d.draw_gaussian_distribution_ellipse_from_dataset()
        gd2d.draw_gaussian_distribution_ellipse(gmme.mu_list, gmme.sigma_list, gmme.cluster_prob_list,
                                                facecolor='none', edgecolor='black', alpha=1.0, linestyle='dashed')
        plt.pause(0.1)
        gmme.update_e_step()
        gmme.update_m_step()
        # check convergence
        log_likelihood = gmme.get_log_likelihood()
        print('log likelihood: {}'.format(log_likelihood))
        if log_likelihood - last_log_likelihood < 1e-7:
            break
        last_log_likelihood = log_likelihood

    print('=== parameter estimation result ===')
    gmme.print_parameter()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class GenerateData2D:
    def __init__(self,
                 data_num=500,
                 mu_list=[np.zeros(2), np.ones(2)],
                 sigma_list=[0.1*np.identity(2), 0.2*np.identity(2)],
                 cluster_prob_list=[0.4, 0.6]):
        self.mu_list = mu_list
        self.sigma_list = sigma_list
        self.cluster_prob_list = np.array(cluster_prob_list) / np.sum(cluster_prob_list)
        self.dataset = []
        for i in range(data_num):
            self.dataset.append(self.generate_data_one())

    def get_random_cluster(self):
        p_rand = np.random.rand()
        p_cluster = 0.0
        for omega_i, p_i in enumerate(self.cluster_prob_list):
            p_cluster += p_i
            if p_rand <= p_cluster:
                omega_ret = omega_i
                break
        return omega_ret

    def generate_data_one(self):
            omega_i = self.get_random_cluster()
            return [np.random.multivariate_normal(self.mu_list[omega_i], self.sigma_list[omega_i]), omega_i]

    def plot_dataset(self, color_list=['red', 'blue', 'green', 'purple', 'orange']):
        for omega_i in range(len(self.cluster_prob_list)):
            dataset_i = filter((lambda x: x[1] == omega_i), self.dataset)
            x_i = [data[0][0] for data in dataset_i]
            y_i = [data[0][1] for data in dataset_i]
            plt.scatter(x_i, y_i, color=color_list[omega_i])

    def draw_gaussian_distribution_ellipse_from_dataset(self):
        self.draw_gaussian_distribution_ellipse(self.mu_list, self.sigma_list, self.cluster_prob_list,
                                                facecolor='none', edgecolor='gray', alpha=0.75)

    def draw_gaussian_distribution_ellipse(self, mu_list, sigma_list, cluster_prob_list=None,
                                           ellipse_size_relative_to_sigma=2, **kwargs):

        def eigsorted(sigma):
            vals, vecs = np.linalg.eigh(sigma)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:,order]

        if cluster_prob_list is None:
            cluster_prob_list = np.array([1.0 / len(mu_list)] * len(mu_list))

        for omega_i in range(len(mu_list)):
            vals, vecs = eigsorted(sigma_list[omega_i])
            theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
            width, height = 2 * ellipse_size_relative_to_sigma * np.sqrt(vals)
            ellipse = Ellipse(xy=mu_list[omega_i], width=width, height=height, angle=theta,
                              linewidth=cluster_prob_list[omega_i]*10.0, **kwargs)
            plt.gca().add_artist(ellipse)

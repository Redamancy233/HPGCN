# import torch
# import numpy as np
from sklearn.metrics import pairwise_distances
from function import *

device = torch.device("cuda:0")

class Coreset_Greedy:
    def __init__(self, all_pts):
        self.all_pts = np.array(all_pts)
        self.dset_size = len(all_pts)
        self.min_distances = None
        self.already_selected = []

    def update_dist(self, centers, only_new=True, reset_dist=False):
        if reset_dist:
            self.min_distances = None
        if only_new:
            centers = [p for p in centers if p not in self.already_selected]

        if centers is not None:
            x = self.all_pts[centers]  # pick only centers
            dist = pairwise_distances(self.all_pts, x, metric='euclidean')

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def sample(self, already_selected, sample_size):

        # initially updating the distances
        self.update_dist(already_selected, only_new=False, reset_dist=True)
        self.already_selected = already_selected

        # epdb.set_trace()

        new_batch = []
        # pdb.set_trace()
        for _ in range(sample_size):
            if self.already_selected == []:
                ind = np.random.choice(np.arange(self.dset_size))
            else:
                ind = np.argmax(self.min_distances)

            assert ind not in already_selected
            self.update_dist([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)

        max_distance = max(self.min_distances)
        print("Max distance from cluster : %0.2f" % max_distance)

        return new_batch, max_distance

def active_sample(laLoader, unlaLodaer, sample_size, model):

        # 获取已标记特征
        labeledFeatures = get_features(model, laLoader)
        labeledFeatures = labeledFeatures.detach().cpu().numpy()
        # 获取未标记特征
        unlabeledFeatures = get_features(model, unlaLodaer)
        unlabeledFeatures = unlabeledFeatures.detach().cpu().numpy()

        allFeatures = np.concatenate((labeledFeatures, unlabeledFeatures), axis=0)
        labeled_indices = np.arange(0, len(labeledFeatures))
        unlabeled_rows = np.arange(0, len(unlabeledFeatures))

        coreset = Coreset_Greedy(allFeatures)
        new_batch, max_distance = coreset.sample(labeled_indices, sample_size)

        new_batch = [i - len(allFeatures) for i in new_batch]

        sample_rows = unlabeled_rows[new_batch]

        return sample_rows



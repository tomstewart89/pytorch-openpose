import sys

sys.path.insert(0, "/home/tom/repos/pytorch-openpose")

import cv2
import numpy as np
import math
import time
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
import torch
from torchvision import transforms
import time

from src import util
from src.model import bodypose_model


# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [
    [2, 3],
    [2, 6],
    [3, 4],
    [4, 5],
    [6, 7],
    [7, 8],
    [2, 9],
    [9, 10],
    [10, 11],
    [2, 12],
    [12, 13],
    [13, 14],
    [2, 1],
    [1, 15],
    [15, 17],
    [1, 16],
    [16, 18],
    [3, 17],
    [6, 18],
]

# the middle joints heatmap correpondence
mapIdx = [
    [31, 32],
    [39, 40],
    [33, 34],
    [35, 36],
    [41, 42],
    [43, 44],
    [19, 20],
    [21, 22],
    [23, 24],
    [25, 26],
    [27, 28],
    [29, 30],
    [47, 48],
    [49, 50],
    [53, 54],
    [51, 52],
    [55, 56],
    [37, 38],
    [45, 46],
]


class Body(object):
    def __init__(self, model_path):
        self.model = bodypose_model()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        model_dict = util.transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

        self.thre1 = 0.1
        self.thre2 = 0.05

        self.stride = 8
        self.padValue = 128

        self.scale_search = [0.5]
        self.boxsize = 368

        self.mid_num = 10

    def build_avg_paf_heatmap(self, oriImg):

        multiplier = [x * self.boxsize / oriImg.shape[0] for x in self.scale_search]
        heatmap_avg = torch.zeros((oriImg.shape[0], oriImg.shape[1], 19)).cuda()
        paf_avg = torch.zeros((oriImg.shape[0], oriImg.shape[1], 38)).cuda()

        h, w = oriImg.shape[:2]

        pad_down = int(np.ceil(h / self.stride) * self.stride - h)
        pad_right = int(np.ceil(w / self.stride) * self.stride - w)

        for scale in multiplier:

            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(size=(int(h * scale), int(w * scale))),
                    transforms.Pad([0, 0, pad_down, pad_right], fill=0.5),
                ]
            )

            data = (transform(oriImg).unsqueeze(0) - 0.5).cuda()

            with torch.no_grad():
                Mconv7_stage6_L1, Mconv7_stage6_L2 = self.model(data)

            paf = transforms.Resize(size=data.shape[-2:])(Mconv7_stage6_L1.squeeze())
            paf = paf[:, : int(h * scale), : int(w * scale)]
            paf = transforms.Resize(size=(oriImg.shape[:2]))(paf).permute(1, 2, 0)

            heatmap = transforms.Resize(size=data.shape[-2:])(Mconv7_stage6_L2.squeeze())
            heatmap = heatmap[:, : int(h * scale), : int(w * scale)]
            heatmap = transforms.Resize(size=(oriImg.shape[:2]))(heatmap).permute(1, 2, 0)

            paf_avg += paf / len(multiplier)
            heatmap_avg += heatmap / len(multiplier)

        return heatmap_avg.cpu().numpy(), paf_avg.cpu().numpy()

    def extract_peaks(self, heatmap_avg):
        all_peaks = []
        peak_counter = 0

        for part in range(18):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(one_heatmap.shape)
            map_left[1:, :] = one_heatmap[:-1, :]
            map_right = np.zeros(one_heatmap.shape)
            map_right[:-1, :] = one_heatmap[1:, :]
            map_up = np.zeros(one_heatmap.shape)
            map_up[:, 1:] = one_heatmap[:, :-1]
            map_down = np.zeros(one_heatmap.shape)
            map_down[:, :-1] = one_heatmap[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (
                    one_heatmap >= map_left,
                    one_heatmap >= map_right,
                    one_heatmap >= map_up,
                    one_heatmap >= map_down,
                    one_heatmap > self.thre1,
                )
            )
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            peak_id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        return all_peaks

    def connect_joints(self, paf_avg, all_peaks):

        connection_all = []
        special_k = []

        for k in range(len(mapIdx)):
            score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0] - 1]
            candB = all_peaks[limbSeq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]
            if nA != 0 and nB != 0:
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        norm = max(0.001, norm)
                        vec = np.divide(vec, norm)

                        startend = list(
                            zip(
                                np.linspace(candA[i][0], candB[j][0], num=self.mid_num),
                                np.linspace(candA[i][1], candB[j][1], num=self.mid_num),
                            )
                        )

                        vec_x = np.array(
                            [
                                score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0]
                                for I in range(len(startend))
                            ]
                        )
                        vec_y = np.array(
                            [
                                score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1]
                                for I in range(len(startend))
                            ]
                        )

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                            0.5 * oriImg.shape[0] / norm - 1, 0
                        )
                        criterion1 = len(np.nonzero(score_midpts > self.thre2)[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]]
                            )

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if len(connection) >= min(nA, nB):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        return connection_all, special_k

    def build_candidates(self, all_peaks, connection_all, special_k):
        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][indexB] != partBs[i]:
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += subset[j2][:-2] + 1
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])
        # delete some rows of subset which has few parts occur
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)

        # subset: n*20 array, 0-17 is the index in candidate, 18 is the total score, 19 is the total parts
        # candidate: x, y, score, id
        return candidate, subset

    def __call__(self, oriImg):

        with TimeIt("build_avg_paf_heatmap"):
            heatmap_avg, paf_avg = self.build_avg_paf_heatmap(oriImg)

        with TimeIt("extract_peaks"):
            all_peaks = self.extract_peaks(heatmap_avg)

        with TimeIt("connect_joints"):
            connection_all, special_k = self.connect_joints(paf_avg, all_peaks)

        with TimeIt("build_candidates"):
            candidate, subset = self.build_candidates(all_peaks, connection_all, special_k)

        return candidate, subset


class TimeIt:
    def __init__(self, s):
        self.s = s
        self.t0 = None
        self.t1 = None

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, t, value, traceback):
        print("%s: %0.1fms" % (self.s, (time.time() - self.t0) * 1000.0))


if __name__ == "__main__":
    body_estimation = Body("/home/tom/repos/pytorch-openpose/model/body_pose_model.pth")

    test_image = "/home/tom/repos/pytorch-openpose/images/ski.jpg"
    oriImg = cv2.imread(test_image)  # B,G,R order

    candidate, subset = body_estimation(oriImg)
    canvas = util.draw_bodypose(oriImg, candidate, subset)
    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.show()

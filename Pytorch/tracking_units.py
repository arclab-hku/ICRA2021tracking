import torch
import statistics
import numpy as np
import cv2

def image_norm(img):
    contrast = img.max() - img.min()
    if contrast > 0:
        img = (img - img.min()) / contrast * 255
        img = img.astype(np.uint8)
    return img

def scores_norm(scores):
    weights = [float(i) / sum(scores) for i in scores]
    return weights

def findAll(matrix, value):
    result = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == value:
                result.append((i,j))
    return result

def getDistictiveScore(heat_map, rect):
    score = 0
    F_area = (rect[2] - rect[0]) * (rect[3] - rect[1])
    target_row = (rect[2] + rect[0])/2
    target_col = (rect[3] + rect[1])/2
    r = np.sqrt( ((rect[2] - rect[0])/2) ** 2 + ((rect[3] - rect[1])/2) ** 2)
    if F_area > 0:
        mask = np.zeros(heat_map.shape, np.uint8)
        mask[rect[1]:rect[3], rect[0]:rect[2]] = 255
        F = cv2.bitwise_and(heat_map, heat_map, mask = mask)
        B = heat_map - F
        DT_list = []
        locations = np.where(heat_map == heat_map.max())
        for row, col in zip(locations[0], locations[1]):
            d_i = np.sqrt((row - target_row) ** 2 + (col - target_col) ** 2)
            DT_list.append(np.exp(1 - (d_i / r) ** 2) - 1)

        DT = statistics.mean(DT_list)
        GT = (np.sum(F) / F_area - np.sum(B) / (heat_map.shape[0] * heat_map.shape[1] - F_area)) ** 2
        # GT = (F.max()/255 - B.max()/255) ** 2
        score = DT * GT
    return score

def feature_recommender(layers_data, layer_list, img_size, rect, top_N_feature = 10, top_N_layer = 2):
    recom_score_list = []
    recom_idx_list = []
    layer_score = []
    for idx in layer_list:
        fmaps = layers_data[idx].clone().detach()
        scores = []
        scale_x = fmaps.shape[3] / img_size[1]
        scale_y = fmaps.shape[2] / img_size[0]
        scaled_rect = [int(scale_x * rect[0]), int(scale_y * rect[1]), int(scale_x * rect[2]), int(scale_y * rect[3])]
        for fmap in fmaps[0, :, :, :]:
            heatmap = fmap.data.cpu().numpy()
            heatmap = image_norm(heatmap)
            score = getDistictiveScore(heatmap, scaled_rect)
            scores.append(score)
        # get idx of top N features in ascending order of their scores
        recom_idx = sorted(range(len(scores)), key=lambda sub: scores[sub])[-top_N_feature:]
        # save idx for this layer
        recom_idx_list.append(recom_idx)
        # get recommend score of selected features
        recom_score = [scores[i] for i in recom_idx]
        # save recommend scores
        recom_score_list.append(recom_score)
        # save overall score of this layer
        layer_score.append(sum(recom_score))

    recom_layers = sorted(range(len(layer_score)), key=lambda sub: layer_score[sub])[-top_N_layer:]

    return recom_idx_list, recom_score_list, layer_score, recom_layers

def reconstruct_target_model(layers_data, layer_list, recom_idx_list, recom_score_list, recom_layers):
    heatmap_list = []
    for idx in recom_layers:
        fmaps = layers_data[layer_list[idx]].clone().detach()
        recom_idx = recom_idx_list[idx]
        scores = recom_score_list[idx]
        weights = scores_norm(scores)
        heatmap = 0
        for fidx, weight in zip(recom_idx, weights):
            fmap = fmaps[0, fidx, :, :]
            heatmap = heatmap + weight * fmap.data.cpu().numpy()

        heatmap = image_norm(heatmap)
        heatmap_list.append(heatmap)

    return heatmap_list
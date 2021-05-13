import torch
import statistics
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

def getDistictiveScore(heatmap, rect, img = None):
    score = 0
    F_area = (rect[2] - rect[0]) * (rect[3] - rect[1])
    target_col = (rect[2] + rect[0])/2
    target_row = (rect[3] + rect[1])/2
    r = np.sqrt(((rect[2] - rect[0])/2) ** 2 + ((rect[3] - rect[1])/2) ** 2)
    # img_resize = cv2.resize(img, heatmap.shape)
    # cv2.rectangle(img_resize, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 1)
    # plt.figure(num=1, figsize=(12, 6), dpi=80)
    if F_area > 0:
        mask = np.zeros(heatmap.shape, np.uint8)
        mask[rect[1]:rect[3], rect[0]:rect[2]] = 255
        F = cv2.bitwise_and(heatmap, heatmap, mask = mask)
        B = heatmap - F
        DT_list = []
        locations = np.where(heatmap == heatmap.max())
        for row, col in zip(locations[0], locations[1]):
            d_i = np.sqrt((row - target_row) ** 2 + (col - target_col) ** 2)
            DT_list.append(np.exp(1 - (1.5 * d_i / r) ** 2) - 1)

        DT = statistics.mean(DT_list)
        GT = (np.sum(F) / F_area - np.sum(B) / (heatmap.shape[0] * heatmap.shape[1] - F_area)) ** 2
        score = DT * GT

        # if score > 0:
        #     plt.clf()
        #     plt.subplot(1, 2, 1)
        #     plt.title(f'score = {score}')
        #     plt.imshow(heatmap, cmap='jet')
        #     plt.gca().add_patch(Rectangle((rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1], edgecolor = 'red', facecolor = 'none', lw = 1))
        #     plt.colorbar(label='activation heatmap')
        #     plt.subplot(1, 2, 2)
        #     plt.title(f'd_i = {d_i}, r = {r}')
        #     plt.imshow(img_resize)
        #     plt.pause(0.01)
        #     print('debug')

    # plt.show()

    return score

def feature_recommender(layers_data, layer_list, img, rect, top_N_feature = 10, top_N_layer = 2):
    recom_score_list = []
    recom_idx_list = []
    layer_score = []
    img_size = img.shape[:2]
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
    if recom_idx_list == 0 or recom_score_list == 0:
        for idx in recom_layers:
            fmaps = layers_data[layer_list[idx]].clone().detach()
            heatmap = 0
            for fmap in fmaps[0, :, :, :]:
                heatmap = heatmap + fmap

            heatmap = heatmap.data.cpu().numpy()
            heatmap_list.append(heatmap)
    else:
        for idx in recom_layers:
            fmaps = layers_data[layer_list[idx]].clone().detach()
            recom_idx = recom_idx_list[idx]
            scores = recom_score_list[idx]
            weights = scores_norm(scores)
            heatmap = 0
            for fidx, weight in zip(recom_idx, weights):
                fmap = fmaps[0, fidx, :, :]
                heatmap = heatmap + weight * fmap.data.cpu().numpy()

            heatmap_list.append(heatmap)

    return heatmap_list

# ======================================================================================================================
# correlation filter utils
# ======================================================================================================================

# ffttools
def fftd(img, backwards=False):
	# shape of img can be (m,n), (m,n,1) or (m,n,2)
	# in my test, fft provided by numpy and scipy are slower than cv2.dft
	return cv2.dft(np.float32(img), flags = ((cv2.DFT_INVERSE | cv2.DFT_SCALE) if backwards else cv2.DFT_COMPLEX_OUTPUT))   # 'flags =' is necessary!


def real(img):
    return img[:, :, 0]


def imag(img):
    return img[:, :, 1]


def complexMultiplication(a, b):
    res = np.zeros(a.shape, a.dtype)

    res[:, :, 0] = a[:, :, 0] * b[:, :, 0] - a[:, :, 1] * b[:, :, 1]
    res[:, :, 1] = a[:, :, 0] * b[:, :, 1] + a[:, :, 1] * b[:, :, 0]
    return res


def complexDivision(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1. / (b[:, :, 0] ** 2 + b[:, :, 1] ** 2)

    res[:, :, 0] = (a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1]) * divisor
    res[:, :, 1] = (a[:, :, 1] * b[:, :, 0] + a[:, :, 0] * b[:, :, 1]) * divisor
    return res


def rearrange(img):
    # return np.fft.fftshift(img, axes=(0,1))
    assert (img.ndim == 2)
    img_ = img.copy()
    # img_ = np.zeros(img.shape, img.dtype)
    xh1 = xh2 = yh1 = yh2 = 0
    if img.shape[1] % 2 == 0:
        xh1 = xh2 = img.shape[1] / 2
    else:
        xh1 = int(img.shape[1] / 2)
        xh2 = xh1 + 1

    if img.shape[0] % 2 == 0:
        yh1 = yh2 = img.shape[0] / 2
    else:
        yh1 = int(img.shape[0] / 2)
        yh2 = yh1 + 1

    img_[0:yh1, 0:xh1] = img[yh2:img.shape[0], xh2:img.shape[1]]
    img_[yh2:img.shape[0], xh2:img.shape[1]] = img[0:yh1, 0:xh1]
    img_[0:yh1, xh2:img.shape[1]] = img[yh2:img.shape[0], 0:xh1]
    img_[yh2:img.shape[0], 0:xh1] = img[0:yh1, xh2:img.shape[1]]

    return img_

# recttools
def x2(rect):
    return rect[0] + rect[2]


def y2(rect):
    return rect[1] + rect[3]


def limit(rect, limit):
    if (rect[0] + rect[2] > limit[0] + limit[2]):
        rect[2] = limit[0] + limit[2] - rect[0]
    if (rect[1] + rect[3] > limit[1] + limit[3]):
        rect[3] = limit[1] + limit[3] - rect[1]
    if (rect[0] < limit[0]):
        rect[2] -= (limit[0] - rect[0])
        rect[0] = limit[0]
    if (rect[1] < limit[1]):
        rect[3] -= (limit[1] - rect[1])
        rect[1] = limit[1]
    if (rect[2] < 0):
        rect[2] = 0
    if (rect[3] < 0):
        rect[3] = 0
    return rect


def getBorder(original, limited):
    res = [0, 0, 0, 0]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = x2(original) - x2(limited)
    res[3] = y2(original) - y2(limited)
    assert (np.all(np.array(res) >= 0))
    return res


def subwindow(img, window, borderType=cv2.BORDER_CONSTANT):
    cutWindow = [x for x in window]
    limit(cutWindow, [0, 0, img.shape[1], img.shape[0]])  # modify cutWindow
    assert (cutWindow[2] > 0 and cutWindow[3] > 0)
    border = getBorder(window, cutWindow)
    # cv2.namedWindow('tracking')
    # cv2.rectangle(img, (cutWindow[0], cutWindow[1]), (cutWindow[0] + cutWindow[2], cutWindow[1] + cutWindow[3]), (0, 255, 0), 1)
    # while True:
    #     cv2.imshow('tracking', img)
    #     c = cv2.waitKey(1) & 0xFF
    #     if c == 27 or c == ord('q'):
    #         break
    res = img[cutWindow[1]:cutWindow[1] + cutWindow[3], cutWindow[0]:cutWindow[0] + cutWindow[2]]

    if (border != [0, 0, 0, 0]):
        res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], borderType)
    return res


# ORCF tracker
class ORCFTracker:
    def __init__(self, multiscale=False):
        self.lambdar = 0.0001  # regularization
        self.padding = 1.5  # extra area surrounding the target
        self.output_sigma_factor = 0.125  # bandwidth of gaussian target

        self.interp_factor = 0.012  # linear interpolation factor for adaptation
        self.sigma = 0.6  # gaussian kernel bandwidth
        self.scale_step = 1 # scale adaptation

        self._x_sz = [0, 0]  # cv::Size, [width,height]  #[int,int]
        self._roi = [0., 0., 0., 0.]  # cv::Rect2f, [x,y,width,height]  #[float,float,float,float]
        self.size_patch = [0, 0, 0]  # [int,int,int]
        self._alphaf = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._yf = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._x = None  # numpy.ndarray    (size_patch[0], size_patch[1])
        self.hann = None  # numpy.ndarray    cos window (size_patch[0], size_patch[1])
        self.cnnFeature = None # size = frame size

    def subPixelPeak(self, left, center, right):
        divisor = 2 * center - right - left  # float
        return (0 if abs(divisor) < 1e-3 else 0.5 * (right - left) / divisor)

    def createHanningMats(self):
        hann2t, hann1t = np.ogrid[0:self.size_patch[0], 0:self.size_patch[1]]
        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (self.size_patch[1] - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (self.size_patch[0] - 1)))
        self.hann = hann2t * hann1t
        self.hann = self.hann.astype(np.float32)

    def createGaussianPeak(self, sizey, sizex):
        syh, sxh = sizey / 2, sizex / 2
        output_sigma = np.sqrt(sizex * sizey) / self.padding * self.output_sigma_factor
        mult = -0.5 / (output_sigma * output_sigma)
        y, x = np.ogrid[0:sizey, 0:sizex]
        y, x = (y - syh) ** 2, (x - sxh) ** 2
        res = np.exp(mult * (y + x))
        return fftd(res)

    # def gaussianCorrelation(self, x1, x2):
    #
    #     c = cv2.mulSpectrums(fftd(x1), fftd(x2), 0, conjB=True)  # 'conjB=' is necessary!
    #
    #     c = fftd(c, True)
    #     c = real(c)
    #     c = rearrange(c)
    #
    #     d = (np.sum(x1 * x1) + np.sum(x2 * x2) - 2.0 * c) / (
    #                 self.size_patch[0] * self.size_patch[1] * self.size_patch[2])
    #
    #     d = d * (d >= 0)
    #     d = np.exp(-d / (self.sigma * self.sigma))
    #
    #     return d

    def getTargetModel(self):
        extracted_roi = [0, 0, 0, 0]  # [int,int,int,int]
        cx = self._roi[0] + self._roi[2] / 2  # float
        cy = self._roi[1] + self._roi[3] / 2  # float

        padded_w = self._roi[2] * self.padding
        padded_h = self._roi[3] * self.padding
        self._x_sz[0] = int(padded_w)
        self._x_sz[1] = int(padded_h)
        self._x_sz[0] = int(self._x_sz[0]) / 2 * 2
        self._x_sz[1] = int(self._x_sz[1]) / 2 * 2

        extracted_roi[2] = int(self.scale_step * self._x_sz[0])
        extracted_roi[3] = int(self.scale_step * self._x_sz[1])
        extracted_roi[0] = int(cx - extracted_roi[2] / 2)
        extracted_roi[1] = int(cy - extracted_roi[3] / 2)

        FeaturesMap = subwindow(self.cnnFeature, extracted_roi, cv2.BORDER_REPLICATE)
        FeaturesMap = FeaturesMap.astype(np.float32) / 255.0 - 0.5
        self.size_patch = [FeaturesMap.shape[0], FeaturesMap.shape[1], 1]

        self.createHanningMats()  # create cos window need size_patch

        target_model_x = self.hann * FeaturesMap

        return target_model_x

    def detect(self, z, x):
        kzf = cv2.mulSpectrums(fftd(x), fftd(z), 0, conjB=True)
        res = real(fftd(complexMultiplication(self._alphaf, kzf), True))

        _, pv, _, pi = cv2.minMaxLoc(res)  # pv:float  pi:tuple of int
        p = [float(pi[0]), float(pi[1])]  # cv::Point2f, [x,y]  #[float,float]

        if (pi[0] > 0 and pi[0] < res.shape[1] - 1):
            p[0] += self.subPixelPeak(res[pi[1], pi[0] - 1], pv, res[pi[1], pi[0] + 1])
        if (pi[1] > 0 and pi[1] < res.shape[0] - 1):
            p[1] += self.subPixelPeak(res[pi[1] - 1, pi[0]], pv, res[pi[1] + 1, pi[0]])

        p[0] -= res.shape[1] / 2.
        p[1] -= res.shape[0] / 2.

        return p, pv

    def init(self, roi, image, cnnFeature):
        self.cnnFeature = cnnFeature
        self._roi = list(map(float, roi))
        assert (roi[2] > 0 and roi[3] > 0)
        self._x = self.getTargetModel()
        self._yf = self.createGaussianPeak(self.size_patch[0], self.size_patch[1])
        xf = fftd(self._x)
        kf = cv2.mulSpectrums(xf, xf, 0, conjB=True)  # 'conjB=' is necessary!
        self._alphaf = complexDivision(self._yf, kf + self.lambdar)

    def update(self, image, cnnFeature):
        self.cnnFeature = cnnFeature
        if (self._roi[0] + self._roi[2] <= 0):  self._roi[0] = -self._roi[2] + 1
        if (self._roi[1] + self._roi[3] <= 0):  self._roi[1] = -self._roi[2] + 1
        if (self._roi[0] >= image.shape[1] - 1):  self._roi[0] = image.shape[1] - 2
        if (self._roi[1] >= image.shape[0] - 1):  self._roi[1] = image.shape[0] - 2

        cx = self._roi[0] + self._roi[2] / 2.
        cy = self._roi[1] + self._roi[3] / 2.

        loc, peak_value = self.detect(self._x, self.getTargetModel())

        self._roi[0] = cx - self._roi[2] / 2.0 + loc[0] * self.scale_step
        self._roi[1] = cy - self._roi[3] / 2.0 + loc[1] * self.scale_step

        if (self._roi[0] >= image.shape[1] - 1):  self._roi[0] = image.shape[1] - 1
        if (self._roi[1] >= image.shape[0] - 1):  self._roi[1] = image.shape[0] - 1
        if (self._roi[0] + self._roi[2] <= 0):  self._roi[0] = -self._roi[2] + 2
        if (self._roi[1] + self._roi[3] <= 0):  self._roi[1] = -self._roi[3] + 2
        assert (self._roi[2] > 0 and self._roi[3] > 0)

        x = self.getTargetModel()
        self._yf = self.createGaussianPeak(self.size_patch[0], self.size_patch[1])
        xf = fftd(x)
        kf = cv2.mulSpectrums(xf, xf, 0, conjB=True)  # 'conjB=' is necessary!
        alphaf = complexDivision(self._yf, kf + self.lambdar)
        self._x = (1 - self.interp_factor) * self._x + self.interp_factor * x
        self._alphaf = (1 - self.interp_factor) * self._alphaf + self.interp_factor * alphaf

        self.scale_step = 1

        return self._roi
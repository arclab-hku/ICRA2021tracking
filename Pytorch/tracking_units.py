import torch
import statistics
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import medfilt
import math

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

def getDistictiveScore(heatmap, rect):
    score = 0
    F_area = (rect[2] - rect[0]) * (rect[3] - rect[1])
    target_col = (rect[2] + rect[0])/2
    target_row = (rect[3] + rect[1])/2
    r = np.sqrt(((rect[2] - rect[0])/2) ** 2 + ((rect[3] - rect[1])/2) ** 2)
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

    return score

def feature_recommender(layers_data, layer_list, img, rect, top_N_feature = 10, top_N_layer = 1):
    recom_score_list = []
    recom_idx_list = []
    layer_score = []
    img_size = img.shape[:2]
    for idx in layer_list:
        fmaps = layers_data[idx].clone().detach()
        scores = []
        scale_x = fmaps.shape[3] / img_size[1]
        scale_y = fmaps.shape[2] / img_size[0]
        scaled_rect = [round(scale_x * rect[0]), round(scale_y * rect[1]), round(scale_x * rect[2]), round(scale_y * rect[3])]
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

def getWeightedFeatures(layers_data, layer_list, recom_idx_list, recom_score_list, recom_layers, map_size = 52):
    weightedFeatures = 0
    if recom_idx_list == 0 or recom_score_list == 0:
        for idx in recom_layers:
            fmaps = layers_data[layer_list[idx]].clone().detach()
            heatmap = 0
            for fmap in fmaps[0, :, :, :]:
                heatmap += fmap

            heatmap = heatmap.data.cpu().numpy()
            weightedFeatures += cv2.resize(heatmap, (map_size, map_size))
    else:
        for idx in recom_layers:
            fmaps = layers_data[layer_list[idx]].clone().detach()
            recom_idx = recom_idx_list[idx]
            weights = recom_score_list[idx]
            # weights = scores_norm(scores)
            heatmap = 0
            for fidx, weight in zip(recom_idx, weights):
                fmap = fmaps[0, fidx, :, :]
                heatmap += weight * fmap.data.cpu().numpy()

            weightedFeatures += sum(weights) * cv2.resize(heatmap, (map_size, map_size))

    weightedFeatures = image_norm(weightedFeatures)

    return weightedFeatures

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


# def rearrange(img):
#     # return np.fft.fftshift(img, axes=(0,1))
#     assert (img.ndim == 2)
#     img_ = img.copy()
#     # img_ = np.zeros(img.shape, img.dtype)
#     xh1 = xh2 = yh1 = yh2 = 0
#     if img.shape[1] % 2 == 0:
#         xh1 = xh2 = int(img.shape[1] / 2)
#     else:
#         xh1 = int(img.shape[1] / 2)
#         xh2 = xh1 + 1
#
#     if img.shape[0] % 2 == 0:
#         yh1 = yh2 = int(img.shape[0] / 2)
#     else:
#         yh1 = int(img.shape[0] / 2)
#         yh2 = yh1 + 1
#
#     img_[0:yh1, 0:xh1] = img[yh2:img.shape[0], xh2:img.shape[1]]
#     img_[yh2:img.shape[0], xh2:img.shape[1]] = img[0:yh1, 0:xh1]
#     img_[0:yh1, xh2:img.shape[1]] = img[yh2:img.shape[0], 0:xh1]
#     img_[yh2:img.shape[0], 0:xh1] = img[0:yh1, xh2:img.shape[1]]
#
#     return img_

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
    res = img[cutWindow[1]:cutWindow[1] + cutWindow[3], cutWindow[0]:cutWindow[0] + cutWindow[2]]

    if (border != [0, 0, 0, 0]):
        res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], borderType)
    return res

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, math.sqrt(variance)

# ORCF tracker
class ORCFTracker:
    def __init__(self):
        self.lambdar = 0.0001  # regularization
        self.padding = 1.5  # extra area surrounding the target
        self.sigma = 0.4  # gaussian kernel bandwidth, coswindow
        self.output_sigma_factor = 0.05  # bandwidth of gaussian target
        self.interp_factor = 0.05  # linear interpolation factor for adaptation
        self.scale_gamma = 0.9
        self.keyFrame = False
        self.hann = None  # numpy.ndarray    cos window (size_patch[0], size_patch[1])
        self.cnnFeature = None  # size = frame size
        self.size_patch = [0, 0, 0]  # current patch size [int,int,int]
        self.confidence = 1
        self.feature_channel = 1

        self._scale2img_x = 1.0
        self._scale2img_y = 1.0
        self._x_sz = [0, 0]  # template cv::Size, [width,height]  #[int,int]
        self._roi = [0., 0., 0., 0.]  # cv::Rect2f, [x,y,width,height]  #[float,float,float,float]
        self._alphaf = []  # list of numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._yf = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._x = None  # numpy.ndarray    (size_patch[0], size_patch[1])
        self._scale2keyframe_x = 1
        self._scale2keyframe_y = 1
        self._scale_x_buffer = []  # store scale of w
        self._scale_y_buffer = []  # store scale of h
        self._buffer_size = 5
        self._keyFrame_buffer = []
        self._keyFrame_meanstd = [0., 0., 0.]  # store [mean, std x, std y] of feature activation

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

    # def gaussianCorrelation(self, x1, x2, enable_guassian = True):
    #     kzf = cv2.mulSpectrums(fftd(x1), fftd(x2), 0, conjB=True)  # 'conjB=' is necessary!
    #     if enable_guassian:
    #         c = fftd(kzf, True)
    #         c = real(c)
    #         c = rearrange(c)
    #         if (x1.ndim == 3 and x2.ndim == 3):
    #             d = (np.sum(x1[:, :, 0] * x1[:, :, 0]) + np.sum(x2[:, :, 0] * x2[:, :, 0]) - 2.0 * c) / (
    #                         self.size_patch[0] * self.size_patch[1] * self.size_patch[2])
    #             d = d * (d >= 0)
    #             d = np.exp(-d / (self.sigma * self.sigma))
    #         elif (x1.ndim == 2 and x2.ndim == 2):
    #             d = (np.sum(x1 * x1) + np.sum(x2 * x2) - 2.0 * c) / (
    #                         self.size_patch[0] * self.size_patch[1] * self.size_patch[2])
    #         else:
    #             print('input patch dimension error')
    #         d = d * (d >= 0)
    #         d = np.exp(-d / (self.sigma * self.sigma))
    #     else:
    #         d = kzf
    #
    #     return d

    def scaleUpdate(self, target_region):
        mean_activation = np.mean(target_region)
        locations = np.where(target_region > 0.5 * target_region.max())
        y_list = locations[0]
        x_list = locations[1]
        pixel_list = target_region[y_list, x_list]
        _, sigma_x = weighted_avg_and_std(x_list, pixel_list)
        _, sigma_y = weighted_avg_and_std(y_list, pixel_list)
        if self.keyFrame:
            self._keyFrame_buffer.append([sum(pixel_list), mean_activation, sigma_x, sigma_y])
            if len(self._keyFrame_buffer) == self._buffer_size:
                self.keyFrame = False
                self._keyFrame_meanstd = np.mean(self._keyFrame_buffer, axis=0)
        else:
            sumRatio = sum(pixel_list) / self._keyFrame_meanstd[0]
            meanRatio = mean_activation / self._keyFrame_meanstd[1]
            r = self.scale_gamma * sumRatio * meanRatio
            scale_step_x = r + (1 - self.scale_gamma) * sigma_x / self._keyFrame_meanstd[2]
            scale_step_y = r + (1 - self.scale_gamma) * sigma_y / self._keyFrame_meanstd[3]
            self._scale_x_buffer.append(scale_step_x)
            self._scale_y_buffer.append(scale_step_y)

        if len(self._scale_x_buffer) == self._buffer_size:
            medfilt(self._scale_x_buffer, 3)
            medfilt(self._scale_y_buffer, 3)
            self._scale2keyframe_x = np.mean(self._scale_x_buffer)
            self._scale2keyframe_y = np.mean(self._scale_y_buffer)
            self._scale_x_buffer.pop()
            self._scale_y_buffer.pop()

    def getTargetModel(self, image):
        extracted_roi = [0, 0, 0, 0]  # [int,int,int,int]
        cx = self._roi[0] + self._roi[2] / 2  # float
        cy = self._roi[1] + self._roi[3] / 2  # float

        padded_w = self._roi[2] * self.padding
        padded_h = self._roi[3] * self.padding

        extracted_roi[2] = round(padded_w)
        extracted_roi[3] = round(padded_h)
        extracted_roi[0] = round(cx - extracted_roi[2] / 2)
        extracted_roi[1] = round(cy - extracted_roi[3] / 2)

        cnnFeature_roi = subwindow(self.cnnFeature, extracted_roi, cv2.BORDER_REPLICATE)
        FeaturesMap = cnnFeature_roi.astype(np.float32) / 255.0 - 0.5

        image_resize = cv2.resize(image, (self.cnnFeature.shape[1], self.cnnFeature.shape[0]))
        rgb_roi = subwindow(image_resize, extracted_roi, cv2.BORDER_REPLICATE)
        hsv_Map = cv2.cvtColor(rgb_roi, cv2.COLOR_BGR2HSV)

        self.size_patch = [FeaturesMap.shape[0], FeaturesMap.shape[1], self.feature_channel]
        self.createHanningMats()  # create cos window need size_patch

        if self.feature_channel > 1:
            target_model_x = np.zeros((FeaturesMap.shape[0], FeaturesMap.shape[1], self.feature_channel),
                                      dtype=np.float32)
            for c in range(self.feature_channel):
                target_model_x[:, :, c] = self.hann * FeaturesMap[:, :, c]
        else:
            target_model_x = self.hann * FeaturesMap

        return target_model_x, cnnFeature_roi

    def init(self, roi, image, cnnFeature):
        if cnnFeature.ndim == 3:
            self.feature_channel = cnnFeature.shape[2]
        self._roi = list(map(float, roi))
        self.cnnFeature = cnnFeature
        self._scale2img_x = image.shape[1] / cnnFeature.shape[1]
        self._scale2img_y = image.shape[0] / cnnFeature.shape[0]
        self._roi[0] = self._roi[0] / self._scale2img_x
        self._roi[1] = self._roi[1] / self._scale2img_y
        self._roi[2] = self._roi[2] / self._scale2img_x
        self._roi[3] = self._roi[3] / self._scale2img_y
        self._alphaf = []
        self._scale_x_buffer = []  # store scale of w
        self._scale_y_buffer = []  # store scale of h
        self._keyFrame_buffer = []
        self._x, searchingRegion = self.getTargetModel(image)
        self._yf = self.createGaussianPeak(self.size_patch[0], self.size_patch[1])
        if self.feature_channel > 1:
            for c in range(self.feature_channel):
                xf = fftd(self._x[:, :, c])
                kf = cv2.mulSpectrums(xf, xf, 0, conjB=True)  # 'conjB=' is necessary!
                self._alphaf.append(complexDivision(self._yf, kf + self.lambdar))
        else:
            xf = fftd(self._x)
            kf = cv2.mulSpectrums(xf, xf, 0, conjB=True)  # 'conjB=' is necessary!
            self._alphaf.append(complexDivision(self._yf, kf + self.lambdar))

        self.keyFrame = True
        # target_region = [int(self._roi[0]), int(self._roi[1]), int(self._roi[2]), int(self._roi[3])]
        # target_feature = subwindow(self.cnnFeature, target_region, cv2.BORDER_REPLICATE)
        # self.scaleUpdate(target_feature)
        self.scaleUpdate(searchingRegion)
        self._x_sz = [self._roi[2], self._roi[3]]

    def detect(self, x):
        res = 0
        if self.feature_channel > 1:
            for c in range(self.feature_channel):
                kzf_cnn = cv2.mulSpectrums(fftd(x[:, :, c]), fftd(self._x[:, :, c]), 0, conjB=True)
                res = res + real(fftd(complexMultiplication(self._alphaf[c], kzf_cnn), True))
        else:
            kzf_cnn = cv2.mulSpectrums(fftd(x), fftd(self._x), 0, conjB=True)
            res = real(fftd(complexMultiplication(self._alphaf[0], kzf_cnn), True))

        _, pv, _, pi = cv2.minMaxLoc(res)  # pv:float  pi:tuple of int
        p = [float(pi[0]), float(pi[1])]  # cv::Point2f, [x,y]  #[float,float]

        self.confidence = pv

        if (pi[0] > 0 and pi[0] < res.shape[1] - 1):
            p[0] += self.subPixelPeak(res[pi[1], pi[0] - 1], pv, res[pi[1], pi[0] + 1])
        if (pi[1] > 0 and pi[1] < res.shape[0] - 1):
            p[1] += self.subPixelPeak(res[pi[1] - 1, pi[0]], pv, res[pi[1] + 1, pi[0]])

        p[0] -= res.shape[1] / 2.
        p[1] -= res.shape[0] / 2.

        return p, pv

    def update(self, image, cnnFeature):
        self.cnnFeature = cnnFeature

        cx = self._roi[0] + self._roi[2] / 2.
        cy = self._roi[1] + self._roi[3] / 2.

        x, searchingRegion = self.getTargetModel(image)
        x = cv2.resize(x, (self._x.shape[1], self._x.shape[0]))

        loc, peak_value = self.detect(x)

        cx = cx + loc[0] * self._scale2keyframe_x
        cy = cy + loc[1] * self._scale2keyframe_y
        self._roi[0] = cx - self._roi[2] / 2.0
        self._roi[1] = cy - self._roi[3] / 2.0

        # target_region = [int(self._roi[0]), int(self._roi[1]), int(self._roi[2]), int(self._roi[3])]
        # target_feature = subwindow(self.cnnFeature, target_region, cv2.BORDER_REPLICATE)
        # self.scaleUpdate(target_feature)
        self.scaleUpdate(searchingRegion)

        self._roi[2] = self._x_sz[0] * self._scale2keyframe_x
        self._roi[3] = self._x_sz[1] * self._scale2keyframe_y
        self._roi[0] = cx - self._roi[2] / 2.0
        self._roi[1] = cy - self._roi[3] / 2.0

        # if out of view
        if (self._roi[0] < 1):
            self._roi[0] = 1
        if (self._roi[1] < 1):
            self._roi[1] = 1
        if (self._roi[0] + self._roi[2] > cnnFeature.shape[1] - 1):
            self._roi[2] = cnnFeature.shape[1] - 1 - self._roi[0]
        if (self._roi[1] + self._roi[3] > cnnFeature.shape[0] - 1):
            self._roi[3] = cnnFeature.shape[0] - 1 - self._roi[1]

        x, searchingRegion = self.getTargetModel(image)
        x = cv2.resize(x, (self._x.shape[1], self._x.shape[0]))
        # self._yf = self.createGaussianPeak(x.shape[0], x.shape[1])
        if self.feature_channel > 1:
            for c in range(self.feature_channel):
                xf = fftd(x[:, :, c])
                kf = cv2.mulSpectrums(xf, xf, 0, conjB=True)  # 'conjB=' is necessary!
                alphaf = complexDivision(self._yf, kf + self.lambdar)
                self._x[:, :, c] = (1 - self.interp_factor) * self._x[:, :, c] + self.interp_factor * x[:, :, c]
                self._alphaf[c] = (1 - self.interp_factor) * self._alphaf[c] + self.interp_factor * alphaf
        else:
            xf = fftd(x)
            kf = cv2.mulSpectrums(xf, xf, 0, conjB=True)  # 'conjB=' is necessary!
            alphaf = complexDivision(self._yf, kf + self.lambdar)
            self._x = (1 - self.interp_factor) * self._x + self.interp_factor * x
            self._alphaf[0] = (1 - self.interp_factor) * self._alphaf[0] + self.interp_factor * alphaf
        # # return tracking result
        target_roi = [self._roi[0] * self._scale2img_x, self._roi[1] * self._scale2img_y, self._roi[2] * self._scale2img_x, self._roi[3] * self._scale2img_y]

        return target_roi, searchingRegion
# This is the Pytorch demo code for the paper:
# "Online Recommendation-based Convolutional Features for Scale-Aware Visual Tracking" ICRA2021
# Ran Duan, Hong Kong PolyU
# rduan036@gmail.com

import torch
import statistics
import numpy as np
import cv2
from matplotlib.patches import Rectangle
from scipy.signal import medfilt, ricker
import math

# ======================================================================================================================
# part 1: cnn feature recommender utils
# ======================================================================================================================
def tensor_norm(tensor):
    contrast = tensor.max() - tensor.min()
    if contrast > 0:
        tn = (tensor - tensor.min()) / contrast * 255
        return tn
    else:
        return tensor

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

# paper equation 1,2,3:
def getDistictiveScore(heatmap, mask, target_row, target_col, r, F_area, B_area):
    score = 0
    F = cv2.bitwise_and(heatmap, heatmap, mask=mask)
    B = heatmap - F
    DT_list = []
    locations = np.where(heatmap == heatmap.max())
    for row, col in zip(locations[0], locations[1]):
        d_i = np.sqrt((row - target_row) ** 2 + (col - target_col) ** 2)
        DT_list.append(np.exp(1 - (1.5 * d_i / r) ** 2) - 1)

    DT = statistics.mean(DT_list)
    GT = (np.sum(F) / F_area - np.sum(B) / B_area) ** 2
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
        F_area = (scaled_rect[2] - scaled_rect[0]) * (scaled_rect[3] - scaled_rect[1])
        target_col = (scaled_rect[2] + scaled_rect[0]) / 2
        target_row = (scaled_rect[3] + scaled_rect[1]) / 2
        r = np.sqrt(((scaled_rect[2] - scaled_rect[0]) / 2) ** 2 + ((scaled_rect[3] - scaled_rect[1]) / 2) ** 2)
        mask = np.zeros([fmaps.shape[2], fmaps.shape[3]], np.uint8)
        B_area = fmaps.shape[2] * fmaps.shape[3] - F_area
        mask[scaled_rect[1]:scaled_rect[3], scaled_rect[0]:scaled_rect[2]] = 255
        for fmap in fmaps[0, :, :, :]:
            heatmap = tensor_norm(fmap)
            heatmap = heatmap.cpu().numpy()
            score = getDistictiveScore(heatmap, mask, target_row, target_col, r, F_area, B_area)
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

# paper equation 4 without coswindow
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
            weights_tensor = torch.Tensor(weights).cuda()
            heatmap = 0
            for fidx, weight in zip(recom_idx, weights_tensor):
                fmap = fmaps[0, fidx, :, :]
                heatmap += weight * fmap
                # heatmap += weight * fmap.data.cpu().numpy()
            weightedFeatures += sum(weights) * cv2.resize(heatmap.data.cpu().numpy(), (map_size, map_size))

    weightedFeatures = image_norm(weightedFeatures)

    return weightedFeatures

# ======================================================================================================================
# part 2: correlation filter utils
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
    res = 0
    cutWindow = [x for x in window]
    limit(cutWindow, [0, 0, img.shape[1], img.shape[0]])  # modify cutWindow
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

# Online Recommended-feature Correlation Filter (ORCF) tracker
class ORCFTracker:
    def __init__(self):
        self.lambdar = 0.0001  # regularization
        self.padding = 1.5  # extra area surrounding the target
        self.sigma = 0.4  # gaussian kernel bandwidth, coswindow
        self.output_sigma_factor = 0.05  # bandwidth of gaussian target
        self.interp_factor = 0.05  # linear interpolation factor for adaptation
        self.scale_gamma = 0.9
        self.keyFrame = False
        self.targetMask = None  # numpy.ndarray    cos window (size_patch[0], size_patch[1])
        self.cnnFeature = None  # size = frame size
        self.size_patch = [0, 0, 0]  # current patch size [int,int,int]
        self.confidence = 1
        self.feature_channel = 1
        self.target_center = [0., 0.]

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

    def roiCheck(self, size_y, size_x):
        # if out of view
        if (self._roi[0] < 1):
            self._roi[0] = 1
        if (self._roi[1] < 1):
            self._roi[1] = 1
        if (self._roi[0] + self._roi[2] > size_x - 1):
            self._roi[2] = size_x - 1 - self._roi[0]
        if (self._roi[1] + self._roi[3] > size_y - 1):
            self._roi[3] = size_y - 1 - self._roi[1]
        if self.target_center[0] < 3 or self.target_center[0] > size_x - 3:
            self.confidence = 0
        if self.target_center[1] < 3 or self.target_center[1] > size_y - 3:
            self.confidence = 0

    def createCosWindow(self):
        cosy, cosx = np.ogrid[0:self.size_patch[0], 0:self.size_patch[1]]
        cosx = 0.5 * (1 - np.cos(2 * np.pi * cosx / (self.size_patch[1] - 1)))
        cosy = 0.5 * (1 - np.cos(2 * np.pi * cosy / (self.size_patch[0] - 1)))
        self.targetMask = cosy * cosx
        self.targetMask = self.targetMask.astype(np.float32)

    def createGaussianMask(self):
        sizey = self.size_patch[0]
        sizex = self.size_patch[1]
        syh, sxh = sizey / 2, sizex / 2
        output_sigma = np.sqrt(sizex * sizey) / self.padding * self.sigma
        mult = -0.5 / (output_sigma * output_sigma)
        y, x = np.ogrid[0:sizey, 0:sizex]
        y, x = (y - syh) ** 2, (x - sxh) ** 2
        self.targetMask = np.exp(mult * (y + x))

    def createRickerMask(self):
        ricker_x = ricker(self.size_patch[1], self.sigma * self.size_patch[1] / self.padding)
        ricker_y = ricker(self.size_patch[0], self.sigma * self.size_patch[0] / self.padding)
        ry = np.ones([1, len(ricker_y)]) * ricker_y
        rx = np.ones([1, len(ricker_x)]) * ricker_x
        self.targetMask = np.transpose(ry) * rx
        self.targetMask = self.targetMask.astype(np.float32)

    def createRickerPeak(self, sizey, sizex):
        ricker_x = ricker(self.size_patch[1], self.output_sigma_factor * sizey / self.padding)
        ricker_y = ricker(self.size_patch[0], self.output_sigma_factor * sizex / self.padding)
        ry = np.ones([1, len(ricker_y)]) * ricker_y
        rx = np.ones([1, len(ricker_x)]) * ricker_x
        ricker_peak = np.transpose(ry) * rx
        return fftd(ricker_peak)

    def createGaussianPeak(self, sizey, sizex):
        syh, sxh = sizey / 2, sizex / 2
        output_sigma = np.sqrt(sizex * sizey) / self.padding * self.output_sigma_factor
        mult = -0.5 / (output_sigma * output_sigma)
        y, x = np.ogrid[0:sizey, 0:sizex]
        y, x = (y - syh) ** 2, (x - sxh) ** 2
        res = np.exp(mult * (y + x))
        return fftd(res)

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
        cx = self.target_center[0]
        cy = self.target_center[1]

        padded_w = self._x_sz[0] * self._scale2keyframe_x * self.padding
        padded_h = self._x_sz[1] * self._scale2keyframe_y * self.padding

        extracted_roi[2] = round(padded_w)
        extracted_roi[3] = round(padded_h)
        extracted_roi[0] = round(cx - extracted_roi[2] / 2)
        extracted_roi[1] = round(cy - extracted_roi[3] / 2)

        cnnFeature_roi = subwindow(self.cnnFeature, extracted_roi, cv2.BORDER_REPLICATE)
        # FeaturesMap = cnnFeature_roi.astype(np.float32) / 255.0 - 0.5
        FeaturesMap = cnnFeature_roi.astype(np.float32) / 255.0
        self.size_patch = [FeaturesMap.shape[0], FeaturesMap.shape[1], self.feature_channel]
        # self.createCosWindow()  # create cos window need size_patch
        # self.createGaussianMask() #
        self.createRickerMask() #

        if self.feature_channel > 1:
            target_model_x = np.zeros((FeaturesMap.shape[0], FeaturesMap.shape[1], self.feature_channel),
                                      dtype=np.float32)
            for c in range(self.feature_channel):
                target_model_x[:, :, c] = self.targetMask * FeaturesMap[:, :, c]
        else:
            target_model_x = self.targetMask * FeaturesMap

        return target_model_x, cnnFeature_roi

    def init(self, roi, image, cnnFeature):
        if cnnFeature.ndim == 3:
            self.feature_channel = cnnFeature.shape[2]
        self.confidence = 1
        self._roi = list(map(float, roi))
        self.cnnFeature = cnnFeature
        self._scale2img_x = image.shape[1] / cnnFeature.shape[1]
        self._scale2img_y = image.shape[0] / cnnFeature.shape[0]
        self._roi[0] = self._roi[0] / self._scale2img_x
        self._roi[1] = self._roi[1] / self._scale2img_y
        self._roi[2] = self._roi[2] / self._scale2img_x
        self._roi[3] = self._roi[3] / self._scale2img_y
        self.target_center[0] = self._roi[0] + self._roi[2] / 2
        self.target_center[1] = self._roi[1] + self._roi[3] / 2
        self._x_sz = [self._roi[2], self._roi[3]]
        self._alphaf = []
        self._scale_x_buffer = []  # store scale of w
        self._scale_y_buffer = []  # store scale of h
        self._keyFrame_buffer = []
        self._x, searchingRegion = self.getTargetModel(image)
        self._yf = self.createGaussianPeak(self.size_patch[0], self.size_patch[1])
        # self._yf = self.createRickerPeak(self.size_patch[0], self.size_patch[1])

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
        self.scaleUpdate(searchingRegion)

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

        cx = self.target_center[0]
        cy = self.target_center[1]

        x, searchingRegion = self.getTargetModel(image)
        if self.confidence == 0:
            return self._roi, cnnFeature

        x = cv2.resize(x, (self._x.shape[1], self._x.shape[0]))

        loc, peak_value = self.detect(x)

        self.target_center[0] = cx + loc[0] * self._scale2keyframe_x
        self.target_center[1] = cy + loc[1] * self._scale2keyframe_y
        self._roi[0] = self.target_center[0] - self._roi[2] / 2.0
        self._roi[1] = self.target_center[1] - self._roi[3] / 2.0

        self.scaleUpdate(searchingRegion)

        self._roi[2] = self._x_sz[0] * self._scale2keyframe_x
        self._roi[3] = self._x_sz[1] * self._scale2keyframe_y
        self._roi[0] = cx - self._roi[2] / 2.0
        self._roi[1] = cy - self._roi[3] / 2.0

        self.roiCheck(cnnFeature.shape[0], cnnFeature.shape[1])

        if self.confidence == 0:
            return [0, 0, 0, 0], searchingRegion

        x, searchingRegion = self.getTargetModel(image)
        x = cv2.resize(x, (self._x.shape[1], self._x.shape[0]))
        # self._yf = self.createGaussianPeak(x.shape[0], x.shape[1])
        # if self.keyFrame:
        #     interp_factor = self.interp_factor
        # else:
        #     interp_factor = 0.2
        interp_factor = self.interp_factor
        if self.feature_channel > 1:
            for c in range(self.feature_channel):
                xf = fftd(x[:, :, c])
                kf = cv2.mulSpectrums(xf, xf, 0, conjB=True)  # 'conjB=' is necessary!
                alphaf = complexDivision(self._yf, kf + self.lambdar)
                self._x[:, :, c] = (1 - interp_factor) * self._x[:, :, c] + interp_factor * x[:, :, c]
                self._alphaf[c] = (1 - interp_factor) * self._alphaf[c] + interp_factor * alphaf
        else:
            xf = fftd(x)
            kf = cv2.mulSpectrums(xf, xf, 0, conjB=True)  # 'conjB=' is necessary!
            alphaf = complexDivision(self._yf, kf + self.lambdar)
            self._x = (1 - interp_factor) * self._x + interp_factor * x
            self._alphaf[0] = (1 - interp_factor) * self._alphaf[0] + interp_factor * alphaf
        # # return tracking result

        target_roi = [self._roi[0] * self._scale2img_x, self._roi[1] * self._scale2img_y, self._roi[2] * self._scale2img_x, self._roi[3] * self._scale2img_y]

        return target_roi, searchingRegion

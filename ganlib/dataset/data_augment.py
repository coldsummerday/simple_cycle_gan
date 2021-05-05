# -*- coding:utf-8 -*-
import os, sys, glob
import os.path as osp
import cv2
import numpy as np
import re
import math
import random


def sort_pts(pts):
    """
    This function used to sort points clockwise.
    :param pts:
    :return: order_pts
    """
    mean_pts = np.mean(pts, axis=0)
    cos_value = np.zeros((4,), dtype=np.float32)
    for i in range(4):
        cos_value[i] = np.arctan2(pts[i,1] - mean_pts[1], pts[i,0] - mean_pts[0])
    order_pts = pts[np.argsort(cos_value)]
    if order_pts[0,0] < mean_pts[0] and order_pts[1,0] < mean_pts[0]:
        order_pts = order_pts[[1,2,3,0]]
    return order_pts


def enlarge_poly(poly, scale=1.0):
    p_list = poly[[0,1,2,3,0]]  # 坐标点
    v_list = []
    for i in range(4):
        v_list.append(p_list[(i+1)%4] - p_list[i])
    v_list = np.array(v_list)
    nv_list = v_list / np.linalg.norm(v_list, 2, axis=1).reshape((4,1))  # 边长
    q_list = []
    for i in range(4):
        q_i = p_list[i] + (scale - 1) / 2 * v_list[((i-1)+4)%4] - (scale - 1) / 2 * v_list[i]
        if q_i[0] < 0:
            q_i[1] = q_i[1] - (q_i[1] - p_list[i,1]) * q_i[0] / (q_i[0] - p_list[i,0])
            q_i[0] = 0
        if q_i[1] < 0:
            q_i[0] = q_i[0] - (q_i[0] - p_list[i,0]) * q_i[1] / (q_i[1] - p_list[i,1])
            q_i[1] = 0
        q_list.append(q_i)
    return np.array(q_list).astype(np.float32)


def crop_roi(im, pts, scale=1.0):
    #order_pts = pts[[0,3,2,1]].astype(np.float32)
    order_pts = sort_pts(pts)
    # order_pts = pts
    enlarge_pts = enlarge_poly(order_pts, scale)
    #enlarge_pts = order_pts.copy()
    #rect = cv2.minAreaRect(enlarge_pts)
    #roi_size = (max(rect[1][0], rect[1][1]), min(rect[1][0], rect[1][1]))
    w = np.linalg.norm(enlarge_pts[1,:] - enlarge_pts[0,:])
    h = np.linalg.norm(enlarge_pts[3,:] - enlarge_pts[0,:])

    canvas = np.array([[0,0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    trans = cv2.getPerspectiveTransform(enlarge_pts, canvas)
    roi = cv2.warpPerspective(im, trans, (int(w), int(h)))
    return roi


def crop_imgs(txt_file, output_folder, output_txt, times):
    """
    对1w行驶证在原图上进行裁剪
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    f = open(txt_file, 'r')
    cnt = 0
    f_new = open(output_txt, 'w')
    for line in f.readlines():
        line = line.strip().split()
        label = line[1]
        img_name = line[0]
        sp = img_name.split('/')
        img_idx = sp[-1].split('_')
        # 行驶证
        # direc = sp[-2].split('_')[1]
        # ori_img = '/5T/xingshizheng/orig_train/xingshizheng_' + direc + '/' + img_idx[0] + '.jpg'
        # 港澳台通行证
        ori_img = '/5T/hmt_pass_data/hmt_pass/' + img_idx[0] + '.jpg'
        idx = img_idx[1].split('.')[0]
        print(ori_img, idx)
        txt_path = ori_img.replace('.jpg', '.txt')
        coord_line = open(txt_path, 'r').readlines()[int(idx)]
        # spt = coord_line.strip().split(',')[:-1]
        spt = coord_line.strip().split()[0].split(',')
        coord = np.array([float(spt[0]), float(spt[1]),
                             float(spt[2]), float(spt[3]),
                             float(spt[4]), float(spt[5]),
                             float(spt[6]), float(spt[7])])
        coord = coord.reshape((4,2)).astype(np.float32)
        rect = cv2.minAreaRect(coord)
        I = cv2.imread(ori_img)
        # padding = [1.05, 1.1, 1.2, 1.3, 0.9]   # 裁剪倍数
        for i in range(times):
            try:
                padding = random.uniform(0.90, 1.35)
                roi = crop_roi(I, coord, padding)
            except Exception as error:
                continue
            cnt += 1
            cv2.imwrite(osp.join(output_folder, str(cnt).zfill(6) + '.jpg'), roi)
            txt = output_folder + str(cnt).zfill(6) + '.jpg' + ' ' + label + '\n'
            print(cnt, txt)
            f_new.write(txt)
    f.close()
    f_new.close()


class Transform(object):
    def __init__(self, img):
        self.img = img

    def AddRotate(self, bg_color=(0,0,0), rot=3.0, **kwargs):
        """
        添加旋转
        """
        center = (self.img.shape[1] // 2, self.img.shape[0] // 2)
        rot = np.random.uniform(-rot, rot)
        rotateMatrix = cv2.getRotationMatrix2D(center, rot, 1.0)
        rot = np.radians(rot)
        new_w = int(self.img.shape[1]*abs(np.cos(rot)) + self.img.shape[0]*abs(np.sin(rot)))
        new_h = int(self.img.shape[1]*abs(np.sin(rot)) + self.img.shape[0]*abs(np.cos(rot)))
        rotateMatrix[0,2] += (new_w - self.img.shape[1]) / 2
        rotateMatrix[1,2] += (new_h - self.img.shape[0]) / 2
        im_final = cv2.warpAffine(self.img, rotateMatrix, (new_w, new_h), borderValue=bg_color)
        return im_final

    def AddBlur(self, level=4, **kwargs):
        """
        添加高斯模糊
        """
        level_ = np.random.randint(1, 1 + level)
        return cv2.blur(self.img, (level_ * 2 + 1, level_ * 2 + 1))

    def AddNoise(self, level=4, **kwargs):
        """
        添加高斯噪声
        """
        diff = 255 - self.img.max()
        level_ = np.random.randint(1, 1 + level)
        noise = np.random.normal(0,1+level_, self.img.shape)
        noise = (noise - noise.min()) / (noise.max()-noise.min()+1e-9)
        noise = diff * noise
        noise = noise.astype(np.uint8)
        dst = self.img + noise
        return dst

    def AddTfactor(self, **kwargs):
        """
        添加饱和度光照的噪声
        """
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        hsv[:,:,0] = hsv[:,:,0]*(0.8+ np.random.random()*0.2)
        hsv[:,:,1] = hsv[:,:,1]*(0.3+ np.random.random()*0.7)
        hsv[:,:,2] = hsv[:,:,2]*(0.5+ np.random.random()*0.5)

        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return out

    def AdjustHue(self, **kwargs):
        """
        调整图片的色度
        添加到色调通道的量在-1和1之间的间隔。
        如果值超过180，则会旋转这些值。
        """
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        val_range = (-0.5, 0.5)
        delta = np.random.uniform(val_range[0], val_range[1])
        hsv[:, :, 0] = np.mod(hsv[:, :, 0] + delta * 255, 255)  # 取余数
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _clip(self, image):
        """
        剪辑图像并将其转换为np.uint8
        """
        return np.clip(image, 0, 255).astype(np.uint8)

    def AdjustContrast(self, **kwargs):
        """
        调整一张图像的对比度
        """
        image = self.img
        contrast_range = (0.5, 1.3)
        factor = np.random.uniform(contrast_range[0], contrast_range[1])
        mean = image.mean(axis=0).mean(axis=0)
        return self._clip((image - mean) * factor + mean)

    def AdjustBrightness(self, **kwargs):
        """
        调整一张图片的亮度
        """
        image = self.img
        brightness_range = (-0.15, 0.15)
        delta = np.random.uniform(brightness_range[0], brightness_range[1])
        return self._clip(image + delta * 255)

    def data_augment(self, noise=True, rotate=True, saturate=True, blur=True,
                     hue=False, contrast=True, brightness=True):
        """
        调用前面几种方式进行数据增强
        """
        transforms = []
        if noise:
            transforms.append(self.AddNoise())
        if rotate:
            transforms.append(self.AddRotate())
        if saturate:
            transforms.append(self.AddTfactor())
        if blur:
            transforms.append(self.AddBlur())
        if hue:
            transforms.append(self.AdjustHue())
        if contrast:
            transforms.append(self.AdjustContrast())
        if brightness:
            transforms.append(self.AdjustBrightness())

        for i in range(len(transforms)):
            prob = random.random()
            if prob >= 0.5:
                self.img = transforms[i]
        return self.img

    def get_affine_matrix(self, srcpoint, h, w, ratio_w=None, ratio_h=None):
        """
        求透视变换矩阵
        """
        dw = max(math.sqrt(pow(srcpoint[0] - srcpoint[2], 2) + pow(srcpoint[1] - srcpoint[3], 2)),
                math.sqrt(pow(srcpoint[6] - srcpoint[4], 2) + pow(srcpoint[7] - srcpoint[5], 2)))
        dh = max(math.sqrt(pow(srcpoint[0] - srcpoint[6], 2) + pow(srcpoint[1] - srcpoint[7], 2)),
                math.sqrt(pow(srcpoint[2] - srcpoint[4], 2) + pow(srcpoint[3] - srcpoint[5], 2)))

        if ratio_w is None:
            ratio_w = random.uniform(0, 1.0)  # 0.3
        if ratio_h is None:
            ratio_h = random.uniform(0, 0.2)  # 0.05

        srcpoint[0] = max(0, srcpoint[0] - dh * ratio_w)
        srcpoint[1] = max(0, srcpoint[1] - dh * ratio_h)

        srcpoint[2] = min(w - 1, srcpoint[2] + dh * ratio_w)
        srcpoint[3] = max(0, srcpoint[3] - dh * ratio_h)

        srcpoint[4] = min(w - 1, srcpoint[4] + dh * ratio_w)
        srcpoint[5] = min(h - 1, srcpoint[5] + dh * ratio_h)

        srcpoint[6] = max(0, srcpoint[6] - dh * ratio_w)
        srcpoint[7] = min(h - 1, srcpoint[7] + dh * ratio_h)

        dw = max(math.sqrt(pow(srcpoint[2] - srcpoint[0], 2) + pow(srcpoint[1] - srcpoint[3], 2)) + 1,
                math.sqrt(pow(srcpoint[6] - srcpoint[4] + 1, 2) + pow(srcpoint[7] - srcpoint[5] + 1, 2)) + 1)
        dh = max(math.sqrt(pow(srcpoint[0] - srcpoint[6], 2) + pow(srcpoint[1] - srcpoint[7], 2)) + 1,
                math.sqrt(pow(srcpoint[2] - srcpoint[4], 2) + pow(srcpoint[3] - srcpoint[5], 2)) + 1)

        pts1 = np.float32([[srcpoint[0], srcpoint[1]], [srcpoint[2], srcpoint[3]], [srcpoint[4], srcpoint[5]],
                        [srcpoint[6], srcpoint[7]]])
        pts2 = np.float32([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]])

        # AffineMatrix = cv2.getAffineTransform(pts1,pts2)
        AffineMatrix = cv2.getPerspectiveTransform(pts1, pts2)
        return AffineMatrix, [dh, dw]

    def crop_affine(self, image, ratio_w=None, ratio_h=None):
        """
        透视变换
        """
        h, w, _ = image.shape
        ptList = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        AffineMatrix, dessize = self.get_affine_matrix(
            [ptList[0][0], ptList[0][1], ptList[1][0], ptList[1][1], ptList[2][0], ptList[2][1], ptList[3][0],
            ptList[3][1]], h, w, ratio_w, ratio_h)
        img_perspective = cv2.warpPerspective(image, AffineMatrix, (int(dessize[1]), int(dessize[0])))
        return img_perspective


if __name__ == '__main__':
    # 在原图上进行裁剪   行驶证
    # txt_file = "/5T/xingshizheng/train_1w_new.txt"
    # output_folder = "/5T/xingshizheng/train_1w_augment/"
    # output_txt = "/5T/xingshizheng/train_1w_augment.txt"
    # crop_imgs(txt_file, output_folder, output_txt)

    # 港澳台通行证
    times = 10
    txt_file = '/5T/hmt_pass_data/train.txt'
    output_folder = '/5T/hmt_pass_data/train_augment/'
    output_txt = '/5T/hmt_pass_data/train_augment.txt'
    crop_imgs(txt_file, output_folder, output_txt, times)

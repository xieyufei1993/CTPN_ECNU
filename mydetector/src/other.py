import sys
import os
reload(sys)
sys.setdefaultencoding('utf-8')
import cv2
import caffe
import numpy as np
from matplotlib import cm
import json
from math import *
import qiniu
import time

def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) +
                    height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) +
                   width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(
        img, matRotation, (widthNew, heightNew), borderValue=(
            255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(
        matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(
        matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    imgOut = imgRotation[int(pt1[1]):int(pt3[1]), int(pt1[0]):int(pt3[0])]
    height, width = imgOut.shape[:2]
    return imgOut

def filter_img(img):
    if img.shape[0] > img.shape[1] * 1.5:
        img = np.rot90(img)
    scale = float(img.shape[0]) / 32.0
    if scale == 0:
        return img
    w = int(float(img.shape[1]) / scale)
    if w > 280:
        w = 280
        img = cv2.resize(img, (w, 32), interpolation=cv2.INTER_LINEAR)
    else:
        img = cv2.resize(img, (w, 32))
        expand = 280 - w

        r = img[:, img.shape[1] - 1, 0].mean()
        g = img[:, img.shape[1] - 1, 1].mean()
        b = img[:, img.shape[1] - 1, 2].mean()

        img = cv2.copyMakeBorder(
            img,
            0,
            0,
            0,
            expand,
            cv2.BORDER_CONSTANT,
            value=(
                r,
                g,
                b))

    return img

def prepare_img(im, mean):
    """
        transform img into caffe's input img.
    """
    im_data = np.transpose(im - mean, (2, 0, 1))
    return im_data


def draw_boxes(
    im,
    bboxes,
    f):
    """
        boxes: bounding boxes
    """
    text_recs = np.zeros((len(bboxes), 8), np.int)
    myret = []
    for box in bboxes:
        single_temp = []
        b1 = box[6] - box[7] / 2
        b2 = box[6] + box[7] / 2
        x1 = box[0]
        y1 = box[5] * box[0] + b1
        x2 = box[2]
        y2 = box[5] * box[2] + b1
        x3 = box[0]
        y3 = box[5] * box[0] + b2
        x4 = box[2]
        y4 = box[5] * box[2] + b2
        disX = x2 - x1
        disY = y2 - y1
        width = np.sqrt(disX * disX + disY * disY)
        fTmp0 = y3 - y1
        fTmp1 = fTmp0 * disY / width
        x = np.fabs(fTmp1 * disX / width)
        y = np.fabs(fTmp1 * disY / width)
        if box[5] < 0:
            x1 -= x
            y1 += y
            x4 += x
            y4 -= y
        else:
            x2 += x
            y2 += y
            x3 -= x
            y3 -= y
        cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
        cv2.line(im, (int(x1), int(y1)), (int(x3), int(y3)),(255,0,0), 2)
        cv2.line(im, (int(x4), int(y4)), (int(x2), int(y2)), (255,0,0), 2)
        cv2.line(im, (int(x3), int(y3)), (int(x4), int(y4)), (255,0,0), 2)
        single_temp.append(int(x1))
        single_temp.append(int(y1))
        single_temp.append(int(x2))
        single_temp.append(int(y2))
        single_temp.append(int(x4))
        single_temp.append(int(y4))
        single_temp.append(int(x3))
        single_temp.append(int(y3))
        myret.append(single_temp)
    #cv2.imwrite("/workspace/0717.JPG",im)
    im = cv2.transpose(im)
    im = cv2.flip(im, 1)
    cv2.imwrite("./0717_rotate_2.jpg", im)
    q = qiniu.Auth("IkhOeL5B6C-vduI1GziOd3dUz2OeOvLfn4Ns34cJ", "mP4A9vbCMNqLAlJ69ylL1MU8t7dGUJAuFrf7QeCP")
    key = str(time.time())+".jpg"
    data = './0717_rotate_2.jpg'
    bucket_name = "xyf0717"
    token = q.upload_token(bucket_name)
    ret, info = qiniu.put_file(token, key, data)
    if ret is not None:
        print('All is OK')
    else:
        print(info)  # error message in info
    os.remove('./0717_rotate_2.jpg')
    uri_result = "http://pbzqwnu26.bkt.clouddn.com/"+key
    return uri_result


def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[:, 0::2]=threshold(boxes[:, 0::2], 0, im_shape[1] - 1)
    boxes[:, 1::2]=threshold(boxes[:, 1::2], 0, im_shape[0] - 1)
    return boxes


def normalize(data):
    if data.shape[0] == 0:
        return data
    max_=data.max()
    min_=data.min()
    return (data - min_) / (max_ - min_) if max_ - min_ != 0 else data - min_


def resize_im(im, scale, max_scale=None):
    f=float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f=float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, (0, 0), fx=f, fy=f), f


class Graph:
    def __init__(self, graph):
        self.graph=graph

    def sub_graphs_connected(self):
        sub_graphs=[]
        for index in xrange(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v=index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v=np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs


class CaffeModel:
    def __init__(self, net_def_file, model_file):
        self.net_def_file=net_def_file
        self.net=caffe.Net(net_def_file, model_file, caffe.TEST)

    def blob(self, key):
        return self.net.blobs[key].data.copy()

    def forward(self, input_data):
        return self.forward2({"data": input_data[np.newaxis, :]})

    def forward2(self, input_data):
        for k, v in input_data.items():
            self.net.blobs[k].reshape(*v.shape)
            self.net.blobs[k].data[...]=v
        return self.net.forward()

    def net_def_file(self):
        return self.net_def_file

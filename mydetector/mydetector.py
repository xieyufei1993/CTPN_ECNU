#coding=utf-8
import os
import base64
import cv2
import numpy as np
from collections import namedtuple
#from bottle import route, run
import urllib
from urlparse import urlparse
from src.detectors import TextProposalDetector, TextDetector
import caffe
from src.other import draw_boxes, resize_im, CaffeModel
from cfg import Config as cfg
from math import *

class ImageDetector(object):
    def __init__(self):
        self.model_dir = "models"
        self.caffemodel = "ctpn_trained_model.caffemodel"
        self.prototxt = "deloy.prototxt"

    def get_image(self,uri):
        resp = urllib.urlopen(uri)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        dst_im = cv2.flip(image, 1)
        dst_im = cv2.transpose(dst_im)
        return dst_im

    def load_model(self,prototxt_dir=None,caffemodel_dir=None):
        if cfg.PLATFORM == "GPU":
            caffe.set_mode_gpu()
            caffe.set_device(cfg.TEST_GPU_ID)
        else:
            caffe.set_mode_cpu()
        if prototxt_dir != None:
            self.prototxt = prototxt_dir
        if caffemodel_dir != None:
            self.caffemodel = caffemodel_dir
        detector_model = TextProposalDetector(CaffeModel(self.prototxt, self.caffemodel))
        return detector_model

    def detect(self,uri,detector_model):
        img = self.get_image(uri)
        text_detector = TextDetector(detector_model)
        im, f=resize_im(img, cfg.SCALE, cfg.MAX_SCALE)
        tmp = im.copy()
        text_lines=text_detector.detect(im)
        myret = draw_boxes(tmp, text_lines, f)
        return myret
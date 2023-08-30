import os
import sys
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data.config import cfg_mnet as cfg
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
# from models_bk.retinaface import RetinaFace
from utils.box_utils import decode
from alfred.utils.log import logger as logging
from alfred.dl.torch.common import device
import glob
import time


force_cpu = False
if force_cpu:
    device = torch.device('cpu')


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path):
    print('Loading pretrained model from {}'.format(pretrained_path))
    pretrained_dict = torch.load(pretrained_path,
                                 map_location=lambda storage, loc: storage if force_cpu else storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class RetinafaceDetector:

    def __init__(self):
        torch.set_grad_enabled(False)
        # net and model
        cfg['leaky'] = True
        self.net = RetinaFace(cfg=cfg, phase='test')
        # net = RetinaFace(phase='test')
        self.net = load_model(self.net, os.path.join(
            os.path.dirname(__file__), 'weights/mobilenet0.25_epoch_245.pth'))
        self.net.eval()
        print('Finished loading model!')
        cudnn.benchmark = True
        self.net = self.net.to(device)
        logging.info('model convert into device: {}'.format(device))

    def detect(self, img):
        target_size = 1280
        max_size = 960
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize,
                             fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        # print(img)
        scale = scale.to(device)

        # print('input tensor shape: {}'.format(img.size()))
        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.5)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:1000, :]
        return dets

    def draw_boxes(self, img_raw, dets):
        for b in dets:
            if b[4] < 0.1:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]),
                          (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        return img_raw

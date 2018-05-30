# ------------------------------------------
# SSH: Single Stage Headless Face Detector
# Demo
# by Mahyar Najibi
# Add mtcnn aligner 
# Add face recognition module
# by Dongfeng Yu and Yun Liu
# ------------------------------------------

from __future__ import print_function
from __future__ import division
from SSH.wrappertest import detect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
from utils.get_config import cfg_from_file, cfg, cfg_print
import caffe
import cv2
import numpy as np
import os
import time
import xml.etree.ElementTree as ET

def inrec(a,b,t1,t2,t3,t4):
    return (t1<=a<=t2) and (t3<=b<=t4)

def inbox(dxmin, dxmax, dymin, dymax, txmin, txmax, tymin, tymax):
    if inrec(dxmin, dxmax, txmin, txmax, tymin, tymax):
        return True
    elif inrec(dxmin, dymax, txmin, txmax, tymin, tymax):
        return True
    elif inrec(dymin, dymax, txmin, txmax, tymin, tymax):
        return True
    elif inrec(dymin, dxmax, txmin, txmax, tymin, tymax):
        return True
    else:
        return False

def parser():
    parser = ArgumentParser('SSH Demo!')
    parser.add_argument('--im',dest='imdir',help='Directory to the image',
                        default='/SSH/detectiontest/evalimg/',type=str)
    parser.add_argument('--trudir',dest='trudir',help='Directory to the ground truth',
                        default='/SSH/detectiontest/groundtruth/',type=str)
    parser.add_argument('--gpu',dest='gpu_id',help='The GPU ide to be used for detection',
                        default=0,type=int)
    parser.add_argument('--proto',dest='prototxt',help='SSH caffe test prototxt',
                        default='SSH/models/test_ssh.prototxt',type=str)
    parser.add_argument('--model',dest='model',help='SSH trained caffemodel',
                        default='data/SSH_models/SSH.caffemodel',type=str)
    parser.add_argument('--cfg',dest='cfg',help='Config file to overwrite the default configs',
                        default='SSH/configs/yliu_wider.yml',type=str)
    parser.add_argument('--iou',dest='iou',help='IOU parameter',
                        default=0.3,type=float)
    parser.add_argument('--opath',dest='out_path',default='data/demo',type=str)
    parser.add_argument('--conf',dest='conf',help='Confidence to get face',default=0.75,type=float)
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parser()
    # Load the external config
    if args.cfg is not None:
        cfg_from_file(args.cfg)
    confidence = args.conf
    # Loading the network
    cfg.GPU_ID = args.gpu_id
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    assert os.path.isfile(args.prototxt),'Please provide a valid path for the prototxt!'
    assert os.path.isfile(args.model),'Please provide a valid path for the caffemodel!'

    net = caffe.Net(args.prototxt, args.model, caffe.TEST)
    net.name = 'SSH'
    pyramid = False 
    assert os.path.exists(args.imdir),'Please provide a valid path for image dir'
    assert os.path.exists(args.trudir),'Please provide a valid path for ground truth dir'
    start = time.time()
    filenum = 0
    #processing start
    for imgfile in os.listdir(args.imdir):
        # Face cut list
        filenum += 1
        tempstart = time.time()
        images = []
        groundtruth = []
        im = cv2.imread(os.path.join(args.imdir,imgfile))
	print('Detecting file {0}'.format(imgfile))
        cls_dets,_ = detect(net,im,visualization_folder=args.out_path,visualize=False,pyramid=pyramid)
        tempperiod = time.time()-tempstart
        print('Detect complete in {0} s'.format(tempperiod))
    print('All detection complete in {0} s Num of files: {1}'.format(time.time()-start, filenum))

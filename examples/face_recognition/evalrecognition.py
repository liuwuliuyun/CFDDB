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
from SSH.test import detect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
from ArcFace.extractor import extractor as Extractor
from ArcFace.aligner import aligner as Aligner
from utils.get_config import cfg_from_file, cfg, cfg_print
import caffe
import cv2
import numpy as np
import os
import time
import xml.etree.ElementTree as ET
import tensorflow as tf

def parser():
    parser = ArgumentParser('SSH Demo!')
    parser.add_argument('--im',dest='imdir',help='Directory to the image',
                        default='/SSH/test/detectiontest/evalimg/',type=str)
    parser.add_argument('--trudir',dest='trudir',help='Directory to the ground truth',
                        default='/SSH/test/detectiontest/groundtruth/',type=str)
    parser.add_argument('--gpu',dest='gpu_id',help='The GPU ide to be used for detection',
                        default=0,type=int)
    parser.add_argument('--proto',dest='prototxt',help='SSH caffe test prototxt',
                        default='/SSH/SSH/models/test_ssh.prototxt',type=str)
    parser.add_argument('--model',dest='model',help='SSH trained caffemodel',
                        default='/SSH/data/SSH_models/SSH.caffemodel',type=str)
    parser.add_argument('--out_path',dest='out_path',help='Output path for figure that not detected',
                        default='data/demo',type=str)
    parser.add_argument('--cfg',dest='cfg',help='Config file to overwrite the default configs',
                        default='/SSH/SSH/configs/yliu_wider.yml',type=str)
    parser.add_argument('--iou',dest='iou',help='IOU parameter',
                        default=0.3,type=float)
    parser.add_argument('--extractor_batch_size',dest='extractor_batch_size',help='Extractor batch size', 
                        default=16, type=int)
    parser.add_argument('--aligner_batch_size',dest='aligner_batch_size',help='Anigner batch size',
                        default=16, type=int)
    parser.add_argument('--devices',dest='devices',help='The GPU device to be used for recognition',
                        default='/gpu:0')
    return parser.parse_args()

def resize_face(im, a, b, c, d, max_height, max_width, enlarge=2.0):
    temp_h = (d - b)/2.0
    c_y = (b + d)/2.0
    temp_w = (c - a)/2.0
    c_x = (a + c)/2.0
    temp_h*=enlarge
    temp_w*=enlarge
    n_a = max(0,int(c_x-temp_w))
    n_b = max(0,int(c_y-temp_h))
    n_c = min(max_width,int(c_x+temp_w))
    n_d = min(max_height,int(c_y+temp_h))
    face_cut = im[n_b:n_d, n_a:n_c]
    temp_length = max(n_d-n_b, n_c-n_a)
    blank_img = np.zeros((temp_length,temp_length,3),np.uint8)
    blank_img[:n_d-n_b,:n_c-n_a,:]=face_cut
    face_cut = cv2.resize(blank_img, (150, 150))
    return face_cut

def batch_process(f, x, s):
    results = []
    for i in range(0,len(x), s):
        x_ = x[i: i + s]
        if len(x_) != s:
            x_ += [x_[0]] * (s - len(x_))
        y_ = f(x_)
        for j in y_:
            if len(results) < len(x):
                results.append(j)
        print(len(results), 'done')
    return results


def do_batch(f, x):
    keys = [i[1] for i in x]
    imgs = [i[0] for i in x]
    imgs = np.stack(imgs, axis=0)
    imgs = f(imgs)
    return list(zip(imgs, keys))



if __name__ == "__main__":
    # Parse arguments
    args = parser()
    args.devices = args.devices.split(',')
    # Load the external config
    if args.cfg is not None:
        cfg_from_file(args.cfg)
    confidence = 0.75
    # Loading the network
    cfg.GPU_ID = args.gpu_id
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    assert os.path.isfile(args.prototxt),'Please provide a valid path for the prototxt!'
    assert os.path.isfile(args.model),'Please provide a valid path for the caffemodel!'

    net = caffe.Net(args.prototxt, args.model, caffe.TEST)
    net.name = 'SSH'
    pyramid = False 
    config = tf.ConfigProto() 
    config.allow_soft_placement = False
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    embedding = []
    emdir = '/SSH/test/recognitiontest/features'
    for embedfile in os.listdir(emdir):
        embedding.append((embedfile.split('.')[0],np.load(os.path.join(emdir,embedfile))))
    aligner = Aligner(session, args.devices, args.aligner_batch_size)
    extractor = Extractor(session, args.devices, args.extractor_batch_size)
    assert os.path.exists(args.imdir),'Please provide a valid path for image dir'
    assert os.path.exists(args.trudir),'Please provide a valid path for ground truth dir'
    alltrueface = 0
    alldetectface = 0
    alldetectfaceright = 0
    allrecognizefaceright = 0
    #processing start
    for imgfile in os.listdir(args.imdir):
        # Face cut list
        images = []
        groundtruth = []
        im = cv2.imread(os.path.join(args.imdir,imgfile))
	print('Detecting File {0}'.format(imgfile))
        cls_dets,_ = detect(net,im,visualization_folder=args.out_path,visualize=False,pyramid=pyramid)
        xmlfile = os.path.join(args.trudir,(imgfile.split('.')[0]+'.xml'))
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        for obj in root.findall('object'):
            objname = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            xmax = int(bbox.find('xmax').text)
            ymin = int(bbox.find('ymin').text)
            ymax = int(bbox.find('ymax').text)
            groundtruth.append((objname,[xmin,xmax,ymin,ymax]))
        

        truefacenum = len(groundtruth)
        detectright = 0
        detectnum = 0
        recognizeright = 0
        images=[]
        for det in cls_dets:
            if det[4]>confidence:
                #images.append(('0',[det[0],det[2],det[1],det[3]]))
                detectnum += 1
                dxmin = det[0]
                dxmax = det[2]
                dymin = det[1]
                dymax = det[3]
                tempiou = args.iou
                findsuccess = False
                people = '0000'
                for truth in groundtruth:
                    txmin = truth[1][0]
                    txmax = truth[1][1]
                    tymin = truth[1][2]
                    tymax = truth[1][3]
                    #if inbox(dxmin, dxmax, dymin, dymax, txmin, txmax, tymin, tymax):
                    ixmin = max(dxmin,txmin)
                    iymin = max(dymin,tymin)
                    ixmax = min(dxmax,txmax)
                    iymax = min(dymax,tymax)
                    targetiou = ((ixmax-ixmin)*(iymax-iymin))/(((dxmax-dxmin)*(dymax-dymin))-(ixmax-ixmin)*(iymax-iymin)+((txmax-txmin)*(tymax-tymin)))
                    if targetiou > tempiou:
                        tempiou = targetiou
                        findsuccess = True
                        people = truth[0]            
                if findsuccess:
                    detectright += 1
                    facetempcut = resize_face(im,dxmin,dymin,dxmax,dymax,im.shape[0],im.shape[1])
                    images.append((facetempcut,people))
        images = batch_process(lambda x:do_batch(aligner.align, x), images, args.aligner_batch_size)
        #[TODO]filler image that has not be idenfified
        images = [(i[0][0],i[1]) for i in images]
        images = batch_process(lambda x:do_batch(extractor.extract, x), images, args.extractor_batch_size)
        for i in images:
            trueid = i[1]
            recid = '00'
            similarity = 0
            for j in embedding:
                if np.dot(i[0],j[1])>similarity:
                    similarity=np.dot(i[0],j[1])
                    recid = j[0]
            if cmp(trueid,recid):
                recognizeright+=1
                
        print('Detection Complete.\n {0} faces detected.\n {1} faces is right.\n {2} faces is wrong.\n {3} faces actually exsits\n {4} faces recognize right'.format(detectnum,detectright,detectnum-detectright,truefacenum,recognizeright)) 
        alltrueface+=truefacenum
        alldetectface+=detectnum
        alldetectfaceright+=detectright
        allrecognizefaceright+=recognizeright
    print('Detection Complete.\n {0} faces detected.\n {1} faces is right.\n {2} faces is wrong.\n {3} faces actually exsits\n {4} faces recognize right'.format(alldetectface,alldetectfaceright,alldetectface-alldetectfaceright,alltrueface,allrecognizefaceright))

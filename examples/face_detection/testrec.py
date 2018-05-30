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
from argparse import ArgumentParser
import os
import cv2
import numpy as np
import os
import time
import tensorflow as tf

from ArcFace.extractor import extractor as Extractor
from ArcFace.aligner import aligner as Aligner
def parser():
    parser = ArgumentParser('SSH Demo!')
    parser.add_argument('--im',dest='imdir',help='Directory to the image',
                        default='/SSH/data/imtruthm/',type=str)
    parser.add_argument('--extractor_batch_size',dest='extractor_batch_size',help='Extractor batch size', 
                        default=64, type=int)
    parser.add_argument('--aligner_batch_size',dest='aligner_batch_size',help='Anigner batch size',
                        default=64, type=int)
    parser.add_argument('--devices',dest='devices',help='The GPU device to be used for recognition',
                        default='/gpu:0')
    return parser.parse_args()


def resize_face(im, width, height):
    temp_length = max(width, height)
    blank_img = np.zeros((temp_length,temp_length,3),np.uint8)
    blank_img[:height,:width,:]=im
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
    emdir = '/SSH/test/recognitiontest/features/'
    embedding = []
    config = tf.ConfigProto() 
    config.allow_soft_placement = False
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    for embedfile in os.listdir(emdir):
        embedding.append((embedfile.split('.')[0],np.load(os.path.join(emdir,embedfile))))
    aligner = Aligner(session, args.devices, args.aligner_batch_size)
    extractor = Extractor(session, args.devices, args.extractor_batch_size)
    assert os.path.exists(args.imdir),'Please provide a valid path for image dir'
    images = []
    for imsubdir in os.listdir(args.imdir):
        path = os.path.join(args.imdir,imsubdir)
        for i in os.listdir(path):
            im = cv2.imread(os.path.join(path,i))
            images.append((resize_face(im,im.shape[1],im.shape[0]),imsubdir))
    recognizeright = 0
    images = batch_process(lambda x:do_batch(aligner.align, x), images, args.aligner_batch_size)
    images = [(i[0][0],i[1]) for i in images]
    images = batch_process(lambda x:do_batch(extractor.extract, x), images, args.extractor_batch_size)
    detable = 0
    for i in images:
        trueid = i[1]
        print
        recid = '00'
        cannotde = 0
        similarity = 0
        for j in embedding:
            if np.dot(i[0],j[1])>similarity:
                similarity=np.dot(i[0],j[1])
                recid = j[0]
        if similarity<0.9:
            cannotde+=1
        elif cmp(trueid,recid):
            detable+=1
        else:
            detable+=1
            recognizeright+=1
            #print('Mis Take {0} as {1}'.format(trueid,recid))
            
    print('{0} total faces, {1} faces recognize right {2} dable'.format(len(images),recognizeright,detable)) 

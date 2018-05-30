from __future__ import print_function
import os
import cv2
import tensorflow as tf
import numpy as np
import demo_0
from argparse import ArgumentParser
from sklearn.preprocessing import normalize

def resize_face(im, width, height):
    temp_length = max(width, height)
    blank_img = np.zeros((temp_length,temp_length,3),np.uint8)
    blank_img[:height,:width,:]=im
    face_cut = cv2.resize(blank_img, (150, 150))
    return face_cut

def parser():
    parser = ArgumentParser('feature extractor from cut images')
    parser.add_argument('--im',dest='ipath',help='Directory to the image',
                        default='/SSH/data/evaldatabase/',type=str)
    parser.add_argument('--out_path',dest='out_path',help='Output path for',
                        default='/SSH/data/evalfeature/',type=str)
    parser.add_argument('--devices',dest='devices',help='The GPU device to be used for recognition',
                        default='/gpu:0')
    parser.add_argument('--extractor_batch_size',dest='extractor_batch_size',help='Extractor batch size', 
                        default=8, type=int)
    parser.add_argument('--aligner_batch_size',dest='aligner_batch_size',help='Anigner batch size',
                        default=8, type=int)
    return parser.parse_args()

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
    


    config = tf.ConfigProto() 
    config.allow_soft_placement = False
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    aligner = demo_0.aligner(session, args.devices, args.aligner_batch_size)
    extractor = demo_0.extractor(session, args.devices, args.extractor_batch_size)
    
    assert os.path.exists(args.ipath),'Error: image file directory does not exist!'
    

    imgdirlist = os.listdir(args.ipath)
    for imdir in imgdirlist:
        images = []
        imgdir_path = os.path.join(args.ipath,imdir)
        imfilelist = os.listdir(imgdir_path)
        for imfile in imfilelist:
            if imfile.endswith('.jpg'):
                im = cv2.imread(os.path.join(imgdir_path,imfile))
                im = resize_face(im,im.shape[1],im.shape[0])
                images.append((im,imfile))

        images = batch_process(lambda x:do_batch(aligner.align, x), images, args.aligner_batch_size)
        #[TODO]filler image that has not be idenfified
        images = [(i[0][0],i[1]) for i in images]
        images = batch_process(lambda x:do_batch(extractor.extract, x), images, args.extractor_batch_size)
        #calculate center
        imagecenter = np.mean([i[0] for i in images],axis=0)
        imagecenter = normalize(imagecenter.reshape(1,-1),axis=1).reshape(256,)
        #find the best sample
        maxval = 0
        keypicname = ''
        keyfeature = 0
        for i in images:
            if np.dot(imagecenter,i[0])>maxval:
                keypicname = i[1]
                maxval = np.dot(imagecenter,i[0])
                keyfeature = i[0]
        
        print('Key picture find : {0}, max confidence is {1}'.format(keypicname,maxval))
        path = os.path.join(args.out_path,((keypicname.split('_')[1]).split('.')[0]))
        np.save(path, keyfeature)   

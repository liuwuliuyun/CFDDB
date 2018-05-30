# ------------------------------------------
# HAAR evaluation
# Author: Yun Liu
# ------------------------------------------

from __future__ import print_function
from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
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
                        default='/SSH/test/detectiontest/evalimg/',type=str)
    parser.add_argument('--trudir',dest='trudir',help='Directory to the ground truth',
                        default='/SSH/test/detectiontest/groundtruth/',type=str)
    parser.add_argument('--out_path',dest='out_path',help='Output path for figure that not detected',
                        default='/SSH/test/haartest/',type=str)
    parser.add_argument('--iou',dest='iou',help='IOU parameter',
                        default=0.3,type=float)
    parser.add_argument('--conf',dest='conf',help='Confidence to get face',default=0.75,type=float)
    parser.add_argument('--msize',dest='msize',default=2,type=int)
    parser.add_argument('--scale',dest='scale',default=1.1,type=float)
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parser()
    # Load the external config
    confidence = args.conf
    alltrueface = 0
    alldetectface = 0
    alldetectfaceright = 0
    face_cascade = cv2.CascadeClassifier('/SSH/test/haarcascade_frontalface_alt.xml')
    start = time.time()
    #processing start
    for imgfile in os.listdir(args.imdir):
        # Face cut list
        images = []
        groundtruth = []
        im = cv2.imread(os.path.join(args.imdir,imgfile))
	print('Detecting File {0}'.format(imgfile))
        gray =cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,args.scale,args.msize)
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
        #visulize detection results
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        if len(groundtruth) != 0:

            for i in range(len(groundtruth)):
                bbox = groundtruth[i][1]
                ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[2]),
                                  bbox[1] - bbox[0],
                                  bbox[3] - bbox[2], fill=False,
                                  edgecolor=(0, 1, 0), linewidth=3))
        
        if len(faces) != 0:
            for i in range(len(faces)):
                bbox = faces[i]
                ax.add_patch(
                    plt.Rectangle((bbox[0],bbox[1]),
                                  bbox[2],
                                  bbox[3], fill=False,
                                  edgecolor=(1, 0, 0), linewidth=2))
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
        plt.savefig(os.path.join(args.out_path,'result_'+imgfile),bbox_inches='tight')
        print('Saving file {0} to {1}'.format('result_'+imgfile,args.out_path))
        #visualize complete 
        truefacenum = len(groundtruth)
        detectright = 0
        detectnum = 0
        for det in faces:
                #images.append(('0',[det[0],det[2],det[1],det[3]]))
                detectnum += 1
                dxmin = det[0]
                dxmax = det[2]+det[0]
                dymin = det[1]
                dymax = det[3]+det[1]
                tempiou = args.iou
                findsuccess = False
                for truth in groundtruth:
                    txmin = truth[1][0]
                    txmax = truth[1][1]
                    tymin = truth[1][2]
                    tymax = truth[1][3]
                    ixmin = max(dxmin,txmin)
                    iymin = max(dymin,tymin)
                    ixmax = min(dxmax,txmax)
                    iymax = min(dymax,tymax)
                    if(xmin>xmax)or(ymin>ymax):
                        continue
                    targetiou = ((ixmax-ixmin)*(iymax-iymin))/(((dxmax-dxmin)*(dymax-dymin))-(ixmax-ixmin)*(iymax-iymin)+((txmax-txmin)*(tymax-tymin)))
                    if targetiou > tempiou:
                        tempiou = targetiou
                        findsuccess = True            
                if findsuccess:
                    detectright += 1
                            
        print('Detection Complete.\n {0} faces detected.\n {1} faces is right.\n {2} faces is wrong.\n {3} faces actually exsits'.format(detectnum,detectright,detectnum-detectright,truefacenum)) 
        
        alltrueface+=truefacenum
        alldetectface+=detectnum
        alldetectfaceright+=detectright
    end = time.time()
    print('Detection Complete.\n {0} faces detected.\n {1} faces is right.\n {2} faces is wrong.\n {3} faces actually exsits\n{4} s used'.format(alldetectface,alldetectfaceright,alldetectface-alldetectfaceright,alltrueface,end-start))

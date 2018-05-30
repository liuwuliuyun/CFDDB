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
                        default='/SSH/test/detectiontest/visual/',type=str)
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parser()
    # Load the external config
    for imgfile in os.listdir(args.imdir):
        # Face cut list
        groundtruth = []
        im = cv2.imread(os.path.join(args.imdir,imgfile))
	print('Detecting File {0}'.format(imgfile))
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
        
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
        plt.savefig(os.path.join(args.out_path,'result_'+imgfile),bbox_inches='tight')
        print('Saving file {0} to {1}'.format('result_'+imgfile,args.out_path))
        #visualize complete

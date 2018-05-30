from __future__ import print_function
import xml.etree.ElementTree as ET
import os
from argparse import ArgumentParser
import cv2
import numpy as np

def resize_face(im, a, b, c, d, max_height, max_width, enlarge=1.5):
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
    face_cut = cv2.resize(blank_img, (500, 500))
    return face_cut

def parser():
	parser = ArgumentParser('xml processing')
	parser.add_argument('-dir',dest='xpath',help='xml directory path',default='/SSH/data/xml/',type=str)
	parser.add_argument('-img',dest='ipath',help='imgs directory path',default='/SSH/data/evalimg/',type=str)
	parser.add_argument('-dbdir',dest='dpath',help='database directory path',default='/SSH/data/imtruth/',type=str)
	return parser.parse_args()

if __name__=="__main__":
	args = parser()
	#imglist = os.listdir(args.ipath)
	xmllist = os.listdir(args.xpath)
	#print(xmllist)
	
	for xmlfile in xmllist:
		if xmlfile.endswith('.xml'):
			xfpath = os.path.join(args.xpath,xmlfile)
			tree = ET.parse(xfpath)
			root = tree.getroot()
			imgname = root.find('filename').text
			print('Opening img {0}'.format(imgname))
			if not imgname.endswith('.jpg'):
				imgname = imgname+'.jpg'
			imgnum = imgname.split('.')
			imgnum = imgnum[0]

			im = cv2.imread(os.path.join(args.ipath,imgname))

			for obj in root.findall('object'):
				objname = obj.find('name').text
				bbox = obj.find('bndbox')
				xmin = int(bbox.find('xmin').text)
				xmax = int(bbox.find('xmax').text)
				ymin = int(bbox.find('ymin').text)
				ymax = int(bbox.find('ymax').text)

				imcut = resize_face(im, xmin, ymin, xmax, ymax, im.shape[0], im.shape[1])
				
				savedir = os.path.join(args.dpath,objname)

				if not os.path.exists(savedir):
					os.makedirs(savedir)
				
				saveimgname = imgnum + '_' + objname +'.jpg'
				saveimgpath = os.path.join(savedir,saveimgname)
				print('Saving {0} to {1}'.format(saveimgname,saveimgpath))
				cv2.imwrite(saveimgpath,imcut)

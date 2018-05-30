from __future__ import print_function
import os


if __name__ == "__main__":

    datadir = '/SSH/data/imtruthm'
    for imname in os.listdir(datadir):
        
        for imfilename in os.listdir(os.path.join(datadir,imname)):
            oldname = (imfilename.split('.')[0]).split('_')[1]
            if not oldname == imname:
                newname = (imfilename.split('.')[0]).split('_')[0]+'_'+imname+'.jpg'
                path = os.path.join(datadir,imname)
                os.rename(os.path.join(path,imfilename),os.path.join(path,newname))
                print('Rename file {0} to {1}'.format(imfilename, newname))

import numpy as np
import cv2
import os
from xml.dom.minidom import parse
import xml.dom.minidom
import pdb

class XgtfHelper:
    def __init__(self,fname):
        self.filename = fname
        self.tree = xml.dom.minidom.parse(fname)
        
    def getPeopleBoxes(self):
        coll = self.tree.documentElement #get root
        boxes = {}
        people = filter(lambda (x):x.getAttribute('name')=='PERSON',
                        coll.getElementsByTagName('object'))
        for peeps in people:
            bboxes = peeps.getElementsByTagName('data:bbox')
            id = peeps.getAttribute('id')

            for bbox in bboxes:
                span = map(int,bbox.getAttribute('framespan').split(':'))
                obj ={}
                obj['height'] = int(bbox.getAttribute('height'))
                obj['width']  = int(bbox.getAttribute('width'))
                obj['x'] =  abs(int(bbox.getAttribute('x')))
                obj['y'] = abs(int(bbox.getAttribute('y')))
                for x in xrange(span[0],span[1]+1):
                    if not id in boxes:
                        boxes[id] = {}
                    boxes[id][x] = obj
                
        return boxes
            

class VideoRunner:
    def __init__(self,fname,labels=None):
        self.filename = fname

        self.vcap = cv2.VideoCapture(fname)
        self.frameno = 0
        if not self.vcap.isOpened():
            self.vcap.open(fname)
        if labels:
            self.labelfilename = labels
            labs = XgtfHelper(labels)
            self.labelmap = labs.getPeopleBoxes()
            
    def getNextImage(self):
        if self.vcap.isOpened():
            #pdb.set_trace()
            code = self.vcap.grab()
            img = None
            if code:
                retval,img = self.vcap.retrieve()
                if retval:
                    self.frameno += 1
                    return img
            else:
                self.vcap.release()

    def getFrameNo(self):
        return self.frameno
    
    def release(self):
        if self.vcap.isOpened():
            self.vcap.release()
    def open(self):
        if not self.vcap.isOpened():
            self.frameno = 0
            self.vcap.open(self.fname)
    def get_class(self,index): #needs a little more thought to be general
        if self.labelmap:
            for key in self.labelmap:
                if index in self.labelmap[key]:
                    return 1
            return 0

    def decomp(self,stepsize):
        ix = 0
        img = self.getNextImage()
        filename = self.filename.split('/')[-1]
        dirname = os.path.splitext(filename)[0]
        f = None
        if os.path.exists(dirname):
            print 'directory exists... terminating'
            return
        else:
            os.makedirs(dirname)
        if self.labelfilename:
            f = open(dirname +'/labels.txt','w')
        while not img is None:
            if ix % stepsize == 0:
                fname = str(ix) + '.jpg' 
                cv2.imwrite(dirname+'/'+fname,img)
                if f:
                    frames = self.subs(img,ix)
                    label = 1 if len(frames) > 0 else 0
                    f.write('{} {}\n'.format(fname,label))
                        
            img = self.getNextImage()
            print ix
            ix += 1
        if f:
            f.close()
    def getRect(self,img,key,ix):
        obj = self.labelmap[key][ix]
        pt1 = obj['x'],obj['y']
        pt2 = obj['x'] + obj['width'], obj['y']+obj['height']
        return pt1,pt2,img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    def subs(self,img,ix):
        frames = []
        negframes = []
        points = []
        if self.labelmap:
            for key in self.labelmap:
                if ix in self.labelmap[key]:
                    p1,p2,frame = self.getRect(img,key,ix)
                    points.append((p1,p2))
                    frames.append((frame,1))
            negframes = select_neg_frame(points,img)
        
        return frames + negframes   
    def annotate_image(self,img,ix):
        if self.labelmap:
            for key in self.labelmap:
                if ix in self.labelmap[key]:
                    pt1,pt2,frame = self.getRect(img,key,ix)
                    rec = cv2.rectangle(img,pt1,pt2,(0,255,0))
                    
    def makeSubs(self,dirname):
        ix = 1
        img = self.getNextImage()
        
        if not os.path.exists(dirname):
            print 'directory must exist'
            return
        filelines = ''
        dirname = os.path.abspath(dirname)
        while not img is None:
            frames = self.subs(img,ix)
            for idx,frame in enumerate(frames):
                subframe,label = frame
                fname = dirname+'/'+str(ix)+'_'+str(idx)+'.jpg'
                cv2.imwrite(fname,subframe)
                filelines += fname +' '+str(label)+'\n'
            img = self.getNextImage()
            print ix
            ix += 1
        with open(dirname+'/labels.txt','w') as f:
            f.write(filelines)
                    
    def playvid(self):
        ix = 0
        img = self.getNextImage()
        while not img is None :
            self.annotate_image(img,ix)
            cv2.imshow('win1',img)
            cv2.waitKey(40)
            img = self.getNextImage()
            ix += 1

# must be careful, assumes image has space for such a box
# will never terminate otherwise
def select_rand_frame(dim,boxes,img):
    frame = None
    flag = True
    while frame is None:
        h,w = dim
        j = max(h,w)
        minx = float('inf')
        maxx = 0
        miny = float('inf')
        maxy = 0
        for (p1,p2) in boxes:
            minx = min(p1[0],minx)
            miny = min(p1[1],miny)
            maxx = max(p2[0],maxx)
            maxy = max(p2[1],maxy)


        # to be safest we select from outside the bounding boxes
        minx = max(minx,1)
        maxx = min(maxx,w-1)
        xend = np.random.randint(0,high=minx)
        xstart = np.random.randint(maxx,high=w)
        length = 0
        ratio = (0.5-np.random.rand())*0.4 + 1
        l2 = 0
        
        if w - xstart > xend:
            hi = max(w-xstart,2)
            length = np.random.randint(1,high=hi)
            l1 = length * ratio
            l2 = length/ratio
            xend = min(w,xstart+l1*length)
        else:
            hi = max(xend,2)
            length = np.random.randint(1,high=hi)
            l1 = length * ratio
            l2 = length/ratio
            xstart = max(0,xend - l1*length)


        ystart = np.random.randint(1,high=h*0.85)
        yend = min((ystart +l2),h)

        dx = xend - xstart
        dy = yend - ystart

        area = dx*dy
        r = dx/dy
        if area < 0.0001*img.size or r > 1.4 or r < 0.8:
            continue

        return (xstart,ystart),(xend,yend),img[ystart:yend, xstart:xend]
            
            
        
            
def select_neg_frame(posboxes,img):
    h = img.shape[0]
    w = img.shape[1]
    negboxes = []
    negframes = []
    for x in range(len(posboxes)): #guarantee equal pos & negs
        pt1,pt2,frame = select_rand_frame((h,w),posboxes,img)
        negboxes.append((pt1,pt2))
        negframes.append((frame,0))
    return negframes
def test():
    import argparse as ag
    parser = ag.ArgumentParser()
    parser.add_argument('vidfile',help='Enter full file path')
    parser.add_argument('--labelfile',help='if label file has a different prefix please include here')
    parser.add_argument('--decomp',type=int,help='use to split the video into images with the step size given')
    parser.add_argument('--subs',help='specify if you want to create and save cropped files of annotations')
    args = parser.parse_args()

    vidfile = args.vidfile
    labelfile = ''
    stepsize = args.decomp
    subsdir = args.subs
    if args.labelfile:
        labelfile = args.labelfile
    else:
        labelfile = os.path.splitext(vidfile)[0] + '.xgtf'

    t = VideoRunner(vidfile,labelfile)
    if stepsize:
        t.decomp(stepsize)
    elif subsdir:
        t.makeSubs(subsdir)
    else:
        t.playvid()

#test()
if __name__ == '__main__':
    test()

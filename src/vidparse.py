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
                obj['x'] =  int(bbox.getAttribute('x'))
                obj['y'] = int(bbox.getAttribute('y'))
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
        
    def annotate_image(self,img,ix):
        if self.labelmap:
            for key in self.labelmap:

                if ix in self.labelmap[key]:
                    obj = self.labelmap[key][ix]
                    pt1 = obj['x'],obj['y']
                    pt2 = obj['x']+obj['width'],obj['y']+obj['height']
                    rec = cv2.rectangle(img,pt1,pt2,(0,255,0))
                    
    def playvid(self):
        ix = 0
        img = self.getNextImage()
        while not img is None :
            self.annotate_image(img,ix)
            cv2.imshow('win1',img)
            cv2.waitKey(40)
            img = self.getNextImage()
            ix += 1
            

def test():
    t = VideoRunner('/Users/dvad/code/deepLearning/drone/datasets/UCF/actions1.mpg',
                    '../../drone/datasets/UCF/actions1.xgtf')
    t.playvid()

#test()
if __name__ == '__main__':
    test()

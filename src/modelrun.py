import numpy as np
import matplotlib.pyplot as plt
import argparse as ag
import os
import sys
import time
import cv2
import pdb
import time
from vidparse import VideoRunner

plt.rcParams['figure.figsize'] = (10,10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
caffe_root = '/Users/dvad/Documents/china/research/caffe/'
proj_root = '/Users/dvad/code/deepLearning/peta/'
sys.path.insert(0,caffe_root+'python')
import caffe


def init_net(weights,model):
    if os.path.isfile(weights) and os.path.isfile(weights):
        print 'Net Found'
    else:
        print 'No net found ... Terminating'
        return
    #Assume we have the model
    caffe.set_mode_cpu() #set NN computation mode to cpu only
    model_def = model
    model_weights = weights

    net = caffe.Net(model_def, #defines the structure of the model
                    model_weights, #contains the trained weights
                    caffe.TEST) #use standard test mode
    
    net.blobs['data'].reshape(1, #batch size
                              3, #3 channel rgb
                              128, # 128 width
                              128) # 128 height, we used these squares for training
    return net
def eval_img(net,imgname):

    try:
        img = load_img(imgname,net)
        #pdb.set_trace()
        clas = get_pclass(net,img)
        return clas
    except Exception as e:
        print 'Error {}'.format(e)
        
def get_pclass(net,img):
    clasmap = {'0':1,'1':0}
    probs =  run_img(net,img)
    
    clas = probs.argmax() #get the class of highest probability
    #print 'predicted class : {}'.format( clasmap[str(clas)] )
    return clasmap[str(clas)]

def eval_vid(net,vid):
    labels = []
    plabels = []
    labelfile = os.path.splitext(vid)[0] + '.xgtf'
    if not os.path.exists(vid):
        print 'video does not exists'
        return
    elif not os.path.exists(labelfile):
        print 'video labels do not exists'
        return
    vid_runner = VideoRunner(vid,labelfile)
    img = vid_runner.getNextImage()
    start = time.time()
    truestart = start
    prevFrames = 0
    fps = 0
    while not img is None:
        ix = vid_runner.getFrameNo() - 1 #1 based index
        print ix
        t_img = trans_opencv_img(img,net)
        #pdb.set_trace()
        clas = get_pclass(net,t_img)
        end = time.time()
        if end - start > 1:
            framecount = ix - prevFrames
            print "fps: {}".format(framecount)
            start = end
        plabels.append(clas)
        labels.append(vid_runner.get_class(ix))
        img = vid_runner.getNextImage()
    #pdb.set_trace()
    labels = np.array(labels)
    plabels = np.array(plabels)
    
    eqs = (labels == plabels)
    grade = float(eqs.sum()*1.0/eqs.size)
    return grade

def eval_imgs(net,dirname):
    files = []
    labels = []
    plabels = []
    if os.path.exists(dirname):
        data = np.genfromtxt(dirname,dtype='str')
    
    for fname in data[:,0]:
        lab = eval_img(net,fname)
        plabels.append(lab)
    labels = data[:,1].astype(np.int)
    plables = np.array(plabels)

    eqs = (labels == plables)
    grade = float(eqs.sum()*1.0/eqs.size)
    return grade
            
def run_img(net,img):
    net.blobs['data'].data[...] = img #... is placeholder for all dims
    ### classification
    t = time.time()
    output = net.forward()
    #pdb.set_trace()
    #print 'forward time : {}'.format(time.time() - t)
    output_prob = output['loss'][0]
    return output_prob

def trans_opencv_img(img,net):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data',(2,0,1))
    transformer.set_channel_swap('data',(2,1,0))
    transformer.set_raw_scale('data',0.00390625)
    transformed_image = transformer.preprocess('data',img)
    print transformed_image.shape
    return transformed_image

def load_img(fname,net):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data',(2,0,1))
    img = caffe.io.load_image(fname)
    transformed_image = transformer.preprocess('data',img)
        
    return transformed_image

def write_results(fname,output):
    if fname and output:
        with open(fname,'w') as f:
            f.write(output)

def main():
    parser = ag.ArgumentParser()
    parser.add_argument('--file',help='Enter the name of the input image')
    parser.add_argument('--list',help='Enter the name of the directory containing images to test')
    parser.add_argument('--vid', help='Enter the name of video to parse and test on')
    parser.add_argument('model', help='Enter the filename of the model you wish to run')
    parser.add_argument('weights',help='Enter the filename for the trained model weights')
    parser.add_argument('--out',help='Enter Filename of the outputfile')
    args = parser.parse_args()

    filename = args.file
    listname = args.list
    vidname = args.vid
    modelname = args.model
    modelweights = args.weights
    outfile = args.out
    grade = None
    if filename :
        net = init_net(modelweights,modelname)
        labels = eval_img(net,filename)
    elif vidname :
        net = init_net(modelweights,modelname)
        grade = eval_vid(net,vidname)
        rez = "Total evaluation: {}%".format(grade)
        print rez
        write_results(outfile,rez)
    elif listname :
        net = init_net(modelweights,modelname)
        grade = eval_imgs(net,listname)
        rez = "Total evaluation: {}%".format(grade)
        print rez
        write_results(outfile,rez)

    else:
        print """Error no images specified
                 Please specify a filename
                 or directory"""
        return
    
    
main()

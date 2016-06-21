import numpy as np
import matplotlib.pyplot as plt
import argparse as ag
import os
import sys
import time
import cv2
import pdb

plt.rcParams['figure.figsize'] = (10,10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
caffe_root = '/Users/dvad/Documents/china/research/caffe/'
proj_root = '/Users/dvad/code/deepLearning/peta/'
sys.path.insert(0,caffe_root+'python')
import caffe


def init_net():
    if os.path.isfile(proj_root + 'src/model_iteration_iter_10000.caffemodel'):
        print 'Net Found'
    else:
        print 'No net found ... Terminating'
        return
    #Assume we have the model
    caffe.set_mode_cpu() #set NN computation mode to cpu only
    model_def = proj_root+'src/deploy.prototxt'
    model_weights = proj_root+'src/model_iteration_iter_10000.caffemodel'

    net = caffe.Net(model_def, #defines the structure of the model
                    model_weights, #contains the trained weights
                    caffe.TEST) #use standard test mode
    
    net.blobs['data'].reshape(1, #batch size
                              3, #3 channel rgb
                              128, # 128 width
                              128) # 128 height, we used these squares for training
    return net
def eval_img(net,imgname):
    clasmap = {'0':1,'1':0}
    try:
        img = load_img(imgname,net)
        probs =  run_img(net,img)
        print probs
        clas = probs.argmax() #get the class of highest probability
        #print 'predicted class : {}'.format( clasmap[str(clas)] )
        return clasmap[str(clas)]
    except Exception as e:
        print 'Error {}'.format(e)
    
        
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
    pdb.set_trace()
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
    
def load_img(fname,net):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data',(2,0,1))
    img = caffe.io.load_image(fname)
    transformed_image = transformer.preprocess('data',img)
    
    print img.shape
    print transformed_image.shape
    
    return transformed_image

def main():
    parser = ag.ArgumentParser()
    parser.add_argument('--file',help='Enter the name of the input image')
    parser.add_argument('--list',help='Enter the name of the directory containing images to test')
    args = parser.parse_args()

    filename = args.file
    listname = args.list
    grade = None
    if filename :
        net = init_net()
        labels = eval_img(net,filename)
    elif listname :
        net = init_net()
        grade = eval_imgs(net,listname)
        print "Total evaluation: {}%".format(grade)
    else:
        print """Error no images specified
                 Please specify a filename
                 or directory"""
        return
    
    
main()

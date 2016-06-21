import os
import shutil as sh
import numpy as np

def __main__():
    data_root = '/data/peta_data/'
    data_train = '/data/peta_data_train/'
    data_test = '/data/peta_data_test/'
    if not os.path.exists(data_train):
            os.mkdir(data_train)
    if not os.path.exists(data_test):
            os.mkdir(data_test)
    numfiles = 0
    for path,dirs,files in os.walk(data_root):
        steps = path.split('/')
        currentdir = steps[-1]
        parent = steps[-2]
        if currentdir == 'archive':
            numfiles = len(files)
            arr = np.array(files)
            np.random.shuffle(arr)
            r =  np.size(arr)
            numtrain = r * 0.9
            trainfiles = arr[0:numtrain]
            testfiles = arr[numtrain:]
            numfiles += r
            for f in trainfiles:
                src = os.path.join(path,f)
                dest = os.path.join(data_train,parent +f)
                print " {} ----> {}".format(src,dest)
                sh.copy( src, dest )
            for f in testfiles:
                src = os.path.join(path,f)
                dest = os.path.join(data_test,parent + f)
                print " {} ----> {}".format(src,dest)
                sh.copy( src, dest )
        
    print numfiles


__main__()


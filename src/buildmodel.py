from pylab import *
import sys
caffe_root='/Users/dvad/Documents/china/research/caffe/'
sys.path.insert(0,caffe_root+'python')
import caffe
import os
import pdb


def init_net(solverfile):
    if os.path.isfile(solverfile):
        print 'Net Found'
    else:
        print 'No net found ... Terminating'
        return
    
    #Assume we have the model
    
    solver = None
    solver = caffe.SGDSolver(solverfile)
    return solver

def run_solver(solver):
    niter = 100
    test_interval = 25
    train_loss = zeros(niter)
    test_acc = zeros(int(np.ceil(niter/test_interval)))
    output = zeros((niter,2,2))

    for it in range(niter):
        solver.step(1) #perform sgd on a minibatch
        train_loss[it] = solver.net.blobs['loss'].data
        # store the output on the first test batch
        # (start the forward pass at conv1 to avoid loading new data)
        solver.test_nets[0].forward(start='conv1')
        output[it] = solver.test_nets[0].blobs['ip2'].data[:2]

        #run a full test every so often
        # (Caffe can also do this for us and write to a log, but we show here
        # how to do it directly in Python, where more complicated things are easier.)
        if it % test_interval == 0:
            print 'Iteration', it, 'testing...'
            correct = 0
            for test_it in range(100):
                solver.test_nets[0].forward()
                correct += sum(solver.test_nets[0].blobs['ip2'].data.argmax(1)
                               == solver.test_nets[0].blobs['label'].data)
                test_acc[it // test_interval] = correct / 1e4
    _, ax1 = subplots()
    ax2 = ax1.twinx()
    ax1.plot(arange(niter),train_loss)
    ax2.plot(test_interval * arange(len(test_acc)),test_acc,'r')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test_accuracy')
    ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))

def main():
    solver = init_net("./solver.prototxt")
    #caffe.set_device(0)
    #caffe.set_mode_gpu()
    caffe.set_mode_cpu()
    run_solver(solver)
    
    
if __name__=='__main__':
    main()

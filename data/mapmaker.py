import json
import argparse as ag
import os


def readfile(fname,dmap):
    if not os.path.exists(fname):
        print "File '{}' does not exist".format(fname)
    else:
        print "Opening {}".format(fname)
    with open(fname,'r') as fil:
        for line in fil.readlines():
            parts = line.split(' ')
            parts = map(lambda(x):x.strip(),parts)
            dmap['labels'][parts[0]] = parts[1]
        
def main():
    parser = ag.ArgumentParser()
    parser.add_argument('files',nargs='+')
    args = parser.parse_args()
    print args
    fnames = args.files
    dmap = {}
    dmap['labels'] = {}
    for f in fnames:
        readfile(f,dmap)
    jsonobj = json.dumps(dmap)
    with open('labelmap.json','w') as fil:
        fil.write(jsonobj)
    

main()

import os
import numpy as np
import argparse as ag


def assign_labels(filename,names,label):
    flag = 'a' if os.path.exists(filename) else 'w'
    
    with open(filename,flag) as fil:
        lines = map(lambda(x):x+' '+label,names)
        pre = '\n' if flag == 'a' else ''
        writestring = pre + '\n'.join(lines)
        fil.write(writestring)

def get_file_names(dir,pflag):
    if not os.path.exists(dir):
        print "Directory does not exist"
        return []
    for p,d,files in os.walk(dir):
        if pflag:
            return map(lambda (x):os.path.join(p,x),files)
        else:
            return files

def main():
    parser = ag.ArgumentParser()
    parser.add_argument('out',help='Specify output file')
    parser.add_argument('label',help='Specify label to assign')
    parser.add_argument('--dir',required=False)
    parser.add_argument('--file',required=False)
    parser.add_argument('--path',type=bool,help='flag to include entire path to file')
    args = parser.parse_args()
    print args
    dir = args.dir
    fname = args.file
    outfile = args.out
    lbl = args.label
    pflag = args.path
    if not outfile :
        print "Must include output file"
        return
    
    if not lbl:
        print "Must include label"
        return

    
    if not fname and not dir:
        print "Must include file or directory name"
        return 
    files = []
    if fname:
        files = [fname]
    elif dir:
        files = get_file_names(dir,pflag)
        if not files:
            return
    assign_labels(outfile,files,lbl)
main()

import re
import os
import numpy as np

# Sort a string with a number inside
# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]


'''
save numpy.array or plt.figure
'''
def save_arr(directory, file, arr):
    path = os.path.join(directory,file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(path,arr)
    print('save numpy array in {}'.format(path))
    
def save_fig(directory,file,fig):
    path = os.path.join(directory,file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(path,file)
    print('save plot figure in {}'.format(path))
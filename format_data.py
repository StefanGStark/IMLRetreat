import os
import sys
import scipy.io as sio
import scipy as sp
import glob
import re
import matplotlib.pyplot as plt
import pdb

# __globals__
# __header__
# __version__
# artifacts
# df
# epochl
# features_names
# fs
# maxep
# Pspec
# stages
# windowl
# X (time points x features)

def format_featnames(matobj):
    names = sp.asarray([x[0] for x in matobj['features_names'][0]])
    return names

def format_data(matobj):
    data = matobj['X'].T
    return data

def has_complex(arr):
    try:
        imag = arr.imag
    except AttributeError:
        return False
    else:
        return sp.any(imag != 0)

def format_exp(path):
    exp = os.path.basename(path)
    exp = re.sub('.mat$', '', exp)
    exp = re.sub(r'\([^)]*\)', '', exp)
    return exp

def format_labels(matobj):
    labarr= matobj['stages'][0]
    labarr = re.sub('R', '4', labarr)
    labarr = sp.array(list(labarr), dtype=int)
    labmat = sp.zeros((labarr.size, 5), dtype=int)
    for indx, l in enumerate(labarr):
        labmat[indx, l] = 1
    return labmat

def plot_data(data, figsize=(18,2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(data)
    plt.tight_layout()
    return ax

wdir = os.path.dirname(__file__)
mat_paths = glob.glob(os.path.join(wdir, 'EEG_data', '*.mat'))

def load_exps():
    _FEATURES = None
    exp_data = dict()
    exp_labs = dict()
    for mpath in mat_paths:
        exp = format_exp(mpath)
        matobj = sio.loadmat(mpath)
        feats = format_featnames(matobj)
        data = format_data(matobj)
        mask = sp.asarray([sp.invert(has_complex(arr)) for arr in data])
        feats = feats[mask]
        order = sp.argsort(feats)
        feats = [f.lower() for f in feats[order]]
        data = data[mask][order].real
        labels = format_labels(matobj)
        if _FEATURES is None:
            _FEATURES = feats
        assert feats == _FEATURES, 'Features do not match! %s\n%s'%(' '.join(feats), ' '.join(_FEATURES))
        assert data.shape[1] == len(labels), 'Labels (%d) do not match time points (%d)' %(len(labels), data.shape[1])
        exp_data[exp] = data   
        exp_labs[exp] = labels
    return exp_data, exp_labs, _FEATURES

if __name__ == '__main__':
    exp_data, exp_labs, features = load_exps()

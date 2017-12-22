import sys
sys.path.append('..')

import os
import json
from time import time
import numpy as np
from tqdm import tqdm

import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv
from theano.tensor.signal.pool import pool_2d as max_pool_2d

from PIL import Image
import h5py
import pickle
from PIL import Image

from lib import activations
from lib import updates
from lib import inits
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.metrics import nnc_score, nnd_score
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data, center_crop, patch

from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream

l2 = 1e-5         # l2 weight decay
nvis = 196        # # of samples to visualize during training
b1 = 0.5          # momentum term of adam
nc = 3            # # of channels in image
nbatch = 128      # # of examples in batch
npx = 64          # # of pixels width/height of images

nx = npx*npx*nc   # # of dimensions in X
niter = 1000        # # of iter at starting learning rate
niter_decay = 30   # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam
ntrain = 25000   # # of examples to train on


relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()
bce = T.nnet.binary_crossentropy

def mse(x,y):
    return T.sum(T.pow(x-y,2), axis = 1)

gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)
sigma_ifn = inits.Normal(loc = -100., scale=0.02)
gain_ifn = inits.Normal(loc=1., scale=0.02)
bias_ifn = inits.Constant(c=0.)

def target_transform(X):
    return floatX(X).transpose(0, 3, 1, 2) / 127.5 - 1.

def input_transform(X):
    return target_transform(X)

def make_conv_layer(X, input_size, output_size, input_filters,
                    output_filters, name, index,
                    weights=None, filter_sz=5):
    is_deconv = output_size >= input_size
    w_size = (input_filters, output_filters, filter_sz, filter_sz) if is_deconv else (
    output_filters, input_filters, filter_sz, filter_sz)

    if weights is None:
        w = gifn(w_size, '%sw%i' % (name, index))
        g = gain_ifn((output_filters), '%sg%i' % (name, index))
        b = bias_ifn((output_filters), '%sb%i' % (name, index))
    else:
        w, g, b = weights

    conv_method = deconv if is_deconv else dnn_conv
    activation = relu if is_deconv else lrelu
    sub = output_size / input_size if is_deconv else input_size / output_size
    if filter_sz == 3:
        bm = 1
    else:
        bm = 2
    layer = activation(batchnorm(conv_method(X, w, subsample=(sub, sub), border_mode=(bm, bm)), g=g, b=b))
    return layer, [w, g, b]

def make_conv_set(input, layer_sizes, num_filters, name, weights=None, filter_szs=None):
    assert (len(layer_sizes) == len(num_filters))
    vars_ = []
    layers_ = []
    current_layer = input
    for i in range(len(layer_sizes) - 1):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i + 1]
        input_filters = num_filters[i]
        output_filters = num_filters[i + 1]
        if weights is not None:
            this_wts = weights[i * 3: i * 3 + 3]
        else:
            this_wts = None
        if filter_szs != None:
            filter_sz = filter_szs[i]
        else:
            filter_sz = 5
        layer, new_vars = make_conv_layer(current_layer, input_size, output_size,
                                          input_filters, output_filters, name, i,
                                          weights=this_wts, filter_sz=filter_sz)
        vars_ += new_vars
        layers_ += [layer]
        current_layer = layer
    return current_layer, vars_, layers_


# Use code below to use a saved model


def inverse(X):
    X_pred = (X.transpose(0, 2, 3, 1) + 1) * 127.5
    X_pred = np.rint(X_pred).astype(int)
    X_pred = np.clip(X_pred, a_min=0, a_max=255)
    return X_pred.astype('uint8')


def load_model():
    [e_params, g_params, d_params] = pickle.load(open("faces_dcgan.pkl", "rb"))
    gwx = g_params[-1]
    dwy = d_params[-1]
    # inputs
    X = T.tensor4()
    ## encode layer
    e_layer_sizes = [128, 64, 32, 16, 8]
    e_filter_sizes = [3, 256, 256, 512, 1024]
    eX, e_params, e_layers = make_conv_set(X, e_layer_sizes, e_filter_sizes, "e", weights=e_params)
    ## generative layer
    g_layer_sizes = [8, 16, 32, 64, 128]
    g_num_filters = [1024, 512, 256, 256, 128]
    g_out, g_params, g_layers = make_conv_set(eX, g_layer_sizes, g_num_filters, "g", weights=g_params)
    g_params += [gwx]
    gX = tanh(deconv(g_out, gwx, subsample=(1, 1), border_mode=(2, 2)))
    ## discrim layer(s)

    df1 = 128
    d_layer_sizes = [128, 64, 32, 16, 8]
    d_filter_sizes = [3, df1, 2 * df1, 4 * df1, 8 * df1]

    def discrim(input, name, weights=None):
        d_out, disc_params, d_layers = make_conv_set(input, d_layer_sizes, d_filter_sizes, name, weights=weights)
        d_flat = T.flatten(d_out, 2)

        disc_params += [dwy]
        y = sigmoid(T.dot(d_flat, dwy))

        return y, disc_params, d_layers

    # target outputs
    target = T.tensor4()

    p_real, d_params, d_layers = discrim(target, "d", weights=d_params)
    # we need to make sure the p_gen params are the same as the p_real params
    p_gen, d_params2, d_layers = discrim(gX, "d", weights=d_params)

    ## GAN costs
    d_cost_real = bce(p_real, T.ones(p_real.shape)).mean()
    d_cost_gen = bce(p_gen, T.zeros(p_gen.shape)).mean()
    g_cost_d = bce(p_gen, T.ones(p_gen.shape)).mean()

    ## MSE encoding cost is done on an (averaged) downscaling of the image
    target_pool = max_pool_2d(target, (4, 4), mode="average_exc_pad", ignore_border=True)
    target_flat = T.flatten(target_pool, 2)
    gX_pool = max_pool_2d(gX, (4, 4), mode="average_exc_pad", ignore_border=True)
    gX_flat = T.flatten(gX_pool, 2)
    enc_cost = mse(gX_flat, target_flat).mean()

    ## generator cost is a linear combination of the discrim cost plus the MSE enocding cost
    d_cost = d_cost_real + d_cost_gen
    g_cost = g_cost_d + enc_cost / 10  ## if the enc_cost is weighted too highly it will take a long time to train

    ## N.B. e_cost and e_updates will only try and minimise MSE loss on the autoencoder (for debugging)
    e_cost = enc_cost

    cost = [g_cost_d, d_cost_real, enc_cost]

    elrt = sharedX(0.002)
    lrt = sharedX(lr)
    d_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
    g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
    e_updater = updates.Adam(lr=elrt, b1=b1, regularizer=updates.Regularizer(l2=l2))

    d_updates = d_updater(d_params, d_cost)
    g_updates = g_updater(e_params + g_params, g_cost)
    e_updates = e_updater(e_params, e_cost)

    print 'COMPILING'
    t = time()
    _train_g = theano.function([X, target], cost, updates=g_updates)
    _train_d = theano.function([X, target], cost, updates=d_updates)
    _train_e = theano.function([X, target], cost, updates=e_updates)
    _get_cost = theano.function([X, target], cost)
    print('%.2f seconds to compile theano functions' % (time() - t))
    img_dir = "gen_images/"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    ae_encode = theano.function([X, target], [gX, target])
    return ae_encode



def infer(path, ae_encode):
    '''
    :param path: path of infer data
    :param ae_encode: compiled theano function
    :return: image saved path in string
    '''

    hf = h5py.File(path, 'r+')

    split_dict = {
        'test': {'input': (0, 1), 'target': (0, 1)},
    }
    hf.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    test_set = H5PYDataset(path, which_sets=('test',))

    batch_size = 1

    test_scheme = SequentialScheme(examples=test_set.num_examples, batch_size=batch_size)
    test_stream = DataStream(test_set, iteration_scheme=test_scheme)

    for te_train, te_target in test_stream.get_epoch_iterator():
        break
    te_out, te_ta = ae_encode(input_transform(te_train), target_transform(te_target))
    te_reshape = inverse(te_out)
    te_target_reshape = inverse(te_ta)

    new_size = (128 * 2, 160)
    new_im = Image.new('RGB', new_size)
    r = np.random.choice(1, 1, replace=False).reshape(1, 1)
    for i in range(1):
        for j in range(1):
            index = r[i][j]

            target_im = Image.fromarray(te_target_reshape[index])
            train_im = Image.fromarray(te_train[index].astype(np.uint8))
            im = Image.fromarray(te_reshape[index])

            new_im.paste(train_im, (128 * (i * 2), 160 * j))
            new_im.paste(im, (128 * (i * 2 + 1), 160 * j))
    img_loc = "gen_images/%i.png" % int(time())
    new_im.save(img_loc)
    return img_loc
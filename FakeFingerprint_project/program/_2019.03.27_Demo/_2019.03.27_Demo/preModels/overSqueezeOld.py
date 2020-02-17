# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 22:52:34 2016
 
@author: eunsoo
"""
#%%
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from scipy.misc import imread, imresize, imsave
from customlayers import Softmax4D
from keras.layers import Input, merge, SeparableConv2D, BatchNormalization
from keras import layers
from keras.layers.pooling import GlobalAveragePooling2D, AveragePooling2D
 
#%%
sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"
batchN = "batch_"
#%%
 
# Modular function for Fire Node
def fire_module(x, fire_id, squeeze=16, expand=64, dim_ordering='tf'):
    s_id = 'fire'+str(fire_id)+'/'
    if dim_ordering is 'tf':
        c_axis = 3
    else:
        c_axis = 1
     
    x = Conv2D(squeeze, (1, 1), padding='valid', name=s_id+sq1x1)(x)
    x = BatchNormalization(name=s_id+batchN+sq1x1)(x)
    x = Activation('relu', name=s_id+relu+sq1x1)(x)
     
    left = Conv2D(expand, (1, 1), padding='valid', name=s_id+exp1x1)(x)
    x = BatchNormalization(name=s_id+batchN+exp1x1)(x)
    left = Activation('relu', name=s_id+relu+exp1x1)(left)
     
    right = Conv2D(expand, (3, 3), padding='same', name=s_id+exp3x3)(x)
    x = BatchNormalization(name=s_id+batchN+exp3x3)(x)
    right = Activation('relu', name=s_id+relu+exp3x3)(right)
    x = layers.concatenate([left, right], axis=c_axis, name=s_id+'concat')
    # x = merge([left, right], mode='concat', concat_axis=c_axis, name=s_id+'concat')
    return x
#%%    
# Original Squeeze from paper. Updated version from squeezenet paper.
 
 
def get_squeezenet(weights_path=None, heatmap=False, nb_classes = 3, input_size=32):
    if heatmap:
        input_img=Input(shape=(None, None, 1))
    else:
        input_img=Input(shape=(input_size, input_size, 1))
 
    if input_size == 32: pool_size = (2,2)
    elif input_size == 48: pool_size = (3,3)
    elif input_size == 64: pool_size = (4,4)
    else :
        print("you have to select among (32, 48, 64)")
        return False
     
    # This part shoud be 7x7 and stride 2
    x = Conv2D(96, (7, 7), strides=(2,2), padding='same',name='conv1_park')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)
    # x = Conv2D(64, 3, 3, strides=(2, 2), padding='valid', name='conv1')(input_img)    
    x = Activation('relu', name='relu_conv1')(x)
    x = ZeroPadding2D((1,1))(x)    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
     
    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = ZeroPadding2D((1,1))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
     
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = ZeroPadding2D((1,1))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
     
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x = Dropout(0.5, name='drop9')(x)    
    x = Conv2D(nb_classes, (1, 1), padding='valid', name='conv10_park')(x)
    x = BatchNormalization(name='batch_conv10')(x)
 
    x = Activation('relu', name='relu_conv10')(x)
 
    if heatmap: 
        x = AveragePooling2D(pool_size=pool_size, strides=(1,1))(x)
        out = Softmax4D(axis=3, name="loss")(x)
    else:
        # x = GlobalAveragePooling2D()(x)
        x = AveragePooling2D(pool_size=pool_size, strides=(1,1))(x)
        x= Flatten(name="flatten")(x)
        out = Activation('softmax', name='loss')(x)
     
    model = Model(inputs=input_img, outputs=[out])
    if weights_path:
        model.load_weights(weights_path)
    return model
 
def shirinking_squeezenet(weights_path=None, heatmap=False, nb_classes = 3, input_size=32):
    if heatmap:
        input_img=Input(shape=(None, None, 1))
    else:
        input_img=Input(shape=(input_size, input_size, 1))
     
    if input_size == 32: pool_size = (7,7)
    elif input_size == 48: pool_size = (11,11)
    elif input_size == 64: pool_size = (15,15)
    else :
        print("you have to select among (32, 48, 64)")
        return False
 
    x = Conv2D(96, (3, 3), padding='same', name='conv1')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)
    x = Activation('relu', name='relu_conv1')(x)
    x = fire_module(x, fire_id=2, squeeze=16, expand=64)    
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    # 32: 15x15, 23x23, 31x31
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x) 
    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
 
    # 32: 7x7, 11x11, 15x15
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x) 
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x = Dropout(0.5, name='drop9')(x)    
    x = Conv2D(nb_classes, (1, 1), padding='valid', name='conv10')(x)
    x = BatchNormalization(name='batch_conv10')(x)
 
    x = Activation('relu', name='relu_conv10')(x)
    if heatmap:
        x = AveragePooling2D(pool_size=pool_size, strides=(2,2))(x)
        out = Softmax4D(axis=3, name="loss")(x)
    else:
        x = AveragePooling2D(pool_size=pool_size, strides=(2,2))(x)
        x= Flatten(name="flatten")(x)
        out = Activation('softmax', name='loss')(x)
    model = Model(inputs=input_img, outputs=[out])
    if weights_path:
        model.load_weights(weights_path)
    return model
 
def convnet(network, weights_path=None, heatmap=False,input_size=32, nb_classes=3):
    """
    Returns a keras model for a CNN.
 
    BEWARE !! : Since the different convnets have been trained in different settings, they don't take
    data of the same shape. You should change the arguments of preprocess_image_batch for each CNN :
    * For AlexNet, the data are of shape (227,227), and the colors in the RGB order (default)
    * For VGG16 and VGG19, the data are of shape (224,224), and the colors in the BGR order
 
    It can also be used to look at the hidden layers of the model.
 
    It can be used that way :
    >>> im = preprocess_image_batch(['cat.jpg'])
 
    >>> # Test pretrained model
    >>> model = convnet('vgg_16', 'weights/vgg16_weights.h5')
    >>> sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    >>> model.compile(optimizer=sgd, loss='categorical_crossentropy')
    >>> out = model.predict(im)
 
    Parameters
    --------------
    network: str
        The type of network chosen. For the moment, can be 'vgg_16' or 'vgg_19'
 
    weights_path: str
        Location of the pre-trained model. If not given, the model will be trained
 
    heatmap: bool
        Says wether the fully connected layers are transformed into Conv2D layers,
        to produce a heatmap instead of a
 
 
    Returns
    ---------------
    model:
        The keras model for this convnet
 
    output_dict:
        Dict of feature layers, asked for in output_layers.
    """
 
 
    # Select the network
    if network == "basic":
        convnet_init = get_squeezenet
    elif network == "ssq":
        convnet_init = shirinking_squeezenet
    else:
        print("We do not have the model")
 
    convnet = convnet_init(weights_path=weights_path, heatmap=heatmap,
        nb_classes=nb_classes, input_size=input_size)
    return convnet
 
if __name__ == "__main__":
    ### Here is a script to compute the heatmap of the dog synsets.
    ## We find the synsets corresponding to dogs on ImageNet website
    s = "n02084071"
    ids = synset_to_dfs_ids(s)
    # Most of the synsets are not in the subset of the synsets used in ImageNet recognition task.
    ids = np.array([id_ for id_ in ids if id_ is not None])
 
    im = preprocess_image_batch(['examples/dog.jpg'], color_mode="rgb")
    print (im.shape)
    # Test pretrained model
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model = convnet('alexnet',weights_path="weights/alexnet_weights.h5", heatmap=True)
    model.compile(optimizer=sgd, loss='mse')
 
    out = model.predict(im)
    heatmap = out[0,ids,:,:].sum(axis=0)
    print (out)
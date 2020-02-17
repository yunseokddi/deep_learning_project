# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 22:52:34 2016

@author: eunsoo
"""
#%%

from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, BatchNormalization
from keras.layers import Dropout, Activation, Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras import layers

#%%
sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"
batchN = "batch_"
#%%
def fireXcep_module(x, fire_id, squeeze=16, expand=64, dim_ordering='tf'):
    s_id = 'fire'+str(fire_id)+'/'
    if dim_ordering is 'tf':
        c_axis = 3
    else:
        c_axis = 1
    
    x = Conv2D(squeeze, (1, 1), padding='valid', name=s_id+sq1x1)(x)
    x = BatchNormalization(name=s_id+batchN+sq1x1)(x)
    x = LeakyReLU(name=s_id+relu+sq1x1)(x)
    
    left = Conv2D(expand, (1, 1), padding='valid', name=s_id+exp1x1)(x)
    left = BatchNormalization(name=s_id+batchN+exp1x1)(left)
    left = LeakyReLU(name=s_id+relu+exp1x1)(left)
    
    right = SeparableConv2D(expand, (3, 3), padding='same', name=s_id+exp3x3)(x)
    right = BatchNormalization(name=s_id+batchN+exp3x3)(right)
    right = LeakyReLU(name=s_id+relu+exp3x3)(right)
    x= layers.concatenate([left, right], axis=3, name=s_id+'concat')
    # x = merge([left, right], mode='concat', concat_axis=c_axis, name=s_id+'concat')
    return x



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
    left = BatchNormalization(name=s_id+batchN+exp1x1)(left)
    left = Activation('relu', name=s_id+relu+exp1x1)(left)
    
    right = Conv2D(expand, (3, 3), padding='same', name=s_id+exp3x3)(x)
    right = BatchNormalization(name=s_id+batchN+exp3x3)(right)
    right = Activation('relu', name=s_id+relu+exp3x3)(right)
    x= layers.concatenate([left, right], axis=3, name=s_id+'concat')
    # x = merge([left, right], mode='concat', concat_axis=c_axis, name=s_id+'concat')
    return x
#%%    
# Original Squeeze from paper. Updated version from squeezenet paper.
def get_squeezenet(nb_classes, dim_ordering='tf', weights_path=None):
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 224, 224))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(224, 224, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only avaiable")
    x = Conv2D(96, (7, 7), strides=(2,2), padding='same',name='conv1_park')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)
    # x = Conv2D(64, 3, 3, strides=(2, 2), padding='valid', name='conv1')(input_img)    
    x = Activation('relu', name='relu_conv1')(x)    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x = Dropout(0.5, name='drop9')(x)    
    x = Conv2D(nb_classes, (1, 1), padding='valid', name='conv10_park')(x)
    x = BatchNormalization(name='batch_conv10')(x)

    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(inputs=input_img, outputs=[out])
    if weights_path:
        model.load_weights(weights_path)
    return model

def get_sqAllXceNetFinalDeep(nb_classes, dim_ordering='tf', weights_path=None):
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 224, 224))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(224, 224, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only avaiable")
    x = SeparableConv2D(96, (7, 7), strides=(2,2), padding='same', name='conv1_park')(input_img)
    # x = Conv2D(64, 3, 3, strides=(2, 2), padding='valid', name='conv1')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)    
    x = Activation('relu', name='relu_conv1')(x)    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
    x = fireXcep_module(x, fire_id=2, squeeze=16, expand=64)
    residual = x
    x = fireXcep_module(x, fire_id=3, squeeze=16, expand=64)
    x= layers.add([x, residual])

    residual = x
    x = fireXcep_module(x, fire_id=4, squeeze=16, expand=64)
    x= layers.add([x, residual])
    residual = x
    x = fireXcep_module(x, fire_id=5, squeeze=16, expand=64)
    x= layers.add([x, residual])
    residual = x
    x = fireXcep_module(x, fire_id=6, squeeze=16, expand=64)
    x= layers.add([x, residual])
    residual = x
    x = fireXcep_module(x, fire_id=7, squeeze=16, expand=64)
    x= layers.add([x, residual])
    
    x = fireXcep_module(x, fire_id=8, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    residual = x
    x = fireXcep_module(x, fire_id=9, squeeze=32, expand=128)
    x= layers.add([x, residual])
    residual = x
    x = fireXcep_module(x, fire_id=10, squeeze=32, expand=128)
    x= layers.add([x, residual])
    residual = x
    x = fireXcep_module(x, fire_id=11, squeeze=32, expand=128)
    x= layers.add([x, residual])
    residual = x
    x = fireXcep_module(x, fire_id=12, squeeze=32, expand=128)
    x= layers.add([x, residual])
    residual = x
    x = fireXcep_module(x, fire_id=13, squeeze=32, expand=128)
    x= layers.add([x, residual])
    
    x = fireXcep_module(x, fire_id=14, squeeze=48, expand=192)
    residual = x
    x = fireXcep_module(x, fire_id=15, squeeze=48, expand=192)
    x= layers.add([x, residual])
    residual = x
    x = fireXcep_module(x, fire_id=16, squeeze=48, expand=192)
    x= layers.add([x, residual])
    residual = x
    x = fireXcep_module(x, fire_id=17, squeeze=48, expand=192)
    x= layers.add([x, residual])
    residual = x
    x = fireXcep_module(x, fire_id=18, squeeze=48, expand=192)
    x= layers.add([x, residual])
    
    x = fireXcep_module(x, fire_id=19, squeeze=64, expand=256)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool20')(x) 
    residual = x
    x = fireXcep_module(x, fire_id=20, squeeze=64, expand=256)
    x= layers.add([x, residual])
    residual = x
    x = fireXcep_module(x, fire_id=21, squeeze=64, expand=256)
    x= layers.add([x, residual])
    residual = x
    x = fireXcep_module(x, fire_id=22, squeeze=64, expand=256)
    x= layers.add([x, residual])
    residual = x
    x = fireXcep_module(x, fire_id=23, squeeze=64, expand=256)
    x= layers.add([x, residual])
    residual = x
    x = fireXcep_module(x, fire_id=24, squeeze=64, expand=256)
    x= layers.add([x, residual])

    x = Dropout(0.5, name='drop25')(x)    
    x = Conv2D(nb_classes, (1, 1), padding='valid', name='conv25_park')(x)
    x = BatchNormalization(name='batch_conv25')(x)
    x = Activation('relu', name='relu_conv25')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(inputs=input_img, outputs=[out])
    if weights_path:
        model.load_weights(weights_path)
    return model

def get_sqAllXceNetFinalWide(nb_classes, dim_ordering='tf', weights_path=None):
    mul = 96
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 224, 224))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(224, 224, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only avaiable")
    x = SeparableConv2D(96+mul, (7, 7), strides=(2,2), padding='same', name='conv1_park')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)    
    x = Activation('relu', name='relu_conv1')(x)    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
    x = fireXcep_module(x, fire_id=2, squeeze=32, expand=64+mul)
    x = fireXcep_module(x, fire_id=3, squeeze=32, expand=64+mul)
    x = fireXcep_module(x, fire_id=4, squeeze=64, expand=128+mul)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
    x = fireXcep_module(x, fire_id=5, squeeze=64, expand=128+mul)
    x = fireXcep_module(x, fire_id=6, squeeze=96, expand=192+mul)
    x = fireXcep_module(x, fire_id=7, squeeze=96, expand=192+mul)
    x = fireXcep_module(x, fire_id=8, squeeze=128, expand=256+mul)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
    x = fireXcep_module(x, fire_id=9, squeeze=128, expand=256+mul)
    x = Dropout(0.5, name='drop9')(x)    
    x = Conv2D(nb_classes, (1, 1), padding='valid', name='conv10_park')(x)
    x = BatchNormalization(name='batch_conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(inputs=input_img, outputs=[out])
    if weights_path:
        model.load_weights(weights_path)
    return model

def get_sqAllXceNetFinalWide2(nb_classes, dim_ordering='tf', weights_path=None):
    mul = 13
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 224, 224))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(224, 224, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only avaiable")
    x = SeparableConv2D(96+mul, (7, 7), strides=(2,2), padding='same', name='conv1_park')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)    
    x = Activation('relu', name='relu_conv1')(x)    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
    x = fireXcep_module(x, fire_id=2, squeeze=16, expand=16*mul)
    x = fireXcep_module(x, fire_id=3, squeeze=16, expand=16*mul)
    x = fireXcep_module(x, fire_id=4, squeeze=32, expand=32*mul)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
    x = fireXcep_module(x, fire_id=5, squeeze=32, expand=32*mul)
    x = fireXcep_module(x, fire_id=6, squeeze=48, expand=48*mul)
    x = fireXcep_module(x, fire_id=7, squeeze=48, expand=48*mul)
    x = fireXcep_module(x, fire_id=8, squeeze=64, expand=64*mul)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
    x = fireXcep_module(x, fire_id=9, squeeze=64, expand=256+mul)
    x = Dropout(0.5, name='drop9')(x)    
    x = Conv2D(nb_classes, (1, 1), padding='valid', name='conv10_park')(x)
    x = BatchNormalization(name='batch_conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(inputs=input_img, outputs=[out])
    if weights_path:
        model.load_weights(weights_path)
    return model

def get_sqAllXceNetFinalWide3(nb_classes, dim_ordering='tf', weights_path=None):
    mul = 5
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 224, 224))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(224, 224, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only avaiable")
    x = SeparableConv2D(96+mul, (7, 7), strides=(2,2), padding='same', name='conv1_park')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)    
    x = Activation('relu', name='relu_conv1')(x)    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
    x = fireXcep_module(x, fire_id=2, squeeze=16, expand=16*mul)
    x = fireXcep_module(x, fire_id=3, squeeze=16, expand=16*mul)
    x = fireXcep_module(x, fire_id=4, squeeze=32, expand=32*mul)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
    x = fireXcep_module(x, fire_id=5, squeeze=32, expand=32*mul)
    x = fireXcep_module(x, fire_id=6, squeeze=48, expand=48*mul)
    x = fireXcep_module(x, fire_id=7, squeeze=48, expand=48*mul)
    x = fireXcep_module(x, fire_id=8, squeeze=64, expand=64*mul)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
    x = fireXcep_module(x, fire_id=9, squeeze=64, expand=256*mul)
    x = Dropout(0.5, name='drop9')(x)    
    x = Conv2D(nb_classes, (1, 1), padding='valid', name='conv10_park')(x)
    x = BatchNormalization(name='batch_conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(inputs=input_img, outputs=[out])
    if weights_path:
        model.load_weights(weights_path)
    return model

def get_sqAllXceNetFinalWide4(nb_classes, dim_ordering='tf', weights_path=None):
    srRatio = 5
    initSq = 32
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 224, 224))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(224, 224, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only avaiable")
    x = SeparableConv2D(96*srRatio, (7, 7), strides=(2,2), padding='same', name='conv1_park')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)    
    x = Activation('relu', name='relu_conv1')(x)    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
    x = fireXcep_module(x, fire_id=2, squeeze=initSq, expand=initSq*srRatio)
    x = fireXcep_module(x, fire_id=3, squeeze=initSq, expand=initSq*srRatio)
    initSq = initSq+16
    x = fireXcep_module(x, fire_id=4, squeeze=initSq, expand=initSq*srRatio)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
    x = fireXcep_module(x, fire_id=5, squeeze=initSq, expand=initSq*srRatio)
    initSq = initSq+16
    x = fireXcep_module(x, fire_id=6, squeeze=initSq, expand=initSq*srRatio)
    x = fireXcep_module(x, fire_id=7, squeeze=initSq, expand=initSq*srRatio)
    initSq = initSq+16
    x = fireXcep_module(x, fire_id=8, squeeze=initSq, expand=initSq*srRatio)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
    x = fireXcep_module(x, fire_id=9, squeeze=initSq, expand=initSq*srRatio)
    x = Dropout(0.5, name='drop9')(x)    
    x = Conv2D(nb_classes, (1, 1), padding='valid', name='conv10_park')(x)
    x = BatchNormalization(name='batch_conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(inputs=input_img, outputs=[out])
    if weights_path:
        model.load_weights(weights_path)
    return model

def get_sqAllXceNetFinalDeepCon(nb_classes, dim_ordering='tf', weights_path=None):
    c_axis = 3
    if dim_ordering is 'th':
        input_img = Input(shape=(3, 224, 224))
    elif dim_ordering is 'tf':
        input_img = Input(shape=(224, 224, 3))
    else:
        raise NotImplementedError("Theano and Tensorflow are only avaiable")
    x = SeparableConv2D(96, (7, 7), strides=(2,2), padding='same', name='conv1_park')(input_img)
    # x = Conv2D(64, 3, 3, strides=(2, 2), padding='valid', name='conv1')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)    
    x = Activation('relu', name='relu_conv1')(x)    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
    x = fireXcep_module(x, fire_id=2, squeeze=16, expand=64)
    residual = x
    x = fireXcep_module(x, fire_id=3, squeeze=16, expand=64)
    x= layers.concatenate([x, residual], axis=c_axis)

    residual = x
    x = fireXcep_module(x, fire_id=4, squeeze=16, expand=64)
    x= layers.concatenate([x, residual], axis=c_axis)
    residual = x
    x = fireXcep_module(x, fire_id=5, squeeze=16, expand=64)
    x= layers.concatenate([x, residual], axis=c_axis)
    residual = x
    x = fireXcep_module(x, fire_id=6, squeeze=16, expand=64)
    x= layers.concatenate([x, residual], axis=c_axis)
    residual = x
    x = fireXcep_module(x, fire_id=7, squeeze=16, expand=64)
    x= layers.concatenate([x, residual], axis=c_axis)
    
    x = fireXcep_module(x, fire_id=8, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    residual = x
    x = fireXcep_module(x, fire_id=9, squeeze=32, expand=128)
    x= layers.concatenate([x, residual], axis=c_axis)

    residual = x
    x = fireXcep_module(x, fire_id=10, squeeze=32, expand=128)
    x= layers.concatenate([x, residual], axis=c_axis)
    residual = x
    x = fireXcep_module(x, fire_id=11, squeeze=32, expand=128)
    x= layers.concatenate([x, residual], axis=c_axis)
    
    x = fireXcep_module(x, fire_id=12, squeeze=48, expand=192)
    x= layers.concatenate([x, residual], axis=c_axis)
    residual = x
    x = fireXcep_module(x, fire_id=13, squeeze=48, expand=192)
    x= layers.concatenate([x, residual], axis=c_axis)
    residual = x
    x = fireXcep_module(x, fire_id=14, squeeze=48, expand=192)
    x= layers.concatenate([x, residual], axis=c_axis)
    
    x = fireXcep_module(x, fire_id=15, squeeze=64, expand=256)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool20')(x) 
    residual = x
    x = fireXcep_module(x, fire_id=16, squeeze=64, expand=256)
    x= layers.concatenate([x, residual], axis=c_axis)
    residual = x
    x = fireXcep_module(x, fire_id=17, squeeze=64, expand=256)

    x = Dropout(0.5, name='drop25')(x)    
    x = Conv2D(nb_classes, (1, 1), padding='valid', name='conv25_park')(x)
    x = BatchNormalization(name='batch_conv25')(x)
    x = Activation('relu', name='relu_conv25')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(inputs=input_img, outputs=[out])
    if weights_path:
        model.load_weights(weights_path)
    return model  


def overSqueezeModel(preTrain, classNum, weights_path=None):
    if preTrain == "sz":
        model = get_squeezenet(classNum, weights_path=weights_path)
    elif preTrain =="sw":
        model = get_sqAllXceNetFinalWide4(classNum, weights_path=weights_path)
    elif preTrain =="dd":
        model = get_sqAllXceNetFinalDeep(classNum, weights_path=weights_path)
    elif preTrain =="dc":
        model = get_sqAllXceNetFinalDeepCon(classNum, weights_path=weights_path)
    else:  
        print("model name is wrong.")
        return False
    return model



# Original Squeeze from paper. Updated version from squeezenet paper.
# def get_squeezeXceNet(nb_classes, dim_ordering='tf'):
#     if dim_ordering is 'th':
#         input_img = Input(shape=(3, 224, 224))
#     elif dim_ordering is 'tf':
#         input_img = Input(shape=(224, 224, 3))
#     else:
#         raise NotImplementedError("Theano and Tensorflow are only avaiable")
#     x = Conv2D(96, (7, 7), strides=(2,2), padding='same', name='conv1_park')(input_img)
#     x = BatchNormalization(name='batch_conv1')(x)    
#     x = Activation('relu', name='relu_conv1')(x)    
#     x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
#     x = fireXcep_module(x, fire_id=2, squeeze=16, expand=64)
#     x = fireXcep_module(x, fire_id=3, squeeze=16, expand=64)
#     x = fireXcep_module(x, fire_id=4, squeeze=32, expand=128)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
#     x = fireXcep_module(x, fire_id=5, squeeze=32, expand=128)
#     x = fireXcep_module(x, fire_id=6, squeeze=48, expand=192)
#     x = fireXcep_module(x, fire_id=7, squeeze=48, expand=192)
#     x = fireXcep_module(x, fire_id=8, squeeze=64, expand=256)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
#     x = fireXcep_module(x, fire_id=9, squeeze=64, expand=256)
#     x = Dropout(0.5, name='drop9')(x)    
#     x = Conv2D(nb_classes, (1, 1), padding='valid', name='conv10_park')(x)
#     x = BatchNormalization(name='batch_conv10')(x)
#     x = Activation('relu', name='relu_conv10')(x)
#     x = GlobalAveragePooling2D()(x)
#     out = Activation('softmax', name='loss')(x)
    
#     model = Model(input=input_img, output=[out])

#     return model

# def get_sqWideSr50(nb_classes, dim_ordering='tf'):
#     if dim_ordering is 'th':
#         input_img = Input(shape=(3, 224, 224))
#     elif dim_ordering is 'tf':
#         input_img = Input(shape=(224, 224, 3))
#     else:
#         raise NotImplementedError("Theano and Tensorflow are only avaiable")
#     x = SeparableConv2D(96, (7, 7), strides=(2,2), padding='same', name='conv1_park')(input_img)
#     x = BatchNormalization(name='batch_conv1')(x)    
#     x = Activation('relu', name='relu_conv1')(x)    
#     x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
#     x = fireXcep_module(x, fire_id=2, squeeze=32, expand=64)
#     x = fireXcep_module(x, fire_id=3, squeeze=32, expand=64)
#     x = fireXcep_module(x, fire_id=4, squeeze=64, expand=128)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
#     x = fireXcep_module(x, fire_id=5, squeeze=64, expand=128)
#     x = fireXcep_module(x, fire_id=6, squeeze=128, expand=256)
#     x = fireXcep_module(x, fire_id=7, squeeze=128, expand=256)
#     x = fireXcep_module(x, fire_id=8, squeeze=256, expand=512)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
#     x = fireXcep_module(x, fire_id=9, squeeze=256, expand=512)
#     x = Dropout(0.5, name='drop9')(x)    
#     x = Conv2D(nb_classes, 1, 1, padding='valid', name='conv10_park')(x)
#     x = BatchNormalization(name='batch_conv10')(x)
#     x = Activation('relu', name='relu_conv10')(x)
#     x = GlobalAveragePooling2D()(x)
#     out = Activation('softmax', name='loss')(x)
    
#     model = Model(input=input_img, output=[out])

#     return model


# def get_sqAllXceNet(nb_classes, dim_ordering='tf'):
#     if dim_ordering is 'th':
#         input_img = Input(shape=(3, 224, 224))
#     elif dim_ordering is 'tf':
#         input_img = Input(shape=(224, 224, 3))
#     else:
#         raise NotImplementedError("Theano and Tensorflow are only avaiable")
#     x = SeparableConv2D(96, 7, 7, strides=(2,2), padding='same', name='conv1_park')(input_img)
#     x = BatchNormalization(name='batch_conv1')(x)    
#     x = Activation('relu', name='relu_conv1')(x)    
#     x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
#     x = fireXcep_module(x, fire_id=2, squeeze=16, expand=64)
#     x = fireXcep_module(x, fire_id=3, squeeze=16, expand=64)
#     x = fireXcep_module(x, fire_id=4, squeeze=32, expand=128)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
#     x = fireXcep_module(x, fire_id=5, squeeze=32, expand=128)
#     x = fireXcep_module(x, fire_id=6, squeeze=48, expand=192)
#     x = fireXcep_module(x, fire_id=7, squeeze=48, expand=192)
#     x = fireXcep_module(x, fire_id=8, squeeze=64, expand=256)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
#     x = fireXcep_module(x, fire_id=9, squeeze=64, expand=256)
#     x = Dropout(0.5, name='drop9')(x)    
#     x = Conv2D(nb_classes, 1, 1, padding='valid', name='conv10_park')(x)
#     x = BatchNormalization(name='batch_conv10')(x)
#     x = Activation('relu', name='relu_conv10')(x)
#     x = GlobalAveragePooling2D()(x)
#     out = Activation('softmax', name='loss')(x)
    
#     model = Model(input=input_img, output=[out])

#     return model

# def get_sqAllXceNet_depth(nb_classes, dim_ordering='tf'):
#     if dim_ordering is 'th':
#         input_img = Input(shape=(3, 224, 224))
#     elif dim_ordering is 'tf':
#         input_img = Input(shape=(224, 224, 3))
#     else:
#         raise NotImplementedError("Theano and Tensorflow are only avaiable")
#     x = SeparableConv2D(96, 7, 7, strides=(2,2), padding='same', name='conv1_park')(input_img)
#     x = BatchNormalization(name='batch_conv1')(x)    
#     x = Activation('relu', name='relu_conv1')(x)    
#     x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
#     x = fireXcep_module2(x, fire_id=2, squeeze=16, expand=64)
#     x = fireXcep_module2(x, fire_id=3, squeeze=16, expand=64)
#     x = fireXcep_module2(x, fire_id=4, squeeze=32, expand=128)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
#     x = fireXcep_module2(x, fire_id=5, squeeze=32, expand=128)
#     x = fireXcep_module2(x, fire_id=6, squeeze=48, expand=192)
#     x = fireXcep_module2(x, fire_id=7, squeeze=48, expand=192)
#     x = fireXcep_module2(x, fire_id=8, squeeze=64, expand=256)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
#     x = fireXcep_module2(x, fire_id=9, squeeze=64, expand=256)
#     x = Dropout(0.5, name='drop9')(x)    
#     x = Conv2D(nb_classes, 1, 1, padding='valid', name='conv10_park')(x)
#     x = BatchNormalization(name='batch_conv10')(x)
#     x = Activation('relu', name='relu_conv10')(x)
#     x = GlobalAveragePooling2D()(x)
#     out = Activation('softmax', name='loss')(x)
    
#     model = Model(input=input_img, output=[out])

#     return model


# def get_sqAllXceNet_2(nb_classes, dim_ordering='tf'):
#     if dim_ordering is 'th':
#         input_img = Input(shape=(3, 224, 224))
#     elif dim_ordering is 'tf':
#         input_img = Input(shape=(224, 224, 3))
#     else:
#         raise NotImplementedError("Theano and Tensorflow are only avaiable")
#     x = SeparableConv2D(96, 7, 7, strides=(2,2), padding='same', name='conv1_park')(input_img)
#     x = BatchNormalization(name='batch_conv1')(x)    
#     x = Activation('relu', name='relu_conv1')(x)    
#     x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
#     x = fireXcep_module(x, fire_id=2, squeeze=16, expand=64)
#     x = fireXcep_module(x, fire_id=3, squeeze=16, expand=64)
#     x = fireXcep_module(x, fire_id=4, squeeze=16, expand=64)
#     x = fireXcep_module(x, fire_id=5, squeeze=32, expand=128)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
#     x = fireXcep_module(x, fire_id=6, squeeze=32, expand=128)
#     x = fireXcep_module(x, fire_id=7, squeeze=32, expand=128)
#     x = fireXcep_module(x, fire_id=8, squeeze=48, expand=192)
#     x = fireXcep_module(x, fire_id=9, squeeze=48, expand=192)
#     x = fireXcep_module(x, fire_id=10, squeeze=48, expand=192)
#     x = fireXcep_module(x, fire_id=11, squeeze=64, expand=256)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
#     x = fireXcep_module(x, fire_id=12, squeeze=64, expand=256)
#     x = fireXcep_module(x, fire_id=13, squeeze=64, expand=256)
#     x = Dropout(0.5, name='drop9')(x)    
#     x = Conv2D(nb_classes, 1, 1, padding='valid', name='conv10_park')(x)
#     x = BatchNormalization(name='batch_conv10')(x)
#     x = Activation('relu', name='relu_conv10')(x)
#     x = GlobalAveragePooling2D()(x)
#     out = Activation('softmax', name='loss')(x)
    
#     model = Model(input=input_img, output=[out])

#     return model


# def get_sqAllXceNetSr50(nb_classes, dim_ordering='tf'):
#     if dim_ordering is 'th':
#         input_img = Input(shape=(3, 224, 224))
#     elif dim_ordering is 'tf':
#         input_img = Input(shape=(224, 224, 3))
#     else:
#         raise NotImplementedError("Theano and Tensorflow are only avaiable")
#     x = SeparableConv2D(96, 7, 7, strides=(2,2), padding='same', name='conv1_park')(input_img)
#     x = BatchNormalization(name='batch_conv1')(x)    
#     x = Activation('relu', name='relu_conv1')(x)    
#     x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
#     x = fireXcep_module(x, fire_id=2, squeeze=32, expand=64)
#     x = fireXcep_module(x, fire_id=3, squeeze=32, expand=64)
#     x = fireXcep_module(x, fire_id=4, squeeze=64, expand=128)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
#     x = fireXcep_module(x, fire_id=5, squeeze=64, expand=128)
#     x = fireXcep_module(x, fire_id=6, squeeze=96, expand=192)
#     x = fireXcep_module(x, fire_id=7, squeeze=96, expand=192)
#     x = fireXcep_module(x, fire_id=8, squeeze=128, expand=256)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
#     x = fireXcep_module(x, fire_id=9, squeeze=128, expand=256)
#     x = Dropout(0.5, name='drop9')(x)    
#     x = Conv2D(nb_classes, 1, 1, padding='valid', name='conv10_park')(x)
#     x = BatchNormalization(name='batch_conv10')(x)
#     x = Activation('relu', name='relu_conv10')(x)
#     x = GlobalAveragePooling2D()(x)
#     out = Activation('softmax', name='loss')(x)
    
#     model = Model(input=input_img, output=[out])

#     return model

# Original Squeeze from paper. Updated version from squeezenet paper.
# def get_sqXceNetRes(nb_classes, dim_ordering='tf'):
#     if dim_ordering is 'th':
#         input_img = Input(shape=(3, 224, 224))
#     elif dim_ordering is 'tf':
#         input_img = Input(shape=(224, 224, 3))
#     else:
#         raise NotImplementedError("Theano and Tensorflow are only avaiable")
#     x = Conv2D(96, 7, 7, strides=(2,2), padding='same', name='conv1_park')(input_img)
#     # x = Conv2D(64, 3, 3, strides=(2, 2), padding='valid', name='conv1')(input_img)
#     x = BatchNormalization(name='batch_conv1')(x)    
#     x = Activation('relu', name='relu_conv1')(x)    
#     x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
#     x = fireXcep_module(x, fire_id=2, squeeze=16, expand=64)
#     residual = x
#     x = fireXcep_module(x, fire_id=3, squeeze=16, expand=64)
#     x = merge([x, residual], mode='sum')
#     x = fireXcep_module(x, fire_id=4, squeeze=32, expand=128)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
#     residual = x
#     x = fireXcep_module(x, fire_id=5, squeeze=32, expand=128)
#     x = merge([x, residual], mode='sum')
#     x = fireXcep_module(x, fire_id=6, squeeze=48, expand=192)

#     residual = x
#     x = fireXcep_module(x, fire_id=7, squeeze=48, expand=192)
#     x = merge([x, residual], mode='sum')

#     x = fireXcep_module(x, fire_id=8, squeeze=64, expand=256)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
#     residual = x
#     x = fireXcep_module(x, fire_id=9, squeeze=64, expand=256)
#     x = merge([x, residual], mode='sum')
    
#     x = Dropout(0.5, name='drop9')(x)    
#     x = Conv2D(nb_classes, 1, 1, padding='valid', name='conv10_park')(x)
#     x = BatchNormalization(name='batch_conv10')(x)
#     x = Activation('relu', name='relu_conv10')(x)
#     x = GlobalAveragePooling2D()(x)
#     out = Activation('softmax', name='loss')(x)
    
#     model = Model(input=input_img, output=[out])

#     return model  

# def get_sqAllXResNet(nb_classes, dim_ordering='tf'):
#     if dim_ordering is 'th':
#         input_img = Input(shape=(3, 224, 224))
#     elif dim_ordering is 'tf':
#         input_img = Input(shape=(224, 224, 3))
#     else:
#         raise NotImplementedError("Theano and Tensorflow are only avaiable")
#     x = SeparableConv2D(96, 7, 7, strides=(2,2), padding='same', name='conv1_park')(input_img)
#     # x = Conv2D(64, 3, 3, strides=(2, 2), padding='valid', name='conv1')(input_img)
#     x = BatchNormalization(name='batch_conv1')(x)    
#     x = Activation('relu', name='relu_conv1')(x)    
#     x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
#     x = fireXcep_module(x, fire_id=2, squeeze=16, expand=64)
#     residual = x
#     x = fireXcep_module(x, fire_id=3, squeeze=16, expand=64)
#     x = merge([x, residual], mode='sum')
#     x = fireXcep_module(x, fire_id=4, squeeze=32, expand=128)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    
#     residual = x
#     x = fireXcep_module(x, fire_id=5, squeeze=32, expand=128)
#     x = merge([x, residual], mode='sum')
#     x = fireXcep_module(x, fire_id=6, squeeze=48, expand=192)

#     residual = x
#     x = fireXcep_module(x, fire_id=7, squeeze=48, expand=192)
#     x = merge([x, residual], mode='sum')

#     x = fireXcep_module(x, fire_id=8, squeeze=64, expand=256)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool8')(x)
    
#     residual = x
#     x = fireXcep_module(x, fire_id=9, squeeze=64, expand=256)
#     x = merge([x, residual], mode='sum')
    
#     x = Dropout(0.5, name='drop9')(x)    
#     x = Conv2D(nb_classes, 1, 1, padding='valid', name='conv10_park')(x)
#     x = BatchNormalization(name='batch_conv10')(x)
#     x = Activation('relu', name='relu_conv10')(x)
#     x = GlobalAveragePooling2D()(x)
#     out = Activation('softmax', name='loss')(x)
    
#     model = Model(input=input_img, output=[out])

#     return model  

# def get_sqAllXResNet_2(nb_classes, dim_ordering='tf'):
#     if dim_ordering is 'th':
#         input_img = Input(shape=(3, 224, 224))
#     elif dim_ordering is 'tf':
#         input_img = Input(shape=(224, 224, 3))
#     else:
#         raise NotImplementedError("Theano and Tensorflow are only avaiable")
#     x = SeparableConv2D(96, 7, 7, strides=(2,2), padding='same', name='conv1_park')(input_img)
#     # x = Conv2D(64, 3, 3, strides=(2, 2), padding='valid', name='conv1')(input_img)
#     x = BatchNormalization(name='batch_conv1')(x)    
#     x = Activation('relu', name='relu_conv1')(x)    
#     x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
#     x = fireXcep_module(x, fire_id=2, squeeze=16, expand=64)
#     residual = x
#     x = fireXcep_module(x, fire_id=3, squeeze=16, expand=64)
#     x = merge([x, residual], mode='sum')
    
#     residual = x
#     x = fireXcep_module(x, fire_id=4, squeeze=16, expand=64)
#     x = merge([x, residual], mode='sum')
    
#     x = fireXcep_module(x, fire_id=5, squeeze=32, expand=128)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)
    
#     residual = x
#     x = fireXcep_module(x, fire_id=6, squeeze=32, expand=128)
#     x = merge([x, residual], mode='sum')
    
#     residual = x
#     x = fireXcep_module(x, fire_id=7, squeeze=32, expand=128)
#     x = merge([x, residual], mode='sum')

#     x = fireXcep_module(x, fire_id=8, squeeze=48, expand=192)

#     residual = x
#     x = fireXcep_module(x, fire_id=9, squeeze=48, expand=192)
#     x = merge([x, residual], mode='sum')
    
#     residual = x
#     x = fireXcep_module(x, fire_id=10, squeeze=48, expand=192)
#     x = merge([x, residual], mode='sum')

#     x = fireXcep_module(x, fire_id=11, squeeze=64, expand=256)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool9')(x)
    
#     residual = x
#     x = fireXcep_module(x, fire_id=12, squeeze=64, expand=256)
#     x = merge([x, residual], mode='sum')

#     residual = x
#     x = fireXcep_module(x, fire_id=13, squeeze=64, expand=256)
#     x = merge([x, residual], mode='sum')

#     x = Dropout(0.5, name='drop9')(x)    
#     x = Conv2D(nb_classes, 1, 1, padding='valid', name='conv14_park')(x)
#     x = BatchNormalization(name='batch_conv10')(x)
#     x = Activation('relu', name='relu_conv10')(x)
#     x = GlobalAveragePooling2D()(x)
#     out = Activation('softmax', name='loss')(x)
    
#     model = Model(input=input_img, output=[out])

#     return model  



#%%    
if __name__ == '__main__':
    import time
#    from keras.utils.visualize_util import plot
    start = time.time()
    model = get_squeezeXceNet(2)
    duration = time.time() - start
    print "{} s to make model".format(duration)
    
    start = time.time()
    model.output
    duration = time.time()- start
    print "{} s to get output.".format(duration)
    
    start = time.time()
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    duration = time.time() - start
    print "{} s to get compile".format(duration)
    
#    plot(model, to_file='images/SqueezeNet_new.png', show_shapes=True)
    
    
    
    
    
    
    
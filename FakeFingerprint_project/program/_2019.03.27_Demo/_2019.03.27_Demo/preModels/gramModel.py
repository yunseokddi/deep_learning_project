from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from scipy.misc import imread, imresize, imsave
from customlayers import Softmax4D, gramMatrix
from keras.layers import Input,BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D, AveragePooling2D
from keras import layers
from keras.layers.advanced_activations import LeakyReLU
#%%
sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"
batchN = "batch_"

def fire_module(x, fire_id, squeeze=16, expand=64, dim_ordering='tf'):
    s_id = 'fire'+str(fire_id)+'/'
    if dim_ordering is 'tf':
        c_axis = 3
    else:
        c_axis = 1

    x = Conv2D(squeeze, (1, 1), padding='valid', name=s_id+sq1x1)(x)
    x = BatchNormalization(name=s_id+batchN+sq1x1)(x)
    x = LeakyReLU(name=s_id+relu+sq1x1)(x)
    
    left = Conv2D(expand, (1, 1), padding='valid', name=s_id+exp1x1)(x)
    x = BatchNormalization(name=s_id+batchN+exp1x1)(x)
    left = LeakyReLU(name=s_id+relu+exp1x1)(left)
    
    right = Conv2D(expand, (3, 3), padding='same', name=s_id+exp3x3)(x)
    x = BatchNormalization(name=s_id+batchN+exp3x3)(x)
    right = LeakyReLU(name=s_id+relu+exp3x3)(right)
    
    x = layers.concatenate([left, right], axis=c_axis, name=s_id+'concat')
    return x

def gramModule(x, inputSize, gramId):
    s_id = 'gram_'+str(gramId)+'/'
    x = Conv2D(inputSize, (1, 1), padding='valid', name=s_id+"conv")(x)
    x = BatchNormalization(name=s_id+"batchNorm")(x)
    x = LeakyReLU(name=s_id+"relu")(x)
    x = gramMatrix()(x)
    return x

def veryShallowGramModel(input_shape, nb_classes = 2, weights_path=None, gram_size=128):
    if input_shape:
        input_img=Input(shape=input_shape+(1,))        
    else:
        input_img=Input(shape=(None, None, 1))

    # This part shoud be 7x7 and stride 2
    x = Conv2D(64, (3, 3), padding='valid',name='conv1')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)
    x = LeakyReLU(name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2,2), name='pool1')(x)
    x = Conv2D(128, (3, 3), padding='valid',name='conv2')(input_img)
    x = BatchNormalization(name='batch_conv2')(x)
    x = LeakyReLU(name='relu_conv2')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2,2), name='pool2')(x)
    x = Conv2D(128, (3, 3), padding='valid',name='conv3')(input_img)
    x = LeakyReLU(name='relu_conv3')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2,2), name='pool3')(x)
    x = BatchNormalization(name='batch_conv3')(x)
    x = gramModule(x, inputSize=gram_size, gramId=2)
    x = Flatten(name='flatten')(x)
    x = Dense(128, name='dense_1')(x)  
    x = BatchNormalization(name='batch_dense1')(x)
    x = LeakyReLU(name='relu_dense1')(x)
    x = Dropout(0.5, name='drop8')(x)
    x = Dense(nb_classes, activation='softmax', name='predic')(x)
    model = Model(inputs=input_img, outputs=x)
    if weights_path:
        model.load_weights(weights_path)
    return model

def fireGramModel(input_shape, nb_classes = 2, weights_path=None, gram_size=32):
    if input_shape:
        input_img=Input(shape=input_shape+(1,))        
    else:
        input_img=Input(shape=(None, None, 1))
    # else:
    #     raise NotImplementedError("Theano and Tensorflow are only avaiable")
    x = Conv2D(96, (7, 7), strides=(2,2), padding='same',name='conv1_park')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)
    # x = Conv2D(64, 3, 3, strides=(2, 2), padding='valid', name='conv1')(input_img)    
    x = LeakyReLU(name='relu_conv1')(x)    
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
    x = gramModule(x, inputSize=gram_size, gramId=2)
    x = Conv2D(nb_classes, (1, 1), padding='valid', name='conv10_park')(x)
    x = BatchNormalization(name='batch_conv10')(x)
    x = LeakyReLU(name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(inputs=input_img, outputs=[out])

    return model

def shallowGramModel(input_shape, nb_classes = 2, weights_path=None, gram_size=128):
    if input_shape:
        input_img=Input(shape=input_shape+(1,))        
    else:
        input_img=Input(shape=(None, None, 1))

    # This part shoud be 7x7 and stride 2
    x = Conv2D(256, (7, 7), strides=(2,2), padding='same',name='conv1')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)
    x = LeakyReLU(name='relu_conv1')(x)
    x = LeakyReLU('relu', name='relu_conv2')(x)
    x = Conv2D(256, (7, 7), strides=(2,2), padding='same',name='conv3')(x)
    x = BatchNormalization(name='batch_conv3')(x)
    x = LeakyReLU(name='relu_conv3')(x)
    x = AveragePooling2D(pool_size=(3, 3), strides=(2,2), name='avg_pool4')(x)
    x = fire_module(x, fire_id=5, squeeze=16, expand=64)
    x = AveragePooling2D(pool_size=(3, 3), strides=(2,2), name='avg_pool5')(x)
    x = fire_module(x, fire_id=6, squeeze=16, expand=64)
    x = gramModule(x, inputSize=gram_size, gramId=2)
    x = Conv2D(nb_classes, (1, 1), padding='valid', name='conv7')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    model = Model(inputs=input_img, outputs=[out])
    if weights_path:
        model.load_weights(weights_path)
    return model

def get_smallGramModel(input_shape, nb_classes = 2, weights_path=None, gram_size=128):
    if input_shape:
        input_img=Input(shape=input_shape+(1,))        
    else:
        input_img=Input(shape=(None, None, 1))

    x = Conv2D(96, (3, 3), strides=(2,2), padding='same',name='conv1')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)
    x = LeakyReLU(name='relu_conv1')(x)
    gram_1 = gramModule(x, inputSize=gram_size, gramId=3)
    x = fire_module(x, fire_id=5, squeeze=48, expand=192)
    # gram_2 = gramModule(x, inputSize=gram_size, gramId=5)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    x = fire_module(x, fire_id=6, squeeze=64, expand=256)
    # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool6')(x)
    gram_3 = gramModule(x, inputSize=gram_size, gramId=6)
    gram_final = layers.concatenate([gram_1, gram_3], axis=3, name='gram_concat')
    # gram_final = merge([gram_1, gram_2, gram_3, gram_4, gram_5], mode='concat', concat_axis=3, name='gram_concat')
    grammer = fire_module(gram_final, fire_id=7, squeeze=16, expand=64)
    grammer = MaxPooling2D(pool_size=(3, 3), strides=(2,2), name='avg_pool7')(grammer)
    grammer = fire_module(grammer, fire_id=8, squeeze=32, expand=128)
    grammer = Dropout(0.5, name='drop8')(grammer)  
    grammer = MaxPooling2D(pool_size=(3, 3), strides=(2,2), name='avg_pool8')(grammer)
    grammer = Conv2D(nb_classes, (1, 1), padding='valid', name='conv9')(grammer)
    grammer = BatchNormalization(name='batch_conv10')(grammer)
    grammer = LeakyReLU(name='relu_conv10')(grammer)
    grammer = GlobalAveragePooling2D()(grammer)
    out = Activation('softmax', name='loss')(grammer)
    
    model = Model(inputs=input_img, outputs=[out])
    if weights_path:
        model.load_weights(weights_path)
    return model

def get_MultipleGramModel(input_shape, nb_classes = 2, weights_path=None, gram_size=64):
    if input_shape:
        input_img=Input(shape=input_shape+(1,))        
    else:
        input_img=Input(shape=(None, None, 1))

    x = Conv2D(96, (3, 3), strides=(2,2), padding='same',name='conv1')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)
    x = LeakyReLU(name='relu_conv1')(x)
    # gram_1 = gramModule(x, inputSize=gram_size, gramId=2)

    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    # gram_2 = gramModule(x, inputSize=gram_size, gramId=3)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    gram_3 = gramModule(x, inputSize=gram_size, gramId=4)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    x = fire_module(x, fire_id=5, squeeze=48, expand=192)
    # x = Dropout(0.5, name='drop9')(x)  
    gram_4 = gramModule(x, inputSize=gram_size, gramId=5)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)
    x = fire_module(x, fire_id=6, squeeze=64, expand=256)
    gram_5 = gramModule(x, inputSize=gram_size, gramId=6)
    gram_final = layers.concatenate([gram_3, gram_4, gram_5], axis=3, name='gram_concat')
    # gram_final = merge([gram_1, gram_2, gram_3, gram_4, gram_5], mode='concat', concat_axis=3, name='gram_concat')
    grammer = fire_module(gram_final, fire_id=7, squeeze=16, expand=64)
    grammer = MaxPooling2D(pool_size=(3, 3), strides=(2,2), name='avg_pool7')(grammer)
    grammer = fire_module(grammer, fire_id=8, squeeze=32, expand=128)
    grammer = Dropout(0.5, name='drop8')(grammer)  
    grammer = MaxPooling2D(pool_size=(3, 3), strides=(2,2), name='avg_pool8')(grammer)
    grammer = Conv2D(nb_classes, (1, 1), padding='valid', name='conv9')(grammer)
    grammer = BatchNormalization(name='batch_conv10')(grammer)
    grammer = LeakyReLU(name='relu_conv10')(grammer)
    grammer = GlobalAveragePooling2D()(grammer)
    out = Activation('softmax', name='loss')(grammer)
    
    model = Model(inputs=input_img, outputs=[out])
    if weights_path:
        model.load_weights(weights_path)
    return model
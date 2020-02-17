from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from scipy.misc import imread, imresize, imsave
from .customlayers import Softmax4D, gramMatrix
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
    left = BatchNormalization(name=s_id+batchN+exp1x1)(left)
    left = LeakyReLU(name=s_id+relu+exp1x1)(left)
    
    right = Conv2D(expand, (3, 3), padding='same', name=s_id+exp3x3)(x)
    right = BatchNormalization(name=s_id+batchN+exp3x3)(right)
    right = LeakyReLU(name=s_id+relu+exp3x3)(right)
    
    x = layers.concatenate([left, right], axis=c_axis, name=s_id+'concat')
    return x

def gramModule(x, inputSize, gramId):
    s_id = 'gram_'+str(gramId)+'/'
    x = Conv2D(inputSize, (1, 1), padding='valid', name=s_id+"conv")(x)
    x = BatchNormalization(name=s_id+"batchNorm")(x)
    x = Activation('tanh',name=s_id+"relu")(x)
    x = gramMatrix()(x)
    return x


def gramClassifier(gram_final, fire_id, nb_classes=2, numSq=16, maxpool=True, depth=1):
    if maxpool: Rectifier = MaxPooling2D
    else: Rectifier = AveragePooling2D
    
    if depth==1:
        grammer = fire_module(gram_final, fire_id=fire_id, squeeze=numSq, expand=numSq*4)
        grammer = Rectifier(pool_size=(3, 3), strides=(2,2), name='pool_'+str(fire_id))(grammer)
        grammer = fire_module(grammer, fire_id=fire_id+1, squeeze=numSq*2, expand=numSq*8)
        grammer = Dropout(0.5, name='drop'+str(fire_id+1))(grammer)  
        grammer = Rectifier(pool_size=(3, 3), strides=(2,2), name='pool_'+str(fire_id+1))(grammer)
    if depth == 2:
        grammer = fire_module(gram_final, fire_id=fire_id, squeeze=numSq, expand=numSq*4)
        grammer = fire_module(grammer, fire_id=fire_id+1, squeeze=numSq, expand=numSq*4)
        grammer = Rectifier(pool_size=(3, 3), strides=(2,2), name='pool_'+str(fire_id+1))(grammer)
        grammer = fire_module(grammer, fire_id=fire_id+2, squeeze=numSq*2, expand=numSq*8)
        grammer = Dropout(0.5, name='drop'+str(fire_id+2))(grammer)  
        grammer = Rectifier(pool_size=(3, 3), strides=(2,2), name='pool_'+str(fire_id+2))(grammer)
    if depth == 3:
        grammer = fire_module(gram_final, fire_id=fire_id, squeeze=numSq, expand=numSq*4)
        grammer = fire_module(grammer, fire_id=fire_id+1, squeeze=numSq, expand=numSq*4)
        grammer = Rectifier(pool_size=(3, 3), strides=(2,2), name='pool_'+str(fire_id+1))(grammer)
        grammer = fire_module(grammer, fire_id=fire_id+2, squeeze=numSq*2, expand=numSq*8)
        grammer = fire_module(grammer, fire_id=fire_id+3, squeeze=numSq*2, expand=numSq*8)
        grammer = Dropout(0.5, name='drop'+str(fire_id+3))(grammer)  
        grammer = Rectifier(pool_size=(3, 3), strides=(2,2), name='pool_'+str(fire_id+3))(grammer)

    grammer = Conv2D(nb_classes, (1, 1), padding='valid', name='conv_'+str(fire_id+4))(grammer)
    grammer = BatchNormalization(name='batcvlabch_'+str(fire_id+4))(grammer)
    grammer = LeakyReLU(name='relu_'+str(fire_id+4))(grammer)
    grammer = GlobalAveragePooling2D()(grammer)
    out = Activation('softmax', name='loss')(grammer)
    return out




def get_smallGramModel(input_shape, gLoc, nb_classes = 2, weights_path=None, gram_size=128, maxpool=True, numSq=16, depth=1):
    if input_shape:
        input_img=Input(shape=input_shape+(1,))        
    else:
        input_img=Input(shape=(None, None, 1))

    x = Conv2D(96, (3, 3), padding='same',name='conv1')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)
    x = LeakyReLU(name='relu_conv1')(x)
    gram_1 = gramModule(x, inputSize=gram_size, gramId=1)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool2')(x)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    gram_2 = gramModule(x, inputSize=gram_size, gramId=2)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool3')(x)
    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    gram_3 = gramModule(x, inputSize=gram_size, gramId=3)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    x = fire_module(x, fire_id=5, squeeze=48, expand=192)
    gram_4 = gramModule(x, inputSize=gram_size, gramId=4)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)
    x = fire_module(x, fire_id=6, squeeze=64, expand=256)
    gram_5 = gramModule(x, inputSize=gram_size, gramId=5)
    grams = [gram_1, gram_2, gram_3, gram_4, gram_5]
    gramCat = []
    for index, put in enumerate(gLoc):
        if put == 1:
            gramCat.append(grams[index])
    gram_final = layers.concatenate(gramCat, axis=3, name='gram_concat')
    out = gramClassifier(gram_final, fire_id=7, numSq=numSq, maxpool=maxpool, depth=depth)
    
    model = Model(inputs=input_img, outputs=[out])
    if weights_path:
        model.load_weights(weights_path, by_name=True)
    return model

def get_MultipleGramModel(input_shape, gLoc, nb_classes = 2, weights_path=None, gram_size=128, depth=1, maxpool=True, numSq=16,):
    if input_shape:
        input_img=Input(shape=input_shape+(1,))        
    else:
        input_img=Input(shape=(None, None, 1))

    x = Conv2D(96, (7, 7), strides=(2,2), padding='same',name='conv1')(input_img)
    x = BatchNormalization(name='batch_conv1')(x)
    x = LeakyReLU(name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    gram_1 = gramModule(x, inputSize=gram_size, gramId=2)

    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)
    gram_2 = gramModule(x, inputSize=gram_size, gramId=3)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    gram_3 = gramModule(x, inputSize=gram_size, gramId=4)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool4')(x)
    x = fire_module(x, fire_id=5, squeeze=48, expand=192)
    # x = Dropout(0.5, name='drop9')(x)  
    gram_4 = gramModule(x, inputSize=gram_size, gramId=5)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)
    x = fire_module(x, fire_id=6, squeeze=64, expand=256)
    gram_5 = gramModule(x, inputSize=gram_size, gramId=6)
    grams = [gram_1, gram_2, gram_3, gram_4, gram_5]
    gramCat = []
    for index, put in enumerate(gLoc):
        if put == 1:
            gramCat.append(grams[index])
    gram_final = layers.concatenate(gramCat, axis=3, name='gram_concat')
    # gram_final = merge([gram_1, gram_2, gram_3, gram_4, gram_5], mode='concat', concat_axis=3, name='gram_concat')
    out = gramClassifier(gram_final, fire_id=7, numSq=numSq, maxpool=maxpool, depth=depth)
    
    model = Model(inputs=input_img, outputs=[out])
    if weights_path:
        model.load_weights(weights_path, by_name=True)
    return model

def new_gram_models(preTrain, target_size, weights_path=None, gram_size=128):
    if preTrain == 'gaa':
        gLoc = [1,1,0,0,0]
        model = get_MultipleGramModel(input_shape=target_size, depth=1, weights_path=weights_path, gLoc=gLoc)
    elif preTrain == 'gab':
        gLoc = [1,1,1,0,0]
        model = get_MultipleGramModel(input_shape=target_size, depth=1, weights_path=weights_path, gLoc=gLoc)
    elif preTrain == 'gac': 
        gLoc = [1,1,1,1,0]
        model = get_MultipleGramModel(input_shape=target_size, depth=1, weights_path=weights_path, gLoc=gLoc)
    elif preTrain == 'gad':
        gLoc = [1,1,1,1,1]
        model = get_MultipleGramModel(input_shape=target_size, depth=1, weights_path=weights_path, gLoc=gLoc)
    elif preTrain == 'gae':
        gLoc = [0,0,0,0,1]
        model = get_MultipleGramModel(input_shape=target_size, depth=1, weights_path=weights_path, gLoc=gLoc)
    elif preTrain == 'gaf': 
        gLoc = [0,0,0,1,1]
        model = get_MultipleGramModel(input_shape=target_size, depth=1, weights_path=weights_path, gLoc=gLoc)
    elif preTrain == 'gag': 
        gLoc = [0,0,1,1,1]
        model = get_MultipleGramModel(input_shape=target_size, depth=1, weights_path=weights_path, gLoc=gLoc)
    elif preTrain == 'gah': 
        gLoc = [0,1,1,1,1]
        model = get_MultipleGramModel(input_shape=target_size, depth=1, weights_path=weights_path, gLoc=gLoc)
    elif preTrain == 'gai': 
        gLoc = [1,0,1,0,1]
        model = get_MultipleGramModel(input_shape=target_size, depth=1, weights_path=weights_path, gLoc=gLoc)
    elif preTrain == 'gaj': 
        gLoc = [1,1,0,1,0]
        model = get_MultipleGramModel(input_shape=target_size, depth=1, weights_path=weights_path, gLoc=gLoc)
    elif preTrain == 'gak': 
        gLoc = [0,1,0,1,0]
        model = get_MultipleGramModel(input_shape=target_size, depth=1, weights_path=weights_path, gLoc=gLoc)
    elif preTrain == 'gba': 
        gLoc = [1,1,0,0,0]
        model = get_smallGramModel(input_shape=target_size, depth=1, weights_path=weights_path, gLoc=gLoc)
    elif preTrain == 'gbb': 
        gLoc = [1,1,1,0,0]
        model = get_smallGramModel(input_shape=target_size, depth=1, weights_path=weights_path, gLoc=gLoc)
    elif preTrain == 'gbc':
        gLoc = [1,1,1,1,0]
        model = get_smallGramModel(input_shape=target_size, depth=1, weights_path=weights_path, gLoc=gLoc)
    elif preTrain == 'gbd':
        gLoc = [1,1,1,1,1]
        model = get_smallGramModel(input_shape=target_size, depth=1, weights_path=weights_path, gLoc=gLoc)
    elif preTrain == 'gbe': 
        gLoc = [0,0,0,1,1]
        model = get_smallGramModel(input_shape=target_size, depth=1, weights_path=weights_path, gLoc=gLoc)
    elif preTrain == 'gbf':
        gLoc = [0,0,1,1,1] 
        model = get_smallGramModel(input_shape=target_size, depth=1, weights_path=weights_path, gLoc=gLoc)
    elif preTrain == 'gbg':
        gLoc = [0,1,1,1,1]
    elif preTrain == 'gbi': 
        gLoc = [1,0,1,0,1]
        model = get_smallGramModel(input_shape=target_size, depth=1, weights_path=weights_path, gLoc=gLoc)
    elif preTrain == 'gbj': 
        gLoc = [1,1,0,1,0]
        model = get_smallGramModel(input_shape=target_size, depth=1, weights_path=weights_path, gLoc=gLoc)
    elif preTrain == 'gbk': 
        gLoc = [0,1,0,1,0]
        model = get_smallGramModel(input_shape=target_size, depth=1, weights_path=weights_path, gLoc=gLoc)
    else:
        return False
    return model
















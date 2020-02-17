import numpy as np
from pandas import DataFrame
from pandas import read_csv
import os
import json
import matplotlib.pyplot as plt
from random import shuffle
from scipy.misc import imread
#import random

def returnSubImageList(testDir):
    """
        return list of imagepaths of all entire sub directory.
    """
    allTestFile = []
    extList = ['.bmp', '.png', '.gif', '.jpg', '.jpeg']
    for (path, dir, files) in os.walk(testDir):
        for fileName in files:
#            print fileName
            ext = os.path.splitext(fileName)[1]
            if ext.lower() not in extList: # extension check
                continue
            else:
                imgPath = os.path.join(path, fileName)
                allTestFile.append(imgPath)
    return allTestFile


def makeOneShotFigure(source):
    datas = read_csv(source)
    # summarize history for accuracy
    fig = plt.Figure()
    fig.set_canvas(plt.gcf().canvas)
    plt.plot(datas['loss'])
    plt.plot(datas['val_loss'])
    #plt.title('model accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    figureName = os.path.basename(source)
    figureName = figureName[:figureName.rfind('.')]+'.png'
    plt.savefig(os.path.join(os.path.dirname(source), figureName))
    plt.gcf().clear()
    
def makeResultFigure(source):
    """
        from the source path (CSV file), draw graph and save it
    """
    datas = read_csv(source)
    
    # summarize history for accuracy
    fig = plt.Figure()
    fig.set_canvas(plt.gcf().canvas)
    plt.subplot(211)
    plt.plot(datas['acc'])
    plt.plot(datas['val_acc'])
    #plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    
    # summarize history for loss
    plt.subplot(212)
    plt.plot(datas['loss'])
    plt.plot(datas['val_loss'])
    #plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    figureName = os.path.basename(source)
    figureName = figureName[:figureName.rfind('.')]+'.png'
    plt.savefig(os.path.join(os.path.dirname(source), figureName))
    plt.gcf().clear()


def jsonDumper(fileName, jsonString):
    with open(fileName, 'w') as f:
        f.write(json.dumps(jsonString))
    return 

def classReturnFromFolder(folder):
    if os.path.isdir(folder):
        direc = os.listdir(folder)
    else:
        print ("Can not find %s directory" % folder)
        return None
    return sorted(direc)

def getSamplePerEpoch(folder, label):
    folders = [os.path.join(folder, lb) for lb in label]
    numFiles = 0
    for ff in folders:
        if os.path.isdir(ff): numFiles += len(os.listdir(ff))
        else:
            print("No folder {}.".format(ff))
            return None
    return numFiles
    # numFiles = 0
    # if os.path.isdir(folder):
    #     for fold in os.listdir(folder):
    #         numFiles += len(os.listdir(os.path.join(folder, fold)))
    # else:
    #     print ("Not Folder : Can not get sample per epoch")
    #     return None
    # return numFiles

def kvInvese(dit):
    # return inverse dictionary (key and value)
    newDict = dict()
    for key, value in dit.items():
        newDict[value] = key
    return newDict

def makeDataFrame(label, binary=True):
    # It only used for testing
    if binary:
        label = binaryLabel(bg=False) 
    invDict = kvInvese(label)
    columns = []
    index = []
    for i in sorted(invDict.keys()):
        columns.append(invDict[i])
        index.append(label[invDict[i]])
    numKey = len(label.keys())
    conf = DataFrame(np.zeros((numKey, numKey)), columns = columns, 
                     index=index)
    return conf

def loadLabelUnkown(year, sensor):
    sensorName = sensor.lower()
    if sensorName == "crossmatch":
        labels ={"Live":0, "Gelatin":1, "OOMOO":2}
    elif sensorName == "digital_persona":
        labels ={"Live":0, "RTV":1, "Liquid Ecoflex":2}
    elif sensorName == "greenbit":
        labels ={"Live":0, "RTV":1, "Liquid Ecoflex":2}
    elif sensorName == "hi_scan":
        labels ={"Live":0, "RTV":1, "Liquid Ecoflex":2}
    else:
        print ("There is no {0} sensor.".format(sensorName))
        return False
    return labels

# Sensor folder load
def loadLabelFromYear(year, sensor):
    year = year.lower()
    if year == "livdet2011":
        label = load2011labels(sensor)
    elif year == "livdet2013":
        label = load2013labels(sensor)
    elif year == "livdet2015":
        label = load2015labels(sensor)
    else:
        print("You may write wrong year : ", year)
        label = None
    return label

# Sensor folder load
def imageSizeLoader(paths):
    """
        return image shape in the paths
    """
    livePath = os.path.join(paths, "Live")
    extList = ['.bmp', '.png', '.gif', '.jpg', '.jpeg']
    imageList = os.listdir(livePath)
    for img in imageList:
        ext = os.path.splitext(img)[1]
        if ext.lower() not in extList: # extension check
            continue
        else:
            imgLoaded = imread(os.path.join(livePath, img), flatten=True)
            return imgLoaded.shape
    print("There are no images in the folder.")


def loadSensorList(year):
    year = year.lower()
    if year == "livdet2011":
        return ["biometrika", "digital", "italdata","sagem"]
    elif year == "livdet2013":
        return ["biometrika", "crossmatch", "italdata"]
    elif year == "livdet2015":
        return ["crossmatch", "digital_persona", "greenbit", "hi_scan"]
    else:
        print ("There is no %s Dataset" % year)
        return False       
        
def binaryLabel(bg=True):
    if bg:
        return {"Live":0, "Fake":1, "BG":2}
    else:
        return {"Live":0, "Fake":1}

def labelOut(label):
    """
        return labels sorted by value of input dictionary label
    """
    invDict = kvInvese(label)
    newLabel = []
    for i in sorted(invDict.keys()):
        newLabel.append(invDict[i])
    return newLabel
    
    
def load2011labels(sensorName):
    labels = {}
    sensorName = sensorName.lower().replace("train",'').replace('test','')
    if sensorName == "biometrika":
        labels ={"Live":0, "EcoFlex":1, "Gelatin":2, "Latex":3, "Silgum":4, "WoodGlue":5}
    elif sensorName == "digital":
        labels ={"Live":0, "Gelatin":1, "Latex":2, "Playdoh":3, "Silicone":4, "Wood Glue":5}
    elif sensorName == "italdata":
        labels ={"Live":0, "EcoFlex":1, "Gelatin":2, "Latex":3, "Silgum":4, "WoodGlue":5}
    elif sensorName == "sagem":
        labels ={"Live":0, "Gelatin":1, "Latex":2, "Playdoh":3, "Silicone":4, "Wood Glue":5}
    else:
        print ("There is no {0} sensor.".format(sensorName))
        return False
    return labels
    
def load2013labels(sensorName):
    labels = {}
    sensorName = sensorName.lower().replace("train",'').replace('test','')
    if sensorName == "biometrika":
        labels ={"Live":0, "Ecoflex":1, "Gelatin":2, "Latex":3, "Modasil":4, "WoodGlue":5}
    elif sensorName == "crossmatch":
        labels ={"Live":0, "BodyDouble":1, "Latex":2, "Playdoh":3, "WoodGlue":4}
    elif sensorName == "italdata":
        labels ={"Live":0, "Ecoflex":1, "Gelatine":2, "Latex":3, "Modasil":4, "WoodGlue":5}
    else:
        print ("There is no {0} sensor.".format(sensorName))
        return False
    return labels
    
def load2015labels(sensorName):
    labels = {}
    sensorName = sensorName.lower().replace("train",'').replace('test','')
    if sensorName == "crossmatch":
        labels ={"Live":0, "Body Double":1, "Ecoflex":2, "Playdoh":3}
    elif sensorName == "digital_persona":
        labels ={"Live":0, "Ecoflex 00-50":1, "Gelatine":2, "Latex":3, "WoodGlue":4}
    elif sensorName == "greenbit":
        labels ={"Live":0, "Ecoflex 00-50":1, "Gelatine":2, "Latex":3, "WoodGlue":4}
    elif sensorName == "hi_scan":
        labels ={"Live":0, "Ecoflex 00-50":1, "Gelatine":2, "Latex":3, "WoodGlue":4}
    else:
        print ("There is no {0} sensor.".format(sensorName))
        return False
    return labels

# def imageSizeLoader(model, img_path):
#     if '2011' in model:
#         if 'Biometrika' in model:
#
#     else:
#

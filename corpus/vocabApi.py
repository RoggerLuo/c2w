from .pkl import read

import os
input_file = os.path.join(os.getcwd(),"corpus","vocab.pkl")

pklData = read(input_file)

def getSize():
    return len(pklData)

def getIndexFromChar(character):
    try:
        return pklData.index(character)
    except:
        return False

def getCharFromIndex(index):
    try:
        return pklData[index]
    except:
        return False


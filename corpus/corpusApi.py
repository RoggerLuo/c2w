from .vocabApi import getIndexFromChar,getSize
import os
import numpy as np
import math
input_file = os.path.join(os.getcwd(),"corpus","harrypotter.txt")
outputArr=[]
data = []
def getData():
    with open(input_file,'rb') as f: 
        data = f.readlines()
    output = []
    for item in data[0:20]:
        item = item.decode('gb18030')
        item = list(map(lambda x:getIndexFromChar(x),item))
        output = output + item
        # output.append(np.array(item))
    print(output)
    units_num = math.floor(len(output) / 32)
    output = np.array(output)
    for i in range(units_num):
        outputArr.append(output[i*32:32*(i+1)])

    print(outputArr)
    return np.array(outputArr)
def getVocabSize():
    return getSize()
#!/usr/bin/env python

import json
import os
from jinja2 import Environment, FileSystemLoader

def create_test():
    
    folderName = os.path.dirname(os.path.realpath(__file__))
    file_loader = FileSystemLoader(folderName)
    env = Environment(loader=file_loader)
    template = env.get_template('DataBlobTest.cpp.template')
    
    pathInput = os.path.join('..', 'data', 'RgOutput.dat')
    with open(pathInput, 'r') as DataBlob:
        DataBlobStr = DataBlob.read()

    output = template.render(DataBlob=DataBlobStr)
    
    with open(os.path.join('..', 'tests', 'DataBlob', 'DataBlobTest.cpp'), 'w') as outputFile:
        outputFile.write(output)

def load_file():
    with open("DataBlob.json", 'r') as DataBlobFile:
        dataBlobJson = json.load(DataBlobFile)

    
    print(dataBlobJson["psrpMatchings"]["numberOfMatchings"])
if __name__ == "__main__":
    #create_test()
    load_file()
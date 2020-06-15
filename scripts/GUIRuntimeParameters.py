from tkinter import *
import os.path
import pandas as pd

# action for set value button
def actionButtonSetValue():
    # get name and value
    parameterName = (str)(entryParameterName.get());
    parameterValue = (float)(entryParameterValue.get());
    # create data frame
    data = {'Name':[parameterName],
            'Value':[parameterValue]
            }
    dataFrame = pd.DataFrame(data)
    if os.path.isfile(dataPath):
        # read data out of file
        readDataFrame = (pd.read_json(dataPath, orient='records')).astype('float', errors='ignore')
        readDataFrame.columns=['Name','Value']
        # check if name already exists in json file
        readDataFrameAnalyzed = readDataFrame['Name'] == parameterName
        entryExists = (readDataFrameAnalyzed.any())
        if not entryExists:
            newDataFrame = readDataFrame.append(dataFrame, ignore_index = True)
        else:
            # get index
            indexOfExistingEntry = readDataFrameAnalyzed.index[readDataFrameAnalyzed.values == True]
            newDataFrame = readDataFrame
            # replace old value
            newDataFrame.at[indexOfExistingEntry, 'Value'] = (float)(parameterValue)
        # write to JSON-file
        newDataFrame.to_json(dataPath, orient='records')
    else:
        # write to JSON-file
        dataFrame.to_json(dataPath, orient='records')

# get path of data file
currentDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.dirname(currentDir)
dataPath = rootDir + "\params\parameters.json"

# create window for GUI
window = Tk()
window.title("Set runtime parameters")

# create elements of GUI
entryParameterName = Entry(window)
entryParameterName.grid(column=0, row=0, padx=10)
entryParameterValue = Entry(window)
entryParameterValue.grid(column=1, row=0, padx=10)

labelParameterName = Label(window, text="Parameter name")
labelParameterName.grid(column=0, row=1, padx=10)
labelParameterValue = Label(window, text="Parameter value")
labelParameterValue.grid(column=1, row=1, padx=10)

buttonSetParameter = Button(window, text="Set Values", command=actionButtonSetValue)
buttonSetParameter.grid(column=2,row=2, padx=10, pady=10)

# run window forever
window.mainloop()

import pandas as pd
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   #if like me you do not have a lot of memory in your GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "" #then these two lines force keras to use your CPU
import math
from sklearn.model_selection import train_test_split
from PIL import Image
from matplotlib import pyplot
import numpy as np
import json

# Function to collect 29 frames for CNN input
def getCnnFrames(foldername, vidBaseName, second):
    file =  str('/Volumes/MULTIPIE/CAR-CAM/' + foldername + '/seconds/' + vidBaseName + '.csv')
    fullDf = pd.read_csv(file)
    fullDf.columns = ['frame', 'sec']
    secDf = fullDf.loc[fullDf['sec'] == second]
    return secDf

# Function to extract a csv file with all the possible inputs for the speed detection network
def getPossInputs(verbose = False):
    # calculate which images to extract
    inputlist = []
    # Iterate through all weeks
    for week in os.listdir('/Volumes/MULTIPIE/CAR-CAM'):
        if verbose == True:
            print(week)
        # Iterate through all videos
        for video in os.listdir(str('/Volumes/MULTIPIE/CAR-CAM/' + week + '/videos')):
            if verbose == True:
                print(video)
            # path name for given video
            baseName = os.path.splitext(video)[0]
            # Ignore hidden files
            if baseName[0:2] == '._':
                if verbose == True:
                    print('Ignoring Hidden File')
            else:
                # Find paths for second data file (which frames occur in each second)
                # and speed data file (what speed is it at each second)
                secPath = str('/Volumes/MULTIPIE/CAR-CAM/' + week + '/seconds/' + baseName + '.csv')
                speedPath = str(str('~/PycharmProjects/Diss/speeds/' + week + '/' + baseName + '.csv'))
                # Read files into dataframes
                secDf = pd.read_csv(secPath)
                speedDf = pd.read_csv(speedPath)
                secDf.columns = ['frame', 'sec']
                speedDf.columns = ['sec', 'speed']
                # Iterate through seconds
                for sec in speedDf['sec']:
                    # Locate speed at each second
                    speed1 = speedDf.loc[speedDf['sec'] == int(sec)]
                    speed2 = speed1['speed'][speed1['speed'].index.tolist()[0]]
                    # Ignore speeds which were not recorded correctly
                    if str(speed2) != 'nan':
                        inList = [week, video, sec, float(speed2)]
                        # Append to the list of possible inputs
                        inputlist.append(inList)
    # Convert list to data frame with all required information and sort
    poss = pd.DataFrame(inputlist, columns=['Week', 'Video', 'Second', 'Speed'])
    possSorted = poss.sort_values(['Week', 'Video', 'Second'])
    # Export sorted dataframe to csv file
    possSorted.to_csv(str('~/PycharmProjects/Diss/possInputs.csv'), index=False, header=True)

# Function to convert rgb to greyscale if dimensionality reduction is required in future to increase speed
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

### DATA EXTRACTION WITHOUT DATA GENERATOR

# Function to extract the image data into numpy arrays ready to use in NN
def getimarrays(X_In, Y_In, depth):
    # Initialize lists
    Xlist = []
    Ylist = []

    # Data comes in as a set of rows, each of which points to a week, video name and second for frame extraction
    # Iterate through these rows
    for index, row in X_In.iterrows():
        # Extract basename to identify video
        baseName = os.path.splitext(row['Video'])[0]
        # Find the csv which contains information about the frames in this second
        secPath = str('/Volumes/MULTIPIE/CAR-CAM/' + row['Week'] + '/seconds/' + baseName + '.csv')
        # Import into data frame
        secDf = pd.read_csv(secPath)
        secDf.columns = ['frame', 'sec']
        sec = row['Second']
        # Locate the frames which correspond to this second
        frames = secDf.loc[secDf['sec'] == sec]['frame']
        # Indides come in as original index in file, so we find the index of the first and iterate through from there
        firstIndex = frames.index.tolist()[0]
        # Initialize numpy array for frame data
        frDf = np.empty((depth, 128, 128, 3))
        # 29.7 fps so we only extract 29 frames (some seconds have 30 corresponding frames, some have 29)
        for i in range(0,5):
            frNums = np.round(np.linspace(0,28,5)).astype('int')
            print(firstIndex)
            print(i)
            print(firstIndex + frNums[i])
            fr = frames[firstIndex + frNums[i]]
            # image names stored as 10 digits (filled with 0s on top of the actual number)
            # so we need to add 0s to frame number
            frName = str(f'{fr:010}' + '.jpg')
            frPath = str('/Volumes/MULTIPIE/CAR-CAM/' + row['Week'] + '/small_Images/' + row['Video'] + '/' + frName)
            # Import image and save as numpy array
            image = Image.open(frPath)
            data = np.asarray(image)/255
            frDf[i] = data
        # Append image data for given second to list
        Xlist.append([row['Week'], row['Video'], row['Second'], frDf])
        # Append speed for given second to list
        Ylist.append([Y_In[index]])

    # Save these lists as arrays
    X = np.asarray(Xlist)
    Y = np.asarray(Ylist)

    return X, Y

# Function to retrieve a specified amount of training data for use WITHOUT A DATA GENERATOR
def getData(numExtract = 'all', depth = 5, verbose = False):
    """

    :type depth: int
    """
    # Import set of all possible inputs
    possSorted = pd.read_csv('~/PycharmProjects/Diss/possInputs.csv')
    if numExtract != 'all':
        # Sample the requested number of input
        somePossSorted = possSorted.sample(numExtract)
        possXSorted = somePossSorted[['Week', 'Video', 'Second']]
        possYSorted = somePossSorted['Speed']
    else:
        possXSorted = possSorted[['Week', 'Video', 'Second']]
        possYSorted = possSorted['Speed']

    # Split into test and training sets
    X_trInfo, X_teInfo, Y_trInfo, Y_teInfo = train_test_split(
                                possXSorted, possYSorted, test_size = 0.33, random_state = 42)

    # Extract image (X) and Speed (Y) data for training and test sets
    X_train, Y_train = getimarrays(X_trInfo, Y_trInfo, depth)
    X_test, Y_test = getimarrays(X_teInfo, Y_teInfo, depth)

    return X_train, X_test, Y_train, Y_test

# Function to extract files for upload to Google Drive, to be later utilised in Google Colab
def getdrivefolder(numSamples, depth = 5):
    X_train, X_test, Y_train, Y_test = getData(numSamples, depth)
    np.save('/Users/joecampbell/PycharmProjects/Diss/For Drive/X_train.npy', X_train)
    np.save('/Users/joecampbell/PycharmProjects/Diss/For Drive/X_test.npy', X_test)
    np.save('/Users/joecampbell/PycharmProjects/Diss/For Drive/Y_train.npy', Y_train)
    np.save('/Users/joecampbell/PycharmProjects/Diss/For Drive/Y_test.npy', Y_test)

### DATA EXTRACTION WITH DATA GENERATOR

# Function to extract the image data into numpy arrays ready to use in NN
def getimarrays2(ID, week, vid, sec, depth, type):
    # Find the csv which contains information about the frames in this second
    secPath = str('/Volumes/MULTIPIE/CAR-CAM/' + week + '/seconds/' + vid + '.csv')
    # Import into data frame
    secDf = pd.read_csv(secPath)
    secDf.columns = ['frame', 'sec']
    # Locate the frames which correspond to this second
    frameList = []
    frames = secDf.loc[secDf['sec'] == sec]['frame']
    frameList.extend(list(frames))
    print(frameList)
    # Initialize numpy array for frame data
    frDf = np.empty((depth, 128, 128, 3))
    # 29.7 fps so we only extract 29 frames (some seconds have 30 corresponding frames, some have 29)
    for i in range(0, depth):
        frNums = np.round(np.linspace(0,len(frameList)-1,depth)).astype('int')
        fr = frameList[frNums[i]]
        print(fr)
        # image names stored as 10 digits (filled with 0s on top of the actual number)
        # so we need to add 0s to frame number
        frName = str(f'{fr:010}' + '.jpg')
        frPath = str('/Volumes/MULTIPIE/CAR-CAM/' + week + '/small_Images/' + vid + '.MP4/' + frName)
        # Import image and save as numpy array
        image = Image.open(frPath)
        data = np.asarray(image) / 255
        frDf[i] = data
    # Append image data for given second to list
    if type == 'fit':
        np.save(str('/Volumes/MULTIPIE/speed data/fit/' + str(ID) + '.npy'),frDf)
    elif type == 'eval':
        np.save(str('/Volumes/MULTIPIE/speed data/eval/' + str(ID) + '.npy'),frDf)

# Function to retrieve a specified amount of training data for use WITH A DATA GENERATOR
def getData2(numExtract = 'all', depth = 5, verbose = False):
    count = 0
    labels_fit = dict()
    IDList_fit = []
    labelList_fit = []
    IDInfo_fit = dict()

    labels_eval = dict()
    IDList_eval = []
    labelList_eval = []
    IDInfo_eval = dict()

    labelCount_fit = 0

    # Import set of all possible inputs
    prePossSorted = pd.read_csv('~/PycharmProjects/Diss/possInputs.csv')
    if numExtract != 'all':
        # Sample the requested number of input
        possSorted = prePossSorted.sample(numExtract)
        fit_num = float(numExtract) * 0.7
        eval_num = numExtract
    else:
        possSorted = prePossSorted
        fit_num = math.floor(float(len(possSorted)) * 0.7)
        eval_num = len(possSorted)


    for index, row in possSorted.iterrows():
        vid = os.path.splitext(row['Video'])[0]
        speed = row['Speed']
        sec = row['Second']
        week = row['Week']

        ID = str(f'{count:04}')
        if labelCount_fit < fit_num:
            try:
                print("FIT")
                labelCount_fit += 1
                print(labelCount_fit)
                labels_fit[ID] = speed
                IDInfo_fit[ID] = [week, vid, sec, speed]
                IDList_fit.append(ID)
                labelList_fit.append(speed)
                getimarrays2(ID, week, vid, sec, 5, 'fit')
                count += 1
            except IndexError:
                print("Index Error")
        elif labelCount_fit < eval_num:
            try:
                print("EVAL")
                labelCount_fit += 1
                print(labelCount_fit)
                labels_eval[ID] = speed
                IDInfo_eval[ID] = [week, vid, sec, speed]
                IDList_eval.append(ID)
                labelList_eval.append(speed)
                getimarrays2(ID, week, vid, sec, 5, 'eval')
                count += 1
            except IndexError:
                print("Index Error")
        else:
            labelCount_fit += 1
            print(labelCount_fit)
            print("FIT AND EVAL FULL")

    print(labelCount_fit)
    X_train, X_test, y_train, y_test = train_test_split(IDList_fit, labelList_fit, test_size = 0.25, random_state = 42)
    partition_fit = {'train': X_train, 'validation': X_test}

    with open('/Volumes/MULTIPIE/speed data/labels_fit.json', 'w') as lab:
        json.dump(labels_fit, lab)

    with open('/Volumes/MULTIPIE/speed data/IDInfo_fit.json', 'w') as lablist:
        json.dump(IDInfo_fit, lablist)

    with open('/Volumes/MULTIPIE/speed data/partition_fit.json', 'w') as part:
        json.dump(partition_fit, part)

    partition_eval = {'evaluate': IDList_eval}

    with open('/Volumes/MULTIPIE/speed data/labels_eval.json', 'w') as lab:
        json.dump(labels_eval, lab)

    with open('/Volumes/MULTIPIE/speed data/IDInfo_eval.json', 'w') as lablist:
        json.dump(IDInfo_eval, lablist)

    with open('/Volumes/MULTIPIE/speed data/partition_eval.json', 'w') as part:
        json.dump(partition_eval, part)

# Function utilised to display frames after they have been extracted as NPY files
# Useful for ensuring that the data has been extracted correctly
def dispData(num):
    count = 0
    with open('/Volumes/MULTIPIE/speed data/labels_eval.json', 'r') as fp:
        labels = json.load(fp)
    for ID in os.listdir('/Volumes/MULTIPIE/speed data/eval/'):
        if ID[0] != '.':
            if count < num:
                baseName = os.path.splitext(ID)[0]
                print(baseName)
                X = np.load('/Volumes/MULTIPIE/speed data/eval/' + ID)
                y = labels[baseName]
                for i in X:
                    pyplot.imshow(i)
                    pyplot.show()
                print(y)
                count += 1
                print(count)


if __name__ == '__main__':
    # print(getCnnFrames('JAG-WEEK-1', 'TS_N0001', 20))
    # getPossInputs(verbose=True)
    # runCNN(200, 5)
    # getdrivefolder(2000, 5)
    getData2(8000, 5)
    # dispData(5)
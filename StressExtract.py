import pandas as pd
import numpy as np
from moviepy.editor import *
import statistics
from PIL import Image
import json
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import xlrd

# Function to extract the image data into numpy arrays ready to use in NN
def getimarrays(ID, week, vid, start, end, depth, type, mode):

    # Find the csv which contains information about the frames in this second
    secPath = str('/Volumes/MULTIPIE/CAR-CAM/' + week + '/seconds/' + vid + '.csv')
    # Import into data frame
    secDf = pd.read_csv(secPath)
    secDf.columns = ['frame', 'sec']
    secs = [i for i in range(int(start), int(end) + 1)]
    # Locate the frames which correspond to this second
    frameList = []
    for sec in secs:
        frames = secDf.loc[secDf['sec'] == sec]['frame']
        frameList.extend(list(frames))
    print(frameList)
    # # Indices come in as original index in file, so we find the index of the first and iterate through from there
    # firstIndex = frames.index.tolist()[0]
    # Initialize numpy array for frame data
    frDf = np.empty((depth, 128, 128, 3))
    # 29.7 fps so we only extract 29 frames (some seconds have 30 corresponding frames, some have 29)
    for i in range(0,depth):
        frNums = np.round(np.linspace(0,len(frameList)- 1,depth)).astype('int')
        fr = frameList[frNums[i]]
        # image names stored as 10 digits (filled with 0s on top of the actual number)
        # so we need to add 0s to frame number
        frName = str(f'{fr:010}' + '.jpg')
        frPath = str('/Volumes/MULTIPIE/CAR-CAM/' + week + '/small_Images/' + vid + '.MP4/' + frName)
        # Import image and save as numpy array
        image = Image.open(frPath)
        data = np.asarray(image)/255
        frDf[i] = data
    # # Append image data for given second to list
    if type == 'fit':
        np.save(str('/Volumes/MULTIPIE/Stress Data ' + mode + '/fit/' + str(ID) + '.npy'),frDf)
    elif type == 'eval':
        np.save(str('/Volumes/MULTIPIE/Stress Data ' + mode + '/eval/' + str(ID) + '.npy'),frDf)

#  Function to extract stress values from Excel questionnaire spreadsheets
# Can either extract as a mean value (for regression) or as either 'high' or 'low' classes (for 2 class classification)
def stressExtract(mode = 'mean'):
    rateDict = {'1': {}, '2': {}, '3': {}, '4': {}, '5': {}, '6': {}}
    for q in os.listdir('/Users/joecampbell/Desktop/Dissertation/Questionnaire Responses'):
        if q[0] == 'D':
            print(q)
            qNum = q.split(' ')[4]
            workbook = xlrd.open_workbook('/Users/joecampbell/Desktop/Dissertation/Questionnaire Responses/' + q)
            worksheet = workbook.sheet_by_name('Questionnaire')
            if qNum == '6':
                endNum = 118
            else:
                endNum = 119
            valueList = []
            for i in range(20, endNum):
                # clipNum = str(worksheet.cell(i, 1)).split(':')[1]
                value = str(worksheet.cell(i, 2)).split(':')[1]
                valueList.append(float(value))
            std = statistics.stdev(valueList)
            mu = statistics.mean(valueList)
            valList2 = [(5 + ((x-mu)*(1/std)))  for x in valueList]
            for clipNum in range(1, len(valList2)+1):
                if clipNum in rateDict[qNum]:
                    rateDict[qNum][clipNum].append(valList2[clipNum-1])
                else:
                    rateDict[qNum][clipNum] = []
                    rateDict[qNum][clipNum].append(valList2[clipNum-1])
    print(rateDict)

    for key in rateDict:
        for key2 in rateDict[key]:
            if mode == '2class':
                print(rateDict[key][key2])
                for i in range(0, len(rateDict[key][key2])):
                    if rateDict[key][key2][i] > 5:
                        # High stress
                        rateDict[key][key2][i] = 'high'
                    else:
                        # Low Stress
                        rateDict[key][key2][i] = 'low'
                print(rateDict[key][key2])
                rateDict[key][key2] = statistics.mode(rateDict[key][key2])
                print(rateDict[key][key2])
            else:
                rateDict[key][key2] = statistics.mean(rateDict[key][key2])

    with open(str('/Volumes/MULTIPIE/Stress_Rating_Dict_'+ mode + '.json'), 'w') as json_file:
        json.dump(rateDict, json_file)

# Function to get data set in the format required by a data generator
# Data can be extracted for regression or 2 class classification tasks by changing 'mode'
def getData(mode = 'mean'):
    count = 0
    labels_fit = dict()
    IDList_fit = []
    labelList_fit = []
    IDInfo_fit = dict()

    labels_eval = dict()
    IDList_eval = []
    labelList_eval = []
    IDInfo_eval = dict()

    stressExtract(mode)
    if mode == '2class':
        labelCount_fit = {'high': 0, 'low': 0}

    with open(str('/Volumes/MULTIPIE/Stress_Rating_Dict_'+ mode + '.json'), 'r') as json_file:
        rateDict = json.load(json_file)
    print(rateDict)
    for filename in os.listdir(str('/Volumes/MULTIPIE/FOR USE/CSV/')):
        baseName = os.path.splitext(filename)[0]
        if baseName[0:2] != '._':
            print(baseName)
            clipDir = pd.read_csv(str('/Volumes/MULTIPIE/FOR USE/CSV/'+ filename))
            # print(clipDir)
            for index, row in clipDir.iterrows():
                    vidName = row['name']
                    vidNum = row['number']
                    info = vidName.split('_')
                    week = info[0]
                    rating = rateDict[str(baseName)][str(vidNum)]
                    if week == 'JAG-WEEK-2':
                        vid = info[1]
                        stEn = info[3].split('-')
                        start = float(stEn[0])
                        end = float(stEn[1])
                    else:
                        vid = str(info[1] + '_' + info[2])
                        stEn = info[4].split('-')
                        start = float(stEn[0])
                        end = float(stEn[1])
                    for i in range(0, 10, 1):
                        ID = str(f'{count:04}')
                        fit = 2000
                        eval = 2300
                        if mode == '2class':
                            counter = labelCount_fit[rating]
                            fit_num = fit/2
                            eval_num = eval/2
                            print(rating + ': ' + str(counter))
                        else:
                            counter = count
                            fit_num = fit
                            eval_num = eval
                            print(counter)
                        if counter < fit_num:
                            try:
                                new_start = end - float(2 * (i + 1))
                                new_end = end - float(2 * i)
                                print(new_start, new_end)
                                print("FIT")
                                labels_fit[ID] = rating
                                IDInfo_fit[ID] = [baseName, vidNum, week, vid, rating, new_start, new_end]
                                IDList_fit.append(ID)
                                labelList_fit.append(rating)
                                getimarrays(ID, week, vid, new_start, new_end, 5, 'fit', mode)
                                if mode == '2class':
                                    labelCount_fit[rating] += 1
                                count += 1
                            except IndexError:
                                print("Index Error")
                        elif counter < eval_num:
                            try:
                                new_start = end - float(2 * (i + 1))
                                new_end = end - float(2 * i)
                                print(new_start, new_end)
                                print("EVAL")
                                labels_eval[ID] = rating
                                IDInfo_eval[ID] = [baseName, vidNum, week, vid, rating, new_start, new_end]
                                IDList_eval.append(ID)
                                labelList_eval.append(rating)
                                getimarrays(ID, week, vid, new_start, new_end, 5, 'eval', mode)
                                if mode == '2class':
                                    labelCount_fit[rating] += 1
                                count += 1
                            except IndexError:
                                print("Index Error")
                        else:
                            print("FIT AND EVAL FULL")


    X_train, X_test, y_train, y_test = train_test_split(IDList_fit, labelList_fit, test_size = 0.25, random_state = 42)
    partition_fit = {'train': X_train, 'validation': X_test}

    with open('/Volumes/MULTIPIE/Stress Data ' + mode + '/labels_fit.json', 'w') as lab:
        json.dump(labels_fit, lab)

    with open('/Volumes/MULTIPIE/Stress Data ' + mode + '/IDInfo_fit.json', 'w') as lablist:
        json.dump(IDInfo_fit, lablist)

    with open('/Volumes/MULTIPIE/Stress Data ' + mode + '/partition_fit.json', 'w') as part:
        json.dump(partition_fit, part)

    partition_eval = {'evaluate': IDList_eval}

    with open('/Volumes/MULTIPIE/Stress Data ' + mode + '/labels_eval.json', 'w') as lab:
        json.dump(labels_eval, lab)

    with open('/Volumes/MULTIPIE/Stress Data ' + mode + '/IDInfo_eval.json', 'w') as lablist:
        json.dump(IDInfo_eval, lablist)

    with open('/Volumes/MULTIPIE/Stress Data ' + mode + '/partition_eval.json', 'w') as part:
        json.dump(partition_eval, part)

# Function utilised to display frames after they have been extracted as NPY files
# Useful for ensuring that the data has been extracted correctly
def dispData(num, fitEval = 'fit'):
    count = 0
    with open(str('/Volumes/MULTIPIE/Stress Data/labels_' + fitEval + '.json'), 'r') as fp:
        labels = json.load(fp)
    for ID in os.listdir(str('/Volumes/MULTIPIE/Stress Data/' + fitEval + '/')):
        if ID[0] != '.':
            if count < num:
                baseName = os.path.splitext(ID)[0]
                print(baseName)
                X = np.load('/Volumes/MULTIPIE/Stress Data/' + fitEval + '/' + ID)
                y = labels[baseName]
                for i in X:
                    pyplot.imshow(i)
                    pyplot.show()
                print(y)
                count += 1
                print(count)


if __name__ == '__main__':
    # stressExtract(mode = '2class')
    getData(mode = '2class')
    # dispData(5, 'fit')


import pynmea2
import pandas as pd
import numpy as np
import gmplot
import googlemaps
from datetime import datetime
import math
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *
import re
from PIL import Image
import json
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

# dataFrame used to store the start times of each clip to be used as a reference point in calculations
starts = pd.DataFrame(columns=['file', 'startTime'])

# function to turn a timestamp into a number of seconds
def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

# Function to extract coordinates for each second in a given video
def ctExtract(fileName, foldername, fileDir, verbose = False):
    # dataframe to store coordinates and time
    ctDf = pd.DataFrame(columns=['latitude', 'longitude', 'second'])
    global starts
    # Open File
    file = open(fileDir)
    # Extract name of file without extension
    baseName = os.path.splitext(fileName)[0]
    # Ignore hidden files
    if baseName[0:2] == '._':
        return
    else:
        # Read specified number of GGA lines
        # We get one reading of each type every second
        cycleCount = 0
        for line in file.readlines():
            # Cycle through RMC lines only
            if line.startswith('$GPGGA'):
                try:
                    gpgga = pynmea2.parse(line)
                    # Setting current frame time
                    if cycleCount == 0:
                        # Setting start time for video from frame 0
                        startTime = datetime.strptime(str(gpgga.timestamp), "%H:%M:%S")
                        starts = starts.append({'file': baseName, 'startTime': startTime}, ignore_index=True)
                    # Setting current frame time
                    print(str(gpgga.timestamp))
                    try:
                        currentTime = datetime.strptime(str(gpgga.timestamp), "%H:%M:%S")
                    except ValueError:
                        pass
                    # Calculate time relative to start
                    currRelSecs = get_sec(str(currentTime - startTime))
                    if verbose == True:
                        print("Timestamp: " + str(currRelSecs))
                        print(repr(gpgga))
                    # Extract Latitude and Longitude (checkig for null values)
                    if (gpgga.latitude == 0) or (gpgga.longitude == 0):
                        lat = 'nan'
                        long = 'nan'
                    else:
                        # Extract lat and long from NMEA
                        lat = gpgga.latitude
                        long = gpgga.longitude
                        # Calculate position
                        print(lat)
                        print(long)
                    if verbose == True:
                        print('Latitude: ' + str(lat) + '\n')
                        print('Longitude: ' + str(long) + '\n')
                    ctDf = ctDf.append({'latitude': lat, 'longitude': long, 'second': currRelSecs}, ignore_index=True)
                    cycleCount += 1
                except pynmea2.ParseError as e:
                    print('Parse error: {}'.format(e))
                    continue
        if verbose == True:
            print(ctDf)
        # extract dataframe to csv
        print('Making CSV')
        ctDf.to_csv(str('~/PycharmProjects/Diss/coords/' + foldername + '/' + baseName + '.csv'), index=False, header=True)

# function to apply stExtract to all videos
def allCtExtract(verbose = False):
    # Iterate through different weeks (folders) and gps files to apply stExtract to all
    for foldername in os.listdir('/Volumes/MULTIPIE/CAR-CAM'):
        for filename in os.listdir(str('/Volumes/MULTIPIE/CAR-CAM/' + foldername + '/gps')):
            fileDir = str('/Volumes/MULTIPIE/CAR-CAM/' + foldername + '/gps/' + filename)
            ctExtract(filename, foldername, fileDir, verbose)

# Function to plot coordinates for full video
def plotRoutes():
    # API Key to access google maps
    apikey = 'Insert API Key Here'
    gmaps = googlemaps.Client(key=apikey)
    for foldername in os.listdir('/Users/joecampbell/PycharmProjects/Diss/coords'):
        print(foldername)
        counter = 0
        if foldername[0:1] != '.':
            for filename in os.listdir(str('/Users/joecampbell/PycharmProjects/Diss/coords/' + foldername)):
                baseName = os.path.splitext(filename)[0]
                if baseName[0:2] != '._':
                    fileDir = str('/Users/joecampbell/PycharmProjects/Diss/coords/'+ foldername + '/' + filename)
                    coordList = []
                    waypointList = []
                    coordDf = pd.read_csv(fileDir)
                    if counter == 0:
                        gmap1 = gmplot.GoogleMapPlotter(coordDf.loc[0]['latitude'], coordDf.loc[0]['longitude'], 14, apikey=apikey)
                    for index, row in coordDf.iterrows():
                        if (str(row['latitude']) != 'nan') and (str(row['longitude']) != 'nan'):
                            coordList.append((row['latitude'], row['longitude']))
                            latLongStr = str(str(row['latitude']) + ', ' + str(row['longitude']))
                            waypointList.append((latLongStr, row['second']))
                    try:
                        triplats, triplongs = zip(*coordList)
                    except ValueError:
                        pass
                    gmap1.plot(triplats, triplongs, color='#3B0B39', size=2, marker=False)
                    wpDf = pd.DataFrame(waypointList, columns=['waypoint', 'second'])
                    wpDf.to_csv(str('/Users/joecampbell/PycharmProjects/Diss/waypoints/' + foldername + '/' + baseName + '.csv'), index=False, header=True)
                    counter += 1
            gmap1.draw('maps/' + str(foldername) + '.html')

# Function which iterates through seconds, checks dataframe for matching common locations and labels if matched
def getPossInputs():
    # Build data frame for potential locations
    # (x - center_x)^2 + (y - center_y)^2 < radius^2 equation to be used to check junction
    inputlist = []
    locDf = pd.read_csv('~/PycharmProjects/Diss/junctions.csv')
    for foldername in os.listdir('/Users/joecampbell/PycharmProjects/Diss/coords'):
        if foldername[0:1] != '.':
            for filename in os.listdir(str('/Users/joecampbell/PycharmProjects/Diss/coords/' + foldername)):
                baseName = os.path.splitext(filename)[0]
                if baseName[0:2] != '._':
                    fileDir = str('/Users/joecampbell/PycharmProjects/Diss/coords/'+ foldername + '/' + filename)
                    coordDf = pd.read_csv(fileDir)
                    for index_coord, row_coord in coordDf.iterrows():
                        x = float(row_coord['latitude'])
                        y = float(row_coord['longitude'])
                        for index_loc, row_loc in locDf.iterrows():
                            centre = eval(row_loc['centre'])
                            centre_x = float(centre[0])
                            centre_y = float(centre[1])
                            radius = (float(row_loc['radius'])/1.11)*0.00001
                            if (x - centre_x)**2 + (y - centre_y)**2 < radius**2:
                                inputlist.append([foldername, filename, row_coord['second'], row_loc['junction']])
    possJunction = pd.DataFrame(inputlist, columns=['Week', 'Video', 'Second', 'junction'])
    possJunctionSorted = possJunction.sort_values(['Week', 'Video', 'Second'])
    # Export sorted dataframe to csv file
    possJunctionSorted.to_csv(str('~/PycharmProjects/Diss/possJunctionInputs.csv'), index=False, header=True)

#Function to extract junction information from google maps using Directions API
def getJunctions():
    # API Key to access google maps
    apikey = 'Insert API Key Here'
    gmaps = googlemaps.Client(key=apikey)
    for foldername in os.listdir('/Users/joecampbell/PycharmProjects/Diss/waypoints'):
        print(foldername)
        if foldername[0:1] != '.':
            for filename in os.listdir(str('/Users/joecampbell/PycharmProjects/Diss/waypoints/' + foldername)):
                baseName = os.path.splitext(filename)[0]
                print(baseName)
                if baseName[0:2] != '._':
                    fileDir = str('/Users/joecampbell/PycharmProjects/Diss/waypoints/'+ foldername + '/' + filename)
                    wpDf = pd.read_csv(fileDir)
                    wpList = wpDf['waypoint'].tolist()
                    secList = wpDf['second'].tolist()
                    juncList = []
                    for i in range(0, len(wpList)):
                        now = datetime.now()
                        framesLeft = float(len(wpList) - i)
                        if framesLeft > 60:
                            directions_result = gmaps.directions(wpList[i],
                                                                 wpList[i+60],
                                                                 mode="driving",
                                                                departure_time=now)
                        else:
                            directions_result = gmaps.directions(wpList[i],
                                                                 wpList[-1],
                                                                 mode="driving",
                                                                departure_time=now)
                        try:
                            man = directions_result[0]['legs'][0]['steps'][1]['maneuver']
                            juncList.append([str(secList[i]), str(man)])
                            print(str(secList[i]) + ': ' + str(man))
                        except IndexError:
                            pass
                        except KeyError:
                            pass
                    juncDf = pd.DataFrame(juncList, columns=['second', 'junction'])
                    juncDf.to_csv(str('/Users/joecampbell/PycharmProjects/Diss/junctions/' + foldername + '/' + baseName + '.csv'), index=False, header=True)

# Function to find sets of consecutive junctions labels for more than a given number of seconds
# Where these consecutive junction labels are found, clips are extracted of a set length
def consecJunc(clipLength):
    halfLength = math.floor(clipLength/2)
    clipList = []
    for foldername in os.listdir('/Users/joecampbell/PycharmProjects/Diss/junctions'):
        print(foldername)
        if foldername[0:1] != '.':
            for filename in os.listdir(str('/Users/joecampbell/PycharmProjects/Diss/junctions/' + foldername)):
                baseName = os.path.splitext(filename)[0]
                print(baseName)
                if baseName[0:2] != '._':
                    fileDir = str('/Users/joecampbell/PycharmProjects/Diss/junctions/' + foldername + '/' + filename)
                    juncDf = pd.read_csv(fileDir).applymap(str)
                    try:
                        df = juncDf
                        # Initialize result lists with the first row of df
                        result1 = [df['second'][0]]
                        result2 = [df['junction'][0]]
                        # Use zip() to iterate over the two columns of df simultaneously,
                        # making sure to skip the first row which is already added
                        for second, junction in zip(df['second'][1:], df['junction'][1:]):
                            if junction == result2[-1]:  # If b matches the last value in result2,
                                if len(result1[-1].split(" ")) > 1:
                                    if float(second) == (float(result1[-1].split(" ")[-1]) + 1):
                                        result1[-1] += " " + second  # add a to the last value of result1
                                    elif (float(second) - (float(result1[-1].split(" ")[-1]))) < 20:
                                        result1[-1] += " " + second  # add a to the last value of result1
                                    else:
                                        result1.append(second)
                                        result2.append(junction)
                                else:
                                    result1[-1] += " " + second  # add a to the last value of result1
                            else:  # Otherwise add a new row with the values
                                result1.append(second)
                                result2.append(junction)

                        # Create a new dataframe using these result lists
                        fullJuncDf = pd.DataFrame({'second': result1, 'junction': result2})
                        fullJuncDf.to_csv(str('/Users/joecampbell/PycharmProjects/Diss/FullJunc/' + foldername + '/' + baseName + '.csv'), index=False, header=True)
                        for index, row in fullJuncDf.iterrows():
                            secList = [float(s) for s in row['second'].split(' ')]
                            if len(secList) > clipLength:
                                indiClip = [foldername, baseName, row['junction'], list([float(secList[-1]-clipLength)]), list([float(secList[-1] + clipLength)]), "yes"]
                                clipList.append(indiClip)
                            elif len(secList) > halfLength:
                                indiClip = [foldername, baseName, row['junction'], list([secList[0]]), list([float(secList[0] + (2*clipLength))]), "no"]
                                clipList.append(indiClip)
                    except IndexError:
                        print("No Junctions")

    clipDf = pd.DataFrame(clipList, columns=['week', 'video', 'junction', 'start', 'end', "full"])
    print(len(clipDf))
    for index, row in clipDf.iterrows():
        start = row['start'][0]
        end = row['end'][0]
        week = row['week']
        video = row['video']
        junction = row['junction']
        videoInDir = str('/Volumes/MULTIPIE/CAR-CAM/' + week + '/videos/' + video + '.MP4')
        if row["full"] == "yes":
            videoOutDir = str('/Volumes/MULTIPIE/CLIPS/' + week + '/' + video + '_' + junction + '_' + str(start) + '-' + str(end) + '.MP4')
        else:
            videoOutDir = str('/Volumes/MULTIPIE/CLIPS/' + week + '/' + "NotFull_" + video + '_' + junction + '_' + str(start) + '-' + str(end) + '.MP4')

        ffmpeg_extract_subclip(videoInDir, float(start), float(end), videoOutDir)

# Function which asks user to input details about adjustments for clips
# Used to speed up the process of adjusting clips through automation
def userClipLabel():
    count = 0
    for foldername in os.listdir('/Volumes/MULTIPIE/CLIPS2/'):
        print(foldername)
        if foldername[0:1] != '.':
            juncList = []
            for filename in os.listdir(str('/Volumes/MULTIPIE/CLIPS2/' + foldername)):
                count += 1
                print(count)
                baseName = os.path.splitext(filename)[0]
                if baseName[0:2] != '._':
                    print(baseName)
                    info = re.split('_+', baseName)
                    videoName = str(info[0])
                    junction = info[1]
                    time = info[2]
                    vidPath = str('/Volumes/MULTIPIE/CLIPS2/' + foldername + '/' + filename)
                    cap = cv2.VideoCapture(vidPath)
                    # Check if camera opened successfully
                    if (cap.isOpened() == False):
                        print("Error opening video stream or file")
                    # Read until video is completed
                    while (cap.isOpened()):

                        # Capture frame-by-frame
                        ret, frame = cap.read()

                        if ret == True:

                            # if count == 0:
                            cv2.namedWindow(baseName, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(baseName, 800, 800)
                            cv2.moveWindow(baseName, 2000, -500)
                            cv2.imshow(baseName, frame)
                            #     count = 1
                            # elif count < 20:
                            #     count += 1
                            # else:
                            #     count = 0


                            # Press Q on keyboard to  exit
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        # Break the loop
                        else:
                            break

                    # When everything done, release the video capture object
                    cap.release()
                    # Closes all the frames
                    cv2.destroyAllWindows()
                    # lt = left turn off straight road, rt = right turn off straight road
                    # rb =  roundabout, cr = cross-roads/intersection
                    # tj = T-Junction, sr = Straight Road
                    # ramp = ramp left
                    juncLabel = input("lt, rt, rb, cr, tj, sr, ramp, useless: ")

                    timeChanges = input("Any time adjustments? ")

                    if timeChanges == 'y':

                        startEarlier = input("Does it need to start earlier? " )
                        if startEarlier == 'n':
                            startLater = input("Does it need to start later? ")
                        else:
                            startLater = 'n'

                        finishEarlier = input("Does it need to finish earlier? ")
                        if finishEarlier == 'n':
                            finishLater = input("Does it need to finish later? ")
                        else:
                            finishLater = 'n'

                    else:

                        startEarlier = 'n'
                        startLater = 'n'
                        finishEarlier = 'n'
                        finishLater = 'n'


                    list = [foldername, videoName, junction, time, juncLabel, startEarlier, startLater, finishEarlier, finishLater]
                    juncList.append(list)
        juncDf = pd.DataFrame(juncList, columns=['week', 'video', 'old junction', 'time', 'new junction', 'startEarlier', 'startLater', 'finishEarlier', 'finishLater'])
        juncDf.to_csv(str('/Users/joecampbell/PycharmProjects/Diss/JUNC LABELS/' + foldername + '.csv'), index=False, header=True)

# Function which takes user inputs from userClipLabel() and extracts a set of finalised clips for junction classifier
def adjustClips():
    clipList = []
    for filename in os.listdir('/Users/joecampbell/PycharmProjects/Diss/JUNC LABELS/'):
        baseName = os.path.splitext(filename)[0]
        if baseName[0:2] != '._':
            print(baseName)
            fileDir = str('/Users/joecampbell/PycharmProjects/Diss/JUNC LABELS/' + filename)
            juncLabels = pd.read_csv(fileDir)
            for index, row in juncLabels.iterrows():
                if row['new junction'] != 'useless':
                    print(row)
                    week = row['week']
                    video = row['video']
                    time = row['time'].split('-')
                    origStart = float(time[0])
                    origEnd = float(time[1])
                    newJunc = row['new junction']
                    if row['new junction'] == 'lt':
                        newJunc = 'left turn'
                    elif row['new junction'] == 'rt':
                        newJunc = 'right turn'
                    elif row['new junction'] == 'rb':
                        newJunc = 'roundabout'
                    elif row['new junction'] == 'cr':
                        newJunc = 'cross roads'
                    elif row['new junction'] == 'tj':
                        newJunc = 'T Junction'
                    elif row['new junction'] == 'sr':
                        newJunc = 'straight road'
                    elif row['new junction'] == 'ramp':
                        newJunc = 'ramp left'


                    if row['startEarlier'] == 'y':
                        if origStart < 5:
                            newStart = 0
                        else:
                            newStart = origStart - 5
                    elif row['startEarlier'] == 'yy':
                        if origStart < 10:
                            newStart = 0
                        else:
                            newStart = origStart - 10
                    elif row['startEarlier'] == 'yyy':
                        if origStart < 15:
                            newStart = 0
                        else:
                            newStart = origStart - 15
                    else:
                        if row['startLater'] == 'y':
                            newStart = origStart + 5
                        elif row['startLater'] == 'yy':
                            newStart = origStart + 10
                        elif row['startLater'] == 'yyy':
                            newStart = origStart + 15
                        else:
                            newStart = origStart

                    if row['finishEarlier'] == 'y':
                        newEnd = origEnd - 5
                    elif row['finishEarlier'] == 'yy':
                        newEnd = origEnd - 10
                    elif row['finishEarlier'] == 'yyy':
                        newEnd = origEnd - 15
                    else:
                        if row['finishLater'] == 'y':
                            newEnd = origEnd + 5
                        elif row['finishLater'] == 'yy':
                            newEnd = origEnd + 10
                        elif row['finishLater'] == 'yyy':
                            newEnd = origEnd + 15
                        else:
                            newEnd = origEnd

                    clipList.append([week, video, newJunc, newStart, newEnd])

    clipDf = pd.DataFrame(clipList, columns=['week', 'video', 'junction', 'start', 'end'])

    for index, row in clipDf.iterrows():
        start = row['start']
        end = row['end']
        week = row['week']
        video = row['video']
        junction = row['junction']
        videoInDir = str('/Volumes/MULTIPIE/CAR-CAM/' + week + '/videos/' + video + '.MP4')
        videoOutDir = str('/Volumes/Seagate Backup Plus Drive/Adjusted Clips/' + week + '/' + video + '_' + junction + '_' + str(start) + '-' + str(end) + '.MP4')
        ffmpeg_extract_subclip(videoInDir, float(start), float(end), videoOutDir)

# Function to extract the image data into numpy arrays ready to use in NN
def getimarrays(ID, week, vid, start, end, depth, type, numClass):

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
    # Initialize numpy array for frame data
    frDf = np.empty((depth, 128, 128, 3))
    # 29.7 fps so we only extract 29 frames (some seconds have 30 corresponding frames, some have 29)
    for i in range(0,depth):
        frNums = np.round(np.linspace(0,len(frameList)-1,depth)).astype('int')
        fr = frameList[frNums[i]]
        # image names stored as 10 digits (filled with 0s on top of the actual number)
        # so we need to add 0s to frame number
        frName = str(f'{fr:010}' + '.jpg')
        frPath = str('/Volumes/MULTIPIE/CAR-CAM/' + week + '/small_Images/' + vid + '.MP4/' + frName)
        # Import image and save as numpy array
        image = Image.open(frPath)
        data = np.asarray(image)/255
        frDf[i] = data
    # Append image data for given second to list
    if type == 'fit':
        np.save(str('/Volumes/MULTIPIE/junc data' + str(numClass) + '/fit/' + str(ID) + '.npy'),frDf)
    elif type == 'eval':
        np.save(str('/Volumes/MULTIPIE/junc data' + str(numClass) + '/eval/' + str(ID) + '.npy'),frDf)

# Function to get data set for the 7 class classification problem in the format required by a data generator
def getData7():
    count = 0
    labelKey = {'straight road': 0, 'cross roads': 1, 'roundabout': 2, 'T Junction': 3, 'right turn': 4, 'left turn': 5, 'ramp left': 6}
    labels_fit = dict()
    IDList_fit = []
    labelList_fit = []
    IDInfo_fit = dict()

    labels_eval = dict()
    IDList_eval = []
    labelList_eval = []
    IDInfo_eval = dict()

    labelCount_fit = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}

    for foldername in os.listdir('/Volumes/MULTIPIE/Adjusted Clips/'):
        print(foldername)
        if foldername[0:1] != '.':
            for filename in os.listdir(str('/Volumes/MULTIPIE/Adjusted Clips/' + foldername)):
                baseName = os.path.splitext(filename)[0]
                if baseName[0:2] != '._':
                    print(baseName)
                    info = baseName.split('_')
                    if foldername == 'JAG-WEEK-2':
                        vid = info[0]
                        junction = info[1]
                        stEn = info[2].split('-')
                        start = float(stEn[0])
                        end = float(stEn[1])
                    else:
                        vid = str(info[0] + '_' + info[1])
                        junction = info[2]
                        stEn = info[3].split('-')
                        start = float(stEn[0])
                        end = float(stEn[1])

                    if junction == 'ramp left':
                        fit_num = 110
                        eval_num = 160
                    else:
                        fit_num = 230
                        eval_num = 330
                    for i in range(0, 10, 1):
                        ID = str(f'{count:04}')
                        if labelCount_fit[labelKey[junction]] < fit_num:
                            try:
                                new_start = end - float(2*(i+1))
                                new_end = end - float(2*i)
                                print(new_start, new_end)
                                print("FIT")
                                labelCount_fit[labelKey[junction]] += 1
                                print(junction + ':' + str(labelCount_fit[labelKey[junction]]))
                                labels_fit[ID] = labelKey[junction]
                                IDInfo_fit[ID] = [foldername, vid, junction, labelKey[junction], new_start, new_end]
                                IDList_fit.append(ID)
                                labelList_fit.append(labelKey[junction])
                                getimarrays(ID, foldername, vid, new_start, new_end, 5, 'fit', 7)
                                count += 1
                            except IndexError:
                                print("Index Error")
                        elif labelCount_fit[labelKey[junction]] < eval_num:
                            try:
                                new_start = end - float(2*(i+1))
                                new_end = end - float(2*i)
                                print(new_start, new_end)
                                print("EVAL")
                                labelCount_fit[labelKey[junction]] += 1
                                print(junction + ':' + str(labelCount_fit[labelKey[junction]]))
                                labels_eval[ID] = labelKey[junction]
                                IDInfo_eval[ID] = [foldername, vid, junction, labelKey[junction], new_start, new_end]
                                IDList_eval.append(ID)
                                labelList_eval.append(labelKey[junction])
                                getimarrays(ID, foldername, vid, new_start, new_end, 5, 'eval', 7)
                                count += 1
                            except IndexError:
                                print("Index Error")
                        else:
                            labelCount_fit[labelKey[junction]] += 1
                            print(junction + ':' + str(labelCount_fit[labelKey[junction]]))
                            print("FIT AND EVAL FULL")

    print(labelCount_fit)
    X_train, X_test, y_train, y_test = train_test_split(IDList_fit, labelList_fit, test_size = 0.25, random_state = 42)
    partition_fit = {'train': X_train, 'validation': X_test}

    with open('/Volumes/MULTIPIE/junc data7/labels_fit_7class.json', 'w') as lab:
        json.dump(labels_fit, lab)

    with open('/Volumes/MULTIPIE/junc data7/IDInfo_fit_7class.json', 'w') as lablist:
        json.dump(IDInfo_fit, lablist)

    with open('/Volumes/MULTIPIE/junc data7/partition_fit_7class.json', 'w') as part:
        json.dump(partition_fit, part)

    partition_eval = {'evaluate': IDList_eval}

    with open('/Volumes/MULTIPIE/junc data7/labels_eval_7class.json', 'w') as lab:
        json.dump(labels_eval, lab)

    with open('/Volumes/MULTIPIE/junc data7/IDInfo_eval_7class.json', 'w') as lablist:
        json.dump(IDInfo_eval, lablist)

    with open('/Volumes/MULTIPIE/junc data7/partition_eval_7class.json', 'w') as part:
        json.dump(partition_eval, part)

# Function to get data set for the 4 class classification problem in the format required by a data generator
def getData4():
    count = 0
    labelKey = {'straight road': 0, 'cross roads': 1, 'roundabout': 2, 'T Junction': 3, 'right turn': 4, 'left turn': 5, 'ramp left': 6}
    labels_fit = dict()
    IDList_fit = []
    labelList_fit = []
    IDInfo_fit = dict()

    labels_eval = dict()
    IDList_eval = []
    labelList_eval = []
    IDInfo_eval = dict()

    labelCount_fit = {0:0, 1:0, 2:0, 3:0}

    for foldername in os.listdir('/Volumes/MULTIPIE/Adjusted Clips/'):
        print(foldername)
        if foldername[0:1] != '.':
            for filename in os.listdir(str('/Volumes/MULTIPIE/Adjusted Clips/' + foldername)):
                baseName = os.path.splitext(filename)[0]
                if baseName[0:2] != '._':
                    print(baseName)
                    info = baseName.split('_')
                    if foldername == 'JAG-WEEK-2':
                        vid = info[0]
                        junction = info[1]
                        stEn = info[2].split('-')
                        start = float(stEn[0])
                        end = float(stEn[1])
                    else:
                        vid = str(info[0] + '_' + info[1])
                        junction = info[2]
                        stEn = info[3].split('-')
                        start = float(stEn[0])
                        end = float(stEn[1])

                    if junction == 'ramp left':
                        print('Not in 4 Class')
                    elif junction == 'left turn':
                        print('Not in 4 Class')
                    elif junction == 'right turn':
                        print('Not in 4 Class')
                    else:
                        fit_num = 230
                        eval_num = 330
                        for i in range(0, 10, 1):
                            ID = str(f'{count:04}')
                            if labelCount_fit[labelKey[junction]] < fit_num:
                                try:
                                    new_start = end - float(2*(i+1))
                                    new_end = end - float(2*i)
                                    print(new_start, new_end)
                                    print("FIT")
                                    labelCount_fit[labelKey[junction]] += 1
                                    print(junction + ':' + str(labelCount_fit[labelKey[junction]]))
                                    labels_fit[ID] = labelKey[junction]
                                    IDInfo_fit[ID] = [foldername, vid, junction, labelKey[junction], new_start, new_end]
                                    IDList_fit.append(ID)
                                    labelList_fit.append(labelKey[junction])
                                    getimarrays(ID, foldername, vid, new_start, new_end, 5, 'fit', 4)
                                    count += 1
                                except IndexError:
                                    print("Index Error")
                            elif labelCount_fit[labelKey[junction]] < eval_num:
                                try:
                                    new_start = end - float(2*(i+1))
                                    new_end = end - float(2*i)
                                    print(new_start, new_end)
                                    print("EVAL")
                                    labelCount_fit[labelKey[junction]] += 1
                                    print(junction + ':' + str(labelCount_fit[labelKey[junction]]))
                                    labels_eval[ID] = labelKey[junction]
                                    IDInfo_eval[ID] = [foldername, vid, junction, labelKey[junction], new_start, new_end]
                                    IDList_eval.append(ID)
                                    labelList_eval.append(labelKey[junction])
                                    getimarrays(ID, foldername, vid, new_start, new_end, 5, 'eval', 4)
                                    count += 1
                                except IndexError:
                                    print("Index Error")
                            else:
                                labelCount_fit[labelKey[junction]] += 1
                                print(junction + ':' + str(labelCount_fit[labelKey[junction]]))
                                print("FIT AND EVAL FULL")

    print(labelCount_fit)
    X_train, X_test, y_train, y_test = train_test_split(IDList_fit, labelList_fit, test_size = 0.25, random_state = 42)
    partition_fit = {'train': X_train, 'validation': X_test}

    with open('/Volumes/MULTIPIE/junc data4/labels_fit_4class.json', 'w') as lab:
        json.dump(labels_fit, lab)

    with open('/Volumes/MULTIPIE/junc data4/IDInfo_fit_4class.json', 'w') as lablist:
        json.dump(IDInfo_fit, lablist)

    with open('/Volumes/MULTIPIE/junc data4/partition_fit_4class.json', 'w') as part:
        json.dump(partition_fit, part)

    partition_eval = {'evaluate': IDList_eval}

    with open('/Volumes/MULTIPIE/junc data4/labels_eval_4class.json', 'w') as lab:
        json.dump(labels_eval, lab)

    with open('/Volumes/MULTIPIE/junc data4/IDInfo_eval_4class.json', 'w') as lablist:
        json.dump(IDInfo_eval, lablist)

    with open('/Volumes/MULTIPIE/junc data4/partition_eval_4class.json', 'w') as part:
        json.dump(partition_eval, part)

# Function utilised to display frames after they have been extracted as NPY files
# Useful for ensuring that the data has been extracted correctly
def dispData(num, numClass):
    count = 0
    with open('/Volumes/MULTIPIE/junc data' +  str(numClass) + '/labels_eval_' +  str(numClass) + 'class.json', 'r') as fp:
        labels = json.load(fp)
    for ID in os.listdir('/Volumes/MULTIPIE/junc data' +  str(numClass) + '/eval/'):
        if ID[0] != '.':
            if count < num:
                baseName = os.path.splitext(ID)[0]
                print(baseName)
                X = np.load('/Volumes/MULTIPIE/junc data' +  str(numClass) + '/eval/' + ID)
                y = labels[baseName]
                for i in X:
                    pyplot.imshow(i)
                    pyplot.show()
                print(y)
                count += 1
                print(count)

if __name__ == '__main__':
    # allCtExtract(verbose=True)
    # plotRoutes()
    # getPossInputs()
    # getJunctions()
    # userClipLabel()
    # adjustClips()
    # getData7()
    dispData(20,4)



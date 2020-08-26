import pandas as pd
import math
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *
import ffmpy

# Cycle through junctions labels and detect each set of consecutive seconds with the same junction
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
                                indiClip = [foldername, baseName, row['junction'], list([float(secList[-1] - clipLength)]), list([float(secList[-1] + clipLength)]), "yes"]
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

        # clip = VideoFileClip(videoInDir).subclip(start, end)
        # clip.to_videofile(videoOutDir, codec="libx264", temp_audiofile='temp-audio.m4a', remove_temp=True, audio_codec='aac')

# Function to extract clips and sort into six different folders, to be used in the six different questionnaires
def clipSort():
    count = 1
    for foldername in os.listdir('/Volumes/MULTIPIE/CLIPS/'):
        if foldername[0:1] != '.':
            for filename in os.listdir(str('/Volumes/MULTIPIE/CLIPS/' + foldername)):
                baseName = os.path.splitext(filename)[0]
                print(baseName)
                if baseName[0:2] != '._':
                        videoInDir = str('/Volumes/MULTIPIE/CLIPS/' + foldername + '/' + filename)
                        videoOutDir = str('/Volumes/MULTIPIE/FOR USE/' + str(count) + '/' + foldername + '_' + filename)
                        ffmpeg_extract_subclip(videoInDir, 4, 18, videoOutDir)
                        if count < 6:
                            count += 1
                        else:
                            count = 1

# Function to used to create each individual video for the six different questionnaires
# Each video is made up of concatenated clips
# Each of clip is given a title card telling the participant which cell to put the stress rating in
def makeMovie(number):
    clips = []
    count = 1
    for filename in os.listdir(str('/Volumes/MULTIPIE/FOR USE/' + str(number) + '/')):
        if filename.endswith(".MP4"):
            print(filename)
            baseName = os.path.splitext(filename)[0]
            print(baseName)
            if baseName[0:2] != '._':
                title = TextClip(str('Clip ' + str(count)), color = 'white', size = (1920, 1080), bg_color= 'black', fontsize= 100)
                titleclip = title.set_duration(3)
                clips.append(titleclip)
                video_path = str('/Volumes/MULTIPIE/FOR USE/' + str(number) + '/' + baseName)
                ff = ffmpy.FFmpeg(executable= '/Users/joecampbell/opt/anaconda3/lib/python3.7/site-packages/imageio_ffmpeg/binaries/ffmpeg-osx64-v4.1', inputs = {str(video_path + '.MP4'): None}, outputs = {str(video_path + '.avi'): '-q:v 1'})
                ff.run()
                clips.append(VideoFileClip(str(video_path + '.avi')))
                count += 1
    video = concatenate_videoclips(clips, method='compose')
    video.write_videofile('/Volumes/MULTIPIE/FOR USE/' + str(number) + '.MP4', fps = 29.970296)
    for filename in os.listdir(str('/Volumes/MULTIPIE/FOR USE/' + str(number) + '/')):
        if filename.endswith(".avi"):
            os.remove(str('/Volumes/MULTIPIE/FOR USE/' + str(number) + '/' + filename))

# Function to iterate through and create all 6 questionnaires
def makeAllMovies():
    for i in range(1,7):
        makeMovie(i)

# Function which extracts the clip numbers along with the name of the clip
# Used as a reference when extracting frames for Neural Network inputs
def getClipLabels():
    for number in range(1,7):
        clipLabels = []
        count = 1
        for filename in os.listdir(str('/Volumes/MULTIPIE/FOR USE/' + str(number) + '/')):
            if filename.endswith(".MP4"):
                print(filename)
                baseName = os.path.splitext(filename)[0]
                print(baseName)
                if baseName[0:2] != '._':
                    clipLabels.append([baseName, count])
                    count += 1
        clipDf = pd.DataFrame(clipLabels, columns=['name', 'number'])
        clipDf.to_csv(str('/Volumes/MULTIPIE/FOR USE/' + str(number) + '.csv'), index=False, header=True)


if __name__ == '__main__':
    # consecJunc(10)
    # clipSort()
    # makeMovie(1)
    # makeAllMovies()
    getClipLabels()
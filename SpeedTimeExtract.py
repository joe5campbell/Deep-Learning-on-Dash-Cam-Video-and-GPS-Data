import pynmea2
import pandas as pd
import imageio
from datetime import datetime
import math
from csv import writer
from concurrent.futures import ProcessPoolExecutor
import cv2
import multiprocessing
import os
import sys

# dataFrame used to store the start times of each clip to be used as a reference point in calculations
starts = pd.DataFrame(columns=['file', 'startTime'])

# function for appending to CSVs
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

# function to turn a timestamp into a number of seconds
def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

# function which prints progress during video_to_frames
def print_progress(iteration, total, prefix='', suffix='', decimals=3, bar_length=100):
    """
    Call in a loop to create standard out progress bar
    :param iteration: current iteration
    :param total: total iterations
    :param prefix: prefix string
    :param suffix: suffix string
    :param decimals: positive number of decimals in percent complete
    :param bar_length: character length of bar
    :return: None
    """
    format_str = "{0:." + str(decimals) + "f}"  # format the % done number string
    percents = format_str.format(100 * (iteration / float(total)))  # calculate the % done
    filled_length = int(round(bar_length * iteration / float(total)))  # calculate the filled bar length
    bar = '#' * filled_length + '-' * (bar_length - filled_length)  # generate the bar string
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),  # write out the bar
    sys.stdout.flush()  # flush to stdout

# function for extracting individual frames from videos
def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
    """
    Extract frames from a video using OpenCVs VideoCapture
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    assert os.path.exists(video_path)  # assert the video file exists

    capture = cv2.VideoCapture(video_path)  # open the video using OpenCV

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved

    while frame < end:  # lets loop through the frames until the end

        _, image = capture.read()  # read an image from the capture

        if _ == True:
            crop_img = image[0:890, 0:1700]
            b = cv2.resize(crop_img, (128, 128), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        else:
            pass

        if while_safety > 500:  # break the while if our safety maxs out at 500
            break

        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip

        if frame % every == 0:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count
            save_path = os.path.join(frames_dir, video_filename, "{:010d}.jpg".format(frame))  # create the save path
            if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                cv2.imwrite(save_path, b)  # save the extracted image
                saved_count += 1  # increment our counter by one

        frame += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture

    return saved_count  # and return the count of the images we saved

# function which utilises multiple cores to quickly extract frames from a video
def video_to_frames(video_path, frames_dir, overwrite=False, every=1, chunk_size=1000):
    """
    Extracts the frames from a video using multiprocessing
    :param video_path: path to the video
    :param frames_dir: directory to save the frames
    :param overwrite: overwrite frames if they exist?
    :param every: extract every this many frames
    :param chunk_size: how many frames to split into chunks (one chunk per cpu core process)
    :return: path to the directory where the frames were saved, or None if fails
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    # make directory to save frames, its a sub dir in the frames_dir with the video name
    os.makedirs(os.path.join(frames_dir, video_filename), exist_ok=True)

    def change_res(width, height):
        capture.set(3, width)
        capture.set(4, height)

    capture = cv2.VideoCapture(video_path)  # load the video

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # get its total frame count
    capture.release()  # release the capture straight away


    if total < 1:  # if video has no frames, might be and opencv error
        print("Video has no frames. Check your OpenCV + ffmpeg installation, can't read videos!!!\n"
              "You may need to install OpenCV by source not pip")
        return None  # return None

    frame_chunks = [[i, i+chunk_size] for i in range(0, total, chunk_size)]  # split the frames into chunk lists
    frame_chunks[-1][-1] = min(frame_chunks[-1][-1], total-1)  # make sure last chunk has correct end frame

    prefix_str = "Extracting frames from {}".format(video_filename)  # a prefix string to be printed in progress bar

    # execute across multiple cpu cores to speed up processing, get the count automatically
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:

        futures = [executor.submit(extract_frames, video_path, frames_dir, overwrite, f[0], f[1], every)
                for f in frame_chunks]  # submit the processes: extract_frames(...

        # for i, f in enumerate(as_completed(futures)):  # as each process completes
        #     print_progress(i, len(frame_chunks)-1, prefix=prefix_str, suffix='Complete')  # print it's progress

    return os.path.join(frames_dir, video_filename)  # when done return the directory containing the frames

# function to extract the speed at all timestamps within for a given video
def stExtract(fileName, foldername, fileDir, verbose = False):
    # dataframe to store speed and time
    stDf = pd.DataFrame(columns=['second', 'speed'])
    global starts
    # Open File
    file = open(fileDir)
    # Extract name of file without extension
    baseName = os.path.splitext(fileName)[0]
    # Ignore hidden files
    if baseName[0:2] == '._':
        return
    else:
        # Read specified number of RMC lines
        # We get one reading of each type every second
        cycleCount = 0
        for line in file.readlines():
            # Cycle through RMC lines only
            if line.startswith('$GPRMC'):
                try:
                    msg = pynmea2.parse(line)
                    if cycleCount == 0:
                        # Setting start time for video from frame 0
                        startTime = datetime.strptime(str(msg.timestamp), "%H:%M:%S")
                        starts = starts.append({'file': baseName, 'startTime': startTime}, ignore_index=True)
                    # Setting current frame time
                    print(str(msg.timestamp))
                    try:
                        currentTime = datetime.strptime(str(msg.timestamp), "%H:%M:%S")
                    except ValueError:
                        pass
                    # Calculate time relative to start
                    currRelSecs = get_sec(str(currentTime - startTime))
                    if verbose == True:
                        print("Timestamp: " + str(currRelSecs))
                        print(repr(msg))
                    # Extract Speed in KPH
                    if msg.spd_over_grnd == None:
                        KPH = 'nan'
                    else:
                        KPH = (msg.spd_over_grnd)*1.852
                    if verbose == True:
                        print('KPH: ' + str(KPH) + '\n')
                    stDf = stDf.append({'second': currRelSecs, 'speed': KPH}, ignore_index=True)
                    cycleCount += 1
                except pynmea2.ParseError as e:
                    print('Parse error: {}'.format(e))
                    continue
        if verbose == True:
            print(stDf)
        # extract dataframe to csv
        print('Making CSV')
        stDf.to_csv(str('~/PycharmProjects/Diss/speeds/' + foldername + '/' + baseName + '.csv'), index=False, header=True)

# function to apply stExtract to all videos
def allStExtract(verbose = False):
    # Iterate through different weeks (folders) and gps files to apply stExtract to all
    for foldername in os.listdir('/Volumes/MULTIPIE/CAR-CAM'):
        for filename in os.listdir(str('/Volumes/MULTIPIE/CAR-CAM/' + foldername + '/gps')):
            fileDir = str('/Volumes/MULTIPIE/CAR-CAM/' + foldername + '/gps/' + filename)
            stExtract(filename, foldername, fileDir, verbose)

# function to apply video_to_frames to all videos
def allImExtract(verbose = False):
    # Iterate through weeks (folders) and videos (files) to apply video_to_frames to all
    for foldername in os.listdir('/Volumes/MULTIPIE/CAR-CAM'):
        for filename in os.listdir(str('/Volumes/MULTIPIE/CAR-CAM/' + foldername + '/videos')):
            if verbose == True:
                print(str(foldername + '/' + filename))
            if filename[0:2] == '._':
                print('Ignoring Hidden File!')
            else:
                video_to_frames(video_path=str('/Volumes/MULTIPIE/CAR-CAM/' + foldername + '/videos/' + filename),
                                frames_dir=str('/Volumes/MULTIPIE/CAR-CAM/' + foldername + '/small_Images'), overwrite=False, every=1,
                                chunk_size=1000)

# function to calculate which second each frame falls within for every video
def calcSeconds(verbose = False):
    # Iterate through weeks (folders) and image sets (videos)
    for foldername in os.listdir('/Volumes/MULTIPIE/CAR-CAM'):
        if verbose == True:
            print(foldername)
        for videoname in os.listdir(str('/Volumes/MULTIPIE/CAR-CAM/' + foldername + '/images')):
            if verbose == True:
                print(videoname)
            # path name for given video
            path = str('/Volumes/MULTIPIE/CAR-CAM/' + foldername + '/images/' + videoname)
            vidFile = str('/Volumes/MULTIPIE/CAR-CAM/' + foldername + '/videos/' + videoname)
            if videoname[0:2] == '._':
                print('Ignoring Hidden File!')
            else:
                # Extracting video to get fps data
                vid = imageio.get_reader(vidFile,  'ffmpeg')
                fps = vid.get_meta_data()['fps']
                vidBaseName = os.path.splitext(videoname)[0]
                # open and close csv to clear
                f1 = str('/Volumes/MULTIPIE/CAR-CAM/' + foldername + '/seconds/' + vidBaseName + '.csv')
                # opening the file with w+ mode truncates the file
                f = open(f1, "w+")
                f.close()
            # iterate through images
            for filename in sorted(os.listdir(path)):
                # if verbose == True:
                #     print(str('/Volumes/MULTIPIE/CAR-CAM/' + foldername + '/images/' + videoname + '/' + filename))
                if filename[0:2] == '._':
                    print('Ignoring Hidden File!')
                else:
                    # get image number without .jpg extension
                    baseName = os.path.splitext(filename)[0]
                    # calculate exact time in seconds of each image
                    timestamp = float(baseName) / fps
                    # round down to get an integer second value
                    second = math.floor(timestamp)
                    # append image number and second in which it occurs to corresponding csv
                    append_list_as_row(f1, [baseName, second])

if __name__ == '__main__':
    # calcSeconds(verbose=True)
    # video_to_frames(video_path='/Volumes/MULTIPIE/CAR-CAM/JAG-WEEK-1/videos/TS_N0005.MP4', frames_dir='/Volumes/MULTIPIE/CAR-CAM/JAG-WEEK-1/small_Images', overwrite=False, every=1, chunk_size=1000)
    # allStExtract(verbose=True)
    allImExtract(verbose=True)
    # print("Nothing Running ATM")


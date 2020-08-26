# Deep-Learning-on-Dash-Cam-Video-and-GPS-Data

This repository contains the five different python files utilised for data preparation within the context of the 2020 Dissertation project, Deep Learning on Dash-Cam Video and GPS Data.

The code in these files can not be run locally because they rely on both having the relevant data set present and having the directory structure for the data formatted correctly. These scripts are purely included to provide more context to the dissertation report and to hopefully make understanding of the data preparation sections of the report easier.

Below is a description of each file:

SPEED DETECTION:

  SpeedTimeExtract.py
    This file contains functions for extracting speed values from GPS data for every second of data in the data set. It also includes functions for extracting still images of         every frame from every video in the data set for later use.
    
  SpeedDetect.py
    This file contains functions to extract data in the format required for a regression based neural network.


JUNCTION CLASSIFICATION:

  JunctionExtract.py
    This file contains functions for the entire data preparation process if the Junction Classification section of this disseration.
    

STRESS LEVEL PREDICTION:

  ClipExtract.py
    This file contains the functions utilised to create the videos for the Stress Level Questionnaire.
   
  StressExtract.py
    This file contains the functions used for extracting information from the stress level questionnaire and processing it into input data for the stress level neural network.
   

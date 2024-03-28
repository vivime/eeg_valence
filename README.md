# BA-Vivien-Mezei

This README provides an overview and explanation of the Python scripts provided for EEG data streaming, handling and analysing. 
The provided scripts utilize the muselsl library for interfacing with Muse EEG headsets.

## Getting started

1. Install requirements.txt
2. If there is a problem with installing pygatt try: conda install -c conda-forge liblsl
3. Make sure you have ROS installed: On Mac see: https://robostack.github.io/GettingStarted.html and follow the instructions there. 
4. Create a ROS-environment

Starting the application:
Make sure you are in your ROS-environment
1. In one terminal start a Roscore with "roscore"
2. In another terminal run the main with "python3 main.py"
3. To see what information the Rosnode is getting run in another terminal: "rostopic echo /topic_name". 
4. To record the data to a Rosbag use "rosbag record -a". This records all topics.
See http://wiki.ros.org/rosbag/Commandline for more information.

Also make sure to turn the Muse device on.

Used package: muselsl
Copyright © 2018, authors of muselsl
All rights reserved.

Disclaimer from the muselsl package:
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Files
### Connection/Streaming
Ensure that you have the necessary hardware (Muse EEG headset) and software dependencies installed, including Python and the muselsl library.
Make sure the Muse headset is properly paired and connected to the computer via Bluetooth.
Run the main.py script to initiate the EEG data streaming process.
The script will automatically detect available Muse devices and start streaming from the first detected device.
You can adjust streaming settings such as enabling/disabling data types (PPG, ACC, GYRO) directly in the main.py script.

#### Main Script ('main.py'):
This script serves as the entry point for starting the EEG data streaming process.
It utilizes functions from the muselsl library to list available Muse devices and initiate streaming from a selected device.

#### Stream Script ('stream.py'):
This script contains the core functionality for streaming EEG data from the Muse device.
It defines a stream function which establishes connections, configures data streams, and starts streaming data.

#### Data Handler ('data_handler.py'):
This script handles the processing and saving of EEG data.
It includes a DataHandler class responsible for managing incoming EEG data, writing it to a CSV file, and publishing it to ROS topics.
The class also includes methods for feature extraction and valence prediction during streaming using a pre-trained TensorFlow model.

### Working with Data
To prepare and analyse EEG data a connection to the Muse headband is not needed.
You can run each script without it.

#### Data Preparation ('data_preparation.py')
This script prepares EEG recordings and movie clip data for analysis by combining and processing them into a unified dataset. 
EEG recordings and the movie clip data both come from CSVs.
The EEG recordings come from CSV files located in the recordings directory.
The movie clip data comes from the CSV file (movieclips.csv).
The combined EEG and movie clip data is saved to a CSV file (eeg_recordings.csv).
A separate CSV files for each video of every participant is created for further analysis.

Run the data_preparation.py script to prepare EEG recordings and movie clip data for analysis.


#### Data Analysis ('data_analysis.py')
This script performs data analysis on EEG recordings by extracting features from labeled CSV files.
It iterates through each CSV file, extracting EEG signal matrices and labels.
Feature extraction is applied to compute various statistical features from the EEG signals.
Combines the extracted features into a single DataFrame.

This script also checks for imbalanced classes:
Computes the frequency of valence, arousal, and dominance values in the extracted feature DataFrame.
Visualizes the frequency distribution of valence, arousal, and dominance values using bar plots.

The extracted features are saved to a CSV file for further analysis.

#### Feature Extraction ('feature_extraction.py')
The functions of this script are called from the data_analysis funktion. This module provides functions for extracting various features from EEG data. 
It includes the following statistical features:

- mean
- standard deviation
- skewness
- kurtosis
- maximum 
- minimum 
- covariance 
- eigenvalues
- FFT-based features

Ensure EEG data is available in labeled CSV files in the specified directory.
Run the data analysis script to extract features from EEG data.

#### Valence Model('valence_model.py')
Contains code for building and evaluating machine learning models to predict valence based on EEG data. 
The models implemented include Support Vector Machine (SVM), Gated Recurrent Unit (GRU), and Convolutional Neural Network (CNN).


#### File Comparison ('ros_csv_comparison.py')
This script compares data saved locally as CSV files with data sent to ROS topics and saved in a rosbag. 
Provide paths to the CSV file containing EEG data and the rosbag file containing ROS data.

#### Visualization ('visualize.py')
This Python script visualizes EEG data for a specified video of a participant. 
It plots the EEG data for selected channels over time.
To visualize replace the filePath variable with the path to your EEG data CSV file.
Specify the channels you want to plot in the channels_to_plot list.
Run the script to visualize EEG data for the specified channels over time.


## Notes

Make sure to replace placeholder paths or adjust file paths as necessary to match your system configuration.
Ensure that ROS is installed and configured properly (see Getting started) if you intend to use ROS functionality.
Additional customization and extension of functionality can be achieved by modifying the provided scripts as needed.

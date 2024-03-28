import pandas as pd
import rosbag
from ast import literal_eval

def compare_ros_to_csv(rosbag_file, csv_file):

    csv_data = pd.read_csv(csv_file)
    bag = rosbag.Bag(rosbag_file)
    rosbag_data = []
    timestamps = []

    for topic, msg, t in bag.read_messages(topics= '/eeg_raw_data'):
        data_str = msg.data
        split_data = data_str.split(', ')

        timestamp_str = split_data[0] #sent time
        values_str = ', '.join(split_data[1:])#time sent: 1, no time sent:0

        timestamp = pd.to_datetime(timestamp_str)#sent time

        values = literal_eval(values_str)
        timestamps.append(timestamp)#sent time
        rosbag_data.append((timestamp, *values))

    bag.close()

    # Create a DataFrame
    column_names = ['timestamp', 'TP9', 'AF7', 'AF8', 'TP10', 'RightAUX']
    rosbag_df = pd.DataFrame(rosbag_data, columns=column_names)

    csv_data= csv_data.rename(columns={'Timestamp': 'timestamp'})
    csv_data['timestamp'] = pd.to_datetime(csv_data['timestamp'])

    # Check if both dataframes have the same columns
    if not csv_data.columns.equals(rosbag_df.columns):
        raise ValueError("Columns in the dataframes are not identical")

    # Check data types
    if not csv_data.dtypes.equals(rosbag_df.dtypes):
        raise ValueError("Data types in the dataframes are not identical")

    # Calculate delay
    delay = rosbag_df['timestamp'] - csv_data['timestamp']

    # Calculate frequency
    csv_frequency = 1 / (csv_data['timestamp'].diff().mean().total_seconds())
    rosbag_frequency = 1 / (rosbag_df['timestamp'].diff().mean().total_seconds())

    # Data missing from rosbag
    unique_data_csv = csv_data[~csv_data['timestamp'].isin(rosbag_df['timestamp'])]

    print(f"Delay between dataframes: {delay.mean()} seconds")
    print(f"Frequency in CSV dataframe: {csv_frequency} Hz")
    print(f"Frequency in ROSbag dataframe: {rosbag_frequency} Hz")
    print("Data only in CSV dataframe:")
    print(unique_data_csv)

    rosbag_first_timestamp = rosbag_df['timestamp'].iloc[0]
    rosbag_latest_timestamp = rosbag_df['timestamp'].iloc[-1]
    filtered_csv_df = csv_data[(csv_data['timestamp'] >= rosbag_first_timestamp) & (csv_data['timestamp'] <= rosbag_latest_timestamp)]
    unique_data = filtered_csv_df[~filtered_csv_df['timestamp'].isin(rosbag_df['timestamp'])]
    print("Data only in CSV dataframe with same starttime:")
    print(unique_data)


compare_ros_to_csv(csv_file='/recordings/recording_2024-01-29 14:56:56.csv', rosbag_file='/rosbags/2024-01-29-14-56-46.bag')

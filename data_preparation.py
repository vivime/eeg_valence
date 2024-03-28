from datetime import datetime
import pandas as pd
import os

# EEG_RECORDINGS
# Load all recordings in one dataframe
eeg_recordings = 'recordings'
csv_files = [file for file in os.listdir(eeg_recordings) if file.endswith('.csv')]
dfs = []

for file in csv_files:
    file_path = os.path.join(eeg_recordings, file)
    df = pd.read_csv(file_path)
    dfs.append(df)

eeg_recordings_df = pd.concat(dfs, ignore_index=True)

# Sort the DataFrame based on the 'Timestamp' column in ascending order
eeg_recordings_df = eeg_recordings_df.sort_values(by='Timestamp')
# Drop the 'Right_AUX' column
eeg_recordings_df = eeg_recordings_df.drop('RightAUX', axis=1)
# Convert Timestamp to CET
eeg_recordings_df['Timestamp'] = pd.to_datetime(eeg_recordings_df['Timestamp'], format='ISO8601')
eeg_recordings_df['Timestamp'] = eeg_recordings_df['Timestamp'] + pd.Timedelta('1 hour')


# MOVIECLIPS_CSV
# Add the end timestamp of the movieclips, and the duration, and the time to which the observation begins(end timestamp-60s)
movieclips_csv_file_path = 'movieclips.csv'
movieclips_df = pd.read_csv(movieclips_csv_file_path, delimiter=';')

clip_duration = {
    'baseline.m4v': '62',
    '1.m4v': '200',
    '2.m4v': '132',
    '3.m4v': '349',
    '4.m4v': '167',
    '5.m4v': '137',
    '6.m4v': '191',
    '7.m4v': '193',
    '8.m4v': '395',
    '9.m4v': '146',
}

movieclips_df['duration'] = movieclips_df['Video'].map(clip_duration)
movieclips_df['duration'] = pd.to_numeric(movieclips_df['duration'], errors='coerce')

movieclips_df['Start_Timestamp'] = movieclips_df['Start_Timestamp'].apply(lambda x: datetime.strptime(x.replace('Di', 'Tue').replace('Mi', 'Wed'), '%a %d %b %Y %H:%M:%S CET'))
movieclips_df['Start_Timestamp'] = movieclips_df['Start_Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
movieclips_df['Start_Timestamp'] = pd.to_datetime(movieclips_df['Start_Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')

movieclips_df['duration_delta'] = pd.to_timedelta(movieclips_df['duration'], unit='s')


movieclips_df['End_Timestamp'] = movieclips_df['Start_Timestamp'] + movieclips_df['duration_delta']
movieclips_df['End_Timestamp'] = movieclips_df['End_Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
movieclips_df['End_Timestamp'] = pd.to_datetime(movieclips_df['End_Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
movieclips_df['Observation_Timestamp'] = movieclips_df['End_Timestamp'] - pd.to_timedelta(60, unit='s')
movieclips_df['Observation_Timestamp'] = movieclips_df['Observation_Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')


# Combine eeg_recordings with movieclips
movieclips_df['Observation_Timestamp'] = pd.to_datetime(movieclips_df['Observation_Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
movieclips_df['End_Timestamp'] = pd.to_datetime(movieclips_df['End_Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
eeg_recordings_df['Timestamp'] = pd.to_datetime(eeg_recordings_df['Timestamp'], format='ISO8601')

for _, row in movieclips_df.iterrows():

    condition = (
            (eeg_recordings_df['Timestamp'] >= row['Observation_Timestamp']) &
            (eeg_recordings_df['Timestamp'] <= row['End_Timestamp']+pd.to_timedelta(60, unit='s'))
    )

    # Update the corresponding rows in eeg_recordings_df
    eeg_recordings_df.loc[condition, ['Valence', 'Arousal', 'Dominace', 'Viewed', 'Video', 'Participant']] = row[
        ['Valence', 'Arousal', 'Dominace', 'Viewed', 'Video', 'Participant']].values


eeg_recordings_df.dropna(subset=['Valence'], inplace=True)
print(eeg_recordings_df)

eeg_recordings_df.to_csv('eeg_recordings.csv', index=False)

# Create a CSV for every video of every participant
# Iterate through all combinations of video and participant
for video in map(str, range(1, 10)):
    for participant in map(float, range(1, 12)):
        condition = (eeg_recordings_df['Video'] == f"{video}.m4v") & (eeg_recordings_df['Participant'] == participant)

        result_df = eeg_recordings_df[condition]

        if not result_df.empty:
            output_filename = f"{'out'}/video_{video}_participant_{participant}.csv"
            result_df.to_csv(output_filename, index=False)



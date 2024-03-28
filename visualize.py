import pandas as pd
from matplotlib import pyplot as plt
"""
Visualize the EEG data for a specified video of a specified participant

"""

filePath = '/output_directory/medium_data/video_6_participant_3.0.csv'
eeg_data = pd.read_csv(filePath, parse_dates=['Timestamp'])
eeg_data['time_seconds'] = (eeg_data['Timestamp'] - eeg_data['Timestamp'].iloc[0]).dt.total_seconds()


channels_to_plot = ['TP9', 'AF7', 'AF8', 'TP10']


fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# Plot EEG data for each channel in a separate subplot
for i, channel in enumerate(channels_to_plot):
    axs[i].plot(eeg_data['time_seconds'], eeg_data[channel], label=channel)
    axs[i].set_ylabel('Amplitude')
    axs[i].legend()

axs[-1].set_xlabel('Time (seconds)')
plt.suptitle('EEG Data - Channels', y=0.92)
plt.show()





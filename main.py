from muselsl import list_muses, stream


muses = list_muses()
stream(address=muses[0]['address'], backend='auto', ppg_enabled=True, acc_enabled=True, gyro_enabled=True, eeg_disabled=False)

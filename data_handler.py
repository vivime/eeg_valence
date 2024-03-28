import csv
import os
from datetime import datetime
import numpy as np
import rospy
import tensorflow as tf
from std_msgs.msg import String
from feature_extraction import generate_feature_vectors_from_samples


def _remove_trailing_items(list_of_objects, limit: int):
    if len(list_of_objects) > limit:
        list_of_objects[:] = list_of_objects[:limit]
    return list_of_objects


class DataHandler:
    def __init__(self):
        rospy.init_node('mac_publisher', anonymous=True)
        self.output_file_name = f'recording_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.csv'
        self.pub_raw_eeg = rospy.Publisher('eeg_raw_data', String, queue_size=10)
        self.pub_processed_egg = rospy.Publisher('eeg_processed_data', String, queue_size=10)
        self.pub_head_ppg = rospy.Publisher('head_ppg', String, queue_size=10)
        self.pub_head_acc = rospy.Publisher('head_acc', String, queue_size=10)
        self.pub_head_gyro = rospy.Publisher('head_gyro', String, queue_size=10)

        self.timestamps = []
        self.data = []
        self.tp9 = []
        self.af7 = []
        self.af8 = []
        self.tp10 = []
        self.right_aux = []
        self.model = tf.keras.models.load_model('models/gru_model_valence.h5')
        self._write_header()

        self.data = {channel: [] for channel in ['TP9', 'AF7', 'AF8', 'TP10']}

        self.valence_mapping = {
            0: 'negative',
            1: 'neutral',
            2: 'positive',
        }

    def handle_eeg_data(self, timestamps, data):
        # Write Datapoint to csv
        with open(os.getcwd() + f'/recordings/{self.output_file_name}', 'a', newline='') as csvfile:
            header = ['Timestamp', 'Timestamp_UNIX', 'TP9', 'AF7', 'AF8', 'TP10', 'RightAUX']
            csv_writer = csv.DictWriter(csvfile, fieldnames=header)

            for index, data_point in enumerate(data):
                data_dict = {
                    'Timestamp': datetime.utcfromtimestamp(timestamps[index]),
                    'Timestamp_UNIX': timestamps[index],
                    'TP9': data_point[0],
                    'AF7': data_point[1],
                    'AF8': data_point[2],
                    'TP10': data_point[3],
                    'RightAUX': data_point[4],
                }

                self._pour_values(data_dict=data_dict)
                csv_writer.writerow(data_dict)

                # Publish Data to ROS
                message = f"{datetime.utcfromtimestamp(timestamps[index])}, {data_point}"
                self.pub_raw_eeg.publish(str(message))

        # Predict valence
        self.timestamps.extend(timestamps)
        if self._prediction_is_due():
            prediction_result = self._predict_valence(self._extract_features())
            print(f"The current valence is: {self.valence_mapping.get(int(prediction_result[0]))}")
            self._reset_data_lists()

    def _write_header(self):
        with open(os.getcwd() + f'/recordings/{self.output_file_name}', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Timestamp', 'Timestamp_UNIX', 'TP9', 'AF7', 'AF8', 'TP10', 'RightAUX'])

    def handle_ppg_data(self, timestamps, ppg_data):
        for index, data_point in enumerate(ppg_data):
            message = f"{datetime.utcfromtimestamp(timestamps[index])}, {data_point}"
            self.pub_head_ppg.publish(str(message))

    def handle_acc_data(self, timestamps, acc_data):
        for index, data_point in enumerate(acc_data):
            message = f"{datetime.utcfromtimestamp(timestamps[index])}, {data_point}"
            self.pub_head_acc.publish(str(message))

    def handle_gyro_data(self, timestamps, gyro_data):
        for index, data_point in enumerate(gyro_data):
            message = f"{datetime.utcfromtimestamp(timestamps[index])}, {data_point}"
            self.pub_head_gyro.publish(str(message))






    def _predict_valence(self, features):
        keras_tensor = tf.expand_dims(features, axis=-1)
        keras_tensor = tf.expand_dims(keras_tensor, axis=0)
        valence_predictions = np.array(list(map(lambda x: np.argmax(x), self.model.predict(keras_tensor))))
        return valence_predictions


    def _generate_array_from_lists(self):
        # Stack the lists horizontally
        result_array = np.column_stack((self.timestamps, self.tp9, self.af7, self.af8, self.tp10, self.right_aux))
        return result_array

    def _extract_features(self) -> np.array:
        temp_np = np.column_stack((self.timestamps, self.tp9, self.af7, self.af8, self.tp10))
        vectors, header = generate_feature_vectors_from_samples(temp_np= temp_np, nsamples=150, period=3., state=None)
        return vectors

    def _prediction_is_due(self):
        if len(self.timestamps) == 0:
            return False
        seconds_difference = (datetime.utcfromtimestamp(round(self.timestamps[0])) - datetime.utcfromtimestamp(round(self.timestamps[-1])))
        if abs(seconds_difference.total_seconds()) >= 5:
            return True
        return False

    def _reset_data_lists(self):
        self.timestamps = []
        self.tp9 = []
        self.af7 = []
        self.af8 = []
        self.tp10 = []
        self.right_aux = []

    def _pour_values(self, data_dict):
        self.tp9.append(data_dict.get('TP9'))
        self.af7.append(data_dict.get('AF7'))
        self.af8.append(data_dict.get('AF8'))
        self.tp10.append(data_dict.get('TP10'))
        self.right_aux.append(data_dict.get('RightAUX'))

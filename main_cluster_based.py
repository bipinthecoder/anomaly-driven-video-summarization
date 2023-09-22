import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

from utils import helper_functions as hf
from utils import change_detection as cd
from utils import video_summarization_helper_functions as vshf
import Config as Config
from sklearn.cluster import KMeans

logging.getLogger().setLevel(logging.INFO)

# Video file
anomaly_video_file_path = Config.TO_PROCESS_VIDEO_FILE_PATH
file_name = anomaly_video_file_path.split('/')[-1].split('.')[0]

logging.info(f'processing video file: {file_name}')

# Extracting the frames from video
video_frames_dict = hf.get_frames_from_video(anomaly_video_file_path)

logging.info(f'Number of Frames in Input Video : {len(video_frames_dict)}')

# Grouping frames into batches of 30, since fps=30
frames_in_batches = hf.get_frames_in_batches(30, video_frames_dict)

# Getting frame batches and corresponding order number in a dict
frame_sequence_tracker = hf.get_frame_sequence_tracker(frames_in_batches)

# Performing change detection to extract significant frame sequence keys
significant_keys = cd.perform_change_detection(frame_sequence_tracker)

if not significant_keys:
    hf.no_anomaly_detected()

# Getting one insignificant entry to check normal prediction score
insignificant_keys = hf.get_insignificant_frame(significant_keys)

# Loading the Fine Tuned Keras Model
model = tf.keras.models.load_model(Config.FINE_TUNED_LSTM_AUTO_ENCODER_PATH)

# Checking the prediction value of insignificant frame
insig_cost_tracker = {}

for key in insignificant_keys:
    insignificant_sequence = frame_sequence_tracker[insignificant_keys[0]]

    # Reshaping to get three 10-frame sequence batches for prediction
    reshaped_insignificant_sequences = insignificant_sequence.reshape(3, 10, 256, 256, 1)

    for index, single_sequence in enumerate(reshaped_insignificant_sequences):
        single_sequence = np.expand_dims(single_sequence, axis=0)
        reconstructed_sequence = model.predict(single_sequence)

        # Calculating reconstruction cost for the single sequence
        # cost = np.linalg.norm(single_sequence - reconstructed_sequence[0])
        cost = hf.get_reconstruction_cost(single_sequence, reconstructed_sequence)
        insig_cost_tracker[f'{key}_{index}'] = cost

# Getting Max Insignificance prediction score
max_insignificant_prediction_score = hf.get_max_from_cost_dict(insig_cost_tracker)

# calculating anomaly threshold
cut_off_threshold_percent = 4
anomaly_threshold = hf.get_anomaly_threshold_from_normal_cost(max_insignificant_prediction_score,
                                                              cut_off_threshold_percent)

# Significant frames cost tracker
cost_tracker = {}
for key in significant_keys:
    significant_sequence = frame_sequence_tracker[key]
    reshaped_sequences = significant_sequence.reshape(3, 10, 256, 256, 1)
    for index, single_sequence in enumerate(reshaped_sequences):
        single_sequence = np.expand_dims(single_sequence, axis=0)
        reconstructed_sequence = model.predict(single_sequence)

        # Calculating reconstruction cost for the single sequence
        # cost = np.linalg.norm(single_sequence - reconstructed_sequence[0])
        cost = hf.get_reconstruction_cost(single_sequence, reconstructed_sequence)
        cost_tracker[f'{key}_{index}'] = cost

# Defining keys to consider as a set to avoid duplicates
keys_to_consider = set()

# Logic to track all the prediction cost of significant frames
cost_list = hf.get_values_from_dictionary(cost_tracker)

# Cost list for K-means clustering
k_means_cost_list = np.array(cost_list).reshape(-1, 1)

# Initializing with 'number_of_clusters' clusters: Anomaly and Non Anomaly
number_of_clusters = Config.NUMBER_OF_CLUSTERS

kmeans = KMeans(n_clusters=number_of_clusters)
kmeans.fit(k_means_cost_list)

labels = kmeans.labels_

cluster_centers = kmeans.cluster_centers_

plt.scatter(range(len(k_means_cost_list)), k_means_cost_list, c=labels)
plt.scatter([0] * len(cluster_centers), cluster_centers, c='orange')
plt.show()

anomalous_center = max(cluster_centers)
anomalies = [cost for cost, label in zip(k_means_cost_list, labels) if cluster_centers[label] == anomalous_center]

# Assuming the video is not anomalous
# anomalous_video = False

for anomaly_cost in anomalies:
    for key, value in cost_tracker.items():
        if value == anomaly_cost:
            # if value > anomaly_threshold:
            #     anomalous_video = True
            keys_to_consider.add(key)

# if not anomalous_video or not keys_to_consider:
if not keys_to_consider:
    hf.no_anomaly_detected()

# Fetching all frame keys in a list
frame_key_list = vshf.get_frame_key_list(keys_to_consider)

# Perform Buffer frame appending for meaningful video summarization
final_frames_key_list = vshf.get_final_frames_key_list(frame_key_list)

# Logic to get original frames for summarization
original_frames_for_summarization = \
    vshf.get_original_frames_for_summarization(final_frames_key_list=final_frames_key_list,
                                               original_frames_dict=video_frames_dict)

logging.info(f'Number of Frames in Output Video: {len(original_frames_for_summarization)}')

save_frames_as_video = \
    vshf.save_summarized_frames_as_video(file_name=file_name,
                                         save_location=Config.TO_SAVE_DIRECTORY,
                                         frames_list=original_frames_for_summarization)

logging.info(save_frames_as_video)

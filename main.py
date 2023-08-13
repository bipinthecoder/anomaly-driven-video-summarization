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

logging.getLogger().setLevel(logging.INFO)

# Video file
anomaly_video_file_path = Config.TO_PROCESS_VIDEO_FILE_PATH
file_name = anomaly_video_file_path.split('/')[-1].split('.')[0]

# Loading the Fine Tuned Keras Model
model = tf.keras.models.load_model(Config.FINE_TUNED_LSTM_AUTO_ENCODER_PATH)

# Extracting the frames from video
video_frames_dict = hf.get_frames_from_video(anomaly_video_file_path)

# Grouping frames into batches of 30, since fps=30
frames_in_batches = hf.get_frames_in_batches(30, video_frames_dict)

# Getting frame batches and corresponding order number in a dict
frame_sequence_tracker = hf.get_frame_sequence_tracker(frames_in_batches)

# Performing change detection to extract significant frame sequence keys
significant_keys = cd.perform_change_detection(frame_sequence_tracker)

# Getting one insignificant entry to check normal prediction score
insignificant_keys = hf.get_insignificant_frame(significant_keys)

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
anomaly_threshold = hf.get_anomaly_threshold_from_normal_cost(max_insignificant_prediction_score)

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

# Calculating the prediction cost based on predicted cost list
cost_threshold = hf.calculate_prediction_cost_threshold(cost_list)

# Logic to extract frames with more reconstruction cost
for index, cost in cost_tracker.items():
    if cost > anomaly_threshold:
        keys_to_consider.add(index)

if len(keys_to_consider) == 0:
    for index, cost in cost_tracker.items():
        if cost > cost_threshold:
            keys_to_consider.add(index)

# Fetching all frame keys in a list
frame_key_list = vshf.get_frame_key_list(keys_to_consider)

# Perform Buffer frame appending for meaningful video summarization
final_frames_key_list = vshf.get_final_frames_key_list(frame_key_list)

# Logic to get original frames for summarization
original_frames_for_summarization = \
    vshf.get_original_frames_for_summarization(final_frames_key_list=final_frames_key_list,
                                               original_frames_dict=video_frames_dict)

logging.info(f'Original Frames for Summarization: {len(original_frames_for_summarization)}')

save_frames_as_video = \
    vshf.save_summarized_frames_as_video(file_name=file_name,
                                         save_location=Config.TO_SAVE_DIRECTORY,
                                         frames_list=original_frames_for_summarization)

logging.info(save_frames_as_video)

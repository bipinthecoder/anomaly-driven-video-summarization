import numpy as np
import utils.helper_functions as hf


def perform_change_detection(all_frames_in_batches_dict):
    """This function takes in a DICTIONARY of frames in batches
     and perform change detection. Returns a list of keys indicating
      the sequences with significant changes"""

    frame_average_tracker = {}
    key_list = []

    for frame_index, frame_sequence in all_frames_in_batches_dict.items():
        frame_sum = None
        frame_count = 0
        for frame in frame_sequence:
            frame_count += 1
            if frame_sum is None:
                frame_sum = frame.astype(np.float64)
            else:
                frame_sum += frame.astype(np.float64)

        frame_sequence_average = frame_sum / frame_count
        frame_average_tracker[frame_index] = frame_sequence_average

    key_list = get_significant_change_frames(frame_average_tracker)

    return key_list


def get_significant_change_frames(average_frames_tracker, threshold=0.002):
    """This function will return the frames with significant changes"""

    significant_change_key_list = []

    for i in range(1, len(average_frames_tracker)):
        frame_difference = average_frames_tracker[i] - average_frames_tracker[i + 1]
        # print(np.mean(frame_difference))
        if np.mean(frame_difference) > threshold:
            significant_change_key_list.append(i)

    return significant_change_key_list


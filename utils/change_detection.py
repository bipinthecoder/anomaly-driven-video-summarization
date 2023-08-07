import numpy as np
import utils.helper_functions as hf


def perform_change_detection(all_frames_in_batches_dict):

    """This function takes in a DICTIONARY of frames in batches
     and perform change detection. Returns a list of keys indicating
      the sequences with significant changes"""

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


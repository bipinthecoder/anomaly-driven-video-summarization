import numpy as np
import cv2
from utils import helper_functions as hf


def remove_duplicates_and_sort(list_data):
    list_set = set(sorted(list_data))
    final_list = sorted(list(list_set))
    return final_list


def get_frame_key_list(keys_to_consider):
    """This function returns a list of frame keys
    from input dictionary: Extracted from form: 'frameKey_index' """
    frame_key_list = []

    for items in keys_to_consider:
        key = int(items.split('_')[0])
        frame_key_list.append(key)
    return frame_key_list


def append_initial_buffer_frames(key_list):
    initial_frame_key = key_list[0]
    if initial_frame_key > 3:
        key_list.append(initial_frame_key - 1)
        key_list.append(initial_frame_key - 2)
    return key_list


def append_middle_buffer_frames(frame_key_list):
    final_frames_list = []
    for i in range(len(frame_key_list)):
        if i == 0 or i == len(frame_key_list) - 1:
            final_frames_list.append(frame_key_list[i])
        elif frame_key_list[i] != frame_key_list[i - 1]:
            final_frames_list.append(frame_key_list[i] - 1)
            final_frames_list.append(frame_key_list[i])
        else:
            final_frames_list.append(frame_key_list[i])
    final_frames_list = set(final_frames_list)
    final_frames_list = sorted(list(final_frames_list))
    return final_frames_list


def get_final_frames_key_list(frame_key_list):
    sorted_key_list = sorted(frame_key_list)

    sorted_key_list = append_initial_buffer_frames(sorted_key_list)
    sorted_key_list = remove_duplicates_and_sort(sorted_key_list)
    final_key_list = append_middle_buffer_frames(sorted_key_list)
    return final_key_list


def get_original_frames_for_summarization(final_frames_key_list, original_frames_dict):
    original_frames_for_summarization = []
    for key in final_frames_key_list:
        start, end = hf.get_original_frame_numbers(key, 30)
        for j in range(start, end + 1):
            original_frame = original_frames_dict[j]
            original_frames_for_summarization.append(original_frame)

    return original_frames_for_summarization


def save_summarized_frames_as_video(file_name, save_location, frames_list, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (frames_list[0].shape[1], frames_list[0].shape[0])
    output_file_name = save_location + '/' + file_name + '_output.mp4'
    out = cv2.VideoWriter(output_file_name, fourcc, fps, frame_size)

    for frame in frames_list:
        out.write(frame)
    out.release()

    return 'File Saved Successfully!'

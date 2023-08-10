from utils import data_extractor as de
import Config as Config
from utils import data_preprocessing as dp
import os
import numpy as np
import cv2
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


def get_all_frames_from_videos_in_directory():
    """This function will return all the frames in the video
     from directory as a LIST with a length as the multiple of provided sequence length, ie disregards
     the remaining frames"""

    all_frames = []

    files_in_directory = de.get_normal_files_in_directory()

    sequence_length = 10

    count = 0
    for file in tqdm(files_in_directory):

        file_path = os.path.join(Config.NORMAL_VIDEO_DIRECTORY, file)
        count += 1

        if str(file_path)[-3:] == "mp4":

            video_file = cv2.VideoCapture(file_path)
            video_frames = []
            while True:
                ret, frame = video_file.read()

                # If the end of frames or any error in reading
                if not ret:
                    break

                frame = dp.perform_frame_preprocessing(frame)
                video_frames.append(frame)

            video_file.release()
            cv2.destroyAllWindows()

            frames_length = len(video_frames)
            num_sequences = frames_length // sequence_length
            video_frames = video_frames[:num_sequences * sequence_length]
            all_frames.extend(video_frames)

        return all_frames


def get_dataset_in_sequences():
    """This function is used only for model training as it is a GENERATOR for
    proper utilization of GPU using TensorFlow"""

    # defining the frame list and a sequence length
    all_frames_in_videos = get_all_frames_from_videos_in_directory()
    frames_list = all_frames_in_videos

    size = len(frames_list)
    sequence_length = 10

    for i in range(0, size, sequence_length):
        sequence_frames = frames_list[i: i + sequence_length]

        if len(sequence_frames) == sequence_length:
            clip = np.zeros(shape=(sequence_length, 256, 256, 1))
            for j in range(sequence_length):
                frame = sequence_frames[j]
                clip[j] = frame

            yield clip, clip


def get_frames_in_batches(batch_size, frames_dict):
    count = 0
    frames_in_batch_size = []
    all_frames_list = []

    for i in range(1, len(frames_dict) + 1):
        count += 1
        frame = frames_dict[i]
        preprocessed_frame = dp.perform_frame_preprocessing(frame)
        frames_in_batch_size.append(preprocessed_frame)

        if count == batch_size:
            count = 0
            all_frames_list.extend(frames_in_batch_size)
            frames_in_batch_size = []

    all_frames_array = np.array(all_frames_list)
    num_batches = all_frames_array.shape[0] // batch_size
    all_frames_in_batches = all_frames_array.reshape((num_batches, batch_size, 256, 256, 1))

    return all_frames_in_batches


def get_frame_sequence_tracker(frames_in_batches):
    """This function will return a DICTIONARY with key as the sequence order number
    and value as the sequence itself"""

    frame_sequence_tracker = {}

    for count, frame_sequence in enumerate(frames_in_batches):
        frame_sequence_tracker[count + 1] = frame_sequence

    return frame_sequence_tracker


def get_frames_from_video(video_file_path):
    video_file = cv2.VideoCapture(video_file_path)

    total_frames = 0
    original_frames_dict = {}

    while True:
        ret, frame = video_file.read()

        # If the end of frames or any error in reading
        if not ret:
            break

        total_frames += 1

        original_frames_dict[total_frames] = frame

    video_file.release()
    cv2.destroyAllWindows()

    return original_frames_dict


def get_original_frame_numbers(sequence_key, batch_size=30):
    """Returns the start and end original frame numbers for the given sequence key."""

    start_frame = ((sequence_key - 1) * batch_size + 1)
    end_frame = sequence_key * batch_size

    return start_frame, end_frame


def display_image(image):
    plt.imshow(image)
    plt.show()


def test_func():
    print('hello world')

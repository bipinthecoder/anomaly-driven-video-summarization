import data_extractor as de
import Config as Config
import data_preprocessing as dp
import os
import numpy as np
import cv2
from tqdm.notebook import tqdm


def get_all_frames_from_videos_in_directory():
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
            video_frames = []

        return all_frames


def get_dataset_in_sequences():
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

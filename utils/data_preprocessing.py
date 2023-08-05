import numpy as np
import cv2


def convert_to_gray(img):
    converted_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return converted_gray


def resize_frame(frame, shape=(256, 256)):
    resized_image = cv2.resize(frame, shape)
    return resized_image


def normalize_frame(frame):
    normalized_frame = np.divide(frame, 255.0).astype(np.float32)
    return normalized_frame


def perform_frame_preprocessing(frame, shape=(256, 256)):
    resized_image = resize_frame(frame, shape)
    gray_converted = convert_to_gray(resized_image)
    normalized_image = normalize_frame(gray_converted)
    normalized_image = np.expand_dims(normalized_image, axis=-1)
    return normalized_image

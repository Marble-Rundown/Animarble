'''
This program extracts features from a synchronized .mp3 file into a .mvid file with landmarks and headpose data
'''

import csv
import os
import argparse
import dlib
import cv2
import numpy as np
from pose_estimator import PoseEstimator

from utils import create_unique_filename

ap = argparse.ArgumentParser(
    description="Extract features from a .mp3 file to a .mvid file")
ap.add_argument('input', help="The input .mp3 file to process")
ap.add_argument('-o', '--output', help="The output .mvid file to write to")

args = ap.parse_args()

video_file = args.input

output_filename = None
if args.output is None:
    output_filename = create_unique_filename(
        f"outputs/VideoFeatureExtractor/{os.path.splitext(os.path.basename(video_file))[0]}.mvid")
else:
    output_filename = args.output

''' Computer Vision Initialization '''
VIDEO_RESCALE = 0.5  # Use half resolution

# Get a face detector from the dlib library
detector = dlib.get_frontal_face_detector()
# Get a shape predictor based on a trained model from the dlib library
predictor = dlib.shape_predictor(
    'assets/shape_predictor_68_face_landmarks.dat')


def main():
    cap = cv2.VideoCapture(args['video'])
    assert cap.isOpened(), 'Failed to open video file'

    _, first_frame = cap.read()
    dimensions = tuple(first_frame.shape[i] * VIDEO_RESCALE for i in range(2))

    pe = PoseEstimator(dimensions)

    with open(output_filename, "w+") as output_file:
        fieldnames = ["timestamp"] + [f"rot_{d}" for d in "xyz"] + [
            f"pos_{d}" for d in "xyz"] + [f"landmark_{i}_{d}" for i in range(68) for d in "xy"]
        mvid_writer = csv.DictWriter(output_file, fieldnames=fieldnames)

        mvid_writer.writeheader()

    print(f"Video feature data exported to: {output_filename}")


if __name__ == '__main__':
    main()

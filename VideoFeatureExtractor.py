'''
This program extracts features from a synchronized .mp4 file into a .mvid file with landmarks and headpose data
'''

import time
import csv
import os
import argparse
import dlib
import cv2
import numpy as np
from pose_estimator import PoseEstimator

from utils import create_unique_filename

ap = argparse.ArgumentParser(
    description="Extract features from a .mp4 file to a .mvid file")
ap.add_argument('input', help="The input .mp4 file to process")
ap.add_argument('-d', action='store_true', default=False, dest='should_display',
                help='Whether or not to display frames during processing')
ap.add_argument('-o', '--output', help="The output .mvid file to write to")

args = ap.parse_args()

video_file = args.input

should_display = args.should_display

output_filename = args.output if args.output is not None else create_unique_filename(
    f"outputs/VideoFeatureExtractor/{os.path.splitext(os.path.basename(video_file))[0]}.mvid")

''' Computer Vision Initialization '''
VIDEO_RESCALE_FACTOR = 0.5  # Use half resolution

WINDOW_TITLE = "Video Feature Extractor"

# Get a face detector from the dlib library
detector = dlib.get_frontal_face_detector()
# Get a shape predictor based on a trained model from the dlib library
predictor = dlib.shape_predictor(
    'assets/shape_predictor_68_face_landmarks.dat')


def detect(frame, mark=False):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(grayFrame)

    # Return failure if no faces found
    if len(faces) == 0:
        return False, None, None, None

    face = faces[0]

    p = predictor(grayFrame, face)

    # Collect facial landmarks
    landmarks = np.vstack(
        [np.array([[p.part(i).x, p.part(i).y]], dtype=np.float32) for i in range(68)])

    # Select landmarks that correspond to shape predictor for pose estimation
    pose_estimation_pts = np.vstack(
        [np.array([[p.part(i).x, p.part(i).y]], dtype=np.float32) for i in [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]])

    # Optionally annotate image
    if mark:
        TL, BR = rect_to_coor(face)
        cv2.rectangle(frame, TL, BR, (0, 255, 0), 3)
        for x, y in landmarks:
            cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)
        for x, y in pose_estimation_pts:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

    return True, frame, landmarks, pose_estimation_pts


def rescale_frame(frame, rescale_factor=1):
    width = int(frame.shape[1] * rescale_factor)
    height = int(frame.shape[0] * rescale_factor)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def rect_to_coor(rect):
    x1 = rect.left()
    y1 = rect.top()
    x2 = rect.right()
    y2 = rect.bottom()
    return (x1, y1), (x2, y2)


def main():
    print("Starting processing...")
    start_time = time.time()

    cap = cv2.VideoCapture(video_file)
    assert cap.isOpened(), 'Failed to open video file'

    s, first_frame = cap.read()
    if not(s):
        print("Failed to read dimensions from video file!")
        return

    dimensions = tuple(
        first_frame.shape[i] * VIDEO_RESCALE_FACTOR for i in range(2))

    pe = PoseEstimator(dimensions)

    with open(output_filename, "w+") as output_file:
        fieldnames = ["timestamp"] + [f"rot_{d}" for d in "xyz"] + [
            f"pos_{d}" for d in "xyz"] + [f"landmark_{i}_{d}" for i in range(68) for d in "xy"]
        mvid_writer = csv.DictWriter(output_file, fieldnames=fieldnames)

        mvid_writer.writeheader()

        last_row_dict = None
        while cap.isOpened():
            s, frame = cap.read()
            if not(s):
                print("No more frames!")
                break

            timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            success, labelled_frame, landmarks, pose_estimation_pts = detect(
                frame, mark=should_display)

            row_dict = {}
            if success:
                rescaled_frame = rescale_frame(
                    labelled_frame, VIDEO_RESCALE_FACTOR)

                head_rotation, head_translation = pe.estimate_pose(
                    pose_estimation_pts)

                row_dict["timestamp"] = timestamp
                for i, d in enumerate("xyz"):
                    row_dict[f"rot_{d}"] = head_rotation[i][0]
                    row_dict[f"pos_{d}"] = head_translation[i][0]
                for i, l in enumerate(landmarks):
                    for j, d in enumerate("xy"):
                        row_dict[f"landmark_{i}_{d}"] = l[j]
            else:
                print(f"Couldn't find face at timestamp {timestamp}")
                row_dict = last_row_dict

            mvid_writer.writerow(row_dict)
            last_row_dict = row_dict

            if should_display:
                cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == 27:     # 27 is ASCII for the Esc key on a keyboard
                break

        cap.release()
        cv2.destroyAllWindows()

    print(f"Time consumed: {time.time() - start_time}")
    print(f"Video feature data exported to: {output_filename}")


if __name__ == '__main__':
    main()

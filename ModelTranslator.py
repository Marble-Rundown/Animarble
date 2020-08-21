'''
This program processes a .mvid file and creates a .mdat file with single-marble pan-tilt angles
'''

import time
import csv
import os
import argparse
import numpy as np
import cv2
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from objloader import *


from utils import create_unique_filename

ap = argparse.ArgumentParser(
    description="Process a .mvid file into a .mdat file")
ap.add_argument('input', help="The input .mvid file to process")
ap.add_argument('calibration', help="The .mvid file to use for calibration")
ap.add_argument('-d', action='store_true', default=False, dest='should_animate',
                help='Whether or not to display marble animation during processing')
ap.add_argument('-v', '--video',
                help="The raw .mp4 file to display alongside")
ap.add_argument('-o', '--output', help="The output .mdat file to write to")

args = ap.parse_args()

video_data_file = args.input
calibration_data_file = args.calibration


raw_mp4_file = args.video
should_display = (args.video != None)

# If displaying video, then animate marble too
should_animate = args.should_animate or should_display

output_filename = args.output if args.output is not None else create_unique_filename(
    f"outputs/ModelTranslator/{os.path.splitext(os.path.basename(video_data_file))[0]}.mdat")

''' Constants '''
ROTATION_FILTER_LENGTH = 5

LIP_DIST_FILTER_LENGTH = 5
LIP_DIST_MINIMUM_THRESHOLD = 10
LIP_DIST_MULTIPLIER = 0.5

DEFAULT_PAN_SETPOINT = 90
DEFAULT_TILT_SETPOINT = 90

VIDEO_RESCALE_FACTOR = 0.5  # Use half resolution

WINDOW_TITLE = "Model Translator"


marble = None
# Initialize pygame and OpenGL if display is requested
if should_animate:
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    # Initialize lighting and rendering settings
    glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glMatrixMode(GL_PROJECTION)

    # Initialize marble OBJ
    marble = OBJ('assets/MarbleHeadset_v11.obj', swapyz=True)
    marble.generate()

    # Initialize camera
    glLoadIdentity()
    width, height = display
    glOrtho(-width / 15, width / 15, -
            height / 15, height / 15, 0.1, 40.0)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_MODELVIEW)

    glTranslatef(0.0, 0.0, -30)
    glRotatef(180, 0, 1, 0)


def get_lip_dist(landmarks):
    return np.linalg.norm(landmarks[51]-landmarks[57])


def parse_row(row):
    timestamp = int(row["timestamp"])

    rotation = np.matrix([float(row[f"rot_{d}"]) for d in "xyz"])

    position = np.matrix([float(row[f"pos_{d}"]) for d in "xyz"])

    landmarks = np.matrix([[float(row[f"landmark_{i}_{d}"])
                            for d in "xy"] for i in range(68)])

    return timestamp, rotation, position, landmarks


def ewma(data, filter_length=None):
    if filter_length == None or filter_length > len(data):
        filter_length = len(data)
    alpha = 2 / (len(data) + 1)

    previous = data[-filter_length]
    for d in data[-filter_length:]:
        previous = alpha * d + (1-alpha) * previous

    return previous


def rotate_marble(pan, tilt):
    glPushMatrix()
    glRotatef(tilt, -1, 0, 0)
    glRotatef(pan, 0, -1, 0)
    marble.render()
    glPopMatrix()


def main():
    print("Starting processing...")
    start_time = time.time()

    # Calibration constants for consistent face detection
    rotation_mean, rotation_std = None, None
    lip_dist_mean, lip_dist_std = None, None

    # Use calibration data file to calibration rotation and lip constants
    with open(calibration_data_file) as cal_file:
        cal_reader = csv.DictReader(cal_file)

        rotation_data = np.empty(shape=(0, 3))
        lip_dist_data = np.empty(shape=0)

        for row in cal_reader:
            timestamp, rotation, position, landmarks = parse_row(row)

            rotation_data = np.vstack(
                (rotation_data, rotation))

            lip_dist_data = np.append(lip_dist_data, get_lip_dist(landmarks))

        rotation_mean = np.mean(rotation_data, axis=0)
        rotation_std = np.std(rotation_data, axis=0)

        lip_dist_mean = np.mean(lip_dist_data)
        lip_dist_std = np.std(lip_dist_data)

    print(
        f"Calibrated Rotation:\nMean: {rotation_mean} Std: {rotation_std}")
    print(
        f"Calibrated Lip Dist:\nMean: {lip_dist_mean} Std: {lip_dist_std}")

    cap = None
    # Initialize OpenCV on the raw video file if enabled
    if should_display:
        cap = cv2.VideoCapture(raw_mp4_file)
        assert cap.isOpened(), 'Failed to open video file'

    with open(output_filename, "w+") as mdat_file:
        fieldnames = ["timestamp", "pan_offset",
                      "tilt_offset", "pan_setpoint", "tilt_setpoint"]
        mdat_writer = csv.DictWriter(mdat_file, fieldnames=fieldnames)

        mdat_writer.writeheader()

        with open(video_data_file) as mvid_file:
            mvid_reader = csv.DictReader(mvid_file)

            shifted_rotation_data = np.empty(shape=(0, 3))
            shifted_lip_dist_data = np.empty(shape=0)

            simulation_time = 0
            for row in mvid_reader:

                # If animation is enabled, check for pygame updates and re-render display
                if should_animate:
                    # 27 is ASCII for the Esc key on a keyboard
                    if cv2.waitKey(1) & 0xFF == 27:
                        pygame.quit()
                        quit()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            quit()

                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                timestamp, rotation, position, landmarks = parse_row(row)

                shifted_rotation = rotation - rotation_mean
                shifted_rotation_data = np.vstack(
                    (shifted_rotation_data, shifted_rotation))
                filtered_rotation = ewma(
                    shifted_rotation_data, ROTATION_FILTER_LENGTH)

                shifted_lip_dist = get_lip_dist(landmarks) - lip_dist_mean

                thresholded_lip_dist = shifted_lip_dist if abs(
                    shifted_lip_dist) > LIP_DIST_MINIMUM_THRESHOLD else 0

                shifted_lip_dist_data = np.append(
                    shifted_lip_dist_data, thresholded_lip_dist)
                filtered_lip_dist = ewma(
                    shifted_lip_dist_data, LIP_DIST_FILTER_LENGTH)

                mouth_offset = LIP_DIST_MULTIPLIER * filtered_lip_dist

                print(f"Mouth size: {mouth_offset}")

                # Rotation about Y axis is pan
                pan_offset = shifted_rotation[0, 1]
                # Rotation about X axis is tilt
                tilt_offset = shifted_rotation[0, 0]

                tilt_offset -= mouth_offset  # Subtract so mouth moves down while talking

                pan_offset, tilt_offset = round(pan_offset), round(tilt_offset)

                # Write these offsets to .mdat file
                row_dict = {
                    "timestamp": timestamp,
                    "pan_offset": pan_offset,
                    "tilt_offset": tilt_offset,
                    "pan_setpoint": DEFAULT_PAN_SETPOINT,
                    "tilt_setpoint": DEFAULT_TILT_SETPOINT
                }
                mdat_writer.writerow(row_dict)

                # If we should display video, actually display the frame
                if should_display:
                    s, frame = cap.read()
                    if not(s):
                        print("Failed to read frame!")
                        break
                    cv2.imshow(WINDOW_TITLE, frame)

                # If we should animate, actually animate the marble
                if should_animate:
                    rotate_marble(pan_offset, tilt_offset)
                    pygame.display.flip()
                    pygame.time.wait(timestamp - simulation_time)
                    simulation_time = timestamp

    print(f"Time consumed: {time.time() - start_time}")
    print(f"Model pantilt data exported to: {output_filename}")


if __name__ == '__main__':
    main()

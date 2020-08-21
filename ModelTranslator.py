'''
This program processes a .mvid file and creates a .mdat file with single-marble pan-tilt angles
'''

import csv
import os
import argparse
import numpy as np

from utils import create_unique_filename

ap = argparse.ArgumentParser(
    description="Process a .mvid file into a .mdat file")
ap.add_argument('input', help="The input .mvid file to process")
ap.add_argument('calibration', help="The .mvid file to use for calibration")
ap.add_argument('-d', action='store_true', default=False, dest='should_display',
                help='Whether or not to display marble animation during processing')
ap.add_argument('-v', '--video',
                help="The original .mp4 file to display alongside")
ap.add_argument('-o', '--output', help="The output .mdat file to write to")

args = ap.parse_args()

video_data_file = args.input
calibration_data_file = args.calibration

should_display = args.should_display

output_filename = args.output if args.output is not None else create_unique_filename(
    f"outputs/ModelTranslator/{os.path.splitext(os.path.basename(video_data_file))[0]}.mdat")

''' Constants '''
ROTATION_FILTER_LENGTH = 5


def get_lip_dist(landmarks):
    return distance(landmarks[51], landmarks[57])


def distance(a, b):
    return np.linalg.norm(a-b)


def parse_row(row):
    timestamp = int(row["timestamp"])

    rotation = np.matrix([float(row[f"rot_{d}"]) for d in "xyz"])

    position = np.matrix([float(row[f"pos_{d}"]) for d in "xyz"])

    landmarks = np.matrix([[float(row[f"landmark_{i}_{d}"])
                            for d in "xy"] for i in range(68)])

    return timestamp, rotation, position, landmarks


def old_ewma(data):
    data.reverse()

    def avg(data):
        alpha = 2 / (len(data) + 1)
        if len(data) == 1:
            return data[0]
        else:
            curr = data[0]
            # print(curr)
            data.pop(0)
            return alpha * curr + (1 - alpha) * avg(data)
    return avg(data)


def ewma(data, filter_length=None):
    if filter_length == None or filter_length > len(data):
        filter_length = len(data)
    alpha = 2 / (len(data) + 1)

    previous = data[-filter_length]
    for d in data[-filter_length:]:
        previous = alpha * d + (1-alpha) * previous

    return previous


def main():

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

    with open(video_data_file) as mvid_file:
        mvid_reader = csv.DictReader(mvid_file)

        shifted_rotation_data = np.empty(shape=(0, 3))
        for row in mvid_reader:
            timestamp, rotation, position, landmarks = parse_row(row)

            shifted_rotation = rotation - rotation_mean

            shifted_rotation_data = np.vstack(
                (shifted_rotation_data, shifted_rotation))

            filtered_rotation = ewma(
                shifted_rotation_data, ROTATION_FILTER_LENGTH)
            print(
                f"Rotation: {shifted_rotation}\nFiltered: {filtered_rotation}\n")

    #                     mouth_size = get_lip_dist(landmarks) - SPEECH_OFFSET
    #                     mouth_size = mouth_size if abs(
    #                         mouth_size) > SPEECH_THRESHOLD else 0
    #                     # print(f"Mouth size: {mouth_size}")
    #                     speech_filter.append(mouth_size)
    #                     speech_filter.pop(0)
    #                     rotation[0] += K_SPEECH * \
    #                         median(speech_filter)        # nodding

    #                     if count % NOISE_PERIOD == 0:
    #                         g_noise = tuple(np.random.normal(scale=NOISE_INTENSITY[i]) for i in range(
    #                             3)) if False and mouth_size > SPEECH_THRESHOLD else (0, 0, 0)
    #                         # print("Adding noise")
    #                 count += 1

    #                 converted_rotation = tuple(rotation[i] + rotation_offset[i] + g_noise[i] * (
    #                     (count % NOISE_PERIOD) + 1) / NOISE_PERIOD for i in range(3))
    #                 tilt, pan = converted_rotation[0], converted_rotation[1]

    #                 # print('timestamp:{3}\npitch:{0}\nyaw:{1}\nroll{2}\n'.format(*converted_rotation, cap.get(cv2.CAP_PROP_POS_MSEC)))
    #                 time = cap.get(cv2.CAP_PROP_POS_MSEC)
    #                 # file.write('{0},{1},{2},0,0\n'.format(int(time), jbr(tilt)[0], jbr(pan)[0]))
    #                 file.write('{0},{1},{2},90,90\n'.format(
    #                     int(time), tilt[0], pan[0]))

    #                 completion = int(round(time / video_length * 100))
    #                 print(video_length)
    #                 print(time)
    #                 print(f'{completion}% done')

    #                 rotate_marble(round(float(tilt)), round(float(pan)))
    #         else:
    #             print('Failed to find image points')

    #         cv2.imshow(WINDOW_TITLE, frame)
    #         if cv2.waitKey(1) & 0xFF == 27:     # 27 is ASCII for the Esc key on a keyboard
    #             break

    #         pygame.display.flip()
    #         pygame.time.wait(10)


if __name__ == '__main__':
    main()

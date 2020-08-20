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
    f"outputs/ModelTranslator/{os.path.splitext(os.path.basename(video_file))[0]}.mdat")

SPEECH_FILTER_LENGTH = 5
K_SPEECH = 1.5
SPEECH_OFFSET = 20
SPEECH_THRESHOLD = 2

# Gaussian Noise in pan and tilt while talking
NOISE_PERIOD = 5
NOISE_INTENSITY = (4.0, 3.0, 0.0)

MOVING_AVERAGE_LENGTH = 3

CALIBRATION_LENGTH = 10

np.random.seed(5327)

moving_average = []
speech_filter = [0] * SPEECH_FILTER_LENGTH

calibration = []
lipCalibration = []
calibrated = False
rotation_offset = (0, 0, 0)

g_noise = (0, 0, 0)
count = 0


def main():
    while True:
        # Handle Pygame quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        s, frame = cap.read()
        if not(s):
            print("exiting!")
            break

            frame, landmarks, img_points = detect(frame, mark=True)
            frame = rescale_frame(frame, percent=rescale)

            # print(len(img_points))

            if img_points.size != 0:
                rotation, translation = pe.estimate_pose(img_points)

                if not calibrated:
                    calibration.append(rotation)
                    lipCalibration.append(get_lip_dist(landmarks))
                    # print(calibration)
                    if len(calibration) > CALIBRATION_LENGTH:
                        averages = [sum([rot[i] for rot in calibration]) /
                                    CALIBRATION_LENGTH for i in range(3)]
                        rotation_offset = tuple(-avg for avg in averages)
                        SPEECH_OFFSET = np.mean(lipCalibration)
                        calibrated = True
                        print(f"Speech offset: {SPEECH_OFFSET}")
                        print("done calibrating")
                else:
                    moving_average.append(rotation)
                    if len(moving_average) > MOVING_AVERAGE_LENGTH:
                        moving_average.pop(0)

                    rotation = [ewma([rot[i] for rot in moving_average])
                                for i in range(3)]
                    if landmarks.size != 0:
                        mouth_size = get_lip_dist(landmarks) - SPEECH_OFFSET
                        mouth_size = mouth_size if abs(
                            mouth_size) > SPEECH_THRESHOLD else 0
                        # print(f"Mouth size: {mouth_size}")
                        speech_filter.append(mouth_size)
                        speech_filter.pop(0)
                        rotation[0] += K_SPEECH * \
                            median(speech_filter)        # nodding

                        if count % NOISE_PERIOD == 0:
                            g_noise = tuple(np.random.normal(scale=NOISE_INTENSITY[i]) for i in range(
                                3)) if False and mouth_size > SPEECH_THRESHOLD else (0, 0, 0)
                            # print("Adding noise")
                    count += 1

                    converted_rotation = tuple(rotation[i] + rotation_offset[i] + g_noise[i] * (
                        (count % NOISE_PERIOD) + 1) / NOISE_PERIOD for i in range(3))
                    tilt, pan = converted_rotation[0], converted_rotation[1]

                    # print('timestamp:{3}\npitch:{0}\nyaw:{1}\nroll{2}\n'.format(*converted_rotation, cap.get(cv2.CAP_PROP_POS_MSEC)))
                    time = cap.get(cv2.CAP_PROP_POS_MSEC)
                    # file.write('{0},{1},{2},0,0\n'.format(int(time), jbr(tilt)[0], jbr(pan)[0]))
                    file.write('{0},{1},{2},90,90\n'.format(
                        int(time), tilt[0], pan[0]))

                    completion = int(round(time / video_length * 100))
                    print(video_length)
                    print(time)
                    print(f'{completion}% done')

                    rotate_marble(round(float(tilt)), round(float(pan)))
            else:
                print('Failed to find image points')

            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == 27:     # 27 is ASCII for the Esc key on a keyboard
                break

            pygame.display.flip()
            pygame.time.wait(10)


if __name__ == '__main__':
    main()

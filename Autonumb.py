'''
This program investigates if a simple volume-based approach can automatically numb a marble when its track volume is low
'''

import pygame
import os
import argparse
import csv
from scipy.io import wavfile
import sys
import numpy as np
import matplotlib.pyplot as plt

from utils import create_unique_filename

AUDIO_AMPLITUDE_NUMB_THRESHOLD = 250
AVERAGE_WINDOW_LENGTH = 10000


def parse_row(row):
    timestamp = int(row["timestamp"])

    pan_offset, pan_setpoint = int(row["pan_offset"]), int(row["pan_setpoint"])

    tilt_offset, tilt_setpoint = int(
        row["tilt_offset"]), int(row["tilt_setpoint"])

    return timestamp, pan_offset, tilt_offset, pan_setpoint, tilt_setpoint


def main():
    sample_rate, audio_data = wavfile.read(wav_file)
    mean_audio_data = np.mean(audio_data, axis=1)

    thresholded_audio_data = np.where(
        np.abs(mean_audio_data) >= AUDIO_AMPLITUDE_NUMB_THRESHOLD, 1, 0)

    averaged_audio_data = np.convolve(
        thresholded_audio_data, np.ones(AVERAGE_WINDOW_LENGTH), mode="same")

    thresholds = np.empty(0)
    timestamps = np.empty(0)

    def get_threshold(ms):
        idx = round(ms * sample_rate / 1000)
        if idx < len(audio_data):
            return True, (averaged_audio_data[idx] > 0)
        else:
            return False, 0

    with open(output_filename, "w+") as mdat_out:
        fieldnames = ["timestamp", "pan_offset",
                      "tilt_offset", "pan_setpoint", "tilt_setpoint"]
        writer = csv.DictWriter(mdat_out, fieldnames=fieldnames)
        writer.writeheader()

        with open(mdat_file) as mdat_in:
            reader = csv.DictReader(mdat_in)

            for row in reader:
                timestamp, pan_offset, tilt_offset, pan_setpoint, tilt_setpoint = parse_row(
                    row)

                in_range, threshold = get_threshold(timestamp)

                if not in_range:
                    break

                timestamps = np.append(timestamps, timestamp)
                thresholds = np.append(thresholds, threshold)

                pan_offset *= threshold
                tilt_offset *= threshold

                row_dict = {
                    "timestamp": timestamp,
                    "pan_offset": pan_offset,
                    "tilt_offset": tilt_offset,
                    "pan_setpoint": pan_setpoint,
                    "tilt_setpoint": tilt_setpoint
                }
                writer.writerow(row_dict)

    if should_graph:
        plt.figure(1)
        plt.subplot(311)
        plt.plot(audio_data, color="red")
        plt.subplot(312)
        plt.plot(thresholded_audio_data, color="blue")
        plt.subplot(313)
        plt.plot(timestamps, thresholds, color="green")
        plt.show()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="Numbs a .mdat file based on a .wav file")
    ap.add_argument('input', help="The original .mdat file to autonumb")
    ap.add_argument('audio', help="The .wav file to numb off of")
    ap.add_argument('-o', '--output',
                    help="The output .mdat file to write to")
    ap.add_argument('-d', action='store_true', default=False, dest='should_graph',
                    help='Whether or not to display results graph after processing')

    args = ap.parse_args()

    should_graph = args.should_graph

    mdat_file = args.input

    wav_file = args.audio

    output_filename = args.output if args.output is not None else create_unique_filename(
        f"outputs/Autonumb/{os.path.splitext(os.path.basename(mdat_file))[0]}.mdat")

    main()

from os import path
from pydub import AudioSegment
import argparse
import wave
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import csv

ap = argparse.ArgumentParser(description="Extract features from a .wav file to a .mvid file")
ap.add_argument('--input', help="The input .wav file to process")
ap.add_argument('--output', help="The output .mvid file to write to")

args = ap.parse_args()

def convert_mp3_to_wav_frames(input_file):
    sound = AudioSegment.from_mp3(input_file)
    sound.export("temp.wav", format="wav")

def wav_to_np_array(input_file):
    obj = wave.open(input_file,'r')
    print( "Number of channels",obj.getnchannels())
    print ( "Sample width",obj.getsampwidth())
    print ( "Frame rate.",obj.getframerate())
    print ("Number of frames",obj.getnframes())
    print ( "parameters:",obj.getparams())
    np_wav = read(input_file)
    return np.array(np_wav[1],dtype=float), [obj.getframerate(), obj.getnframes()]

def view_audio(filename):
    spf = wave.open(filename, "r")
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, "Int16")
    if spf.getnchannels() == 2:
        print("Just mono files")
        sys.exit(0)
    plt.figure(1)
    plt.title("Signal Wave...")
    plt.plot(signal)
    plt.show()

def magnify_peaks(frames, degree):
    return pow(frames, degree)

def normalize(frames):
    if frames.max() != 0:
        frames = frames/abs(frames).max()
    return frames

def identify_peaks(frames, slack, spread):
    peaks = []
    inPeak = False
    startIndex = 0
    for i in range(len(frames)):
        if i > spread:
            if abs(frames[(i-spread):i]).max() < 1-slack:
                if inPeak == True:
                    peaks += [[startIndex, i]]
                    inPeak = False
            elif abs(frames[(i-spread):i]).max() > 1-slack:
                if inPeak == False:
                    startIndex = i
                    inPeak = True
    return peaks

def create_base_movement_angles(peaks, max_range):
    pan_offset_angles = np.random.normal(size=(len(peaks)))
    tilt_offset_angles = np.random.normal(size=(len(peaks)))

    pan_offset_angles = pan_offset_angles/abs(pan_offset_angles).max()*max_range
    tilt_offset_angles = tilt_offset_angles/abs(tilt_offset_angles).max()*max_range

    return pan_offset_angles, tilt_offset_angles

def get_highest_amplitudes(peaks, frames, scale):
    max_values = []
    for i in peaks:
        max_values += [math.sqrt(frames[i[0]:i[1]].max())*scale]
    return max_values

def create_writable_data(max_values, pan_offset_angles, tilt_offset_angles, frame_info, peaks, write_every_ms, pan_setpoint, tilt_setpoint, max_range):
    pan_offsets = []
    tilt_offsets = []
    
    timestamps = []

    for i in peaks:
        timestamps += [int((i[0]+i[1])/2/frame_info[0]*1000/write_every_ms)]

    for i in range(len(max_values)):
        pan_offsets += [max_values[i]*pan_offset_angles[i]]
        tilt_offsets += [max_values[i]*tilt_offset_angles[i]]

    scale_factor_pan = max_range/abs(np.array(pan_offsets)).max()
    scale_factor_tilt = max_range/abs(np.array(tilt_offsets)).max()

    pan_offsets = np.array(pan_offsets)*scale_factor_pan
    tilt_offsets = np.array(tilt_offsets)*scale_factor_tilt

    actual_pan_offset = []
    actual_tilt_offset = []

    for i in range(int((frame_info[1]/frame_info[0]*1000)/write_every_ms)):
        was_timestamp = False
        for j in range(len(timestamps)):
            if i == timestamps[j]:
                actual_pan_offset += [pan_offsets[j]]
                actual_tilt_offset += [tilt_offsets[j]]
                was_timestamp = True
        if was_timestamp == False:
            actual_pan_offset += [0]
            actual_tilt_offset += [0]

    return np.array([actual_pan_offset, actual_tilt_offset, np.full((len(actual_pan_offset)), pan_setpoint), np.full((len(actual_pan_offset)), tilt_setpoint)])

def smooth_data(writable_data, smooth_scale):
    indexes = []
    for i in range(len(writable_data[0])):
        if writable_data[0][i] != 0:
            indexes += [i]
    
    future_index = indexes[0]
    prev_val_pan = 0
    prev_val_tilt = 0
    print(indexes)
    for i in range(len(writable_data[0])):
        if not (i in indexes):
            distance_to_next_index = (future_index-i)/smooth_scale
            change_in_value_pan = (writable_data[0][indexes[indexes.index(future_index)]]-prev_val_pan)/distance_to_next_index
            change_in_value_tilt = (writable_data[1][indexes[indexes.index(future_index)]]-prev_val_tilt)/distance_to_next_index
            if change_in_value_pan > 0:
                writable_data[0][i] = writable_data[0][i-1]-change_in_value_pan
            else:
                writable_data[0][i] = writable_data[0][i-1]+change_in_value_pan
            if change_in_value_tilt > 0:
                writable_data[1][i] = writable_data[1][i-1]-change_in_value_tilt
            else:
                writable_data[1][i] = writable_data[1][i-1]+change_in_value_tilt
            prev_val_pan = writable_data[0][i-1]
            prev_val_tilt = writable_data[1][i-1]
        elif i in indexes:
            prev_val_pan = writable_data[0][i-1]
            prev_val_tilt = writable_data[1][i-1]
            try:
                future_index = indexes[indexes.index(i)+1]
            except:
                future_index = indexes[indexes.index(i)]
    
    return np.around(writable_data, 2)


def create_output_file(writable_data, output_filename):
    with open(output_filename, "w+") as output_file:
        fieldnames = ["timestamp"] + ["pan_offset"] + ["tilt_offset"] + ["pan_setpoint"] + ["tilt_setpoint"]
        mdat_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        mdat_writer.writeheader()

        row_dict = {}
        for i in range(len(writable_data[0])):
            row_dict["timestamp"] = str(i*50)
            row_dict["pan_offset"] = str(writable_data[0][i])
            row_dict["tilt_offset"] = str(writable_data[1][i])
            row_dict["pan_setpoint"] = str(writable_data[2][i]) 
            row_dict["tilt_setpoint"] = str(writable_data[3][i])
            mdat_writer.writerow(row_dict)


def generate_movement(input_file, output_file):
    #Getting frames for processing (converts to wav if neccesary)
    if input_file[(len(input_file)-4):len(input_file)] == ".mp3":
        convert_mp3_to_wav_frames(input_file)
        frames, frame_info = wav_to_np_array("temp.wav")
        #view_audio("temp.wav")
    else:
        frames = wavToNpArray(args.input_file)

    frames = magnify_peaks(frames, 2)
    frames = normalize(frames)

    peaks = identify_peaks(frames, 0.99, 1000)
    pan_offset_angles, tilt_offset_angles = create_base_movement_angles(peaks, 10)
    max_values = normalize(np.array(get_highest_amplitudes(peaks, frames, 10)))
    
    writable_data = create_writable_data(max_values, pan_offset_angles, tilt_offset_angles, frame_info, peaks, 50, 90, 90, 10)

    smoothed_data = smooth_data(writable_data, 1)
    
    create_output_file(smoothed_data, output_file)

generate_movement(args.input, args.output)
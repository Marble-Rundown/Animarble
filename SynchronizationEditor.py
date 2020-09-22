'''
This program enables the synchronization of 2 .mdat files with .wav audio files to create an edited .msync file
'''

import pygame
import os
import argparse
import csv
from scipy.io import wavfile
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from objloader import *
from scipy.interpolate import interp1d
from math import ceil
from OffsetTrack import *
from SetpointTrack import *

from utils import create_unique_filename, millisec


#############################
#         Constants         #
#############################
PLAYBACK_SPEED = 2
START_TIMESTAMP = 0
END_TIMESTAMP = 0
MARBLE_DISPLAY_X_OFFSET = 20
ARDUINO_SAMPLING_INTERVAL = 50


#############################
#      Initialization       #
#############################
initialized = False
left_pan_offset_track = left_tilt_offset_track = left_pan_setpoint_track = left_tilt_setpoint_track = None
right_pan_offset_track = right_tilt_offset_track = right_pan_setpoint_track = right_tilt_setpoint_track = None

lp_regions, lp_control_points = [], []
lt_regions, lt_control_points = [], []
rp_regions, rp_control_points = [], []
rt_regions, rt_control_points = [], []

sample_rate, audio_data = wavfile.read('assets/wavfile.wav')
left_wave = right_wave = np.mean(audio_data, axis=1)

# pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

# Initialize marble OBJ
marble = OBJ('assets/MarbleHeadset_v11.obj', swapyz=True)
marble.generate()


def rcp(ms, angle):
    global right_pan_setpoint_track
    index = round(ms / ARDUINO_SAMPLING_INTERVAL)
    right_pan_setpoint_track.add_control_point(index, angle)


def rca(start_ms, start_angle, end_ms, end_angle):
    rcp(start_ms, start_angle)
    rcp(end_ms, end_angle)


def rn(startms, endms):
    global right_pan_offset_track, right_tilt_offset_track
    start_index = round(startms / ARDUINO_SAMPLING_INTERVAL)
    end_index = round(endms / ARDUINO_SAMPLING_INTERVAL)
    right_pan_offset_track.add_region_modifier(start_index, end_index, 0)
    right_tilt_offset_track.add_region_modifier(start_index, end_index, 0)


def lcp(ms, angle):
    global left_pan_setpoint_track
    index = round(ms / ARDUINO_SAMPLING_INTERVAL)
    left_pan_setpoint_track.add_control_point(index, angle)


def lca(start_ms, start_angle, end_ms, end_angle):
    lcp(start_ms, start_angle)
    lcp(end_ms, end_angle)


def ln(startms, endms):
    global left_pan_offset_track, left_tilt_offset_track
    start_index = round(startms / ARDUINO_SAMPLING_INTERVAL)
    end_index = round(endms / ARDUINO_SAMPLING_INTERVAL)
    left_pan_offset_track.add_region_modifier(start_index, end_index, 0)
    left_tilt_offset_track.add_region_modifier(start_index, end_index, 0)


def workspace():
    global left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track
    ########################################
    #           EDIT ME HERE!!!            #
    ########################################

    def both():
        # Dan, you're slipping 
        lca(18791, 90, 19250, 130)
        rca(18791+400, 90, 19250+400, 50)
        
        # So that brings us to a quick
        lca(218588, 130, 219056, 90)
        rca(218588+300, 50, 219056+300, 90)

        # Man, I'm jealous
        lca(252774, 90, 253222, 130)
        rca(252774+400, 90, 253222+400, 50)

        # So last week
        lca(263803, 130, 264360, 90)
        rca(263803+300, 50, 264360+300, 90)

        # How many things?
        rca(348890, 90, 349323, 50)
        lca(348890+400, 90, 349323+400, 130)

        # Anyways
        lca(361962, 130, 362447, 90)
        rca(361962+300, 50, 362447+300, 90)

    # both()

#############################
#           Main            #
#############################

def main():
    global left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track

    raw_left_timestamps, raw_left_pan_offset, raw_left_tilt_offset, raw_left_pan_setpoint, raw_left_tilt_setpoint = parse_mdat_file(
        left_mdat_file)
    raw_right_timestamps, raw_right_pan_offset, raw_right_tilt_offset, raw_right_pan_setpoint, raw_right_tilt_setpoint = parse_mdat_file(
        right_mdat_file)

    max_timestamp = max(raw_left_timestamps[-1], raw_right_timestamps[-1])

    left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track = resample_tracks(
        raw_left_timestamps, raw_left_pan_offset, raw_left_tilt_offset, raw_left_pan_setpoint, raw_left_tilt_setpoint, ARDUINO_SAMPLING_INTERVAL, max_timestamp)
    right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track = resample_tracks(
        raw_right_timestamps, raw_right_pan_offset, raw_right_tilt_offset, raw_right_pan_setpoint, raw_right_tilt_setpoint, ARDUINO_SAMPLING_INTERVAL, max_timestamp)

    command_map = {
        'acp': lambda tokens : print("dummy acp function"),
        'ar': lambda tokens : print("dummy ar function"),
        'l': lambda tokens : print("dummy l function"),
        'r': lambda tokens : print("dummy r function")
    }

    while True:
        tokens = input('>>> ').lower().split()
        if len(tokens) == 0:
            continue
        
        command, tokens = tokens[0], tokens[1:] # split off first element as command
        if command in command_map:
            command_map[command](tokens)
        else:
            print("Unknown command:", command)


            # if args['acp']:
            #     words = args['acp'].split()
            #     if words[0].lower() == 'left':
            #         if words[1].lower() == 'pan':
            #             lp_control_points.append(left_pan_setpoint_track.add_control_point(*words[2:4]))
            #         elif words[1].lower() == 'tilt':
            #             lt_control_points.append(left_tilt_setpoint_track.add_control_point(*words[2:4]))
            #     elif words[0].lower() == 'right':
            #         if words[1].lower() == 'pan':
            #             rp_control_points.append(right_pan_setpoint_track.add_control_point(*words[2:4]))
            #         elif words[1].lower() == 'tilt':
            #             rt_control_points.append(right_tilt_setpoint_track.add_control_point(*words[2:4]))
            # if args['ar']:
            #     words = args['ar'].split()
            #     words[2:] = [int(w) for w in words[2:]]
            #     if words[0].lower() == 'left':
            #         if words[1].lower() == 'pan':
            #             lp_regions.append(left_pan_offset_track.add_region_modifier(*words[2:5]))
            #         elif words[1].lower() == 'tilt':
            #             lt_regions.append(left_tilt_offset_track.add_region_modifier(*words[2:5]))
            #     elif words[0].lower() == 'right':
            #         if words[1].lower() == 'pan':
            #             rp_regions.append(right_pan_offset_track.add_region_modifier(*words[2:5]))
            #         elif words[1].lower() == 'tilt':
            #             rt_regions.append(right_tilt_offset_track.add_region_modifier(*words[2:5]))
            # if args['l']:
            #     words = args['ar'].split()
            #     if words[0].lower() == 'left':
            #         if words[1].lower() == 'pan':
            #             print(f'Left Pan Control Points:\n{lp_control_points}')
            #             print(f'Left Pan Regions:\n{lp_regions}')
            #         elif words[1].lower() == 'tilt':
            #             print(f'Left Tilt Control Points:\n{lt_control_points}')
            #             print(f'Left Tilt Regions:\n{lt_regions}')
            #     elif words[0].lower() == 'right':
            #         if words[1].lower() == 'pan':
            #             print(f'Right Pan Control Points:\n{rp_control_points}')
            #             print(f'Right Pan Regions:\n{rp_regions}')
            #         elif words[1].lower() == 'tilt':
            #             print(f'Right Tilt Control Points:\n{rt_control_points}')
            #             print(f'Right Tilt Regions:\n{rt_regions}')




        
# def split_args(args)
#     words = args['acp'].split()
#     assert len(words) == 4, f'Incorrect number of parameters. Expected 4 arguments but received {len(words)}.'
#     assert words[0].lower() in ['left', 'right'], f"'{words[0]}' is not a valid marble."
#     assert all([type(w) == int for w in words[1:]])
#     return 


def pygame_main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

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
    glOrtho(-width / 15, width / 15, -height / 15, height / 15, 0.1, 40.0)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_MODELVIEW)

    glTranslatef(0.0, 0.0, -30)
    glRotatef(180, 0, 1, 0)

    left_pan_angle, left_tilt_angle, right_pan_angle, right_tilt_angle = export(
        left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track)

    start_index = int(round(START_TIMESTAMP / ARDUINO_SAMPLING_INTERVAL))
    end_index = int(round(END_TIMESTAMP / ARDUINO_SAMPLING_INTERVAL)) if END_TIMESTAMP != 0 else len(left_pan_angle) - 1
    # print(start_index, end_index)
    # start_index = 50
    # print(left_pan_angle)

    start_ms = millisec()
    timer = 0
    for left_pan, left_tilt, right_pan, right_tilt in zip(left_pan_angle, left_tilt_angle, right_pan_angle, right_tilt_angle):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        rotate_marbles(left_pan, left_tilt, right_pan, right_tilt)

        pygame.display.flip()
        timer += ARDUINO_SAMPLING_INTERVAL/PLAYBACK_SPEED
        elapsed_ms = millisec() - start_ms
        waitTime = round(timer - elapsed_ms)
        if waitTime >= 0:
            pygame.time.wait(waitTime)
            print(timer)
        else:
            print(f"Too slow at timer {timer} with waitTime {waitTime}")
    pygame.quit()
    return


#############################
#        Functions          #
#############################
def create_file(file_name, file_type, n=0):
    # Add:    if n != 0 else ''    after f' ({n})' if you don't want the first file to have a number
    destination = './outputs/{0}{1}.{2}'.format(file_name, f'({n})', file_type)
    if not os.path.isfile(destination):
        return open(destination, 'w+')
    else:
        return create_file(file_name, file_type, n+1)

def display_plots():
    plt.close('all')
    plt.figure(1)

    plt.subplot(9, 9, (1, 9))
    plt.title('left pan offsets')
    plt.plot(left_pan_offset_track.apply_modifiers(), color='blue')
    plt.subplot(9, 9, (10, 18))
    plt.title('left tilt offsets')
    plt.plot(left_tilt_offset_track.apply_modifiers(), color='blue')
    plt.subplot(9, 9, (19, 27))
    plt.title('left pan setpoints')
    plt.plot(left_pan_setpoint_track.apply_control_points(), color='blue')
    plt.subplot(9, 9, (28, 36))
    plt.title('left audio waveform')
    plt.plot(left_wave, color='blue')

    plt.subplot(9, 9, (46, 54))
    plt.title('right pan offsets')
    plt.plot(right_pan_offset_track.apply_modifiers(), color='red')
    plt.subplot(9, 9, (55, 63))
    plt.title('right tilt offsets')
    plt.plot(right_tilt_offset_track.apply_modifiers(), color='red')
    plt.subplot(9, 9, (64, 72))
    plt.title('right pan setpoints')
    plt.plot(right_pan_setpoint_track.apply_control_points(), color='red')
    plt.subplot(9, 9, (73, 81))
    plt.title('right audio waveform')
    plt.plot(right_wave, color='red')

    plt.show()

def parse_row(row):
    timestamp = int(float(row["timestamp"]))

    pan_offset, pan_setpoint = int(float(row["pan_offset"])), int(float(row["pan_setpoint"]))

    tilt_offset, tilt_setpoint = int(float(row["tilt_offset"])), int(float(row["tilt_setpoint"]))

    return timestamp, pan_offset, tilt_offset, pan_setpoint, tilt_setpoint


def parse_mdat_file(mdat_file):
    raw_timestamps = np.empty(0)
    raw_pan_offset = np.empty(0)
    raw_pan_setpoint = np.empty(0)
    raw_tilt_offset = np.empty(0)
    raw_tilt_setpoint = np.empty(0)

    with open(mdat_file) as mdat:
        reader = csv.DictReader(mdat)

        for row in reader:
            timestamp, pan_offset, tilt_offset, pan_setpoint, tilt_setpoint = parse_row(row)

            raw_timestamps = np.append(raw_timestamps, timestamp)
            raw_pan_offset = np.append(raw_pan_offset, pan_offset)
            raw_pan_setpoint = np.append(raw_pan_setpoint, pan_setpoint)
            raw_tilt_offset = np.append(raw_tilt_offset, tilt_offset)
            raw_tilt_setpoint = np.append(raw_tilt_setpoint, tilt_setpoint)

    return raw_timestamps, raw_pan_offset, raw_tilt_offset, raw_pan_setpoint, raw_tilt_setpoint


def resample_tracks(raw_timestamps, raw_pan_offset, raw_tilt_offset, raw_pan_setpoint, raw_tilt_setpoint, sampling_interval, max_timestamp):
    interp_pan_offset = interp1d(
        raw_timestamps, raw_pan_offset, kind="cubic", fill_value="extrapolate")
    interp_tilt_offset = interp1d(
        raw_timestamps, raw_tilt_offset, kind="cubic", fill_value="extrapolate")

    interp_pan_setpoint = interp1d(
        raw_timestamps, raw_pan_setpoint, kind="cubic", fill_value="extrapolate")
    interp_tilt_setpoint = interp1d(
        raw_timestamps, raw_tilt_setpoint, kind="cubic", fill_value="extrapolate")

    resampled_max_timestamp = ceil(
        max_timestamp / sampling_interval) * sampling_interval
    num_timestamps = int(resampled_max_timestamp / sampling_interval + 1)

    resampled_pan_offset = interp_pan_offset(np.linspace(
        0, resampled_max_timestamp, num_timestamps, endpoint=True))
    resampled_tilt_offset = interp_tilt_offset(np.linspace(
        0, resampled_max_timestamp, num_timestamps, endpoint=True))

    resampled_pan_setpoint = interp_pan_setpoint(np.linspace(
        0, resampled_max_timestamp, num_timestamps, endpoint=True))
    resampled_tilt_setpoint = interp_tilt_setpoint(np.linspace(
        0, resampled_max_timestamp, num_timestamps, endpoint=True))

    return OffsetTrack(resampled_pan_offset), OffsetTrack(resampled_tilt_offset), SetpointTrack(resampled_pan_setpoint), SetpointTrack(resampled_tilt_setpoint)


def get_angled(offset_track, setpoint_track):
    modified_offset_track = offset_track.apply_modifiers()
    modified_control_points = setpoint_track.apply_control_points()
    angle = modified_offset_track + modified_control_points
    return angle


def export(left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track):
    left_pan_angle = get_angled(left_pan_offset_track, left_pan_setpoint_track)
    left_tilt_angle = get_angled(
        left_tilt_offset_track, left_tilt_setpoint_track)

    right_pan_angle = get_angled(
        right_pan_offset_track, right_pan_setpoint_track)
    right_tilt_angle = get_angled(
        right_tilt_offset_track, right_tilt_setpoint_track)

    return left_pan_angle, left_tilt_angle, right_pan_angle, right_tilt_angle


def save_to_msync(left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track):
    with open(output_filename, "w+") as output_file:
        fieldnames = ["timestamp", "left_pan",
                      "left_tilt", "right_pan", "right_tilt"]
        msync_writer = csv.DictWriter(output_file, fieldnames=fieldnames)

        msync_writer.writeheader()

        left_pan_angle, left_tilt_angle, right_pan_angle, right_tilt_angle = export(
            left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track)

        MIN_PAN = 30
        MAX_PAN = 150
        MIN_TILT = 70
        MAX_TILT = 110

        time = 0
        for left_pan, left_tilt, right_pan, right_tilt in zip(left_pan_angle, left_tilt_angle, right_pan_angle, right_tilt_angle):
            lp = round(min(max(left_pan, MIN_PAN), MAX_PAN))
            lt = round(min(max(left_tilt, MIN_TILT), MAX_TILT))
            rp = round(min(max(right_pan, MIN_PAN), MAX_PAN))
            rt = round(min(max(right_tilt, MIN_TILT), MAX_TILT))

            row_dict = {
                "timestamp": time,
                "left_pan": lp,
                "left_tilt": lt,
                "right_pan": rp,
                "right_tilt": rt
            }
            msync_writer.writerow(row_dict)
            time += ARDUINO_SAMPLING_INTERVAL


def rotate_marbles(left_pan, left_tilt, right_pan, right_tilt):
    OPENGL_PAN_OFFSET = 90
    OPENGL_TILT_OFFSET = 90

    glPushMatrix()
    glTranslatef(MARBLE_DISPLAY_X_OFFSET, 0, 0)
    glRotatef(left_tilt - OPENGL_TILT_OFFSET, -1, 0, 0)
    glRotatef(left_pan - OPENGL_PAN_OFFSET, 0, 1, 0)
    marble.render()
    glPopMatrix()

    glPushMatrix()
    glTranslatef(-MARBLE_DISPLAY_X_OFFSET, 0, 0)
    glRotatef(right_tilt - OPENGL_TILT_OFFSET, -1, 0, 0)
    glRotatef(right_pan - OPENGL_PAN_OFFSET, 0, 1, 0)
    marble.render()
    glPopMatrix()


def playback(event, speed):
    global PLAYBACK_SPEED
    PLAYBACK_SPEED = speed
    plt.close('all')
    pygame_main()
    matplot_main()


def submit(event, user_input, start):
    global START_TIMESTAMP, END_TIMESTAMP
    try:
        user_input = int(user_input)
    except:
        print("Please enter a integer value")
        return
    if start:
        START_TIMESTAMP = user_input
    else:
        END_TIMESTAMP = user_input


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="Process 2 .mdat files into a .msync file")
    ap.add_argument('left', help="The left marble's .mdat file to process")
    ap.add_argument('right', help="The right marble's .mdat file to process")
    ap.add_argument('-o', '--output',
                    help="The output .msync file to write to")
    # TODO(JS): Actually use the output file argument

    args = ap.parse_args()

    left_mdat_file = args.left
    right_mdat_file = args.right

    output_filename = args.output if args.output is not None else create_unique_filename(
        f"outputs/SynchronizationEditor/{os.path.splitext(os.path.basename(left_mdat_file))[0]}_{os.path.splitext(os.path.basename(right_mdat_file))[0]}.msync")

    main()

'''
This program enables the synchronization of 2 .mdat files with .wav audio files to create an edited .msync file
'''

import pygame
from pygame import mixer
import os
import argparse
import csv
import wave
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

pygame.init()
mixer.init()
mixer.music.load("drew_sample.wav")
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
        # What'd you do with extra week?
        lca(4510 , 90, 4870, 130)
        rca(4510+400, 90, 4870+400, 50)
        
        # If you feel like you're missing out
        rca(38344, 50, 38911, 90)
        lca(38344+300, 130, 38911+300, 90)

        # So the Sand Marble Rally is back
        lca(50709, 90, 51276, 130)
        rca(50709+400, 90, 51276+400, 50)

        # We'll see you in a moment
        lca(257501, 130, 257901, 90)

        # (reset back to angle)
        lca(259474, 90, 260000, 130)

        # And now
        rca(305969, 50, 306405, 90)
        lca(305969+400, 130, 306405+400, 90)

        # So it's time to bring out
        rca(484579, 90, 485100, 50)
        lca(484579+400, 90, 485100+400, 130)

        # So last episode
        lca(538037, 130, 538187, 90)
        rca(538037+300, 50, 538187+300, 90)

    # both()

#############################
#           Main            #
#############################


def matplot_main():
    global initialized
    global left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track
    if not initialized:
        raw_left_timestamps, raw_left_pan_offset, raw_left_tilt_offset, raw_left_pan_setpoint, raw_left_tilt_setpoint = parse_mdat_file(
            left_mdat_file)
        raw_right_timestamps, raw_right_pan_offset, raw_right_tilt_offset, raw_right_pan_setpoint, raw_right_tilt_setpoint = parse_mdat_file(
            right_mdat_file)

        max_timestamp = max(raw_left_timestamps[-1], raw_right_timestamps[-1])

        left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track = resample_tracks(
            raw_left_timestamps, raw_left_pan_offset, raw_left_tilt_offset, raw_left_pan_setpoint, raw_left_tilt_setpoint, ARDUINO_SAMPLING_INTERVAL, max_timestamp)
        right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track = resample_tracks(
            raw_right_timestamps, raw_right_pan_offset, raw_right_tilt_offset, raw_right_pan_setpoint, raw_right_tilt_setpoint, ARDUINO_SAMPLING_INTERVAL, max_timestamp)

        initialized = True

    spf = wave.open('assets/wavfile.wav', 'r')

    signal = spf.readframes(-1)
    signal = np.fromstring(signal, "Int16")

    workspace()
    # plt.ion()
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
    plt.plot(signal, color='blue')

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
    plt.plot(signal, color='red')

    seek_start = TextBox(
        plt.axes([0.1, 0.05, 0.1, 0.050]), 'start', initial='0')
    seek_start.on_submit(lambda e, i: submit(e, i, True))
    seek_end = TextBox(plt.axes([0.2, 0.05, 0.1, 0.050]), 'end', initial='0')
    seek_end.on_submit(lambda e, i: submit(e, i, False))

    b_playback_half = Button(plt.axes([0.4, 0.05, 0.1, 0.050]), '0.5x')
    b_playback_half.on_clicked(lambda e: playback(e, 0.5))
    b_playback_1 = Button(plt.axes([0.5, 0.05, 0.1, 0.050]), '1x')
    b_playback_1.on_clicked(lambda e: playback(e, 1))
    b_playback_2 = Button(plt.axes([0.6, 0.05, 0.1, 0.050]), '2x')
    b_playback_2.on_clicked(lambda e: playback(e, 2))
    b_playback_3 = Button(plt.axes([0.7, 0.05, 0.1, 0.050]), '3x')
    b_playback_3.on_clicked(lambda e: playback(e, 3))
    b_playback_4 = Button(plt.axes([0.8, 0.05, 0.1, 0.050]), '4x')
    b_playback_4.on_clicked(lambda e: playback(e, 4))
    b_playback_10 = Button(plt.axes([0.9, 0.05, 0.1, 0.050]), 'exp')
    b_playback_10.on_clicked(lambda e: save_to_msync(left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track,
                                                     left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track))
    # plt.pause(0.1)
    plt.show()


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
    end_index = int(round(END_TIMESTAMP / ARDUINO_SAMPLING_INTERVAL)
                    ) if END_TIMESTAMP != 0 else len(left_pan_angle) - 1

    start_ms = millisec()
    timer = 0
    mixer.music.play()
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


def parse_row(row):
    timestamp = int(row["timestamp"])

    pan_offset, pan_setpoint = int(row["pan_offset"]), int(row["pan_setpoint"])

    tilt_offset, tilt_setpoint = int(
        row["tilt_offset"]), int(row["tilt_setpoint"])

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
            timestamp, pan_offset, tilt_offset, pan_setpoint, tilt_setpoint = parse_row(
                row)

            raw_timestamps = np.append(raw_timestamps, timestamp)
            raw_pan_offset = np.append(raw_pan_offset, pan_offset)
            raw_pan_setpoint = np.append(
                raw_pan_setpoint, pan_setpoint)
            raw_tilt_offset = np.append(raw_tilt_offset, tilt_offset)
            raw_tilt_setpoint = np.append(
                raw_tilt_setpoint, tilt_setpoint)

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

    matplot_main()

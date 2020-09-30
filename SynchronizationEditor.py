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
import keyboard
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
    global left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track
    index = round(ms / ARDUINO_SAMPLING_INTERVAL)
    right_pan_setpoint_track.add_control_point(index, angle)


def rca(start_ms, start_angle, end_ms, end_angle):
    global left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track
    rcp(start_ms, start_angle)
    rcp(end_ms, end_angle)


def rn(startms, endms):
    global right_pan_offset_track, right_tilt_offset_track
    global left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track
    start_index = round(startms / ARDUINO_SAMPLING_INTERVAL)
    end_index = round(endms / ARDUINO_SAMPLING_INTERVAL)
    right_pan_offset_track.add_region_modifier(start_index, end_index, 0)
    right_tilt_offset_track.add_region_modifier(start_index, end_index, 0)


def lcp(ms, angle):
    global left_pan_setpoint_track
    global left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track
    index = round(ms / ARDUINO_SAMPLING_INTERVAL)
    left_pan_setpoint_track.add_control_point(index, angle)


def lca(start_ms, start_angle, end_ms, end_angle):
    global left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track
    lcp(start_ms, start_angle)
    lcp(end_ms, end_angle)


def ln(startms, endms):
    global left_pan_offset_track, left_tilt_offset_track
    global left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track
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

  
    

def matplot_main():
    global left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track

    '''
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
    '''

    display_plots()

    seek_start = TextBox(
        plt.axes([0.1, 0.05, 0.1, 0.050]), 'start', initial='0')
    seek_start.on_submit(lambda e, i: submit(e, i, True))
    seek_end = TextBox(plt.axes([0.2, 0.05, 0.1, 0.050]), 'end', initial='0')
    seek_end.on_submit(lambda e, i: submit(e, i, False))

    '''
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
    '''
    #plt.show()


def pygame_main():
    global left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track
    #raw_left_timestamps, raw_left_pan_offset, raw_left_tilt_offset, raw_left_pan_setpoint, raw_left_tilt_setpoint = parse_mdat_file(left_mdat_file)
    #raw_right_timestamps, raw_right_pan_offset, raw_right_tilt_offset, raw_right_pan_setpoint, raw_right_tilt_setpoint = parse_mdat_file(right_mdat_file)
    max_timestamp = max(raw_left_timestamps[-1], raw_right_timestamps[-1])

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
    
    '''
    for i in range(1, 0, -1):
        pygame.display.set_caption(str(i))
        pygame.display.flip()  
        pygame.time.wait(1000)
    '''
    
    start_ms = 0
    end_ms = 0

    option = input("Please enter 'p' for entire animation, 's' for a slice, 'c' to add a control point pair, or 'q' to quit and save changes (p/s/c/q): ")
    if option == 'p':
        start_ms = 0
        end_ms = max_timestamp
    elif option == 's':
        start_ms = float(input("Please enter a starting timestamp (must be greater than or equal to 0): "))
        end_ms = float(input("Please enter an ending timestamp (must be less than or equal to " + str(max_timestamp) + "): "))
    elif option == 'c':
        add_cp_pair()
        print("Saving changes...")
        return
    elif option == 'q':
        save_to_msync(left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track,
                                                     left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track)
        return False
    
    start_index = int(round(start_ms / ARDUINO_SAMPLING_INTERVAL))
    end_index = int(round(end_ms / ARDUINO_SAMPLING_INTERVAL)
                    ) if end_ms != 0 else len(left_pan_angle) - 1
    timer = 0
    kool_start_ms = start_ms #just setting another dummy variable to start_ms so i don't lose the value
    start_ms = millisec()
    for i in range(start_index, (end_index + 1)):
        left_pan = left_pan_angle[i]
        left_tilt = left_tilt_angle[i]
        right_pan = right_pan_angle[i]
        right_tilt = right_tilt_angle[i]
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
            print(timer + kool_start_ms)
        else:
            print(f"Too slow at timer {timer} with waitTime {waitTime}")
        if keyboard.is_pressed('q'):
            break
    pygame.quit()
    return

#############################
#        Functions          #
#############################
def create_file(file_name, file_type, n=0):
    global left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track
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

    my_speed = input("At what speed would you like to play the animation [any positive float or integer]? Or, you can type 'e' to exit: ")
    print("Close the matplot window to continue.")
    plt.show()
    if my_speed == 'e':
        return
    playback(float(my_speed))

def parse_row(row):
    global left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track
    timestamp = int(row["timestamp"])

    pan_offset, pan_setpoint = int(float(row["pan_offset"])), int(float(row["pan_setpoint"]))

    tilt_offset, tilt_setpoint = int(float(row["tilt_offset"])), int(float(row["tilt_setpoint"]))

    return timestamp, pan_offset, tilt_offset, pan_setpoint, tilt_setpoint


def parse_mdat_file(mdat_file):
    global left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track
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
    global left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track
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
    global left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track
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
    print("Your data has been saved to a .msync file!")


def rotate_marbles(left_pan, left_tilt, right_pan, right_tilt):
    global left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track
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


def playback(speed):
    global left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track
    my_answer = 'y'
    while my_answer == 'y':
        global PLAYBACK_SPEED
        PLAYBACK_SPEED = speed
        plt.close('all')
        my_var = pygame_main()
        if my_var == False:
            break
    matplot_main()


def submit(event, user_input, start):
    global START_TIMESTAMP, END_TIMESTAMP
    global left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track
    try:
        user_input = int(user_input)
    except:
        print("Please enter a integer value")
        return
    if start:
        START_TIMESTAMP = user_input
    else:
        END_TIMESTAMP = user_input

def add_cp_pair():
    global left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track, right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track
    
    max_timestamp = max(raw_left_timestamps[-1], raw_right_timestamps[-1])

    marble = input("Which marble would you like to add a control point pair (Enter 'l' for left, 'r' for right)? ")
    direction = input("Which direction would you like to turn (turnleft, turncenter, turnright)? ")
    start_ms = float(input("Enter a starting timestamp (ms): "))
    end_ms = float(input("Enter an ending timestamp (ms): "))
    ending_angle = 90 #by default
    if direction == "turnleft":
        ending_angle = 50
    elif direction == "turncenter":
        ending_angle = 90
    elif direction == "turnright":
        ending_angle = 130

    
    if marble == 'r': 
        corrected_start_index = np.abs(raw_right_timestamps - start_ms).argmin()
        corrected_end_index = np.abs(raw_right_timestamps - end_ms).argmin()
        starting_angle = raw_right_pan_setpoint[corrected_start_index]
        rca(start_ms, starting_angle, end_ms, ending_angle)
        right_pan_setpoint_track.apply_control_points()
        '''
        rate_of_change = float( (ending_angle - raw_right_pan_setpoint[corrected_start_index]) / (corrected_end_index - corrected_start_index))
        curr_angle = raw_right_pan_setpoint[corrected_start_index]
        for row_index in range(corrected_start_index, (corrected_end_index + 1)):
            print(curr_angle)
            raw_right_pan_setpoint[row_index] = curr_angle
            curr_angle += rate_of_change
        '''
    elif marble == 'l':
        corrected_start_index = np.abs(raw_left_timestamps - start_ms).argmin()
        corrected_end_index = np.abs(raw_left_timestamps - end_ms).argmin()
        starting_angle = raw_left_pan_setpoint[corrected_start_index]
        lca(start_ms, starting_angle, end_ms, ending_angle)
        left_pan_setpoint_track.apply_control_points()
        '''
        rate_of_change = float( (ending_angle - raw_left_pan_setpoint[corrected_start_index]) / (corrected_end_index - corrected_start_index))
        curr_angle = raw_left_pan_setpoint[corrected_start_index]
        for row_index in range(corrected_start_index, (corrected_end_index + 1)):
            print(curr_angle)
            raw_left_pan_setpoint[row_index] = curr_angle
            curr_angle += rate_of_change
        '''
    '''
    left_pan_offset_track, left_tilt_offset_track, left_pan_setpoint_track, left_tilt_setpoint_track = resample_tracks(
        raw_left_timestamps, raw_left_pan_offset, raw_left_tilt_offset, raw_left_pan_setpoint, raw_left_tilt_setpoint, ARDUINO_SAMPLING_INTERVAL, max_timestamp)
    right_pan_offset_track, right_tilt_offset_track, right_pan_setpoint_track, right_tilt_setpoint_track = resample_tracks(
        raw_right_timestamps, raw_right_pan_offset, raw_right_tilt_offset, raw_right_pan_setpoint, raw_right_tilt_setpoint, ARDUINO_SAMPLING_INTERVAL, max_timestamp)
    '''
    print("Control point pair added!", marble, direction, start_ms, end_ms)

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

    raw_left_timestamps, raw_left_pan_offset, raw_left_tilt_offset, raw_left_pan_setpoint, raw_left_tilt_setpoint = parse_mdat_file(left_mdat_file)
    raw_right_timestamps, raw_right_pan_offset, raw_right_tilt_offset, raw_right_pan_setpoint, raw_right_tilt_setpoint = parse_mdat_file(right_mdat_file)
    print("mdats have been parsed")

    output_filename = args.output if args.output is not None else create_unique_filename(
        f"outputs/SynchronizationEditor/{os.path.splitext(os.path.basename(left_mdat_file))[0]}_{os.path.splitext(os.path.basename(right_mdat_file))[0]}.msync")

    main()
    matplot_main()
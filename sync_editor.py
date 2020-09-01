import pygame, csv, wave, sys
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from objloader import *
from math import *


#############################
#         Constants         #
#############################
FRAME_RATE = 100
MARBLE_OFFSET = 20
ARDUINO_INTERVAL = 50


#############################
#      Initialization       #
#############################
frame_time = int(round(1 / FRAME_RATE * 1000))

# Pygame and openGL marble rendering
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

# Initialize the Camera
glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
glEnable(GL_LIGHT0)
glEnable(GL_LIGHTING)
glEnable(GL_COLOR_MATERIAL)
glEnable(GL_DEPTH_TEST)
glShadeModel(GL_SMOOTH) 

marble = OBJ('MarbleHeadset_v11.obj', swapyz=True)
marble.generate()

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
width, height = display
glOrtho(-display[0] / 15, display[0]/ 15, -display[1] / 15, display[1] / 15, 0.1, 40.0)
glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_MODELVIEW)

glTranslatef(0.0, 0.0, -30)
glRotatef(180, 0, 1, 0)


# Matplotlib .wav audio
spf = wave.open("wavfile.wav", "r")

signal = spf.readframes(-1)
signal = np.fromstring(signal, "Int16")

if spf.getnchannels() == 2:
    print("Just mono files")
    sys.exit(0)

plt.figure(1)
plt.plot(signal)
plt.figure(1)
plt.plot(signal)
plt.show()


#############################
#           Main            #
#############################
def main():
    left_data = []
    with open('outputs/Bob_Cropped_converted.csv', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
        for row in csv_reader:
            if 'timestamp' not in row:
                left_data.append([float(x) for x in row])
    
    right_data = []
    with open('outputs/Dan_Cropped_converted.csv', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
        for row in csv_reader:
            if 'timestamp' not in row:
                right_data.append([float(x) for x in row])
    
    max_timestamp = max(left_data[-1][0], right_data[-1][0])
    i_times = [ARDUINO_INTERVAL * i for i in range(ceil(max_timestamp / ARDUINO_INTERVAL) + 1)]
    i_left_tilt = np.interp(i_times, [left_data[i][0] for i in range(len(left_data))], [left_data[i][1] for i in range(len(left_data))], right=left_data[-1][1])
    i_left_pan = np.interp(i_times, [left_data[i][0] for i in range(len(left_data))], [left_data[i][2] for i in range(len(left_data))], right=left_data[-1][2])
    i_right_tilt = np.interp(i_times, [right_data[i][0] for i in range(len(right_data))], [right_data[i][1] for i in range(len(right_data))], right=right_data[-1][1])
    i_right_pan = np.interp(i_times, [right_data[i][0] for i in range(len(right_data))], [right_data[i][2] for i in range(len(right_data))], right=right_data[-1][2])

    i_left = np.dstack((i_left_tilt, i_left_pan))[0]
    i_right = np.dstack((i_right_tilt, i_right_pan))[0]

    converted_i_times = np.array([[t] for t in i_times])
    i_combined = np.append(converted_i_times, i_left, 1)
    i_combined = np.append(i_combined, i_right, 1)

    left_prev, left_curr = (0, 0), (0, 0)
    right_prev, right_curr = (0, 0), (0, 0)
    time = 0
    while len(i_combined) > 0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        rotate_marbles(*i_combined[0][1:])
        i_combined = i_combined[1:]

        pygame.display.flip()
        pygame.time.wait(ARDUINO_INTERVAL)
        time += ARDUINO_INTERVAL


#############################
#        Functions          #
#############################
def create_file(file_name, file_type, n=0):
    destination = './outputs/{0}{1}.{2}'.format(file_name, f'({n})', file_type)       # Add:    if n != 0 else ''    after f' ({n})' if you don't want the first file to have a number
    if not os.path.isfile(destination):
        return open(destination, 'w+')
    else:
        return create_file(file_name, file_type, n+1)

def rotate_marbles(left_x, left_y, right_x, right_y):
    glPushMatrix()
    glTranslatef(-MARBLE_OFFSET, 0, 0)
    glRotatef(left_x, 1, 0, 0)
    glRotatef(left_y, 0, -1, 0)
    marble.render()
    glPopMatrix()

    glPushMatrix()
    glTranslatef(MARBLE_OFFSET, 0, 0)
    glRotatef(right_x, 1, 0, 0)
    glRotatef(right_y, 0, -1, 0)
    marble.render()
    glPopMatrix()


if __name__ == '__main__':
    main()

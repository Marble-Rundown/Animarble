# import numpy as np
# import matplotlib.pyplot as plot
# import wave
# import sys

# spf = wave.open("wavfile.wav", "r")

# signal = spf.readframes(-1)
# signal = np.fromstring(signal, "Int16")


# # if spf.getnchannels() == 2:
# #     print("Only mono channel files are allowed")
# #     sys.exit(0)

# plot.figure(1)
# plot.title("Signal Wave...")
# plot.plot(signal)
# plot.show()

import pygame, csv
from threading import Thread
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from mlib import *
from objloader import *


#############################
#         Constants         #
#############################
FRAME_RATE = 100


#############################
#      Initialization       #
#############################
frame_time = int(round(1 / FRAME_RATE * 1000))

marble = OBJ('MarbleHeadset v11.obj')
marble.generate()

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
glMatrixMode(GL_MODELVIEW)
glShadeModel(GL_SMOOTH)

gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
# glOrtho(-50.0, 50.0, -50.0, 50.0, 0.1, 50.0)
# glOrtho(0.0f, display[0], display[1], 0.0f, 0.0f, 1.0f)
glTranslatef(0.0, 0.0, -20)
glRotatef(0, 0, 0, 0)


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
    
    

    left_prev, left_curr = (0, 0), (0, 0)
    right_prev, right_curr = (0, 0), (0, 0)
    time = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        marble.render()

        if time > left_data[0][0]:
            print(left_data[0][1], left_data[0][2])
            left_curr = (left_data[0][1], left_data[0][2])
            rotate_marble(-1, left_curr[0], left_curr[1])
            left_prev = left_curr
            left_data.pop(0)
        else:
            rotate_marble(-1, left_prev[0], left_prev[1]) if left_prev else rotate_marble(-1, 90, 90)
        
        if time > right_data[0][0]:
            print(right_data[0][1], right_data[0][2])
            right_curr = (right_data[0][1], right_data[0][2])
            rotate_marble(1, right_curr[0], right_curr[1])
            right_prev = right_curr
            right_data.pop(0)
        else:
            rotate_marble(1, right_prev[0], right_prev[1]) if right_prev else rotate_marble(1, 90, 90)

        pygame.display.flip()
        pygame.time.wait(frame_time)
        time += frame_time

if __name__ == '__main__':
    main()

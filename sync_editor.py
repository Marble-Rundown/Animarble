import pygame, csv
import numpy as np
from threading import Thread
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from objloader import *


#############################
#         Constants         #
#############################
FRAME_RATE = 100
MARBLE_OFFSET = 20


#############################
#      Initialization       #
#############################
frame_time = int(round(1 / FRAME_RATE * 1000))

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
# gluPerspective(90.0, width/float(height), 1, 100.0)
glOrtho(-display[0] / 15, display[0]/ 15, -display[1] / 15, display[1] / 15, 0.1, 40.0)
glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_MODELVIEW)

glTranslatef(0.0, 0.0, -30)
glRotatef(180, 0, 1, 0)


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
    
    # max_timestamp = max(left_data[-1][0], right_data[-1][0])
    # x_vals = [50 * i for i in range(max_timestamp / )]


    left_prev, left_curr = (0, 0), (0, 0)
    right_prev, right_curr = (0, 0), (0, 0)
    time = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

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


#############################
#        Functions          #
#############################
def rotate_marble(side, x, y):
    glPushMatrix()
    glTranslatef(side * 20, 0, 0)
    glRotatef(x, 1, 0, 0)
    glRotatef(y, 0, -1, 0)
    # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # draw_sphere()
    # draw_cylinder()
    marble.render()
    glPopMatrix()

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

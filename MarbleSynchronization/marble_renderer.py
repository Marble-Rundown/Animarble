import pygame, csv
from threading import Thread
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

def draw_sphere():
    glColor3f(1.0, 1.0, 0.0)
    sphere = gluNewQuadric()
    gluQuadricNormals(sphere, GLU_SMOOTH)
    gluQuadricTexture(sphere, GL_TRUE)
    gluSphere(sphere, 1.0, 32, 16)

def draw_cylinder():
    #glRotatef(1, 1, 1.25, 12.5)
    glColor3f(1.0, 0.0, 0.0)
    cylinder = gluNewQuadric()
    gluQuadricNormals(cylinder, GLU_SMOOTH)
    gluQuadricTexture(cylinder, GL_TRUE)
    gluCylinder(cylinder, 0.15, 0.15, 2.5, 32, 32)

def renderer_main():
    data = []
    with open('outputs/rotations (1).csv', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
        for row in csv_reader:
            if 'timestamp' not in row:
                #print([float(x) for x in row])
                data.append([float(x) for x in row])


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
    
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -20)
    glRotatef(0, 0, 0, 0)

    time = 0;
    prev, curr = data[0], data[1]
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        #data = [[timestamp, rotx, roty], ...]
        if not len(data) < 2 and time > data[1][0]:
            
            d_x = curr[1] - prev[1]
            glRotatef(d_x, 1, 0, 0)       # Rotates everything
            d_y = curr[2] - prev[2]
            glRotatef(d_y, 0, 1, 0)       # Rotates everything

            if abs(d_x) > 3.5 or abs(d_y) > 3.5:
                print('{0}:\nd_x = {1}\nd_y = {2}'.format(data[1][0], d_x, d_y))       # prints the timestamp, which at index 0

            data.pop(0)
            if len(data) < 2:
                break
            prev, curr = data[0], data[1]
        

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        #Cube()
        draw_sphere()
        draw_cylinder()

        pygame.display.flip()
        pygame.time.wait(10)
        time += 10


t = Thread(renderer_main())
t.start()
t.join
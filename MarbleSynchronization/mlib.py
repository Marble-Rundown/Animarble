# A library of functions related to displaying and manipulating marbles

from OpenGL.GL import *
from OpenGL.GLU import *


def draw_sphere():
    # glPushMatrix()
    glColor3f(1.0, 1.0, 0.0)
    sphere = gluNewQuadric()
    gluQuadricNormals(sphere, GLU_SMOOTH)
    gluQuadricTexture(sphere, GL_TRUE)
    gluSphere(sphere, 1.0, 32, 16)
    # glPopMatrix()

def draw_cylinder():
    # glPushMatrix()
    glColor3f(1.0, 0.0, 0.0)
    cylinder = gluNewQuadric()
    gluQuadricNormals(cylinder, GLU_SMOOTH)
    gluQuadricTexture(cylinder, GL_TRUE)
    gluCylinder(cylinder, 0.15, 0.15, 2.5, 32, 32)
    # glPopMatrix()

def rotate_marble(marble, x, y):
    glPushMatrix()
    glTranslatef(marble * 5, 0, 0)
    glRotatef(x, 1, 0, 0)
    glRotatef(y, 0, -1, 0)
    # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_sphere()
    draw_cylinder()
    glPopMatrix()

# def test(x, y):
#     glPushMatrix()

#     glTranslatef(-5, 0, 0)

#     glRotatef(x, 1, 0, 0)
#     glRotatef(y, 0, -1, 0)
#     # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#     draw_sphere()
#     draw_cylinder()
#     glPopMatrix()
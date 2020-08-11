import pygame, threading
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from obj_loader import *
import shader_loader as ShaderLoader

#class MarbleRenderer(object):
vertices= (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
    )

edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )

def Cube():
    glBegin(GL_LINES)
    for edge in edges:
        for vertext in edge:
            glVertex3fv(verticies[vertex])
    glEnd()


def main(self):
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    # Initialize the Camera
    glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
    #glLightfv(GL_LIGHT0, GL_DIFFUSE, (-40, 200, , 1.0))
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -40)
    glRotatef(0, 0, 0, 0)

    #obj = OBJ('MarbleHeadset.obj')

    clock = pygame.time.Clock()

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    width, height = viewport
    gluPerspective(90.0, width/float(height), 1, 100.0)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_MODELVIEW)

    rx, ry = (0,0)
    tx, ty = (0,0)
    zpos = 5
    rotate = move = False
    #obj = ObjLoader()
    #obj.load_model("MarbleHeadset.obj")

    #texture_offset = len(obj.vertex_index)*12


    #shader = ShaderLoader.compile_shader("assets/video_17_vert.vs", "assets/video_17_frag.fs")

    #VBO = glGenBuffers(1)
    #glBindBuffer(GL_ARRAY_BUFFER, VBO)
    #glBufferData(GL_ARRAY_BUFFER, obj.model.itemsize * len(obj.model), obj.model, GL_STATIC_DRAW)

    ##position
    #glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, obj.model.itemsize * 3, ctypes.c_void_p(0))
    #glEnableVertexAttribArray(0)
    ##texture
    #glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, obj.model.itemsize * 2, ctypes.c_void_p(texture_offset))
    #glEnableVertexAttribArray(1)

    #glUseProgram(shader)


    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        Cube()

        glLoadIdentity()

        # RENDER OBJECT
        glTranslate(tx/20., ty/20., - zpos)
        glRotate(ry, 1, 0, 0)
        glRotate(rx, 0, 1, 0)
        glCallList(obj.gl_list)
        #transformLoc = glGetUniformLocation(shader, "transform")
        #glUniformMatrix4fv(transformLoc, 1, GL_FALSE, rot_x * rot_y)

        #glDrawArrays(GL_TRIANGLES, 0, len(obj.vertex_index))

        pygame.display.flip()
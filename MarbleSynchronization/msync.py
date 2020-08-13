import dlib, pygame
import cv2
import argparse
import numpy as np
import os
from pose_estimator import PoseEstimator
#import marble_renderer
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


#############################
#         Constants         #
#############################
WINDOW_TITLE = 'Facial Landmark Detector'
IMAGE_RESCALE = 75
VIDEO_RESCALE = 50

class Profile:
    def __init__(self, name):
        if name == "Jaiveer":
            self.resting_lip_distance = 0
            self.resting_eyebrow_elevation = 10
            self.weights = {
                "measured_tilt_degrees": 1,
                "lip_spacing": 0,
                "eyebrow_elevation": 0
            }

def get_tilt_features(profile, head_rotation, head_translation, landmarks):
    return {
        "measured_tilt_degrees": head_rotation[0],
        "lip_spacing": landmarks[51][1] - landmarks[57][1] + profile.resting_lip_distance,
        "eyebrow_elevation": (landmarks[19][1] + landmarks[24][1]) / 2 - landmarks[27][1] + profile.resting_eyebrow_elevation
    }

def get_tilt_weights(profile):
    return profile.weights

def dot_product(features, weights):
    return sum([feature_val * weights[feature_label] for feature_label, feature_val in features.items()])


#############################
#       Initialization      #
#############################
# Computer Vision Initialization
detector = dlib.get_frontal_face_detector()     # Get a face detector from the dlib library
predictor = dlib.shape_predictor('assets/shape_predictor_68_face_landmarks.dat')      # Get a shape predictor based on a trained model from the dlib library

def check_str(value):
    if type(value) != str:
        raise argparse.ArgumentTypeError(f"'{value}' is not a string")
    return value;
ap = argparse.ArgumentParser()      # Create an instance of an ArgumentParser object
ap.add_argument('-i', '--image', type=check_str, help='The path to the image')      # Adds an argument '--image' that describes the image file path
ap.add_argument('-v', '--video', type=check_str, help='The path to the video')      # Adds an argument '--video' that describes the video file path
ap.add_argument('-s', '--stream', help='Set this to True to start in livestream mode')      # Adds an argument '--stream' that determines whether to use the webcam
args = vars(ap.parse_args())        # Vars() returns the __dict__ attribute of an object, so args is a dictionary of the command line parameters passed to this program
if not any(args.values()):
    raise TypeError('Expected 1 argument, but received 0 arguments')
num_args = len([a for a in args.values() if a])
if num_args > 1:
    raise TypeError('Expected only 1 argument, but received {0} arguments'.format(num_args))

file_type = ''      # Initialize variables based on target type
target = None
rescale = 100
dimensions = ()
if args['image']:
    file_type = 'image'
    image = cv2.imread('media/' + args['image'])
    dimensions = tuple(image.shape[i] * IMAGE_RESCALE / 100 for i in range(2))
    rescale = IMAGE_RESCALE
elif args['video']:
    file_type = 'video'
    cap = cv2.VideoCapture('media/' + args['video'])
    assert cap.isOpened(), 'Failed to open video file'
    _, first_frame = cap.read()
    dimensions = tuple(first_frame.shape[i] * VIDEO_RESCALE / 100 for i in range(2))
    rescale = VIDEO_RESCALE
elif args['stream']:
    file_type = 'stream'
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), 'Failed to open stream'
    _, first_frame = cap.read()
    dimensions = tuple(first_frame.shape[i] for i in range(2))

pe = PoseEstimator(dimensions)


# Pygame and OpenGL Initialization

def draw_sphere():
    glColor3f(1.0, 1.0, 0.0)
    sphere = gluNewQuadric()
    gluQuadricNormals(sphere, GLU_SMOOTH)
    gluQuadricTexture(sphere, GL_TRUE)
    gluSphere(sphere, 1.0, 32, 16)

def draw_cylinder():
    glColor3f(1.0, 0.0, 0.0)
    cylinder = gluNewQuadric()
    gluQuadricNormals(cylinder, GLU_SMOOTH)
    gluQuadricTexture(cylinder, GL_TRUE)
    gluCylinder(cylinder, 0.15, 0.15, 2.5, 32, 32)

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

def render_marble(pan, tilt):
    glPushMatrix()
    glRotatef(tilt, 1, 0, 0)
    glRotatef(pan, 0, -1, 0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_sphere()
    draw_cylinder()
    glPopMatrix()


#############################
#           Main            #
#############################
def main():

    with create_file('rotations', 'csv') as file:
        file.write('timestamp,pitch,yaw\n')

        if file_type == 'image':
            frame, landmarks, img_points = detect(image, mark=True)
            frame = rescale_frame(frame, percent=rescale)
        
            if img_points.size != 0:
                rotation, translation = pe.estimate_pose(img_points)
        
            cv2.imshow(WINDOW_TITLE, frame)     # Shows the image in a new window
            cv2.waitKey(0)
        else:
            sliding_window = []
            WINDOW_LENGTH = 6

            profile = Profile("Jaiveer")

            while cap.isOpened():           # Main while loop
                for event in pygame.event.get():        # Pygame
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                
                s, frame = cap.read()
                assert s, "Failed to read from webcam"

                frame, landmarks, img_points = detect(frame, mark=True)
                frame = rescale_frame(frame, percent=rescale)

                if img_points.size != 0:
                    rotation, translation = pe.estimate_pose(img_points)
                
                    sliding_window.append(rotation)
                    if len(sliding_window) > WINDOW_LENGTH:
                        sliding_window.pop(0)
                    filtered_rotation = [ewma([rot[i] for rot in sliding_window]) for i in range(3)]

                    tilt = dot_product(
                        get_tilt_features(profile, filtered_rotation, translation, landmarks), 
                        get_tilt_weights(profile))
                    pan = 0

                    # file.write('{0},{1},{2}\n'.format(int(cap.get(cv2.CAP_PROP_POS_MSEC)), converted_rotation[0][0], converted_rotation[1][0]))

                    render_marble(pan, tilt)
                else:
                    print("Failed to find image points")

                cv2.imshow(WINDOW_TITLE, frame)
                if cv2.waitKey(1) & 0xFF == 27:     # 27 is ASCII for the Esc key on a keyboard
                    break     

                pygame.display.flip()
                pygame.time.wait(10)
    cap.release()
    cv2.destroyAllWindows()


#############################
#         Functions         #
#############################
jbr = lambda angle: -angle + 90

def create_file(file_name, file_type, n=0):
    destination = './outputs/{0}{1}.{2}'.format(file_name, f' ({n})' if n != 0 else '', file_type)
    if not os.path.isfile(destination):
        return open(destination, 'w+')
    else:
        return create_file(file_name, file_type, n+1)

def ewma(data):
    data.reverse()
    def avg(data):
        alpha = 2 / (len(data) + 1)
        if len(data) == 1:
            return data[0]
        else:
            curr = data[0]
            print(curr)
            data.pop(0)
            return alpha * curr + (1 - alpha) * avg(data)
    return avg(data)

def rect_to_coor(rect):
    x1 = rect.left()        # These assignments grab the coordinates of the top left and bottom right points of the rectangle[] object
    y1 = rect.top()
    x2 = rect.right()
    y2 = rect.bottom()
    return (x1, y1), (x2, y2)

def detect(frame, mark=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    landmarks = np.empty((0, 2), dtype=np.float32)      # landmarks will default to Empty if the detector cannot detect a face, in which case the for-loop below wouldn't run
    img_points = np.empty((0, 2), dtype=np.float32)
    for face in faces:      # 'faces' is an iterable of human faces (so with more than one face it will have more than one element)
        # Boxing out the faces
        if mark:
            TL, BR = rect_to_coor(face)
            cv2.rectangle(frame, TL, BR, (0, 255, 0), 3)        # From these two points, we can draw a rectangle
        
        # Calculating landmarks
        p = predictor(gray, face)
        for i in [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]:       # range(68)   [33, 8, 45, 36, 54, 48]
            x = p.part(i).x
            y = p.part(i).y
            img_points = np.append(img_points, np.array([[x, y]]), axis=0)
            if mark:
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        for i in range(68):       # range(68)   [33, 8, 45, 36, 54, 48]
            x = p.part(i).x
            y = p.part(i).y
            landmarks = np.append(landmarks, np.array([[x, y]]), axis=0)
            if i in range(17, 22) or i in range(22, 27):
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    return frame, landmarks, img_points

def rescale_frame(frame, percent=100):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


#############################
#          MAIN          #
#############################
if __name__ == '__main__':
    main()
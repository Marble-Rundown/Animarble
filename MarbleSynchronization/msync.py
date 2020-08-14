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
            self.resting_tilt = 32
            self.resting_lip_distance = 0
            self.chatter_lip_distance = 10
            self.resting_eyebrow_elevation = 43
            self.tilt_weights = {
                "measured_tilt_degrees": 1,
                "lip_spacing": 3,
                "eyebrow_elevation": -5,
                "chatter": 0
            }

def ewma(feature_data, exceptions):
    def avg(data):
        alpha = 2 / (len(data) + 1)
        if len(data) == 1:
            return data[-1]
        else:
            curr = data[-1]
            prev = avg(data[:-1])
            return {key: (alpha * curr[key] + (1-alpha) * prev[key] if key not in exceptions else curr[key]) for key in curr.keys()}
    return avg(feature_data)


sliding_window = []
WINDOW_LENGTH = 6
def get_tilt_features(profile, head_rotation, head_translation, landmarks):
    features = {}
    features["measured_tilt_degrees"] = float(head_rotation[0] - profile.resting_tilt)
    features["lip_spacing"] = float(landmarks[66][1] - landmarks[62][1] - profile.resting_lip_distance)
    features["eyebrow_elevation"] = float(landmarks[27][1] - (landmarks[19][1] + landmarks[24][1]) / 2 - profile.resting_eyebrow_elevation)
    features["chatter"] = np.random.normal() if features["lip_spacing"] > profile.chatter_lip_distance else 0

    sliding_window.append(features)
    if len(sliding_window) > WINDOW_LENGTH:
        sliding_window.pop(0)
    filtered_features = ewma(sliding_window, set("eyebrow_elevation"))
    return filtered_features

def get_tilt_weights(profile):
    return profile.tilt_weights

def dot_product(features, weights):
    return sum([feature_val * weights[feature_label] for feature_label, feature_val in features.items()])

def log_features_weights(features, weights):
    description = ""
    for feature_label, feature_val in features.items():
      description += f"{feature_label}: {feature_val:.3f} * {weights[feature_label]} = {feature_val * weights[feature_label]:.3f} | "
    description = f"Final Angle: {dot_product(features, weights):.3f}\n" + description
    print(description)


#############################
#       Initialization      #
#############################

# Numpy random seeding
np.random.seed(5327)

# Computer Vision Initialization
detector = dlib.get_frontal_face_detector()     # Get a face detector from the dlib library
predictor = dlib.shape_predictor('assets/shape_predictor_68_face_landmarks.dat')      # Get a shape predictor based on a trained model from the dlib library

def check_str(value):
    if type(value) != str:
        raise argparse.ArgumentTypeError(f"'{value}' is not a string")
    return value
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
                
                   
                    tilt_weights = get_tilt_weights(profile)

                    tilt_features = get_tilt_features(profile, rotation, translation, landmarks)
                    
                    log_features_weights(tilt_features, tilt_weights)

                    tilt = dot_product(tilt_features, tilt_weights)
                    pan = 0

                    file.write('{0},{1},{2}\n'.format(int(cap.get(cv2.CAP_PROP_POS_MSEC)), tilt, pan))

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

def create_file(file_name, file_type, n=0):
    destination = './outputs/{0}{1}.{2}'.format(file_name, f' ({n})' if n != 0 else '', file_type)
    if not os.path.isfile(destination):
        return open(destination, 'w+')
    else:
        return create_file(file_name, file_type, n+1)

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
        for i in [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]: 
            x = p.part(i).x
            y = p.part(i).y
            img_points = np.append(img_points, np.array([[x, y]]), axis=0)
            if mark:
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        for i in range(68):
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

import dlib, pygame, math
import cv2
import argparse
import numpy as np
import os
from pose_estimator import PoseEstimator
#import marble_renderer
from threading import Thread
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


#############################
#         Constants         #
#############################
WINDOW_TITLE = 'Facial Landmark Detector'
IMAGE_RESCALE = 75
VIDEO_RESCALE = 50

#LOW_PASS = 20

WINDOW_LENGTH = 3
K_SPEECH = 1.0
#K_EYEBROW = 1.0

#tilt_weight = {'speech_exaggeration_multiplier': 1.0,
#            'resting_lip_spacing': 20
#    }
#tilt

#pan_template





#############################
#       Initialization      #
#############################
np.random.seed(5327)

# Computer Vision Initialization
detector = dlib.get_frontal_face_detector()     # Get a face detector from the dlib library
predictor = dlib.shape_predictor('assets/shape_predictor_68_face_landmarks.dat')      # Get a shape predictor based on a trained model from the dlib library
sliding_window = []

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
d_x = 0
d_y = 0

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


#############################
#           Main            #
#############################
def main():
    rotation, translation = (), ()

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
            moving_average = []
            n = 6

            calibration = []
            calibrated = False
            c_n = 30
            rotation_offset = (0, 0, 0)

            profile = Profile("Jeffrey")

            while cap.isOpened():           # Main while loop
                for event in pygame.event.get():        # Pygame
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                
                s, frame = cap.read()
                assert s, 'Failed to read next frame'
                frame, landmarks, img_points = detect(frame, mark=True)
                frame = rescale_frame(frame, percent=rescale)

                #print(len(img_points))

                if img_points.size != 0:
                    rotation, translation = pe.estimate_pose(img_points)

                    if not calibrated:
                        calibration.append(rotation)
                        #print(calibration)
                        if len(calibration) > c_n:
                            averages = [sum([rot[i] for rot in calibration]) / c_n for i in range(3)]
                            rotation_offset = tuple(-avg for avg in averages)
                            calibrated = True
                    
                    #print(img_points)
                    #if len(rotation) != 0:
                    #moving_average.append(rotation)
                    #if len(moving_average) > n:
                    #    moving_average.pop(0)
                    ##rotation = [sum([rot[i] for rot in moving_average]) / n for i in range(3)] 
                    #rotation = [ewma([rot[i] for rot in moving_average]) for i in range(3)]
                    #if landmarks.size != 0:
                    #    rotation[0] += -K_SPEECH * (landmarks[51][1] - landmarks[57][1] + 20)        # nodding
                    #print(landmarks[51][1] - landmarks[57][1])
                    #print(landmarks[24][1] - landmarks[44][1])

                    #g_noise = tuple(numpy.random.normal(scale=3.0) for _ in range(3)) if abs(landmarks[51][1] - landmarks[57][1] + 20) > 5 else (0, 0, 0)

                    #converted_rotation = tuple(jbr(rot) for rot in rotation)
                    converted_rotation = tuple(rotation[i] + rotation_offset[i] for i in range(3)) 

                    tilt_weights = get_tilt_weights(profile)

                    tilt_features = get_tilt_features(profile, converted_rotation, translation, landmarks)
                    
                    log_features_weights(tilt_features, tilt_weights)

                    tilt = dot_product(tilt_features, tilt_weights)
                    pan = 0

                    #rot_frame.append((converted_rotation[0], converted_rotation[1]))
                    #if len(rot_frame) > 2:
                    #    rot_frame.pop(0)
                    #    d_x, d_y = rot_frame[0][0] - rot_frame[1][0], rot_frame[1][1] - rot_frame[0][1]
                    #    if abs(d_x) > LOW_PASS or abs(d_y) > LOW_PASS:
                    #        converted_rotation = rot_frame[0]
                            #rot_frame[1] = rot_frame[0]
                                

                    #print('timestamp:{3}\npitch:{0}\nyaw:{1}\nroll{2}\n'.format(*converted_rotation, cap.get(cv2.CAP_PROP_POS_MSEC)))
                    #print(rotation_offset)
                    #file.write('{0},{1},{2}\n'.format(int(cap.get(cv2.CAP_PROP_POS_MSEC)), converted_rotation[0][0], converted_rotation[1][0]))
                    file.write('{0},{1},{2}\n'.format(int(cap.get(cv2.CAP_PROP_POS_MSEC)), tilt, pan))
                    #rotate_marble(pan, tilt)
                    rotate_marble(tilt + rotation_offset[0], pan + rotation_offset[1])
                else:
                    print('Failed to find image points')

                cv2.imshow(WINDOW_TITLE, frame)
                if cv2.waitKey(1) & 0xFF == 27:     # 27 is ASCII for the Esc key on a keyboard
                    break

                pygame.display.flip()
                pygame.time.wait(10)
    cap.release()
    cv2.destroyAllWindows()


#############################
#          Classes          #
#############################
class Profile:
    def __init__(self, name):
        if name == 'Jaiveer':
            self.resting_tilt = 32
            self.resting_lip_distance = 0
            self.chatter_lip_distance = 5
            self.resting_eyebrow_elevation = 43
            self.weights = {
                'measured_tilt_degrees': 1,
                'lip_spacing': 3,
                'eyebrow_elevation': -5, 
                'chatter': 0
            }
        elif name == 'Jeffrey':
            self.resting_tilt = 20
            self.resting_lip_distance = 20
            self.chatter_lip_distance = 5
            self.resting_eyebrow_elevation = 32
            self.weights = {
                'measured_tilt_degrees': 1,
                'lip_spacing': 0,
                'eyebrow_elevation': 0,
                'chatter': 0
            }


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

def get_tilt_features(profile, head_rotation, head_translation, landmarks):
    #NOSE_LENGTH = distance(landmarks[])
    features = {
        "measured_tilt_degrees": float(head_rotation[0]),
        "lip_spacing": float(landmarks[51][1] - landmarks[57][1] - profile.resting_lip_distance),
        "eyebrow_elevation": float(landmarks[27][1] - (landmarks[19][1] + landmarks[24][1]) / 2 - profile.resting_eyebrow_elevation)
    }

    features['chatter'] = np.random.normal() if features['lip_spacing'] > profile.chatter_lip_distance else 0

    sliding_window.append(features)
    if len(sliding_window) > WINDOW_LENGTH:
        sliding_window.pop(0)
    filtered_features = ewma(sliding_window, set("eyebrow_elevation"))
    return filtered_features

def get_tilt_weights(profile):
    return profile.weights

def dot_product(features, weights):
    return sum([feature_val * weights[feature_label] for feature_label, feature_val in features.items()])

def log_features_weights(features, weights):
    description = ""
    for feature_label, feature_val in features.items():
      description += f"{feature_label}: {feature_val:.3f} * {weights[feature_label]} = {feature_val * weights[feature_label]:.3f} | "
    description = f"Final Angle: {dot_product(features, weights):.3f}\n" + description
    print(description)

#def ewma(data):
#    data.reverse()
#    def avg(data):
#        alpha = 2 / (len(data) + 1)
#        if len(data) == 1:
#            return data[0]
#        else:
#            curr = data[0]
#            #print(curr)
#            data.pop(0)
#            return alpha * curr + (1 - alpha) * avg(data)
#    return avg(data)

def rect_to_coor(rect):
    x1 = rect.left()        # These assignments grab the coordinates of the top left and bottom right points of the rectangle[] object
    y1 = rect.top()
    x2 = rect.right()
    y2 = rect.bottom()
    return (x1, y1), (x2, y2)

def distance(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def detect(frame, mark=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    #landmarks, img_points = 0, 0
    landmarks = np.empty((0, 2), dtype=np.float32)      # landmarks and img_points will default to Empty if the detector cannot detect a face, in which case the for-loops below wouldn't run
    img_points = np.empty((0, 2), dtype=np.float32)
    
    face = faces[0] if faces else None    # Only operate on the first face detected. If you want to do multiple faces, 'landmarks' and 'img_points' would have to be LISTS of the landmarks and image points for each face!

    if face:
        # Boxing out the faces
        if mark:
            TL, BR = rect_to_coor(face)
            cv2.rectangle(frame, TL, BR, (0, 255, 0), 3)        # From these two points, we can draw a rectanngle

        # Calculating landmarks
        p = predictor(gray, face)
        for i in [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]:       # range(68)   [33, 8, 45, 36, 54, 48]
            x = p.part(i).x
            y = p.part(i).y
            image_points = np.append(img_points, np.array([[x, y]]), axis=0)
            img_points = image_points
            print(len(img_points))
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

def rotate_marble(tilt, pan):
    glPushMatrix()
    glRotatef(tilt, 1, 0, 0)
    glRotatef(pan, 0, -1, 0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_sphere()
    draw_cylinder()
    glPopMatrix()


#############################
#          Special          #
#############################
#__name__ is a special Python variable
#1. If you run THIS script using Gitbash: the __name__ variable within this script equals '__main__'
#2. If you create another script import_script.py that IMPORTS THIS SCRIPT using 'import msync':the __name__ variable within import_script.py equals 'msync'
if __name__ == '__main__':
    main()
import dlib
import pygame
import math
import cv2
import argparse
import numpy as np
from pose_estimator import PoseEstimator
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


#############################
#         Constants         #
#############################
WINDOW_TITLE = 'Facial Landmark Detector'
IMAGE_RESCALE = 75
VIDEO_RESCALE = 50

# Lip/Speech Feature
SPEECH_FILTER_LENGTH = 5
K_SPEECH = 1.5
SPEECH_OFFSET = 20
SPEECH_THRESHOLD = 2

# Gaussian Noise in pan and tilt while talking
NOISE_PERIOD = 5
NOISE_INTENSITY = (4.0, 3.0, 0.0)

MOVING_AVERAGE_LENGTH = 3

CALIBRATION_LENGTH = 10


#############################
#       Initialization      #
#############################
np.random.seed(5327)

# Computer Vision Initialization
# Get a face detector from the dlib library
detector = dlib.get_frontal_face_detector()
# Get a shape predictor based on a trained model from the dlib library
predictor = dlib.shape_predictor(
    'assets/shape_predictor_68_face_landmarks.dat')
sliding_window = []


def check_str(value):
    if type(value) != str:
        raise argparse.ArgumentTypeError(f"'{value}' is not a string")
    return value


# Create an instance of an ArgumentParser object
ap = argparse.ArgumentParser()
# Adds an argument '--image' that describes the image file path
ap.add_argument('-i', '--image', type=check_str, help='The path to the image')
# Adds an argument '--video' that describes the video file path
ap.add_argument('-v', '--video', type=check_str, help='The path to the video')
# Adds an argument '--stream' that determines whether to use the webcam
ap.add_argument('-s', '--stream',
                help='Set this to True to start in livestream mode')
# ap.add_argument('-m', '--marble', required=True, help='The output filename to write to')
# Vars() returns the __dict__ attribute of an object, so args is a dictionary of the command line parameters passed to this program
args = vars(ap.parse_args())
# num_args = len([a for a in args.values() if a])
# if num_args != 2:
#     raise TypeError(f'Expected 2 arguments, but received {num_args} argument(s)')
filename = "unnamed"
# if marble not in ['bob', 'dan', 'drew']:
#     raise TypeError(f"'{marble}' is not a valid marble")

file_type = ''      # Initialize variables based on target type
target = None
rescale = 100
dimensions = ()
video_length = 0
if args['image']:
    file_type = 'image'
    image = cv2.imread('media/' + args['image'])
    dimensions = tuple(image.shape[i] * IMAGE_RESCALE / 100 for i in range(2))
    rescale = IMAGE_RESCALE
elif args['video']:
    filename = os.path.splitext(os.path.basename(args['video']))[0]
    print(filename)
    file_type = 'video'
    cap = cv2.VideoCapture(args['video'])
    assert cap.isOpened(), 'Failed to open video file'
    _, first_frame = cap.read()
    dimensions = tuple(
        first_frame.shape[i] * VIDEO_RESCALE / 100 for i in range(2))
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    video_length = cap.get(cv2.CAP_PROP_POS_MSEC)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
    rescale = VIDEO_RESCALE
elif args['stream']:
    file_type = 'stream'
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), 'Failed to open stream'
    _, first_frame = cap.read()
    dimensions = tuple(first_frame.shape[i] for i in range(2))

pe = PoseEstimator(dimensions)


# Pygame and OpenGL Initializationß
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

    with create_file(filename, 'csv') as file:
        file.write('timestamp,tilt,pan,tilt_setpoint,pan_setpoint\n')

        if file_type == 'image':
            frame, landmarks, img_points = detect(image, mark=True)
            frame = rescale_frame(frame, percent=rescale)

            if img_points.size != 0:
                rotation, translation = pe.estimate_pose(img_points)

            # Shows the image in a new window
            cv2.imshow(WINDOW_TITLE, frame)
            cv2.waitKey(0)
        else:
            calibration = []
            lipCalibration = []
            calibrated = False
            CALIBRATION_COUNT = 30
            rotation_offset = (0, 0, 0)

            profile = Profile("Jaiveer")

            while cap.isOpened():           # Main while loop
                for event in pygame.event.get():        # Pygame
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()

                s, frame = cap.read()
                if not(s):
                    print("exiting!")
                    break

                frame, landmarks, img_points = detect(frame, mark=True)
                frame = rescale_frame(frame, percent=rescale)

                if img_points.size != 0:
                    rotation, translation = pe.estimate_pose(img_points)

                    if not calibrated:
                        calibration.append(rotation)
                        if len(calibration) > CALIBRATION_COUNT:
                            print([rot[0] for rot in calibration])
                            averages = [
                                sum([rot[i] for rot in calibration]) / CALIBRATION_COUNT for i in range(3)]
                            rotation_offset = tuple(avg for avg in averages)
                            calibrated = True

                            print(
                                f"Calibration complete! Offsets: ({rotation_offset[0]}, {rotation_offset[1]}, {rotation_offset[2]})")
                    else:
                        converted_rotation = tuple(
                            rotation[i] - rotation_offset[i] for i in range(3))

                        tilt_weights = get_tilt_weights(profile)

                        tilt_features = get_tilt_features(
                            profile, converted_rotation, translation, landmarks)

                        log_features_weights(tilt_features, tilt_weights)

                        tilt = dot_product(tilt_features, tilt_weights)
                        pan = 0

                        tilt = converted_rotation[0]

                        file.write('{0},{1},{2}\n'.format(
                            int(cap.get(cv2.CAP_PROP_POS_MSEC)), tilt, pan))
                        rotate_marble(tilt, pan)
                else:
                    print('Failed to find image points')

                cv2.imshow(WINDOW_TITLE, frame)
                # 27 is ASCII for the Esc key on a keyboard
                if cv2.waitKey(1) & 0xFF == 27:
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
            self.lip_spacing_to_nose_width_ratio = 0.5
            self.chatter_lip_distance = 5
            self.resting_eyebrow_elevation = 43
            self.weights = {
                'measured_tilt_degrees': 1,
                'lip_spacing': -3,
                'eyebrow_elevation': 0,
                'chatter': 0
            }
        elif name == 'Jeffrey':
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

def create_file(file_name, file_type, n=0):
    # Add:    if n != 0 else ''    after f' ({n})' if you don't want the first file to have a number
    destination = './outputs/{0}{1}.{2}'.format(
        file_name, f' ({n})', file_type)
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
    nose_width = distance(landmarks[31], landmarks[35])
    lip_spacing = distance(landmarks[51], landmarks[57])
    print(f"nose width {nose_width} lip spacing {lip_spacing}")
    features = {
        "measured_tilt_degrees": float(head_rotation[0]),
        "lip_spacing": lip_spacing - nose_width * profile.lip_spacing_to_nose_width_ratio,
        "eyebrow_elevation": float(landmarks[27][1] - (landmarks[19][1] + landmarks[24][1]) / 2 - profile.resting_eyebrow_elevation)
    }

    features['chatter'] = np.random.normal(
    ) if features['lip_spacing'] > profile.chatter_lip_distance else 0

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
    description = f"Final Angle: {dot_product(features, weights):.3f}\n" + \
        description
    print(description)


def rect_to_coor(rect):
    # These assignments grab the coordinates of the top left and bottom right points of the rectangle[] object
    x1 = rect.left()
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
    # landmarks and img_points will default to Empty if the detector cannot detect a face, in which case the for-loops below wouldn't run
    landmarks = np.empty((0, 2), dtype=np.float32)
    img_points = np.empty((0, 2), dtype=np.float32)

    # Only operate on the first face detected. If you want to do multiple faces, 'landmarks' and 'img_points' would have to be LISTS of the landmarks and image points for each face!
    face = faces[0] if faces else None

    if face:
        # Boxing out the faces
        if mark:
            TL, BR = rect_to_coor(face)
            # From these two points, we can draw a rectanngle
            cv2.rectangle(frame, TL, BR, (0, 255, 0), 3)

        # Calculating landmarks
        p = predictor(gray, face)
        # range(68)   [33, 8, 45, 36, 54, 48]
        for i in [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]:
            x = p.part(i).x
            y = p.part(i).y
            img_points = np.append(img_points, np.array([[x, y]]), axis=0)
            if mark:
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        for i in range(68):       # range(68)   [33, 8, 45, 36, 54, 48]
            x = p.part(i).x
            y = p.part(i).y
            landmarks = np.append(landmarks, np.array([[x, y]]), axis=0)
            if True or i in range(17, 22) or i in range(22, 27):
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    return frame, landmarks, img_points


def rescale_frame(frame, percent=100):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
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
if __name__ == '__main__':
    main()


import os, argparse, dlib, cv2, pygame, math
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
K_SPEECH = 2.0
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
# ap.add_argument('-m', '--marble', required=True, help='The output filename to write to')
args = vars(ap.parse_args())        # Vars() returns the __dict__ attribute of an object, so args is a dictionary of the command line parameters passed to this program
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

    with create_file(filename, 'csv') as file:
        file.write('timestamp,tilt,pan,tilt_setpoint,pan_setpoint\n')

        if file_type == 'image':
            frame, landmarks, img_points = detect(image, mark=True)
            frame = rescale_frame(frame, percent=rescale)
        
            if img_points.size != 0:
                rotation, translation = pe.estimate_pose(img_points)
        
            cv2.imshow(WINDOW_TITLE, frame)     # Shows the image in a new window
            cv2.waitKey(0)
        else:
            moving_average = []
            speech_filter = [0] * SPEECH_FILTER_LENGTH

            calibration = []
            lipCalibration = []
            calibrated = False
            rotation_offset = (0, 0, 0)

            g_noise = (0, 0, 0)
            count = 0

            profile = Profile("Jeffrey")

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

                #print(len(img_points))

                if img_points.size != 0:
                    rotation, translation = pe.estimate_pose(img_points)

                    if not calibrated:
                        calibration.append(rotation)
                        lipCalibration.append(get_lip_dist(landmarks))
                        #print(calibration)
                        if len(calibration) > CALIBRATION_LENGTH:
                            averages = [sum([rot[i] for rot in calibration]) / CALIBRATION_LENGTH for i in range(3)]
                            rotation_offset = tuple(-avg for avg in averages)
                            SPEECH_OFFSET = np.mean(lipCalibration)
                            calibrated = True
                            print(f"Speech offset: {SPEECH_OFFSET}")
                            print("done calibrating")
                    else:
                        moving_average.append(rotation)
                        if len(moving_average) > MOVING_AVERAGE_LENGTH:
                            moving_average.pop(0)

                        rotation = [ewma([rot[i] for rot in moving_average]) for i in range(3)]
                        if landmarks.size != 0:
                            mouth_size = get_lip_dist(landmarks) - SPEECH_OFFSET
                            mouth_size = mouth_size if abs(mouth_size) > SPEECH_THRESHOLD else 0
                            print(f"Mouth size: {mouth_size}")
                            speech_filter.append(mouth_size)
                            speech_filter.pop(0)
                            rotation[0] += K_SPEECH * median(speech_filter)        # nodding

                            if count % NOISE_PERIOD == 0:
                                g_noise = tuple(np.random.normal(scale=NOISE_INTENSITY[i]) for i in range(3)) if False and mouth_size > SPEECH_THRESHOLD else (0, 0, 0)
                                # print("Adding noise")
                        count += 1

                        converted_rotation = tuple(rotation[i] + rotation_offset[i] + g_noise[i] * ((count % NOISE_PERIOD) + 1) / NOISE_PERIOD for i in range(3))
                        tilt, pan = converted_rotation[0], converted_rotation[1]

                        # print('timestamp:{3}\npitch:{0}\nyaw:{1}\nroll{2}\n'.format(*converted_rotation, cap.get(cv2.CAP_PROP_POS_MSEC)))
                        file.write('{0},{1},{2},0\n'.format(int(cap.get(cv2.CAP_PROP_POS_MSEC)), jbr(tilt), jbr(pan)))
                        
                        rotate_marble(round(float(tilt)), round(float(pan)))
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
    destination = './outputs/{0}{1}.{2}'.format(file_name, f' ({n})', file_type)       # Add:    if n != 0 else ''    after f' ({n})' if you don't want the first file to have a number
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
            #print(curr)
            data.pop(0)
            return alpha * curr + (1 - alpha) * avg(data)
    return avg(data)

def median(data):
    m = np.median(data) 
    # print(f"Max: {max(data)} Min: {min(data)} Median: {m}")
    return m

def get_lip_dist(landmarks):
    return float(distance(landmarks[52], landmarks[58]))

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
if __name__ == '__main__':
    main()
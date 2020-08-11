
import dlib
import cv2
import argparse
import numpy as np
import os
from pose_estimator import PoseEstimator

#############################
#         Constants         #
#############################
WINDOW_TITLE = 'Facial Landmark Detector'
IMAGE_RESCALE = 75
VIDEO_RESCALE = 50


#############################
#       Initialization      #
#############################
detector = dlib.get_frontal_face_detector()     # Get a face detector from the dlib library
predictor = dlib.shape_predictor('assets/shape_predictor_68_face_landmarks.dat')      # Get a shape predictor based on a trained model from the dlib library

def check_str(value):
    if type(value) != str:
        raise argparse.ArgumentTypeError(f"'{value}' is not a string")
    return value;
ap = argparse.ArgumentParser()     
ap.add_argument('-i', '--image', type=check_str, help='The path to the image')     
ap.add_argument('-v', '--video', type=check_str, help='The path to the video')   
ap.add_argument('-s', '--stream', help='Set this to True to start in livestream mode')     
args = vars(ap.parse_args()) 
if not any(args.values()):
    raise TypeError('Expected 1 argument, but received 0 arguments')
num_args = len([a for a in args.values() if a])
if num_args > 1:
    raise TypeError('Expected only 1 argument, but received {0} arguments'.format(num_args))

# Initialize variables based on target type
file_type = ''
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


#############################
#           Main            #
#############################
def main():
    rotation, translation = (), ()
    file = create_file('rotations', 'csv')
    file.write('timestamp,pitch,yaw\n')

    if file_type == 'image':
        frame, landmarks = detect(image, mark=True)
        frame = rescale_frame(frame, percent=rescale)
        
        if landmarks.size != 0:
            rotation, translation = pe.estimate_pose(landmarks)
        
        cv2.imshow(WINDOW_TITLE, frame)   
        cv2.waitKey(0)
    else:
        moving_average = []
        n = 10
        while cap.isOpened():
            s, frame = cap.read()
            if s:
                frame, landmarks = detect(frame, mark=True)
                frame = rescale_frame(frame, percent=rescale)

                if landmarks.size != 0:
                    rotation, translation = pe.estimate_pose(landmarks)
                
                moving_average.append(rotation)
                if len(moving_average) > n:
                    moving_average.pop(0)
                rotation = tuple(sum([rot[i] for rot in moving_average]) / n for i in range(3))
                
                rotation = tuple(jbr(rot) for rot in rotation)
                print('timestamp:{3}\npitch:{0}\nyaw:{1}\nroll{2}\n'.format(*rotation, cap.get(cv2.CAP_PROP_POS_MSEC)))
                file.write('{0},{1},{2}\n'.format(int(cap.get(cv2.CAP_PROP_POS_MSEC)), rotation[0][0], rotation[1][0]))
                #print(landmarks)

                cv2.imshow(WINDOW_TITLE, frame)
                if cv2.waitKey(1) & 0xFF == 27:     # 27 is ASCII for the Esc key on a keyboard
                    break
            else:
                break
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

def rect_to_coor(rect):
    x1 = rect.left()        # These assignments grab the coordinates of the top left and bottom right points of the rectangle[] object
    y1 = rect.top()
    x2 = rect.right()
    y2 = rect.bottom()
    return (x1, y1), (x2, y2)

def detect(frame, mark=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    landmarks = np.empty((0, 2), dtype=np.float32)        # landmarks will default to Empty if the detector cannot detect a face, in which case the for-loop below wouldn't run
    for face in faces:
        # Boxing out the faces
        if mark:
            TL, BR = rect_to_coor(face)
            cv2.rectangle(frame, TL, BR, (0, 255, 0), 3)       
        
        # Calculating landmarks
        p = predictor(gray, face)
        for i in [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]:   
            x = p.part(i).x
            y = p.part(i).y
            landmarks = np.append(landmarks, np.array([[x, y]]), axis=0)
            if mark:
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
    return frame, landmarks

def rescale_frame(frame, percent=100):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


#############################
#          Special          #
#############################
if __name__ == '__main__':
    main()
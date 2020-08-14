import cv2
import numpy as np
import math

class PoseEstimator:
    def __init__ (self, img_size=(1920, 1080)):
        self.size = img_size
        # self.obj_points = self._get_obj_points('assets/object_points.txt')
        # self.obj_points = np.array([
        #    (0.0, -10.0, -50.0),         # Nose base
        #    (0.0, 0.0, 0.0),             # Nose tip
        #    (0.0, -330.0, -65.0),        # Chin
        #    (-225.0, 170.0, -135.0),     # Left corner of the left eye
        #    (225.0, 170.0, -135.0),      # Right corner of the right eye
        #    (-150.0, -150.0, -125.0),    # Mouth left corner
        #    (150.0, -150.0, -125.0)      # Mouth right corner
        # ])

        self.obj_points = np.float32([[6.825897, 6.760612, 4.402142],
                                     [1.330353, 7.122144, 6.903745],
                                     [-1.330353, 7.122144, 6.903745],
                                     [-6.825897, 6.760612, 4.402142],
                                     [5.311432, 5.485328, 3.987654],
                                     [1.789930, 5.393625, 4.413414],
                                     [-1.789930, 5.393625, 4.413414],
                                     [-5.311432, 5.485328, 3.987654],
                                     [2.005628, 1.409845, 6.165652],
                                     [-2.005628, 1.409845, 6.165652],
                                     [2.774015, -2.080775, 5.048531],
                                     [-2.774015, -2.080775, 5.048531],
                                     [0.000000, -3.116408, 6.097667],
                                     [0.000000, -7.415691, 4.070434]])

        # self.focal_length = self.size[1]
        camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.size[0], 0, camera_center[0]],          # POTENTIAL ERROR!!!
             [0, self.size[1], camera_center[1]],
             [0, 0, 1]], dtype="double")

        self.dist_coeffs = np.zeros((4, 1))

    def _get_obj_points(self, file_path):         # Use a _ before the function name to indicate it is for internal use only
       contents = []
       with open(file_path) as file:
           for line in file:
               contents.append(line)
       obj_points = np.array(contents, dtype=np.float32)
       obj_points = np.reshape(obj_points, (3, -1)).T      # 

       # Transform the model into a front view.
       obj_points[:, 2] *= -1

       return obj_points

    def estimate_pose(self, img_points):
        #img_points = np.array([img_points])
        #print(self.obj_points)
        #print(img_points)
        assert self.obj_points.shape[0] == img_points.shape[0], 'Image points incompatible with object points'
        #_, camera_matrix, dist, rotation, translation = cv2.calibrateCamera(self.obj_points, img_points, self.size, None, None)
        #print(f'{camera_matrix}\n{dist}\n{rotation}\n{translation}')
        _, rotation, translation = cv2.solvePnP(self.obj_points, img_points, self.camera_matrix, None)
        return rotation / math.pi * 180, translation
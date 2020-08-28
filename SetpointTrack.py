import numpy as np
from scipy.interpolate import interp1d


class SetpointTrack:

    class ControlPoint:
        def __init__(self, time_index, angle):
            self.time_index = time_index
            self.angle = angle

        def __str__(self):
            return f"Index: {self.time_index}\tAngle: {self.angle}"

    def __init__(self, resampled_data):
        self.resampled_data = resampled_data
        self.control_points = {}
        self.next_control_point_key = 0

        self.add_control_point(0, resampled_data[0])
        self.add_control_point(len(resampled_data) - 1, resampled_data[-1])

    def add_control_point(self, time_index, angle):
        control_point = self.ControlPoint(time_index, angle)
        control_point_key = self.next_control_point_key
        self.control_points[control_point_key] = control_point

        self.next_control_point_key += 1

        return control_point_key

    def edit_control_point(self, control_point_key, time_index, angle):
        control_point = self.control_points[control_point_key]
        control_point.time_index = time_index
        control_point.angle = angle

    def delete_control_point(self, control_point_key):
        del self.control_points[control_point_key]

    def apply_control_points(self):
        sorted_control_points = sorted(
            self.control_points.values(), key=lambda cp: cp.time_index)

        control_point_time_indices = np.array(
            [cp.time_index for cp in sorted_control_points])
        control_point_angles = np.array(
            [cp.angle for cp in sorted_control_points])

        interp_angles = interp1d(
            control_point_time_indices, control_point_angles)

        return interp_angles(np.linspace(
            0, len(self.resampled_data) - 1, len(self.resampled_data), endpoint=True))

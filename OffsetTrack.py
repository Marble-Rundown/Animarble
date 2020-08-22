import numpy as np


class OffsetTrack:

    class RegionModifier:
        def __init__(self, start_index, end_index, multiplier):
            self.start_index = start_index
            self.end_index = end_index
            self.multiplier = multiplier

    def __init__(self, resampled_data):
        self.resampled_data = resampled_data
        self.region_modifiers = {}
        self.next_region_key = 0

    def add_region_modifier(self, start_index, end_index, multiplier):
        rm = self.RegionModifier(start_index, end_index, multiplier)
        key = self.next_region_key
        self.region_modifiers[key] = rm
        self.next_region_key += 1
        return key

    def edit_region_modifier(self, region_key, start_index, end_index, multiplier):
        rm = self.region_modifiers[region_key]
        rm.start_index = start_index
        rm.end_index = end_index
        rm.multiplier = multiplier

    def delete_region_modifier(self, region_key):
        self.region_modifiers.pop(region_key)

    def apply_modifiers(self):
        copy_resampled_data = np.copy(self.resampled_data)
        for key in self.region_modifiers:
            rm = self.region_modifiers[key]
            for i in range(rm.start_index, rm.end_index):
                copy_resampled_data[i] *= rm.multiplier
        return copy_resampled_data

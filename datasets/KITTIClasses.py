from collections import namedtuple

import numpy as np


KittiClass = namedtuple("KittiClass", ["label", "color"])

KITTI_CLASSES = {
    0: KittiClass("unlabeled", [0, 0, 000]),
    1: KittiClass("outlier", [0, 0, 255]),
    10: KittiClass("car", [245, 150, 100]),
    11: KittiClass("bicycle", [245, 230, 100]),
    13: KittiClass("bus", [250, 80, 100]),
    15: KittiClass("motorcycle", [150, 60, 30]),
    16: KittiClass("on-rails", [255, 0, 0]),
    18: KittiClass("truck", [180, 30, 80]),
    20: KittiClass("other vehicle", [255, 0, 0]),
    30: KittiClass("person", [30, 30, 255]),
    31: KittiClass("bicyclist", [200, 40, 255]),
    32: KittiClass("motorcyclist", [90, 30, 150]),
    40: KittiClass("road", [255, 0, 255]),
    44: KittiClass("parking", [255, 150, 255]),
    48: KittiClass("sidewalk", [75, 0, 75]),
    49: KittiClass("other ground", [75, 0, 175]),
    50: KittiClass("building", [0, 200, 255]),
    51: KittiClass("fence", [50, 120, 255]),
    52: KittiClass("other structure", [0, 150, 255]),
    60: KittiClass("lane marking", [170, 255, 150]),
    70: KittiClass("vegetation", [0, 175, 0]),
    71: KittiClass("trunk", [0, 60, 135]),
    72: KittiClass("terrain", [80, 240, 150]),
    80: KittiClass("pole", [150, 240, 255]),
    81: KittiClass("traffic sign", [0, 0, 255]),
    99: KittiClass("other object", [255, 255, 50]),
    252: KittiClass("moving car", [245, 150, 100]),
    253: KittiClass("moving bicyclist", [255, 0, 0]),
    254: KittiClass("moving person", [200, 40, 255]),
    255: KittiClass("moving motorcyclist", [30, 30, 255]),
    256: KittiClass("moving on-rails", [90, 30, 150]),
    257: KittiClass("moving bus", [250, 80, 100]),
    258: KittiClass("moving truck", [180, 30, 80]),
    259: KittiClass("moving other vehicle", [255, 0, 0])
}

KITTI_COLORS = np.zeros((max(KITTI_CLASSES.keys()) + 1, 3), dtype=np.float)
for k, v in KITTI_CLASSES.items():
    KITTI_COLORS[k] = np.array(v.color)

KITTI_COLORS /= 255
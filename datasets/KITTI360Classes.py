from collections import namedtuple

import numpy as np

KITTI360_CATEGORIES = {
    0: "void",
    1: "flat",
    2: "construction",
    3: "object",
    4: "nature",
    5: "sky",
    6: "human",
    7: "vehicle"
}

KITTI360Class = namedtuple( "KITTI360Class", [
    'name',
    'categoryId',
    'hasInstances',
    'ignoreInEval',
    'color',
])

KITTI360_CLASSES = {
    #    |                                     |cat| has |Ignore|                 |
    # ID |                                name |ID |Inst | Eval |  color          |
    # ---+-------------------------------------+--+------+------+-----------------+
        0: KITTI360Class(           "unlabeled", 0, False,  True, (  0,   0,   0)),
        1: KITTI360Class(         "ego vehicle", 0, False,  True, (  0,   0,   0)),
        2: KITTI360Class("rectification border", 0, False,  True, (  0,   0,   0)),
        3: KITTI360Class(          "out of roi", 0, False,  True, (  0,   0,   0)),
        4: KITTI360Class(              "static", 0, False,  True, (  0,   0,   0)),
        5: KITTI360Class(             "dynamic", 0, False,  True, (111,  74,   0)),
        6: KITTI360Class(              "ground", 0, False,  True, ( 81,   0,  81)),
        7: KITTI360Class(                "road", 1, False, False, (128,  64, 128)),
        8: KITTI360Class(            "sidewalk", 1, False, False, (244,  35, 232)),
        9: KITTI360Class(             "parking", 1, False,  True, (250, 170, 160)),
       10: KITTI360Class(          "rail track", 1, False,  True, (230, 150, 140)),
       11: KITTI360Class(            "building", 2, False, False, ( 70,  70,  70)),
       12: KITTI360Class(                "wall", 2, False, False, (102, 102, 156)),
       13: KITTI360Class(               "fence", 2, False, False, (190, 153, 153)),
       14: KITTI360Class(          "guard rail", 2, False,  True, (180, 165, 180)),
       15: KITTI360Class(              "bridge", 2, False,  True, (150, 100, 100)),
       16: KITTI360Class(              "tunnel", 2, False,  True, (150, 120,  90)),
       17: KITTI360Class(                "pole", 3, False, False, (153, 153, 153)),
       18: KITTI360Class(           "polegroup", 3, False,  True, (153, 153, 153)),
       19: KITTI360Class(       "traffic light", 3, False, False, (250, 170,  30)),
       20: KITTI360Class(        "traffic sign", 3, False, False, (220, 220,   0)),
       21: KITTI360Class(          "vegetation", 4, False, False, (107, 142,  35)),
       22: KITTI360Class(             "terrain", 4, False, False, (152, 251, 152)),
       23: KITTI360Class(                 "sky", 5, False, False, ( 70, 130, 180)),
       24: KITTI360Class(              "person", 6,  True, False, (220,  20,  60)),
       25: KITTI360Class(               "rider", 6,  True, False, (255,   0,   0)),
       26: KITTI360Class(                 "car", 7,  True, False, (  0,   0, 142)),
       27: KITTI360Class(               "truck", 7,  True, False, (  0,   0,  70)),
       28: KITTI360Class(                 "bus", 7,  True, False, (  0,  60, 100)),
       29: KITTI360Class(             "caravan", 7,  True,  True, (  0,   0,  90)),
       30: KITTI360Class(             "trailer", 7,  True,  True, (  0,   0, 110)),
       31: KITTI360Class(               "train", 7,  True, False, (  0,  80, 100)),
       32: KITTI360Class(          "motorcycle", 7,  True, False, (  0,   0, 230)),
       33: KITTI360Class(             "bicycle", 7,  True, False, (119,  11,  32)),
       34: KITTI360Class(              "garage", 2,  True,  True, ( 64, 128, 128)),
       35: KITTI360Class(                "gate", 2, False,  True, (190, 153, 153)),
       36: KITTI360Class(                "stop", 2,  True,  True, (150, 120,  90)),
       37: KITTI360Class(           "smallpole", 3,  True,  True, (153, 153, 153)),
       38: KITTI360Class(                "lamp", 3,  True,  True, (  0,  64,  64)),
       39: KITTI360Class(           "trash bin", 3,  True,  True, (  0, 128, 192)),
       40: KITTI360Class(     "vending machine", 3,  True,  True, (128,  64,   0)),
       41: KITTI360Class(                 "box", 3,  True,  True, ( 64,  64, 128)),
       42: KITTI360Class("unknown construction", 0, False,  True, (102,   0,   0)),
       43: KITTI360Class(     "unknown vehicle", 0, False,  True, ( 51,   0,  51)),
       44: KITTI360Class(      "unknown object", 0, False,  True, ( 32,  32,  32)),
       -1: KITTI360Class(       "license plate", 7, False,  True, (  0,   0, 142)),
    # 65535: KITTI360Class(       "license plate", 7, False,  True, (  0,   0, 142)),  # Allow for two's complement
}

_min_id = min(KITTI360_CLASSES.keys())
_max_id = max(KITTI360_CLASSES.keys())

KITTI360_COLORS = np.zeros((_max_id - _min_id + 1, 3), dtype=np.float)
for k, v in KITTI360_CLASSES.items():
    KITTI360_COLORS[k - _min_id] = np.array(v.color)

KITTI360_COLORS /= 255


def color_points(semantic_labels):
    shifted_semantic_labels = semantic_labels - _min_id
    shifted_semantic_labels = shifted_semantic_labels.detach().cpu().numpy().astype(int)
    return KITTI360_COLORS[shifted_semantic_labels]

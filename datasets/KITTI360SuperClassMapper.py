import numpy as np

from datasets.KITTI360Classes import KITTI360_CLASSES, KITTI360_CATEGORIES


class SemanticSuperclassMapper:

    def __init__(self):

        self.class_map = {k: v.categoryId for k, v in KITTI360_CLASSES.items()}
        self.default_super_class = 0

        class_ids = sorted(list(set(self.class_map.keys())))
        self._class_one_hot_map = {k: i for i, k in enumerate(class_ids)}

        self.superclass_labels = KITTI360_CATEGORIES

        superclass_ids = sorted(list(set(self.superclass_labels.keys())))
        self._superclass_one_hot_map = {k: i for i, k in enumerate(superclass_ids)}

        self.one_hot_maps = {
            "class_one_hot_map": self._class_one_hot_map,
            "superclass_one_hot_map": self._superclass_one_hot_map
        }

    # self.class_map = {k: v['superclass'] for k, v in self.cfg['classes'].items()}

    def get_superclass(self, semantic_class):
        superclass = self.default_super_class * np.ones_like(semantic_class)

        for k, s in self.class_map.items():
            superclass[semantic_class == k] = s

        return superclass

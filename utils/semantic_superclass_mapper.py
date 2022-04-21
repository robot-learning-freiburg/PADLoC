from os import path

import numpy as np
import yaml


class SemanticSuperclassMapper:

	def __init__(self, **kwargs):

		default_cfg_file = path.join(path.abspath(path.dirname(__file__)), "cityscapes_superclasses.yaml")
		self.cfg_file = kwargs.get("cfg_file") or default_cfg_file

		with open(self.cfg_file, "r") as f:
			self.cfg = yaml.load(f, yaml.SafeLoader)

		tmp_cfg = self.cfg['classes']
		column_names = ["class", "superclass"]
		column_indices = {c: tmp_cfg['columns'].index(c) for c in column_names}

		self.class_map = {d[column_indices['class']]: d[column_indices['superclass']] for d in tmp_cfg['data']}

		class_ids = sorted(list(set(self.class_map.keys())))
		self._class_one_hot_map = {k: i for i, k in enumerate(class_ids)}

		tmp_cfg = self.cfg['superclasses']
		column_names = ["superclass", "label"]
		column_indices = {c: tmp_cfg['columns'].index(c) for c in column_names}

		self.superclass_labels = {d[column_indices['superclass']]: d[column_indices['label']] for d in tmp_cfg['data']}

		superclass_ids = sorted(list(set(self.superclass_labels.keys())))
		self._superclass_one_hot_map = {k: i for i, k in enumerate(superclass_ids)}

		self.one_hot_maps = {
			"class_one_hot_map": self._class_one_hot_map,
			"superclass_one_hot_map": self._superclass_one_hot_map
		}

		# self.class_map = {k: v['superclass'] for k, v in self.cfg['classes'].items()}

	def get_superclass(self, semantic_class):

		superclass = np.zeros_like(semantic_class)

		for k, s in self.class_map.items():
			superclass[semantic_class == k] = s

		return superclass

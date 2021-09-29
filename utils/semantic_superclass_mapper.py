from os import path

import numpy as np
import yaml


class SemanticSuperclassMapper:

	def __init__(self, **kwargs):

		default_cfg_file = path.join(path.abspath(path.dirname(__file__)), "cityscapes_superclasses.yaml")
		self.cfg_file = kwargs.get("cfg_file") or default_cfg_file

		with open(self.cfg_file, "r") as f:
			self.cfg = yaml.load(f, yaml.SafeLoader)

		self.class_map = {k: v['superclass'] for k, v in self.cfg['classes'].items()}

	def get_superclass(self, semantic_class):

		superclass = np.zeros_like(semantic_class)

		for k, s in self.class_map.items():
			superclass[semantic_class == k] = s

		return superclass

from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import object
import numpy as np
from .note import Note
from .technique import *
from os import path

class Song(object):
	def __init__(self, sr=44100, hop=256, raw_audio=None, 
				 es_note_list=None, melody=None, 
				 smooth_melody=None, name=None):
		self.sr = sr
		self.hop = hop
		self.raw_audio = raw_audio
		self.es_note_list = es_note_list
		self.melody = melody
		self.smooth_melody = smooth_melody
		self.name = name

	def esn2ts(self):
		ts_list = np.empty([0,3], dtype=float)
		for i, esn in enumerate(self.es_note_list):
			for tech in esn.all_techs:
				if tech.t_type in [T_PULL, T_HAMMER, T_SLIDE]:
					if tech.value == 1 and i < len(self.es_note_list): 
						n_esn = self.es_note_list[i+1]
						if n_esn.tech(tech.t_type) == 2:
							ts_list = np.vstack([ts_list, [esn.onset, n_esn.offset, tech.t_type]])
				else: ts_list = np.vstack([ts_list, [esn.onset, esn.offset, tech.t_type]])
		return ts_list

	def esn_matrix(self):
		return np.array([esn.array_repr() for esn in detected_esns], dtype=float)

	def load_smooth_melody(self, file_path):
		try:
			self.smooth_melody = np.loadtxt(file_path)
		except IOError:
			print('Smooth melody file {} does not exists!'.format(file_path))

	def load_melody(self, file_path):
		try:
			self.melody = np.loadtxt(file_path)
		except IOError:
			print('Melody file {} does not exists!'.format(file_path))

	def load_note_list(self, file_path):
		try:
			raw_notes = np.loadtxt(file_path)
		except IOError:
			print('Note file {} does not exists!'.format(file_path))
		self.es_note_list = np.array([Note(p, on, dur) for p, on, dur in raw_notes], dtype=object)

	def load_esn_list(self, file_path):
		try:
			esn_list = np.loadtxt(file_path)
		except IOError:
			print('ES_Note file {} does not exists!'.format(file_path))
		self.es_note_list = np.array([Note(array=esn) for esn in esn_list], dtype=object)

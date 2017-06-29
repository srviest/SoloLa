import numpy as np
from technique import *

class Note(object):
	def __init__(self, pitch=0, onset=0.0, duration=0.0, 
				 techs=[], array=None, note=None):
		if array is not None:
			self.arr = array.astype(float)
		elif note is not None:
			self.arr = note.array_repr().astype(float)
		else:
			self.arr = np.zeros(12, dtype=float)
			self.arr[0] = float(pitch)
			self.arr[1] = onset
			self.arr[2] = duration
			for t in techs:
				self.arr[t.t_type] = t.value

	def __repr__(self):
		return repr(self.array_repr())

	def __str__(self):
		return str(self.array_repr())

	def array_repr(self):
		return self.arr.copy()

	def add_tech(self, tech):
		self.arr[tech.t_type] = tech.value

	def equal_tech(self, other):
		return (self.arr[3:] == other.array_repr()[3:]).all()

	def pitch():
	    doc = "The pitch property."
	    def fget(self):
	        return self.arr[0]
	    def fset(self, value):
	        self.arr[0] = value
	    return locals()
	pitch = property(**pitch())

	def onset():
	    doc = "The onset property."
	    def fget(self):
	        return self.arr[1]
	    def fset(self, value):
	        self.arr[1] = value
	    return locals()
	onset = property(**onset())

	def duration():
	    doc = "The duration property."
	    def fget(self):
	        return self.arr[2]
	    def fset(self, value):
	        self.arr[2] = value
	    return locals()
	duration = property(**duration())	
	
	@property
	def offset(self):
		return self.arr[1] + self.arr[2]

	@property
	def all_techs(self):
		return [Tech(idx+3, t) for idx, t in enumerate(self.arr[3:])]

	def tech(self, t_num):
		if t_num in range(3, 12):
			return Tech(t_num, self.arr[t_num])
		elif t_num == T_NORMAL:
			value = 1 if np.count_nonzero(self.arr[3:]) == 0 else 0
			return Tech(t_num, value)
		else:
			raise ValueError('ERROR: number of tech should be 3 ~ 12, not {}.'.format(t_num))
			return None

	@staticmethod
	def merge(first, second):
		lead, back = (first.array_repr(), second.array_repr()) \
					if first.onset <= second.onset else (second.array_repr(), first.array_repr())
		note = np.zeros(12, dtype=float)
		note[0] = lead[0]
		note[1] = lead[1]
		note[2] = back[1] + back[2] - lead[1]

		note[T_VIBRATO] = lead[T_VIBRATO] if lead[T_VIBRATO] > 0 else back[T_VIBRATO]
		note[T_SLIDE_IN] = lead[T_SLIDE_IN]
		note[T_SLIDE_OUT] = back[T_SLIDE_OUT]

		if lead[T_SLIDE] == 1 and back[T_SLIDE] == 2:
			note[T_SLIDE] = 0
		elif lead[T_SLIDE] == 2 and back[T_SLIDE] == 1:
			note[T_SLIDE] = 3
		elif lead[T_SLIDE] > 0:
			note[T_SLIDE] = lead[T_SLIDE]
		else:
			note[T_SLIDE] = back[T_SLIDE]

		if lead[T_HAMMER] == 1 and back[T_HAMMER] == 2:
			note[T_HAMMER] = 0
		elif lead[T_HAMMER] > 0:
			note[T_HAMMER] = lead[T_HAMMER]
		else:
			note[T_HAMMER] = back[T_HAMMER]

		if lead[T_PULL] == 1 and back[T_PULL] == 2:
			note[T_PULL] = 0
		elif lead[T_PULL] > 0:
			note[T_PULL] = lead[T_PULL]
		else:
			note[T_PULL] = back[T_PULL]

		if lead[T_BEND] > 0 and back[T_BEND] > 0 and \
		   lead[T_RELEASE] > 0 and back[T_RELEASE] > 0:
			note[T_PREBEND] = lead[T_PREBEND]
			note[T_VIBRATO] = lead[T_BEND]
		elif lead[T_BEND] > 0 and back[T_PREBEND] > 0 and back[T_RELEASE] > 0:
			note[T_BEND] = lead[T_BEND]
			note[T_RELEASE] = back[T_RELEASE]
		elif back[T_BEND] > 0 and lead[T_PREBEND] > 0 and lead[T_RELEASE] > 0:
			note[T_BEND] = back[T_BEND]
			note[T_RELEASE] = lead[T_RELEASE]
			note[T_PREBEND] = lead[T_PREBEND]
		else:
			note[3:6] = lead[3:6]

		return type(first)(array=note)


class DiscreteNote(Note):
	def __init__(self, pitch=0, onset=0, duration=0, 
				 techs=[], array=None, note=None):
		if array is not None:
			self.arr = array.astype(int)
		elif note is not None:
			self.arr = note.array_repr().astype(int)
		else:
			self.arr = np.zeros(12, dtype=int)
			self.arr[0] = int(round(pitch))
			self.arr[1] = int(round(onset))
			self.arr[2] = int(round(duration))
			try:
				for t in techs:
					self.arr[t.t_type] = t.value
			except Exception, e:
				print 't:', t.value, t.t_type
				raise e

	def discrete_to_cont(self, hop_size, sr):
		ratio = float(hop_size) / float(sr)
		return Note(self.pitch, self.onset*ratio, self.duration*ratio, self.all_techs)

class CandidateNote(DiscreteNote):
	def __init__(self, pitch=0, onset=0, duration=0, 
				 techs=[], segs=[], array=None, note=None):
		if array is not None:
			self.arr = array.astype(int)
		elif note is not None:
			self.arr = note.array_repr().astype(int)
		else:
			self.arr = np.zeros(12, dtype=int)
			self.arr[0] = int(round(pitch))
			self.arr[1] = int(round(onset))
			self.arr[2] = int(round(duration))
			for t in techs:
				self.arr[t.t_type] = t.value
		self.segs = segs

def merge_notes(notes):
    for i in range(len(notes)-1, 0, -1):
        nt = notes[i-1]
        ### Merge notes if there is bend or release
        if (nt.tech(T_BEND).value > 0 and \
            nt.offset == notes[i].onset and \
            nt.pitch + nt.tech(T_BEND).value == notes[i].pitch) or \
           (nt.tech(T_RELEASE).value > 0 and \
            nt.offset == notes[i].onset and \
            nt.pitch == notes[i].pitch):
            notes[i-1] = Note.merge(nt, notes[i])
            notes.remove(notes[i])
        elif nt.tech(T_SLIDE).value == 1:
            t_val = 3 if notes[i].tech(T_SLIDE).value in (1, 3) else 2
            notes[i].add_tech(Tech(T_SLIDE, t_val))
        elif nt.tech(T_HAMMER).value == 1:
            t_val = 3 if notes[i].tech(T_HAMMER).value in (1, 3) else 2
            notes[i].add_tech(Tech(T_HAMMER, t_val))
        elif nt.tech(T_PULL).value == 1:
            t_val = 3 if notes[i].tech(T_PULL).value in (1, 3) else 2
            notes[i].add_tech(Tech(T_PULL, t_val))

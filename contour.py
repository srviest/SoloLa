import numpy as np

class Contour(object):
	def __init__(self, start_idx=0, seq=np.array([])):
		self.start_idx = int(start_idx)
		self.seq = np.array(seq).copy()

	def __repr__(self):
		return 'start_idx: ' + str(self.start_idx) + '\nseq: ' + repr(self.seq)

	def __str__(self):
		return 'start_idx: ' + str(self.start_idx) + '\nseq: ' + repr(self.seq)

	def __getitem__(self, arg):
		return self.seq[arg]

	@property
	def length(self):
		return len(self.seq)

	@property
	def end_idx(self):
		return self.start_idx + len(self.seq) - 1

	@property
	def max(self):
		return np.max(self.seq)

	@property
	def min(self):
		return np.min(self.seq)

	def estimated_pitch(self, indices=None):
		x = self.seq[indices] if indices else self.seq
		return int(round(x.mean()))

	def append(self, val):
		self.seq = np.append(self.seq, val)

	def sub_contour(self, indices):
		if len(indices) == 0: return None
		idx = self.start_idx + indices[0]
		return type(self)(idx, self.seq[indices])


class Slope(Contour):
	def __init__(self, start_idx=0, seq=np.array([]), contour=None):
		if contour != None:
			self.start_idx = int(contour.start_idx)
			self.seq = contour.seq.copy()
		else:
			self.start_idx = int(start_idx)
			self.seq = np.array(seq).copy()

	@property
	def end_pitch(self):
		return self.estimated_pitch(indices=range(self.length-4, self.length))

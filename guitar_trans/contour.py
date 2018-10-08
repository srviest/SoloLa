from __future__ import unicode_literals
from __future__ import division
from builtins import str
from past.utils import old_div
from builtins import object
import numpy as np
from itertools import groupby

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

class Segment(object):
    def __init__(self, val=0, pos=0, length=0, ref_con=None, seg=None):
        if seg is not None:
            self.val = seg.val # value
            self.pos = seg.pos # position
            self.length = seg.length # length
            self.ref_con = seg.ref_con # referenced contour
        else:
            self.val = val # value
            self.pos = pos # position
            self.length = length # length
            self.ref_con = ref_con # referenced contour

    def __repr__(self):
        return '(val: ' + str(self.val) + ', pos: ' + str(self.pos) + ', length: ' + str(self.length) + ')'

    def __str__(self):
        return '(val: ' + str(self.val) + ', pos: ' + str(self.pos) + ', length: ' + str(self.length) + ')'

    @property 
    def end(self):
        return self.pos + self.length
    @property
    def mid(self):
        return self.pos + int(old_div((self.length + 1), 2))

    def diff(self):
        ct = Contour(self.ref_con.start_idx+self.pos, 
                     self.ref_con.seq[self.pos:self.pos+self.length+1])
        return ct.max - ct.min

    def contour(self):
        return Contour(self.ref_con.start_idx+self.pos, 
                       self.ref_con.seq[self.pos:self.pos+self.length])

class SegmentedContour(Contour):
    def __init__(self, start_idx, seq, trend=[]):
        super(SegmentedContour, self).__init__(start_idx, seq)
        self.__seg_dict = {}
        if len(trend) > 0:
            p = 0
            for val, _s in groupby(trend[:len(self.seq)]):
                length = len(list(_s))
                if val != 0:
                    self.__seg_dict[p] = Segment(val, p, length, self)
                p += length

    def seg(self, key):
        return self.__seg_dict[key]

    def all_segs(self, sort=False):
        if sort:
            return sorted(list(self.__seg_dict.values()), key=lambda x: x.pos)
        else:
            return list(self.__seg_dict.values())
    
    def seg_keys(self):
        return list(self.__seg_dict.keys())

    @property
    def n_segs(self):
        return len(self.__seg_dict)

    def merge_segs(self, keys):
        if len(keys) > 1:
            keys.sort()
            s = self.seg(keys[0])
            s.length = self.seg(keys[-1]).end - s.pos
            for i in keys[1:]:
                self.delete_seg(i)

    def delete_seg(self, key):
        if isinstance(key, Segment):
            self.__seg_dict.pop(key.pos)
        else:
            self.__seg_dict.pop(key)

    def get_trend(self):
        trend = np.zeros(self.length)
        for s in self.all_segs():
            trend[s.pos:s.pos+s.length] = s.val
        return trend

    def sub_contour(self, indices):
        if len(indices) == 0: return None
        idx = self.start_idx + indices[0]
        return type(self)(idx, self.seq[indices], self.get_trend()[indices])


from __future__ import print_function
from __future__ import unicode_literals
from builtins import str
from builtins import object
T_NORMAL = 12
T_PREBEND = 3
T_BEND = 4
T_RELEASE = 5
T_PULL = 6
T_HAMMER = 7
T_SLIDE = 8
T_SLIDE_IN = 9
T_SLIDE_OUT = 10
T_VIBRATO = 11

T_STR_DICT = {  T_NORMAL: 'Normal',
				T_PREBEND: 'Pre-bend',
				T_BEND: 'Bend',
				T_RELEASE: 'Release',
				T_PULL: 'Pull',
				T_HAMMER: 'Hammer',
				T_SLIDE: 'Slide',
				T_SLIDE_IN: 'Slide-in',
				T_SLIDE_OUT: 'Slide-out',
				T_VIBRATO: 'Vibrato'}

class Tech(object):
	def __init__(self, t_type=T_NORMAL, value=0):
		if t_type > T_NORMAL or t_type < T_PREBEND: 
			print('ERROR: No Tech type {}. Will assign to normal type(12).'.format(t_type))
			t_type = 12
		self.t_type = int(t_type)
		# if self.t_type != T_NORMAL:
		# 	if value != 1 and value != 2:
		# 		print('ERROR: Value of Tech type {} should be 1 or 2 not {}. Will assign to 1.'.format(self.t_type, value))
		# 		value = 1
		# elif value != 0:
		# 	print('ERROR: Value of Tech type {} should be 0 not {}. Will assign to 0.'.format(self.t_type, value))
		# 	value = 0
		self.value = value

	def __eq__(self, other):
		return (self.t_type == other.t_type and self.value == other.value)

	def __str__(self):
		return 'Tech(t_type: ' + T_STR_DICT[self.t_type] + '(' + str(self.t_type) + '), value: ' + str(self.value) + ')'

	def __repr__(self):
		return 'Tech(t_type: ' + T_STR_DICT[self.t_type] + '(' + str(self.t_type) + '), value: ' + str(self.value) + ')'

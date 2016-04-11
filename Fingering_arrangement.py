#!/usr/bin/env python
# encoding: utf-8
"""
Author: Yuan-Ping Chen
Data: 2016/03/15
-------------------------------------------------------------------------------
Fingering arrangement: automatically arrange the guitar fingering.
-------------------------------------------------------------------------------
Args:
    input_files:    files to be processed. 
                    Only the .expression_style_note files would be considered.
    output_dir:     Directory for storing the results.

Optional args:
    Please refer to --help.
-------------------------------------------------------------------------------
Returns:
    Raw melody contour:         Text file of estimated melody contour 
                                in Hz with extenion of .raw.melody.

"""
import numpy as np
import os
import networkx as nx
import itertools

class GuitarEvent(object):

    def __init__(self, **kwargs):
        # optional timing information
        # timestamp start
        self.ts_start = kwargs.get('ts_start')
        # beat start
        self.beat_start = kwargs.get('beat_start')
        # beat duration
        self.dur = kwargs.get('dur')


class Pluck(GuitarEvent):

    def __init__(self, string, fret, **kwargs):
        super(Pluck, self).__init__(**kwargs)

        self.string = string
        self.fret = fret

    def distance(self, other):
        '''
        Get the distance between this pluck with a pluck or strum
        '''

        if isinstance(other, Pluck):
            if self.fret == 0 or other.fret == 0:
                distance = 0
            else:
                distance = self.fret - other.fret
        elif isinstance(other, Strum):
            other_frets = [p.fret for p in other.plucks]
            min_other_frets = min(other_frets)
            max_other_frets = max(other_frets)

            if self.fret <= min_other_frets:
                distance = min_other_frets - self.fret
            elif self.fret >= max_other_frets:
                distance = self.fret - max_other_frets
            else:
                distance = self.fret - (min_other_frets + max_other_frets)/2
        else:
            raise ValueError('Must compare to a pluck or strum')

        return abs(distance)

    def is_open(self):
        '''
        True if the pluck is an open string
        '''
        return self.fret == 0

    def __eq__(self, other_pluck):
        return self.string == other_pluck.string and self.fret == other_pluck.fret

    def __str__(self):
        return '<pluck: string: %d, fret: %d>' % (self.string+1, self.fret)

    def __repr__(self):
        return self.__str__()


class ScoreEvent(object):

    def __init__(self, **kwargs):
        # optional timing information
        # onset timestamp
        self.onset_ts = kwargs.get('onset_ts')
        # offset timestamp
        self.offset_ts = kwargs.get('offset_ts')
        # beat start
        self.beat_start = kwargs.get('beat_start')
        # beat duration
        self.dur = kwargs.get('dur')

class Note(ScoreEvent):

    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def __init__(self, pname, oct, **kwargs):
        '''
        pname {String}: pitch name
        oct {Integer}: octave
        kwargs is for passing in timing information
        '''
        super(Note, self).__init__(**kwargs)

        # pitch class
        if pname.upper() in Note.pitch_classes:
            self.pname = pname.upper()
        else:
            raise ValueError('Invalid pitch name')

        # octave
        self.oct = oct

    def toMidi(self):
        '''
        Convert the pitch name and octave to a MIDI note number
        between 0 and 127
        '''

        p_ind = Note.pitch_classes.index(self.pname)
        num_chroma = len(Note.pitch_classes)

        midi = (self.oct-1)*num_chroma + 24 + p_ind
        
        if midi >= 0 and midi <= 127:
            return midi
        else:
            return None

    def __add__(self, step):
        '''
        Add an integer number of semitones to the note
        '''

        num_chroma = len(Note.pitch_classes)
        step_up = True
        if step < 0:
            step_up = False

        note = Note(self.pname, self.oct, self.id)
        p_ind = Note.pitch_classes.index(self.pname)
        new_p_ind = (p_ind + step) % num_chroma

        note.pname = Note.pitch_classes[new_p_ind]
        oct_diff = int(step / 12)

        note.oct = self.oct + oct_diff

        if oct_diff == 0:
            if step_up:
                if new_p_ind >= 0 and new_p_ind < p_ind:
                    note.oct += 1
            else:
                if new_p_ind > p_ind and new_p_ind < num_chroma:
                    note.oct -= 1

        return note

    def __sub__(self, step):
        '''
        Subtract an integer number of semitones to the note
        '''

        return self.__add__(-step)

    def __eq__(self, other_note):
        return self.pname == other_note.pname and self.oct == other_note.oct

    def __lt__(self, other_note):
        return self.oct < other_note.oct or (self.oct == other_note.oct and Note.pitch_classes.index(self.pname) < Note.pitch_classes.index(other_note.pname))

    def __le__(self, other_note):
        return self.__lt__(other_note) or self.__eq__(other_note)

    def __gt__(self, other_note):
        return self.oct > other_note.oct or (self.oct == other_note.oct and Note.pitch_classes.index(self.pname) > Note.pitch_classes.index(other_note.pname))

    def __ge__(self, other_note):
        return self.__gt__(other_note) or self.__eq__(other_note)

    def __str__(self):
        return "<note@: %s%d>" % (self.pname, self.oct)

    def __repr__(self):
        return self.__str__()



class Score(object):

    def __init__(self, note):
        '''
        Initialize a score 
        '''

        # musical events occuring in the input score
        self.score_events = []
        self.doc = None         # container for parsed music document
        for n in note:
            Note = self._handle_note(n)
            self.score_events.append(Note)

    def engrave(self):
        '''
        Call after self.score_events has been populated from file
        to print the internal data structure to the terminal.
        Mostly used for debugging.
        '''

        for e in self.score_events:
            print e

    def _handle_note(note):
        '''
        Helper function that takes an mei note element
        and creates a Note object out of it.
        '''
        MIDI_num = int(note[0])
        pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        MIDI_num%12
        pname = pitch_names[MIDI_num%12]
        oct = int(MIDI_num/12-1)

        return Note(pname, oct)


class ArrangeTabAstar(object):
    '''
    AStar class that forms a graph from a music score
    '''

    def __init__(self, score, num_frets):
        self.score = score
        self.num_frets = num_frets
        self.graph = None
        self.path = None
    def gen_tab(self):
        self.graph = self._gen_graph()

        # run the A* algorithm
        self.path = nx.astar_path(self.graph, 1, self.graph.number_of_nodes())
        # remove start and end nodes
        del self.path[0], self.path[-1]

        strums = []
        for n in self.path:
            n = self.graph.node[n]
            guitar_event = n['guitar_event']
            score_event = n['score_event']

            plucks = []
            if isinstance(guitar_event, Pluck):
                plucks.append((score_event.pname, score_event.oct, guitar_event))
            else:
                for pluck, note in zip(guitar_event.plucks, score_event.notes):
                    plucks.append((note.id, pluck))
            strums.append(plucks)

        fingering = np.empty([0,2])
        for s in strums:            
            for ss in s: 
                fingering = np.append(fingering,[[ss[2].string, ss[2].fret]], axis=0)

        return fingering
        np.savetxt(output_path, fingering, fmt='%s')
        

    def _gen_graph(self):
        dg = nx.DiGraph()

        # start node for the search agent
        dg.add_node(1, guitar_event='start')

        prev_node_layer = [1]
        node_num = 2
        num_nodes = len(self.score.score_events)
        for i, e in enumerate(self.score.score_events):
            
            # generate all possible fretboard combinations for this event
            candidates = self._get_candidates(e)
            if len(candidates) == 0:
                continue

            node_layer = []
            for c in candidates:
                # each candidate position becomes a node on the graph
                dg.add_node(node_num, guitar_event=c, score_event=e)
                node_layer.append(node_num)

                # form edges between this node and nodes in previous layer
                edges = []
                for prev_node in prev_node_layer:
                    # calculate edge weight
                    w = ArrangeTabAstar.biomechanical_burlet(dg.node[prev_node]['guitar_event'], dg.node[node_num]['guitar_event'])
                    edges.append((prev_node, node_num, w))
                dg.add_weighted_edges_from(edges)

                node_num += 1

            prev_node_layer = node_layer

        # end node for the search agent
        dg.add_node(node_num, guitar_event='end')
        edges = [(prev_node, node_num, 0) for prev_node in prev_node_layer]
        dg.add_weighted_edges_from(edges)

        return dg
    
    @staticmethod
    def biomechanical_burlet(n1, n2):
        '''
        Evaluate the biomechanical cost of moving from one node to another.

        PARAMETERS
        ----------
        n1: GuitarEvent
        n2: following GuitarEvent
        '''        

        distance = 0            # biomechanical distance
        w_distance = 2          # distance weight

        if n1 != 'start':
            # calculate distance between nodes
            if not n1.is_open():
                distance = n1.distance(n2)

        fret_penalty = 0
        w_fret_penalty = 1      # fret penalty weight
        fret_threshold = 7      # start incurring penalties above fret 7

        chord_distance = 0
        w_chord_distance = 2

        chord_string_distance = 0       # penalty for holes between string strums
        w_chord_string_distance = 1

        if isinstance(n2, Pluck):
            if n2.fret > fret_threshold:
                fret_penalty += 1
        else:
            frets = [p.fret for p in n2.plucks]
            if max(frets) > fret_threshold:
                fret_penalty += 1

            chord_distance = max(frets) - min(frets)

            strings = sorted([p.string for p in n2.plucks])
            for i in range(len(strings)-1,-1,-1):
                if i-1 < 0:
                    break
                s2 = strings[i]
                s1 = strings[i-1]
                chord_string_distance += (s2-s1)
                
            chord_string_distance -= len(strings)-1
        
        return w_distance*distance + w_fret_penalty*fret_penalty + w_chord_distance*chord_distance + w_chord_string_distance*chord_string_distance

    def _get_candidates(self, score_event):
        '''
        Calculate guitar pluck or strum candidates for a given note or chord event
        '''

        candidates = []
        if isinstance(score_event, Note):
            candidates = self._get_candidate_frets(score_event)
        return candidates

    def _get_candidate_frets(self, note):
        '''
        Given a note, get all the candidate (string, fret) pairs
        where it could be played given the current guitar properties
        (number of strings, and tuning).
        '''

        candidates = []
        num_chroma = len(Note.pitch_classes)
        strings = [Note('E', 4), Note('B', 3), Note('G', 3), Note('D', 3), Note('A', 2), Note('E', 2)]
        # get open string pitches with capo position
        open_strings = [n for n in strings]

        for i, s in enumerate(open_strings):
            # calculate pitch difference from the open string note
            oct_diff = note.oct - s.oct
            pname_diff = Note.pitch_classes.index(note.pname) - Note.pitch_classes.index(s.pname)
            pitch_diff = pname_diff + num_chroma*oct_diff

            if pitch_diff >= 0 and pitch_diff <= self.num_frets:
                candidates.append(Pluck(i, pitch_diff))

        return candidates


def parse_input_files(input_files, ext):
    """
    Collect all files by given extension.

    :param input_files:  list of input files or directories.
    :param ext:          the string of file extension.
    :returns:            a list of stings of file name.
    
    """
    from os.path import basename, isdir
    import fnmatch
    import glob
    files = []

    # check what we have (file/path)
    if isdir(input_files):
        # use all files with .raw.melody in the given path
        files = fnmatch.filter(glob.glob(input_files+'/*'), '*'+ext)
    else:
        # file was given, append to list
        if basename(input_files).find(ext)!=-1:
            files.append(input_files)
    print '  Input files: '
    for f in files: print '    ', f
    return files

def parser():
    """
    Parses the command line arguments.

    :param lgd:       use local group delay weighting by default
    :param threshold: default value for threshold

    """
    import argparse
    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="""
    If invoked without any parameters, the software S1 Extract melody contour,
     track notes and timestmaps of intersection of ad continuous pitch sequence
     inthe given files, the pipeline is as follows,

    """)
    # general options
    p.add_argument('input_files', type=str, metavar='input_files',
                   help='files to be processed')
    p.add_argument('output_dir', type=str, metavar='output_dir',
                   help='output directory.')
    p.add_argument('-fn', '--fret_number', type=int, dest='fn',  help="the fret number of guitar finger board.",  default=22)
    # version
    p.add_argument('--version', action='version',
                   version='%(prog)spec 1.03 (2016-03-30)')
    # parse arguments
    args = p.parse_args()

    # return args
    return args
    

def main(args):
    print 'Running fingering arrangement...'
    
    # parse and list files to be processed
    files = parse_input_files(args.input_files, ext='.expression_style_note')
    
    # create result directory
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    print '  Output directory: ', '\n', '    ', args.output_dir

    # processing
    for f in files:
        # parse file name and extension
        ext = os.path.basename(f).split('.')[-1]
        name = os.path.basename(f).split('.')[0]

        # extract the pitch, onset and duration
        note_nparray = expression_style_note[:,0:3]
        # convert numpy array to list
        note = np.ndarray.tolist(note_nparray)    
        # generate the score model
        score = Score(note)
        astar = ArrangeTabAstar(score, num_frets=args.fn)
        fingering = astar.gen_tab()
        np.savetxt(args.output_dir+os.sep+name+'.fingering', fingering, fmt='%s')

        
if __name__ == '__main__':
    args = parser()
    main(args)
import collections
import math
from os.path import join as oj

import mat4py


# auxilin_dir = '/accounts/grad/xsli/auxilin_data'
# auxilin_dir = '/scratch/users/vision/data/abc_data/auxilin_data/'

class signal(object):
    
    """
    class of detected signals
    
    Attributes:
        x: int
            x coordinate of signal
        y: int
            y coordinate of signal
        amplitude: float
            amplitude of fitted Gaussian
        frame: int
            index of frame that the signal belongs to
        track_id: int
            index of track (among all tracks) that the signal belongs to
        index_in_track: int
            frame - (index of first frame of corrsponding track)
        neighbors: dict
            key: float
            value: list of signal objects: signals whose distance to the current signal is smaller than key
            
    please feel free to add other stuff
    """
    
    def __init__(self, x, y, amplitude, frame, track_id, index_in_track):
        self.x = x
        self.y = y
        self.amplitude = amplitude
        self.frame = frame
        self.track_id = track_id
        self.index_in_track = index_in_track
        self.neighbors = collections.defaultdict(list)
        
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class clathrinSignal(signal):
    
    """
    class of clathrin signals
    """
    
    def get_neighbors(self, dist):
        if dist in self.neighbors:
            return self.neighbors[dist]

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class auxilinSignal(signal):
        
    def get_neighbors(self, dist):
        if dist in self.neighbors:
            return self.neighbors[dist]        
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
        
class track(object):
    
    """
    class of detected tracks
    
    Attributes:
        clathrin_signals: list
            list of clathrinSignal objects
        auxilin_signals: list
            list of auxilinSignal
        lifetime: int
            lifetime of track
            
    please feel free to add other stuff
    """
    
    def __init__(self, clathrin_signals, auxilin_signals, lifetime):
        
        self.clathrin_signals = clathrin_signals
        self.auxilin_signals = auxilin_signals
        self.lifetime = lifetime
        
    def add_signal(self, cla_signal, aux_signal):
        
        self.clathrin_signals.append(cla_signal)
        self.auxilin_signals.append(aux_signal)
        self.lifetime += 1

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
        
def read_tracks(auxilin_dir, cell_name):
    
    """
    function to read .mat tracks into python objects
    
    Input:
        auxilin_dir, cell_name: str
        
    Output:
        tracks: list of track objects
            each element in the list corresponding to one track
        tracks_raw: dict
            mat['tracks']: dict with everything
        tracks_by_frame: dict
            key: int (#frame)
            value: list of dict {'cla':clathrinSignal, 'aux': auxilinSignal"}
                all signals in the corresponding frame, use key='cla' to access clathrin signal
    """
    
    tracks_fname = oj(auxilin_dir, 'A7D2', cell_name, 'TagRFP/Tracking/ProcessedTracks.mat')
    mat = mat4py.loadmat(tracks_fname)
    tracks_raw = mat['tracks']
    t, x, y, A, n_tracks = tracks_raw['t'], tracks_raw['x'], tracks_raw['y'], tracks_raw['A'], len(tracks_raw['t'])
    tracks, tracks_by_frame = [], collections.defaultdict(list)
    for tr in range(n_tracks):
        tracks.append(track(clathrin_signals=[], auxilin_signals=[], lifetime=0))
        for i in range(len(t[tr])):
            if not math.isnan(x[tr][0][i]) and not math.isnan(y[tr][0][i]):
                new_cla_signal = clathrinSignal(x=int(x[tr][0][i]),
                                                y=int(y[tr][0][i]),
                                                amplitude=A[tr][0][i],
                                                frame=t[tr][i],
                                                track_id=tr,
                                                index_in_track=i
                                               )
                new_aux_signal = auxilinSignal(x=int(x[tr][1][i]),
                                               y=int(y[tr][1][i]),
                                               amplitude=A[tr][1][i],
                                               frame=t[tr][i],
                                               track_id=tr,
                                               index_in_track=i
                                               )
                tracks_by_frame[t[tr][i]].append({'cla':new_cla_signal, 'aux':new_aux_signal})
                tracks[-1].add_signal(new_cla_signal, new_aux_signal)
                
    return tracks, tracks_raw, tracks_by_frame
    

#def get_neighbors():
    


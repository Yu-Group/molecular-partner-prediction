import numpy as np
import pandas as pd
import sys
sys.path.append('../src')
from os.path import join as oj
import config
import pickle as pkl


class TestPipelineOriginal:
    '''Tests that original pipeline properly assigns outcomes / preprocesses based on lifetime
    '''
    def setup(self,  metadata_file=oj(config.DIR_PROCESSED, 'metadata_clath_aux+gak_a7d2.pkl')):
        np.random.seed(13)
        self.m = pkl.load(open(metadata_file, 'rb'))
        '''
            print(
                f'valid:\t\t{:>4.0f} aux+ / {:>4.0f} ({m["num_aux_pos_valid"] / m["num_tracks_valid"]:.3f})')
            print('----------------------------------------')
            print(f'hotspots:\t{m["num_hotspots_valid"]:>4.0f} aux+ / {m["num_hotspots_valid"]:>4.0f}')
            print(
                f'short:\t\t{m["num_short"] - m["num_short"] * m["acc_short"]:>4.0f} aux+ / {m["num_short"]:>4.0f} ({m["acc_short"]:.3f})')
            print(f'long:\t\t{m["num_long"] * m["acc_long"]:>4.0f} aux+ / {m["num_long"]:>4.0f} ({m["acc_long"]:.3f})')
            print(
                f'hard:\t\t{m["num_aux_pos_hard"]:>4.0f} aux+ / {m["num_tracks_hard"]:>4.0f} ({m["num_aux_pos_hard"] / m["num_tracks_hard"]:.3f})')

            if acc is not None:
                print('----------------------------------------')
                print(f'hard acc:\t\t\t  {acc:.3f}')
                num_eval = m["num_tracks_valid"] - m["num_hotspots_valid"]
            #         print(
            #             f'total acc (no hotspots):\t  {(m["num_short"] * m["acc_short"] + m["num_long"] * m["acc_long"] + acc * m["num_tracks_hard"]) / num_eval:.3f}')
            print('\nlifetime threshes', m['thresh_short'], m['thresh_long'])
        '''
    def test_metadata(self):
        '''Test metadata matches original
        '''
        assert self.m['num_aux_pos_valid'] == 2066
        assert self.m['num_tracks_valid'] == 7594
        
        assert self.m['num_hotspots_valid'] == 349
        assert self.m['num_short'] == 5697
        assert self.m['num_long'] == 113
        assert self.m['num_tracks_hard'] == 2936
        
        
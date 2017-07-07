# Step 3 in the chain process: Make OtherLargeS2s minitree
# By using hax.minitrees.load_single_dataset, you don't actually make a .root minitree
# instead you get a dataframe in return and pickle it to where you want

# Updated to get interior_split_fraction branch

import sys, os
import numpy as np
import pandas as pd
import hax
from pax import units, configuration, datastructure

pax_config = configuration.load_configuration('XENON1T')

v_drift = pax_config['DEFAULT']['drift_velocity_liquid']
drift_time_gate = pax_config['DEFAULT']['drift_time_gate']
sample_duration = pax_config['DEFAULT']['sample_duration']
electron_life_time = pax_config['DEFAULT']['electron_lifetime_liquid']

class Peaks(hax.minitrees.TreeMaker):
    """Information on the large s2 peaks other than the interaction peak. (e.g. s2_1_area)
    Provides:
     - s2_x_y
    x denotes the order of the area of the peak:
     - 1: The 1st largest among peaks other than main interaction peak
     - 2: The 2nd largest among peaks other than main interaction peak
     ...
     - 5: The 5th largest among peaks other than main interaction peak
    
    y denotes the property of the peak:
     - area: The uncorrected area in pe of this peak
     - range_50p_area: The width, duration of region that contains 50% of the area of the peak
     - area_fraction_top: The fraction of uncorrected area seen by the top array
     - x: The x-position of this peak (by TopPatternFit)
     - y: The y-position of this peak
     - z: The z-position of this peak (computed using configured drift velocity)
     - corrected_area: The corrected area in pe of the peak
     - delay_is1: The hit time mean minus main s1 hit time mean
     - delay_is2: The hit time mean minus main s2 hit time mean
     - *interior_split_fraction: Area fraction of the smallest of the two halves considered in the best split inside the peak
    
    Notes:
     - 'largest' refers to uncorrected area, also excluding main interation peak
     - 'z' calculated from the time delay from main interaction s1
     - 'main interaction' is event.interactions[0]
                          (currently is largest S2 + largest S1 before it)
     - 'corrected_area' only corrected by lifetime and spatial 
        
    """
    extra_branches = ['peaks.*']
    peak_name = ['s2_%s_' % order for order in ['1','2','3','4','5']]
    peak_fields = ['area', 'range_50p_area', 'x', 'y', 'z', 'delay_is2','goodness_of_fit']
    main_fields = ['s2','s2_delay_is2','s2_range_50p_area',
                  's1','x','y','z','cs2','cs1','goodness_of_fit']
    __version__ = '0.1.1'
    
    def extract_data(self, event):
        event_data = dict()
        
        # At least one s1 is needed to anchor all s2s
        if len(event.interactions) != 0:
            # find out which peak are in main interaction
            interaction = event.interactions[0]
            s1 = event.peaks[interaction.s1]
            s2 = event.peaks[interaction.s2]
            other_s2s = [ix for ix in event.s2s 
                         if (ix != interaction.s2) and 
                            (event.peaks[ix].index_of_maximum - s1.index_of_maximum > 0)]
            
            _mains = {}
            
            for field in self.main_fields:
                drift_time = (s2.index_of_maximum - s1.index_of_maximum) * sample_duration
                
                if field == 's2':
                    _x = s2.area
                elif field == 's2_delay_is2':
                    _x = (s2.hit_time_mean - s1.hit_time_mean)
                elif field == "s2_range_50p_area":
                    _x = list(s2.range_area_decile)[5]
                elif field == "s1":
                    _x = s1.area
                elif field in 'x':
                    _x = interaction.x
                elif field in 'y':
                    _x = interaction.y
                elif field == 'z':
                    _x = (- drift_time + drift_time_gate) * v_drift
                elif field == 'cs2':
                    s2_lifetime_correction = np.exp(drift_time / electron_life_time)
                    _x = s2.area * s2_lifetime_correction * s2.s2_spatial_correction
                elif field == 'goodness_of_fit':
                        # In case of x and y need to get position from reconstructed_positions
                        for rp in s2.reconstructed_positions:
                            if rp.algorithm == 'PosRecTopPatternFit':
                                _x = getattr(rp, field)
                                break
                            else:
                                _x = float('nan')
                elif field =='cs1':
                    _x = s1.area * interaction.s1_area_correction
                else:
                    _x = getattr(s2, field)
                    
                _mains[field] = _x

            event_data.update(_mains)
                
            # Start looking for the properties we want
            for order, ix in enumerate(other_s2s):
                peak = event.peaks[ix]
                if order >= 5:
                    break
                _current_peak = {}
                
                drift_time = (peak.index_of_maximum - s1.index_of_maximum) * sample_duration
                
                for field in self.peak_fields: # assuming peaks are already sorted
                    # Deal with special cases
                    if field == 'range_50p_area':
                        _x = list(peak.range_area_decile)[5]
                    elif field in ('x', 'y','goodness_of_fit'):
                        # In case of x and y need to get position from reconstructed_positions
                        for rp in peak.reconstructed_positions:
                            if rp.algorithm == 'PosRecTopPatternFit':
                                _x = getattr(rp, field)
                                break
                            else:
                                _x = float('nan')
                    elif field == 'z':
                        _x = (- drift_time + drift_time_gate) * v_drift
                    elif field == 'corrected_area':
                        s2_lifetime_correction = np.exp(drift_time / electron_life_time)
                        _x = peak.area * s2_lifetime_correction * peak.s2_spatial_correction
                    elif field == 'delay_is1':
                        _x = (peak.hit_time_mean - s1.hit_time_mean)
                    elif field == 'delay_is2':
                        _x = (peak.hit_time_mean - s2.hit_time_mean)
                    else:
                        _x = getattr(peak, field)
                    
                    field = self.peak_name[order] + field
                    _current_peak[field] = _x

                event_data.update(_current_peak)
         
        return event_data
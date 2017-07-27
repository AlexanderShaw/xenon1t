import ROOT
import root_pandas
import numpy as np
import pandas as pd
import sys
from root_pandas import read_root

def better_row(df_row,i):
    """ Make a better row to analyze what is ms or ss from truth data, i is the index of the row """
    ed = []
    ed.append(max(df_row.Fax_ed))
    
    s_a = np.where(df_row.Fax_ed == ed[0])
    
    s_a_x = df_row.Fax_x[s_a]
    s_a_y = df_row.Fax_y[s_a]
    s_a_z = df_row.Fax_z[s_a]
    
    if len(df_row.Fax_ed) > 1:
        ed.append(sorted(df_row.Fax_ed)[- 2])
        s_b = np.where(df_row.Fax_ed == ed[1])
        s_b_x = df_row.Fax_x[s_b]
        s_b_y = df_row.Fax_y[s_b]
        s_b_z = df_row.Fax_z[s_b]
        
        s_dist = np.sqrt((s_a_x - s_b_x)**2 + (s_a_y - s_b_y)**2 + (s_a_z - s_b_z)**2)
        
    else:
        ed.append(float('nan'))
        s_dist = float('nan')


    idx = df_row.index
    
    eventid = df_row.eventid
    
    return pd.DataFrame({
        'eventid' : eventid,
        'ed_a' : ed[0],
        'ed_b' : ed[1],
        's_dist' : s_dist
    }, index = [i])

def better_df(df):
	""" takes pre-fax waveform root file, prepares events to better
	classified as SS or not-SS """
	col_needed = better_row(df.iloc[0], 0)
	ndf = pd.DataFrame(columns = list(col_needed))
	for i in range(len(df)):
	    row = df.iloc[i]
	    br = better_row(row, i)
	    ndf = ndf.append(br)
	return ndf

def classify_df(df):
	""" 
	takes a "better_df" ready for classification
	and classifies the events as  SS or not-SS 

	ed_a is the largest energy deposition
	ed_b is second largest energy deposition

	s_dist is the distance between these scatters

	"""
	# dummy
	df['class'] = 4

	ed_a_lower_lim = 5.
	ed_a_upper_lim = 100.

	ed_b_lower_lim = 5.

	s_dist_lim = 200.

	for index, event in df.iterrows():
		if (
			ed_a_lower_lim < event.ed_a and
			 event.ed_a < ed_a_upper_lim and 
			 ed_b_lower_lim < event.ed_b and 
			 event.s_dist > s_dist_lim
			):
			event['class'] = 1
		else:
			event['class'] = 2
	return df

# -----------------------------------------------------------------
# need a minitree to extract features from root file

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
    peak_fields = ['area', 'range_50p_area', 'x', 'y', 'z', 'delay_is2','delay_is1','goodness_of_fit']
    main_fields = ['s2','s2_delay_is1','s2_range_50p_area',
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
                elif field == 's2_delay_is1':
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


def extract_minitree(processed_filename, processed_filepath):

	""" Take processed fax data and turn it into useful
	dataframe to add classification """

	hax.init(experiment='XENON1T',
         use_runs_db=False,
         pax_version_policy='loose',
         main_data_paths=[processed_filepath],
         minitree_paths = [processed_filepath])

	pr_df = hax.minitrees.load(processed_filename, [Peaks])

	return pr_df

""" Inputs """

input_filepath = str(sys.argv[1])
input_filename = str(sys.argv[2])
processed_filepath = str(sys.argv[3])
processed_filename = str(sys.argv[4])
output_filepath = str(sys.argv[5])
output_filename = processed_filename


for df in read_root(input_filepath+input_filename, chunksize = 2000):
	pre_fax = df
	break

class_scatter = better_df(pre_fax)
class_scatter = classify_df(class_scatter)

fax_processed_df = extract_minitree(processed_filename, processed_filepath)

fax_processed_df['class'] = class_scatter['class']


# export the classified, ready to use data
fax_processed_df.to_pickle(output_filepath+output_filename+".pkl")
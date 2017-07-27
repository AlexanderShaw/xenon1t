"""

Important functions for the learner

prepare_data :  input - classified dataframe to fill nan's, reduce
						dimensionality, and pretty up for
						the learner to work on.
				output - prepared dataframe for the learner

"""

import pandas as pd
import numpy as np
import os

# ------------------------------------------------ prepare_data

def prepare_data(df):
	"""
	- df is input dataframe, with classification
	- nafill is an array of values to fill the NaN's with

	"""

	# Need to add a class feature if df doesn't have one.
	# The df shouldn't have one if it is a set one wants
	# to predict, but giving it a dummy value will make it all easier.
	# 
	# It should already have a class feature if it is
	# the training set.

	if 'class' not in list(df):
		df['class'] = 5

	nafill = pd.DataFrame({'class' : 0.0,
	 'cs1' : 0.0,
	 'cs2': 0.0,
	 'drift_time': 0.0,
	 'event_number': 0.0,
	 'run_number': 0.0,
	 's1': 0.0,
	 's1_area_fraction_top': 0.0,
	 's1_pattern_fit': 0.0,
	 's1_range_50p_area': 0.0,
	 's1_rise_time': 0.0,
	 's2': 0.0,
	 's2_1_area': 0.0,
	 's2_1_area_fraction_top': 0.0,
	 's2_1_corrected_area': 0.0,
	 's2_1_delay_is1': 0.0,
	 's2_1_delay_is2': 0.0,
	 's2_1_range_50p_area': 0.0,
	 's2_1_goodness_of_fit': 0.0,
	 's2_1_interior_split_fraction': 0.0,
	 's2_1_x': 1000.0,
	 's2_1_y': 1000.0,
	 's2_1_z': 1000.0,
	 's2_2_area': 0.0,
	 's2_2_area_fraction_top': 0.0,
	 's2_2_corrected_area': 0.0,
	 's2_2_delay_is1': 0.0,
	 's2_2_delay_is2': 0.0,
	 's2_2_range_50p_area': 0.0,
	 's2_2_goodness_of_fit': 0.0,
	 's2_2_interior_split_fraction': 0.0,
	 's2_2_x': 1000.0,
	 's2_2_y': 1000.0,
	 's2_2_z': 1000.0,
	 's2_3_area': 0.0,
	 's2_3_area_fraction_top': 0.0,
	 's2_3_corrected_area': 0.0,
	 's2_3_delay_is1': 0.0,
	 's2_3_delay_is2': 0.0,
	 's2_3_range_50p_area': 0.0,
	 's2_3_goodness_of_fit': 0.0,
	 's2_3_interior_split_fraction': 0.0,
	 's2_3_x': 1000.0,
	 's2_3_y': 1000.0,
	 's2_3_z': 1000.0,
	 's2_4_area': 0.0,
	 's2_4_area_fraction_top': 0.0,
	 's2_4_corrected_area': 0.0,
	 's2_4_delay_is1': 0.0,
	 's2_4_delay_is2': 0.0,
	 's2_4_range_50p_area': 0.0,
	 's2_4_goodness_of_fit': 0.0,
	 's2_4_interior_split_fraction': 0.0,
	 's2_4_x': 1000.0,
	 's2_4_y': 1000.0,
	 's2_4_z': 1000.0,
	 's2_5_area': 0.0,
	 's2_5_area_fraction_top': 0.0,
	 's2_5_corrected_area': 0.0,
	 's2_5_delay_is1': 0.0,
	 's2_5_delay_is2': 0.0,
	 's2_5_range_50p_area': 0.0,
	 's2_5_goodness_of_fit': 0.0,
	 's2_5_interior_split_fraction': 0.0,
	 's2_5_x': 1000.0,
	 's2_5_y': 1000.0,
	 's2_5_z': 1000.0,
	 's2_area_fraction_top': 0.0,
	 's2_pattern_fit': 0.0,
	 's2_range_50p_area': 0.0,
	 's2_rise_time': 0.0,
	 'x': 1000.0,
	 'y': 1000.0,
	 'z': 1000.0}, index = [df.index])

	df = df.fillna(nafill)

	keep = ['class','s2','s2_range_50p_area', 's2_1_x','s2_1_y','s2_2_x','s2_2_y',
	        's2_1_z','s2_2_z',
	        's2_1_delay_is2','s2_2_delay_is2',
	        's2_1_range_50p_area', 's2_1_area',
	       's2_2_range_50p_area', 's2_2_area']

	todrop = ['s2_1_x','s2_1_y','s2_2_x','s2_2_y','s2_3_x','s2_3_y','s2_4_x','s2_4_y']

	toadd = ['s2_1_r','s2_2_r']


	# applying simple cuts and giving r values to all df's

	keep = sorted(list((set(keep) - set(todrop)).union(set(toadd))))

	df['s2_1_r'] = np.sqrt(df['s2_1_x']**2 + df['s2_1_y']**2)
	df['s2_2_r'] = np.sqrt(df['s2_2_x']**2 + df['s2_2_y']**2)

	df = df[keep]

	return df

#################################################################################

import my_lax_copy_sr1 as lc

def chosen_cuts(df):

	""" Choose and apply cuts defined in  my_lax_copy_sr1 """

	width = lc.S2Width()
	df = width.process(df)

	fc = lc.AmBeFiducial()
	df = fc.process(df)

	return df
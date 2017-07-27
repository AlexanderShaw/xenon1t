
# coding: utf-8

# In[2]:

from collections import OrderedDict

import numpy as np
import pandas as pd

#from pax import units, configuration, datastructure
#pax_config = configuration.load_configuration('XENON1T')

class Lichen(object):
    version = np.NaN

    def pre(self, df):
        return df

    def process(self, df):
        df = self.pre(df)
        df = self._process(df)
        df = self.post(df)
        return df

    def _process(self, df):
        raise NotImplementedError()

    def post(self, df):
        if 'temp' in df.columns:
            return df.drop('temp', 1)
        return df

    def name(self):
        return 'Cut%s' % self.__class__.__name__
    
class ManyLichen(Lichen):
    lichen_list = []
    def _process(self, df):
        for lichen in self.lichen_list:
            df = lichen.process(df)
        return df


# In[11]:

class S2Width(Lichen):
    def s2_width_model(self, z):
        diffusion_constant = 2.280000e-08
        v_drift = 1.440000e-04
        w0 = 309.7
        return np.sqrt(w0 ** 2 - 5.880 * diffusion_constant * z / v_drift ** 3)
    
    def relative_s2_width_bounds(self, s2, kind='high'):
        x = 0.5 * np.log10(np.clip(s2, 150, 4500 if kind == 'high' else 2500))
        if kind == 'high':
            return 3 - x
        elif kind == 'low':
            return -0.9 + x
        raise ValueError("kind must be high or low")
    
    def pre(self, df):
        df['temp'] = df['s2_range_50p_area'] / self.s2_width_model(df.z)
        return df
        
    def _process(self, df):
        df = df[df.temp <= self.relative_s2_width_bounds(df.s2,kind='high')]
        df = df[df.temp >= self.relative_s2_width_bounds(df.s2,kind='low')]
        return df


# In[4]:

class FiducialCylinder1T(Lichen):

    def pre(self, df):
        df['temp'] = np.sqrt(df['x'] * df['x'] + df['y'] * df['y'])
        return df
    
    def _process(self, df):
        df = df[(df.z > -92.9) & (df.z < -9) & (df.temp < 36.94)]
        return df


# In[5]:

class DAQVeto(ManyLichen):
    version = 1

    def __init__(self):
        self.lichen_list = [#self.EndOfRunCheck(),
                            self.BusyTypeCheck(),
                            self.BusyCheck(),
                            self.HEVCheck()]

    class EndOfRunCheck(Lichen):
        """Check that the event does not come in the last 21 seconds of the run
        """
        def _process(self, df):
            import hax          # noqa
            if not len(hax.config):
                # User didn't init hax yet... let's do it now
                hax.init()

            # Get the end times for each run
            # The datetime -> timestamp logic here is the same as in the pax event builder
            run_numbers = np.unique(df.run_number.values)
            run_end_times = [int(q.replace(tzinfo=pytz.utc).timestamp() * int(1e9))
                             for q in hax.runs.get_run_info(run_numbers.tolist(), 'end')]
            run_end_times = {run_numbers[i]: run_end_times[i]
                             for i in range(len(run_numbers))}

            # Pass events that occur before (end time - 21 sec) of the run they are in
            df = df[df.apply(lambda row: row['event_time'] <
                             run_end_times[row['run_number']] - 21e9, axis=1)]
            return df

    class BusyTypeCheck(Lichen):
        def _process(self, df):
            df = df[((~(df['previous_busy_on'] < 60e9)) |
                       (df['previous_busy_off'] < df['previous_busy_on']))]
            return df

    class BusyCheck(Lichen):
        def _process(self, df):
            df = df[(abs(df['nearest_busy']) > df['event_duration'] / 2)]
            return df

    class HEVCheck(Lichen):
        def _process(self, df):
            df = df[ (abs(df['nearest_hev']) >
                                      df['event_duration'] / 2)]
            return df


# In[6]:

class AmBeFiducial(Lichen):
    
    def pre(self, df):
        source_position = (55.965311731903, 43.724893639103577, -50)
        df['temp'] = ((source_position[0] - df['x']) ** 2 +
                      (source_position[1] - df['y']) ** 2 +
                      (source_position[2] - df['z']) ** 2) ** 0.5
        return df
    
    def _process(self, df):
        df = df[(df.temp < 80) & (df.z > -83.45) & (df.z < -13.45) &
                (np.sqrt(df.x * df.x + df.y * df.y) < 42.00)
               ]
        return df


# In[7]:

class InteractionExists(Lichen):
    def _process(self, df):
        df = df[df.cs1 > 0]
        return df


# In[8]:

class S2AreaFractionTop(Lichen):
    def _not_in_use_process(self, df):
        df = df[(df.s2_area_fraction_top > 0.5) &
                (df.s2_area_fraction_top < 0.72)
               ]
        return df
    
    def _process(self, df):
        df['s2aft_up_lim'] = (0.6177399420527526 + 3.713166211522462e-08 * df.s2 + 0.5460484265254656 / np.log(df.s2))
        df['s2aft_low_lim'] = (0.6648160611018054 - 2.590402853814859e-07 * df.s2 - 0.8531029789184852 / np.log(df.s2))
        df = df[df.s2_area_fraction_top < df.s2aft_up_lim]
        df = df[df.s2_area_fraction_top > df.s2aft_low_lim]
        df.drop(['s2aft_up_lim','s2aft_low_lim'], axis = 1, inplace = True)
        return df


# In[9]:

class S2PatternLikelihood(Lichen):
    def _process(self, df):
        df = df[df.s2_pattern_fit < 75 + 10 * df.s2 ** 0.45]
        return df


# In[10]:

class S2Threshold(Lichen):
    def _process(self, df):
        df = df[df.s2 > 200]
        return df


# In[ ]:




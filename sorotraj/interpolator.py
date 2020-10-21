#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
import os


class Interpolator:
    def __init__(self, trajectory):
        self.import_traj(trajectory)
        self.final_time = None


    # Import the trajectory and parse it for interpolation
    def import_traj(self, trajectory):
        prefix = trajectory['prefix']
        suffix = trajectory['suffix']

        if prefix is not None:
            prefix = np.array(prefix)
            self.prefix =  {'time' : prefix[:,0]  ,
                            'values': prefix[:,1:]  }
        else:
            self.prefix = None


        if suffix is not None:
            suffix = np.array(suffix)
            self.suffix =  {'time' : suffix[:,0]  ,
                            'values': suffix[:,1:]  }
        else:
            self.suffix =  None

        main   = np.array(trajectory['setpoints'])
        self.main =    {'time' : main[:,0]  ,
                        'values': main[:,1:]  }


    # Generate an interpolation function based on the trajectory
    def get_interp_function(self, num_reps=1, speed_factor = 1.0, invert_direction=False, as_list=False):
        if num_reps<1:
            raise ValueError("The number of reps must be greater than 0")

        if speed_factor<=0:
            raise ValueError("The speed factor must be greater than 0")

        num_reps = int(num_reps)
        
        # Duplicate the time vector
        time_use = self.main['time'][1:]
        len_traj = time_use.shape[0]
        times  = np.zeros((len_traj)*num_reps)

        for idx in range(num_reps):
            new_times=time_use + idx*time_use[-1]
            times[idx*len_traj: (idx+1)*len_traj]= new_times
        times = np.insert(times, 0, self.main['time'][0])
        times = times/speed_factor

        # Duplicate the values
        values = np.tile(self.main['values'][1:,:],(num_reps,1))
        values = np.insert(values, 0, self.main['values'][0,:], axis=0)

        # Insert the prefix and suffix
        if self.prefix is not None:
            times=times+self.prefix['time'][-1]
            times = np.insert(times, range(self.prefix['time'].shape[0]), self.prefix['time'], axis=0)
            values = np.insert(values, range(self.prefix['values'].shape[0]), self.prefix['values'], axis=0)

        if self.suffix is not None:
            times = np.append(times, self.suffix['time']+times[-1], axis=0)
            values = np.append(values, self.suffix['values'], axis=0)

        self.final_time = times[-1]

        if isinstance(invert_direction,list):
            values[:,invert_direction] = -values[:,invert_direction]
        elif invert_direction == True:
            values = -values

        # Make an the interpolation function
        if as_list:
            # Make an array of 1D functions, one for each channel
            num_channels = values.shape[1]
            self.interp_fun = []
            for idx in range(num_channels):
                self.interp_fun.append(interp1d(times,values[:,idx],bounds_error=False,fill_value=values[-1,idx], axis=0))
        else:
            # Make one function returning an array of channel values
            self.interp_fun = interp1d(times,values,bounds_error=False,fill_value=values[-1,:], axis=0)

        return self.interp_fun


    # Generate a function to return the current cycle number given times
    def get_cycle_function(self, num_reps=1, speed_factor = 1.0, invert_direction=False, as_list=False):

        if self.prefix is not None:
            prefix_dur = self.prefix['time'][-1]
        else:
            prefix_dur = 0.0

        if self.suffix is not None:
            suffix_dur = self.suffix['time'][-1]
        else:
            suffix_dur = 0.0

        main_dur = self.main['time'][-1]
        total_main_dur = main_dur/speed_factor*num_reps

        def cycle_fn(t):
            if not isinstance(t, list):
                tl=np.asarray([t])
            else:
                tl=np.asarray(t)
            
            out=[]
            for t_curr in tl:
                if t_curr<prefix_dur:
                    out.append(-2)
                elif t_curr>prefix_dur+total_main_dur:
                    out.append(-1)
                else:
                    t_test = t_curr-prefix_dur
                    out.append(np.ceil(num_reps*t_test/total_main_dur)-1)
            
            if len(tl)==1 :
                out = out[0]
            return out

        return cycle_fn


    # Get the final time of the interpolation
    def get_final_time(self):
        return self.final_time
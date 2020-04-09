#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
import os


class Interpolator:
    def __init__(self, trajectory):
        self.import_traj(trajectory)


    # Import the trajectory and parse it for interpolation
    def import_traj(self, trajectory):
        prefix = np.array(trajectory['prefix'])
        suffix = np.array(trajectory['suffix'])
        main   = np.array(trajectory['setpoints'])


        self.prefix =  {'time' : prefix[:,0]  ,
                        'values': prefix[:,1:]  }

        self.suffix =  {'time' : suffix[:,0]  ,
                        'values': suffix[:,1:]  }

        self.main =    {'time' : main[:,0]  ,
                        'values': main[:,1:]  }


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
        times=times+self.prefix['time'][-1]
        times = np.insert(times, range(self.prefix['time'].shape[0]), self.prefix['time'], axis=0)
        times = np.append(times, self.suffix['time']+times[-1], axis=0)

        values = np.insert(values, range(self.prefix['values'].shape[0]), self.prefix['values'], axis=0)
        values = np.append(values, self.suffix['values'], axis=0)

        if invert_direction:
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
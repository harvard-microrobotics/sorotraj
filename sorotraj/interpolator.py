#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
import os
import sorotraj
import copy


class Interpolator:
    """
    Trajectory interpolator

    Attributes
    ----------
    trajectory : dict
        Trajectory to interpolate

    Examples
    --------
    >>> interp = sorotraj.Interpolator(traj)
    ... actuation_fn = interp.get_interp_function(
    ...     num_reps=1,
    ...     speed_factor=1.0,
    ...     invert_direction=False)
    ... interp.get_final_time()
    8.0
    """
    def __init__(self, trajectory):
        self._import_traj(trajectory)
        self.final_time = None


    def _import_traj(self, trajectory):
        """
        Parse a trajectory for interpolation

        Parameters
        ----------
        trajectory : dict
            Trajectory to interpolate

        Raises
        ------
        ValueError
            If the 'main' trajectory segment is None 
        """
        prefix = trajectory.get('prefix', None)
        suffix = trajectory.get('suffix', None)
        main = trajectory.get('setpoints', None)

        if main is not None:
            main   = np.array(main)
            self.main =    {'time' : main[:,0]  ,
                            'values': main[:,1:],
                            'max_time': np.max(main[:,0]),
                            'min_time': np.min(main[:,0]) }
        else:
            self.main = None


        if prefix is not None:
            prefix = np.array(prefix)
            self.prefix =  {'time' : prefix[:,0]  ,
                            'values': prefix[:,1:],
                            'max_time': np.max(prefix[:,0]),
                            'min_time': np.min(prefix[:,0])}
        else:
            self.prefix = None


        if suffix is not None:
            suffix = np.array(suffix)
            self.suffix =  {'time' : suffix[:,0]  ,
                            'values': suffix[:,1:],
                            'max_time': np.max(suffix[:,0]),
                            'min_time': np.min(suffix[:,0])  }
        else:
            self.suffix =  None

        self.trajectory_unpacked = {
            'main':self.main,
            'prefix':self.prefix,
            'suffix':self.suffix,
        }


    def get_interp_function(self, num_reps=1, speed_factor = 1.0, invert_direction=False, as_list=False):
        #self._build_interp_function(num_reps, speed_factor, invert_direction, as_list)
        self.build_traj_funtions(num_reps=1, speed_factor = 1.0, invert_direction=False)
        return self.interp_fun

    # Generate an interpolation function based on the trajectory
    def _build_interp_function(self, num_reps=1, speed_factor = 1.0, invert_direction=False, as_list=False):
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
                self.interp_fun.append(interp1d(times,values[:,idx],bounds_error=False,fill_value=(values[0,idx],values[-1,idx]), axis=0))
        else:
            # Make one function returning an array of channel values
            #self.interp_fun = interp1d(times,values,bounds_error=False,fill_value=values[-1,:], axis=0)
            self.interp_fun = interp1d(times,values,bounds_error=False,fill_value=(values[0,:],values[-1,:]), axis=0)
    

    def get_traj_function(self, num_reps=1, speed_factor = 1.0, invert_direction=False):

        traj_interp = TrajectoryInterpolator(self.trajectory_unpacked,
                                                num_reps, speed_factor, invert_direction)

        interp_fun = traj_interp.get_traj_function()
        final_time = traj_interp.get_final_time()
        return interp_fun, final_time


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



class TrajectoryInterpolator:
    def __init__(self, traj_unpacked, num_reps=1, speed_factor = 1.0, invert_direction=False):
        self.main = copy.deepcopy(traj_unpacked['main'])
        self.prefix = copy.deepcopy(traj_unpacked['prefix'])
        self.suffix = copy.deepcopy(traj_unpacked['suffix'])

        # Invert values
        if isinstance(invert_direction,list):
            self.main['values'][:,invert_direction] = -self.main['values'][:,invert_direction]
            self.prefix['values'][:,invert_direction] = -self.prefix['values'][:,invert_direction]
            self.suffix['values'][:,invert_direction] = -self.suffix['values'][:,invert_direction]
        elif invert_direction == True:
            self.main['values'] = -self.main['values']
            self.prefix['values'] = -self.prefix['values']
            self.suffix['values'] = -self.suffix['values']

        # Generate interpolation functions
        if self.main is not None:
            self.main_duration=(self.main['time'][-1])*num_reps
            self.interp_main = sorotraj.WrappedInterp1d(self.main['time'],
                                                    self.main['values'],
                                                    axis=0)
        else:
            self.main_duration=0
    
        if self.prefix is not None:
            self.prefix_duration=self.prefix['time'][-1] - self.prefix['time'][0]
            self.interp_prefix = interp1d(self.prefix['time'],
                                      self.prefix['values'],
                                      bounds_error=False,
                                      fill_value=(self.prefix['values'][0,:],self.prefix['values'][-1,:]),
                                      axis=0)
        else:
            self.prefix_duration=0
        
        if self.suffix is not None: 
            self.suffix_duration=self.suffix['time'][-1] - self.suffix['time'][0]
            self.interp_suffix = interp1d(self.suffix['time'],
                                        self.suffix['values'],
                                        bounds_error=False,
                                        fill_value=(self.suffix['values'][0,:],self.suffix['values'][-1,:]),
                                        axis=0)
        else:
            self.suffix_duration=0

        if not(self.prefix_duration + self.main_duration + self.suffix_duration >0):
            raise ValueError("The trajectory is empty (it is 0 seconds long)")


    def get_traj_function(self):
        return self.traj_function


    def traj_function(self, x0):
        # TODO make this responsive to cases where only one or two of the trajectory segments exist
        if x0<self.prefix_duration:
            return self.interp_prefix(x0)
        elif x0>(self.prefix_duration+self.main_duration):
            return self.interp_suffix(x0)
        else:
            return self.interp_main(x0)

    def get_final_time(self):
        return self.prefix_duration + self.main_duration + self.suffix_duration


class WrappedInterp1d:
    def __init__(self, x, y, **kwargs):
        self.x_min = np.min(x)
        self.x_max = np.max(x)
        self.x_diff = self.x_max - self.x_min
        self.interp_fun = interp1d(x,y,**kwargs)
    
    def get_function(self):
        return self.wrapped_interp1d

    def min_wrap(self, x0):
        underflow = self.x_min-x0
        num_underflows = np.floor(underflow/self.x_diff)
        leftover = underflow - num_underflows*self.x_diff
        x0 = self.x_max - leftover
        return x0

    def max_wrap(self, x0):
        overflow = x0-self.x_max
        num_overflows = np.floor(overflow/self.x_diff)
        leftover = overflow - num_overflows*self.x_diff
        x0 = self.x_min + leftover
        return x0

    def wrapped_interp1d(self, x0):
        x0 = np.array(x0)
        shape = x0.shape
        if len(shape) ==1:
            pass
        elif len(shape)==2:
            if not np.any(shape == 1):
                raise ValueError("Input array must be 1D")
        else:
            raise ValueError("Input array must be 1D")

        min_check = x0<self.x_min
        max_check = x0>self.x_max

        x0_wrapped = np.array(x0)
        x0_wrapped[min_check] = self.min_wrap(x0[min_check])
        x0_wrapped[max_check] = self.max_wrap(x0[max_check])
            
        return self.interp_fun(x0_wrapped)
        
        
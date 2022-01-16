#!/usr/bin/env python

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
import os
import sorotraj
import copy
import warnings


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
        trajectory = copy.deepcopy(trajectory)
        if trajectory.get('setpoints',False):
            trajectory['main'] = trajectory.pop('setpoints')

        # Unpack the trajectory and make sure there is at least one waypoint
        # in each component 
        key_list = ['prefix', 'main', 'suffix']
        self.trajectory_unpacked = {}
        for key in key_list:
            curr_unpacked = None
            curr_data = trajectory.get(key, None)
            if curr_data is not None:
                if len(curr_data)>0:
                    curr_data   = np.array(curr_data)
                    curr_unpacked =    {'time' : curr_data[:,0]  ,
                                    'values': curr_data[:,1:],
                                    'max_time': np.max(curr_data[:,0]),
                                    'min_time': np.min(curr_data[:,0]) }
                
            self.trajectory_unpacked[key] = curr_unpacked
        

    def get_interp_function(self, num_reps=1, speed_factor = 1.0, invert_direction=False, as_list=None):
        """
        Get a trajectory interpolation function with the specified parameters

        (This function exists for backward compatibillity. In the future, use
        "get_traj_function" instead.)

        Parameters
        ----------
        num_reps : int
            Number of times to repeat the "main" trajectory segment
        speed_factor : float
            Speed multiplier (times are multiplied by inverse of this)
        invert_direction : Union[bool, list]
            Invert the sign of the interpolated values. If True, all signs are
            flipped. If list, invert_direction is treated as a list of indices.

        Returns
        -------
        traj_function
            The trajectory interpolation function

        Raises
        ------
        ValueError
            If num_reps is less than 0, or if speed_factor is 0 or less
        """
        num_reps=int(num_reps)
        if num_reps<0:
            raise ValueError("The number of reps must be at least 0")

        if speed_factor<=0:
            raise ValueError("The speed factor must be strictly greater than 0")

        if as_list is not None:
            raise DeprecationWarning('Using "as_list" to get a list of interpolation functions is no longer supported')

        interp_fun, final_time = self.get_traj_function(num_reps=1, speed_factor = 1.0, invert_direction=False)
        self.interp_fun = interp_fun
        self.final_time = final_time
        return self.interp_fun


    def get_traj_function(self, num_reps=1, speed_factor = 1.0, invert_direction=False):
        """
        Get a trajectory interpolation function with the specified parameters

        Parameters
        ----------
        num_reps : int
            Number of times to repeat the "main" trajectory segment
        speed_factor : float
            Speed multiplier (times are multiplied by inverse of this)
        invert_direction : Union[bool, list]
            Invert the sign of the interpolated values. If True, all signs are
            flipped. If list, invert_direction is treated as a list of indices.

        Returns
        -------
        traj_function : function
            The trajectory interpolation function
        final_time : float
            The end time of the trajectory
        """
        traj_interp = TrajectoryInterpolator(self.trajectory_unpacked,
                                                num_reps, speed_factor, invert_direction)

        interp_fun = traj_interp.get_traj_function()
        final_time = traj_interp.get_final_time()
        return interp_fun, final_time


    def get_cycle_function(self, num_reps=1, speed_factor = 1.0, invert_direction=False, as_list=None):
        """
        Get a function to return the current cycle number given time as an input

        Parameters
        ----------
        num_reps : int
            Number of times to repeat the "main" trajectory segment
        speed_factor : float
            Speed multiplier (times are multiplied by inverse of this)
        invert_direction : Union[bool, list]
            Invert the sign of the interpolated values. If True, all signs are
            flipped. If list, invert_direction is treated as a list of indices.

        Returns
        -------
        cycle_function : function
            The cycle function
        final_time : float
            The end time of the trajectory
        """

        if as_list is not None:
            raise DeprecationWarning('Using "as_list" to get a list of interpolation functions is no longer supported')

        if self.trajectory_unpacked['prefix'] is not None:
            prefix_dur = self.trajectory_unpacked['prefix']['time'][-1]
        else:
            prefix_dur = 0.0

        if self.trajectory_unpacked['suffix'] is not None:
            suffix_dur = self.trajectory_unpacked['suffix']['time'][-1]
        else:
            suffix_dur = 0.0

        main_dur =self.trajectory_unpacked['main']['time'][-1]
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


    def get_final_time(self):
        """
        Get the final time of the most-recent interpolator

        (This function exists for backward compatibillity. In the future, obtain the
        final time from the "get_traj_function" instead.)

        Parameters
        ----------


        """
        return self.final_time



class TrajectoryInterpolator:
    """
    A trajectory interpolator based on specified parameters

    Parameters
    ----------
    traj_unpacked : dict
        Unpacked trajectory object (dict where keys are trajectory components
        with fields "time" and ""values")
    num_reps : int, optional
        Number of times to repeat the "main" trajectory segment
    speed_factor : float, optional
        Speed multiplier (times are multiplied by inverse of this)
    invert_direction : Union[bool, list], optional
        Invert the sign of the interpolated values. If True, all signs are
        flipped. If list, invert_direction is treated as a list of indices.
    fill_value : Union[list, np.ndarray], optional
        Default value of signals (only used when prefix and main are empty 
        in the trajectory)

    Raises
    ------
    ValueError
        If all trajectory components are empty
    """
    def __init__(self, traj_unpacked, num_reps=1, speed_factor = 1.0, invert_direction=False, fill_value=None):
        self.main = copy.deepcopy(traj_unpacked['main'])
        self.prefix = copy.deepcopy(traj_unpacked['prefix'])
        self.suffix = copy.deepcopy(traj_unpacked['suffix'])

        seg_list = [self.prefix, self.main, self.suffix]
        segs_valid = [None]*len(seg_list)
        num_empty = 0
        for idx, segment in enumerate(seg_list):
            if segment is None:
                segs_valid[idx] = False
            else:
                segs_valid[idx] = True

        if sum(segs_valid)<3:
            warnings.warn("One or more trajectory components is empty.", UserWarning)
        
        if sum(segs_valid)==0:
            raise ValueError("All trajectory components are empty. Please check your trjectory definition")

        first_valid = segs_valid.index(True)
        num_channels = seg_list[first_valid]['values'].shape[-1]

        if fill_value is None:
            fill_value=[0]*num_channels

        self.fill_value = np.array(fill_value)

        # Invert values
        if isinstance(invert_direction,list):
            self.main['values'][:,invert_direction] = -self.main['values'][:,invert_direction]
            self.prefix['values'][:,invert_direction] = -self.prefix['values'][:,invert_direction]
            self.suffix['values'][:,invert_direction] = -self.suffix['values'][:,invert_direction]
        elif invert_direction == True:
            self.main['values'] = -self.main['values']
            self.prefix['values'] = -self.prefix['values']
            self.suffix['values'] = -self.suffix['values']

        self._generate_interp_functions(num_reps, speed_factor)

    def _generate_interp_functions(self, num_reps, speed_factor):
        """
        Generate nessecary interpolator functions

        Parameters
        ----------
        traj_unpacked : dict
            Unpacked trajectory object (dict where keys are trajectory components
            with fields "time" and ""values")
        num_reps : int, optional
            Number of times to repeat the "main" trajectory segment
        speed_factor : float, optional
            Speed multiplier (times are multiplied by inverse of this)

        Raises
        ------
        ValueError
            If the trajectory is 0 seconds long
        """
        # Generate interpolation functions
        if self.main is not None:
            self.main_duration=(self.main['time'][-1])*num_reps
            interpolator = WrappedInterp1d(self.main['time'],
                                                    self.main['values'],
                                                    axis=0)
            self.interp_main = interpolator.get_function()
        else:
            self.main_duration=0
            self.interp_main = None
    
        if self.prefix is not None:
            self.prefix_duration=self.prefix['time'][-1] - self.prefix['time'][0]
            self.interp_prefix = interp1d_patched(self.prefix['time'],
                                      self.prefix['values'],
                                      bounds_error=False,
                                      fill_value=(self.prefix['values'][0,:],self.prefix['values'][-1,:]),
                                      axis=0)
        else:
            self.prefix_duration=0
            self.interp_prefix = None
        
        if self.suffix is not None:
            # Insert last waypoint before suffix into time 0 in the suffix
            if self.suffix['time'][0]>0:
                self.suffix['time'] = np.hstack(([0], self.suffix['time']))
                if self.main is not None:
                    self.suffix['values'] = np.vstack((self.main['values'][-1,:], self.suffix['values']))
                elif self.prefix is not None:
                    self.suffix['values'] = np.vstack((self.prefix['values'][-1,:], self.suffix['values']))
                else:
                    self.suffix['values'] = np.vstack((self.fill_value, self.suffix['values']))

            self.suffix_duration=self.suffix['time'][-1] - self.suffix['time'][0]
            self.interp_suffix = interp1d_patched(self.suffix['time'],
                                        self.suffix['values'],
                                        bounds_error=False,
                                        fill_value=(self.suffix['values'][0,:],self.suffix['values'][-1,:]),
                                        axis=0)
        else:
            self.suffix_duration=0
            self.interp_suffix = None

        if not(self.prefix_duration + self.main_duration + self.suffix_duration >0):
            raise ValueError("The trajectory is empty (it is 0 seconds long)")


    def get_traj_function(self):
        """
        Get the trajectory function

        Returns
        -------
        traj_function : function
            The trajectory interpolation meta-function.
        """
        return self.traj_function


    def traj_function(self, x0):
        """
        The trajectory interpolation function

        Parameters
        ----------
        x0 : Union[float, list, np.ndarray]

        Returns
        -------
        output : np.ndarray
            The trajectory at the given time point(s)

        Raises
        ------
        ValueError
            If input is not a 1D array-like object
        RuntimeError
            If the length of the output does not equal the length of the input
        """
        x0 = np.array(x0)
        shape = x0.shape
        if x0.ndim ==0:
            pass
        elif x0.ndim ==1:
            pass
        elif x0.ndim ==2:
            if not np.any(shape == 1):
                raise ValueError("Input array must be 1D, got %dD"%(x0.ndim))
        else:
            raise ValueError("Input array must be 1D, got %dD"%(x0.ndim))

        prefix_check = x0<self.prefix_duration
        suffix_check = x0>(self.prefix_duration+self.main_duration)
        main_check = np.logical_not(np.logical_or(prefix_check, suffix_check))

        output = []
        if self.interp_prefix is not None:
            prefix_vals = self.interp_prefix(x0[prefix_check])
            if len(output)>0:
                output = np.vstack((output,prefix_vals))
            else:
                output = prefix_vals

        if self.interp_main is not None:
            main_vals = self.interp_main(x0[main_check])
            if len(output)>0:
                output = np.vstack((output,main_vals))
            else:
                prefix_vals = np.tile(self.interp_main(min(self.main['time'])),(len(x0[prefix_check]),1))
                output = np.vstack((prefix_vals,main_vals))
        
        
        if self.interp_suffix is not None:
            suffix_vals = self.interp_suffix(x0[suffix_check])
            if len(output)>0:
                output = np.vstack((output,suffix_vals))
            else:
                prefix_vals = np.tile(self.interp_suffix(min(self.suffix['time'])),(len(x0[prefix_check]),1))
                main_vals = np.tile(self.interp_suffix(min(self.suffix['time'])),(len(x0[main_check]),1))
                output = np.vstack(prefix_vals, main_vals, suffix_vals)

        if len(output) != len(x0):
            raise RuntimeError("The length of the output does not equal the length of the input")

        return output


    def get_final_time(self):
        """
        Get the final time of the trajectory

        Returns
        -------
        final_time : float
            The final time
        """
        return self.prefix_duration + self.main_duration + self.suffix_duration


class WrappedInterp1d:
    """
    Create a wrapping 1D interpolator

    Parameters
    ----------
    x : dict
        x points to use in interpolation
    y : int, optional
        Values to use for interpolation
    **kwargs : optional
        kwargs to pass to scipy.interpolate.interp1d().
    """
    def __init__(self, x, y, **kwargs):
        self.x_min = np.min(x)
        self.x_max = np.max(x)
        self.x_diff = self.x_max - self.x_min
        self.interp_fun = interp1d_patched(x,y,**kwargs)
    
    def get_function(self):
        """
        Get the wrapped interpolation function

        Returns
        -------
        wrapped_interp1d : function
            The wrapped interpolator function
        """
        return self.wrapped_interp1d

    def min_wrap(self, x0):
        """
        Calculate wrapped x values when x is less than the wrapping bounds

        Parameters
        ----------
        x0 : np.ndarray
            Values of x

        Returns
        -------
        wrapped_x0 : np.ndarray
            Values of x wrapped.
        """
        underflow = self.x_min-x0
        num_underflows = np.floor(underflow/self.x_diff)
        leftover = underflow - num_underflows*self.x_diff
        x0 = self.x_max - leftover
        return x0

    def max_wrap(self, x0):
        """
        Calculate wrapped x values when x is greater than the wrapping bounds

        Parameters
        ----------
        x0 : np.ndarray
            Values of x

        Returns
        -------
        wrapped_x0 : np.ndarray
            Values of x wrapped.
        """
        overflow = x0-self.x_max
        num_overflows = np.floor(overflow/self.x_diff)
        leftover = overflow - num_overflows*self.x_diff
        x0 = self.x_min + leftover
        return x0

    def wrapped_interp1d(self, x0):
        """
        The wrapped interp1d function. Input x0, return interpolated cyclic values 

        Parameters
        ----------
        x0 : Union[float, list, np.ndarray]
            Values of x where you want to interpolate
        
        Returns
        -------
        output : np.ndarray
            The interpolated values of y at the given time point(s)

        Raises
        ------
        ValueError
            If the input is not 1D
        """
        x0 = np.array(x0)
        shape = x0.shape
        if x0.ndim ==0:
            pass
        elif x0.ndim ==1:
            pass
        elif x0.ndim ==2:
            if not np.any(shape == 1):
                raise ValueError("Input array must be 1D, got %dD"%(x0.ndim))
        else:
            raise ValueError("Input array must be 1D, got %dD"%(x0.ndim))

        min_check = x0<self.x_min
        max_check = x0>self.x_max

        x0_wrapped = np.array(x0)
        x0_wrapped[min_check] = self.min_wrap(x0[min_check])
        x0_wrapped[max_check] = self.max_wrap(x0[max_check])
            
        return self.interp_fun(x0_wrapped)
        

def interp1d_patched(x, y, **kwargs):
    """
    Get a 1D interpolation function where single-length input data are handled

    When the length of x and y is greater than 1, interp1d is used. When the 
    legnth of x and y is 1, use the value of y for all values of x0.

    Parameters
    ----------
    x0 : Union[float, list, np.ndarray]
        Values of x where you want to interpolate
    
    Returns
    -------
    patched_interp1d : function
        The patched interp1d function (same way the regular interp1d works)
    """
    x, y = map(np.asarray, (x, y ))
    
    def fun (x0):
        if x.size == 1:
            return np.tile(y,(x0.size,1))
        else:
            return interp1d(x, y, **kwargs)(x0)
    
    return fun
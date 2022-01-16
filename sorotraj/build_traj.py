#!/usr/bin/env python

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import scipy.signal as signal
import matplotlib.pyplot as plt
import yaml
import sys
import os
import copy
import sorotraj


class TrajBuilder:
    """
    Trajectory builder

    Attributes
    ----------
    verbose : bool
        Flag used to turn on verbose printing

    Examples
    --------
    >>> def_file = 'examples/traj_setup/setpoint_traj_demo.yaml'
    ... builder = TrajBuilder()
    ... builder.load_traj_def(def_file)
    ... traj = builder.get_trajectory()
    ... out_file = 'examples/traj_built/setpoint_traj_demo.traj'
    ... builder.save_traj(out_file)
    """
    def __init__(self, verbose=False):
        self.filename = None
        self.definition = None
        self.full_trajectory = None
        self.verbose=verbose


    def load_traj_def(self, filename):
        """
        Load a trajectory definition from a file.

        Once lodaed, the trajectory definition is set, and the trajectory is built.

        Parameters
        ----------
        filename : str
            The file to load

        Raises
        ------
        ValueError
            If the filename is not of type 'str'
        """
        if not isinstance(filename, str):
            raise ValueError("filename must be a string")

        # Strip yaml file prefix
        filename=filename.replace('.yaml','')

        # Save the filename with no prefix
        self.filename = os.path.abspath(filename+'.yaml').replace('.yaml','')

        # Read in the setpoint file
        self.definition = sorotraj.load_yaml(filename+'.yaml')

        if self.verbose:  
            print('Trajectory Loaded: %s'%(filename+'.yaml'))

        # Build the trajectory
        self.build_traj()
    

    def build_traj(self):
        """
        Build the current trajectory

        Raises
        ------
        RuntimeError
            If the trajectory definition has not been set
        """
        if self.definition is None:
            raise RuntimeError("Trajectory definition has not been set or loaded.")

        self.settings = self.definition.get("settings",None)
        self.config = self.definition.get("config",None)
        
        self.traj_type = str(self.settings.get("traj_type"))
        self.subsample_num = self.config.get("subsample_num")

        # Generate the trajectory based on the file definition
        self._expand_traj()
        self._validate_traj()



    def set_definition(self, definition):
        """
        Set the trajectory definition manually.

        The trajectory definition is set, and the trajectory is rebuilt.

        Parameters
        ----------
        definition : dict
            The trajectory definition to set

        Raises
        ------
        ValueError
            If the trajectory definition is not of type 'dict'
        """
        if not isinstance(definition, dict):
            raise ValueError("Trajectory definition must be of type 'dict'.")

        self.definition = copy.deepcopy(definition)
        self.build_traj()


    # Pass out the trajectory definition
    def get_definition(self, use_copy=False):
        """
        Get the trajectory definition.

        Parameters
        ----------
        use_copy : bool
            Decide whether to pass the trajectory by referece. If True,
            the actual trajectory object is returned, otherwise a copy
            of the trajectory is returned.

        Returns
        -------
        trajectory_definition : dict
            The trajectory definition

        Raises
        ------
        RuntimeError
            If the trajectory definition is not set set
        """
        if self.definition is None:
            raise RuntimeError("Trajectory definition has not been set or loaded.")
        
        if use_copy:
            return self.definition
        else:
            return copy.deepcopy(self.definition)

    
    def get_trajectory(self, use_copy=False):
        """
        Get the built trajectory.

        Parameters
        ----------
        use_copy : bool
            Decide whether to pass the trajectory by referece. If True,
            the actual trajectory object is returned, otherwise a copy
            of the trajectory is returned.

        Returns
        -------
        trajectory : dict
            The full trajectory
        
        Raises
        ------
        RuntimeError
            If the trajectory has not been built
        """
        if self.full_trajectory is None:
            raise RuntimeError("Trajectory has not been built.")

        if use_copy:
            return self.full_trajectory
        else:
            return copy.deepcopy(self.full_trajectory)


    def save_definition(self, filename):
        """
        Save the trajectory definition to a file.

        Parameters
        ----------
        filename : str
            The file to save

        Raises
        ------
        ValueError
            If the filename is not of type 'str'
        RuntimeError
            If the trajectory definition is not set
        """
        if not isinstance(filename, str):
            raise ValueError("filename must be a string")

        if self.definition is None:
            raise RuntimeError("Trajectory definition has not been set or loaded.")

        # Get rid of file extension if it exists
        basename = os.path.splitext(filename)
        filename = basename[0]

        # Save the trajectory definition
        sorotraj.save_yaml(self.definition, filename+".yaml")

        if self.verbose:    
            print('Trajectory Definition Saved: %s'%(filename+".yaml"))


    def save_traj(self, filename):
        """
        Save the trajectory to a file.

        Parameters
        ----------
        filename : str
            The file to save

        Raises
        ------
        ValueError
            If the filename is not of type 'str'
        RuntimeError
            If the trajectory has not been built
        """
        if not isinstance(filename, str):
            raise ValueError("filename must be a string")

        if self.full_trajectory is None:
            raise RuntimeError("Trajectory has not been built.")

        # Get rid of file extension if it exists
        basename = os.path.splitext(filename)
        filename = basename[0]

        # Save the trajectory
        sorotraj.save_yaml(self.full_trajectory, filename+".traj")

        if self.verbose:       
            print('Trajectory Saved: %s'%(filename+".traj"))


    def _validate_traj(self):
        """
        Validate the current trajectory

        Raises
        ------
        RuntimeError
            If the trajectory has not been built
        ValueError
            If the trajectory is invalid
        """
        if self.full_trajectory is None:
            raise RuntimeError("Trajectory has not been built.")

        traj_comp = self.get_traj_components()

        for traj_segment_key in traj_comp:
            times = traj_comp[traj_segment_key]['time']
            # Check that times do not contain duplicates
            if len(times) != len(set(times)):
                raise ValueError("Duplicate times found in trajectory segment: %s"%(traj_segment_key))

            # Check that times are monotonic
            if not np.all(np.diff(times) > 0):
                raise ValueError("Times are not monotonically increasing in trajectory segment: %s"%(traj_segment_key))




    def _expand_traj(self):
        """
        Generate the trajectory based on the definition

        Raises
        ------
        ValueError
            If the trajectory type is invalid
        RuntimeError
            If the trajectory fails to build
        """
        if self.traj_type == "waveform":
            success = self._do_waveform()
        elif self.traj_type == "interp":
            success = self._do_interp()
        elif self.traj_type == "direct":
            success = self._do_direct()
        else:
            success = False
            raise ValueError('Invalid trajectory type: %s.'%(self.traj_type))

        if not success:
            raise RuntimeError('Trajectory failed to build')

        if self.verbose:
            print("Trajectory has %d lines"%(len(self.full_trajectory['setpoints'])))


    # Generate a waveform trajectory
    def _do_waveform(self):
        """
        Generate a waveform trajectory based on the trajectory definition

        Raises
        ------
        ValueError
            If the waveform type is invalid
        """
        freq_in = self.config.get("waveform_freq")

        # Convert the frequency to floats
        if isinstance(freq_in, list):
            freq = []
            for item in freq_in:
                freq.append(float(item))
        else:
            freq = float(freq_in)


        press_max = np.array(self.config.get("waveform_max"))
        press_min = np.array(self.config.get("waveform_min"))
        waveform_type = self.config.get("waveform_type")


        setpts = self.config.get("setpoints",None)
        prefix = setpts.get("prefix",None)
        suffix = setpts.get("suffix",None)


        channels =  self.config.get("channels")
        num_cycles = int(self.config.get("num_cycles"))

        press_amp = (press_max-press_min)/2.0 *channels
        press_off = (press_max+press_min)/2.0 * channels

        # Make the waveform
        traj = []
        if waveform_type == "square-sampled":
            time_samp = np.linspace(0,num_cycles/freq, self.subsample_num+1 )
            traj = signal.square(2.0*np.pi * freq*time_samp)

        elif waveform_type == "square":
            time_samp_0 = np.linspace(0, num_cycles/freq, num_cycles +1)

            time_samp_1 = np.linspace(1/freq -0.51/freq, num_cycles/freq - 0.51/freq, num_cycles )
            time_samp = np.append(time_samp_0, time_samp_1)

            time_samp_2 = np.linspace(1/freq -0.50/freq ,num_cycles/freq - 0.50/freq, num_cycles)
            time_samp = np.append(time_samp, time_samp_2)

            time_samp_3 = np.linspace(1/freq -0.01/freq ,num_cycles/freq - 0.01/freq, num_cycles)
            time_samp = np.append(time_samp, time_samp_3)

            time_samp = np.sort(time_samp)

            traj = np.array([-1,-1,1,1])
            traj = np.tile(traj,int(num_cycles))
            traj = np.append(traj, traj[0])


        elif waveform_type == "sin":
            time_samp = np.linspace(0,num_cycles/freq, self.subsample_num+1 )
            traj = np.sin(2.0*np.pi * freq*time_samp)

        elif waveform_type == "cos-up":
            time_samp = np.linspace(0,num_cycles/freq, self.subsample_num+1 )
            traj = np.cos(2.0*np.pi * freq*time_samp)

        elif waveform_type == "cos-down":
            time_samp = np.linspace(0,num_cycles/freq, self.subsample_num+1 )
            traj = -np.cos(2.0*np.pi * freq*time_samp)

        elif waveform_type == "triangle":
            time_samp = np.linspace(0,num_cycles/freq, num_cycles*2 +1)
            traj = np.array([-1,1])
            traj = np.tile(traj,int(num_cycles))
            traj = np.append(traj, traj[0])

        elif waveform_type == "sawtooth-f":
            time_samp_1 = np.linspace(0,num_cycles/freq, num_cycles +1)
            time_samp_2 = np.linspace(1/freq -0.01/freq ,num_cycles/freq - 0.01/freq, num_cycles)
            time_samp = np.sort(np.append(time_samp_1, time_samp_2))

            traj = np.array([-1,1])
            traj = np.tile(traj,int(num_cycles))
            traj = np.append(traj, traj[0])

        elif waveform_type == "sawtooth-r":
            time_samp_1 = np.linspace(0,num_cycles/freq, num_cycles +1)
            time_samp_2 = np.linspace(1/freq -0.99/freq ,num_cycles/freq - 0.99/freq, num_cycles)
            time_samp = np.sort(np.append(time_samp_1, time_samp_2))

            traj = np.array([-1,1])
            traj = np.tile(traj,int(num_cycles))
            traj = np.append(traj, traj[0])

        else:
            raise ValueError("Invald waveform type: %s"%(waveform_type))


        #Stick everything together
        out_times = np.vstack(time_samp)
        traj_arr = np.reshape(np.array(traj),(1,len(traj)))
        press_amp_arr = np.reshape(np.array(press_amp),(1,len(press_amp)))
        print(traj_arr.shape)

        out_traj  = np.matmul(traj_arr.T,press_amp_arr)
        press_off_all=np.tile(press_off,(len(traj),1))
        out_traj = press_off_all +out_traj


        out_traj_whole = np.append(out_times,out_traj,axis=1)
        
        self.full_trajectory = {}
        self.full_trajectory['prefix'] = prefix
        self.full_trajectory['suffix'] = suffix
        self.full_trajectory['setpoints']   = out_traj_whole.tolist()

        return True


    def _do_direct(self):
        """
        Generate a trajectory directly from waypoints in the trajectory definition
        """
        # Get trajectory components
        setpts = self.config.get("setpoints",None)
        traj_setpoints = setpts.get("main",  None)
        prefix = setpts.get("prefix",None)
        suffix = setpts.get("suffix",None)

        # Copy trajectory components over to the final trajectory 
            
        self.full_trajectory = {}
        self.full_trajectory['prefix'] = prefix
        self.full_trajectory['suffix'] = suffix
        self.full_trajectory['setpoints']   = traj_setpoints

        return True


    def _do_interp(self):
        """
        Generate an interpolation trajectory based on the trajectory definition

        Raises
        ------
        ValueError
            If the interpolation type is invalid
        """
        interp_type = str(self.config.get("interp_type"))
        if interp_type == "none":
            return self._do_direct()


        setpts = self.config.get("setpoints",None)
        traj_setpoints = setpts.get("main",  None)
        prefix = setpts.get("prefix",None)
        suffix = setpts.get("suffix",None)
        


        if interp_type == "linear":
            t_step = (traj_setpoints[-1][0]-traj_setpoints[0][0])/self.subsample_num
            # Calculate the longer trajectory
            allOut=[]
            for idx in range(0,len(traj_setpoints)-1):
                seg = self._calculate_lin_segment(traj_setpoints[idx],traj_setpoints[idx+1],t_step)
                allOut.extend(seg)
            
            # Add the last entry to finish out the trajectory
            allOut.append(traj_setpoints[-1])


        

        elif interp_type == "cubic":
            t_step = (traj_setpoints[-1][0]-traj_setpoints[0][0])/self.subsample_num
            traj_setpoints = np.array(traj_setpoints)
            times=traj_setpoints[:,0]
            pres=traj_setpoints[:,1:]

            # Replace nearest points with the original knot points
            t_intermediate = np.linspace(times[0],times[-1], self.subsample_num+1 )
            idx = self._find_nearest(t_intermediate, times)
            t_intermediate[idx] = times

            # Generate a cubic spline
            cs = CubicSpline(times, pres, bc_type = "periodic")
            traj = cs(t_intermediate)
            allOut = np.insert(traj,0, t_intermediate ,axis = 1)
            allOut = allOut.tolist()

        else:
            raise ValueError("Invalid interpolation type: %s"%(interp_type))

        self.full_trajectory = {}
        self.full_trajectory['prefix'] = prefix
        self.full_trajectory['suffix'] = suffix
        self.full_trajectory['setpoints']   = allOut

        return True


    def _find_nearest(self, array, values):
        """
        Find the nearest points in interpolated trajectory

        Parameters
        ----------
        array : array-like
            The array to search through
        values : list
            list of values

        Returns
        -------
        idx : list
            list of indices in interpolated array where values are closest
        """
        array = np.asarray(array)
        idx=[]
        for val in values:
            idx.append(np.argmin(np.abs(array - val)))
        return idx


    def _calculate_lin_segment(self,start_point,end_point,t_step):
        """
        Calculate a linear trajectory segment

        Parameters
        ----------
        start_point : array-like
            1D array of start values (idx 0 is the time)
        end_point : list
            1D array of end values (idx 0 is the time)
        t_step : float
            Timestep

        Returns
        -------
        segment : list of lists
            List of trajectory points
        """
        # Calculate the linear interpolation time vector
        t_intermediate = np.arange(start_point[0],end_point[0],t_step)
        
        # Turn the incomming setpoints into arrays
        time_vec = np.asarray([end_point[0], start_point[0]])
        state_vec = np.transpose(np.asarray([end_point[1:], start_point[1:]]))
        
        # Create an interpolation function and use it
        fun = interp1d(time_vec,state_vec,fill_value="extrapolate")
        seg = np.transpose(fun(t_intermediate))
        
        # put the time back at the beginning of the array
        seg  = np.insert(seg,0,t_intermediate, axis=1)
        
        return seg.tolist()


    # Convert a trajectory using a conversion function
    def convert_traj(self,conversion_fun):
        """
        Convert a trajectory line-by-line using a conversion function

        Parameters
        ----------
        conversion_fun : function
            Conversion function taking in one trajectory line (list) and returning one line (list)

        Raises
        ------
        RuntimeError
            If the trajectory has not been built
        """

        if self.full_trajectory is None:
            raise RuntimeError("Trajectory has not been built.")
        
        full_trajectory_new = dict()
        for traj_segment_key in self.full_trajectory:
            # If the trajectory segment key is the metadata tag, skip it.
            if traj_segment_key == 'meta':
                continue

            traj_segment = self.full_trajectory[traj_segment_key]

            # If the trajectory segment is empty, pass that along
            if traj_segment is None:
                full_trajectory_new[traj_segment_key] = None
                continue

            # If the trajectory has lines, convert them
            full_trajectory_new[traj_segment_key] = []
            for line in traj_segment:
                full_trajectory_new[traj_segment_key].append(conversion_fun(line))

        full_trajectory_new['meta'] = {'converted': True}
        self.full_trajectory = full_trajectory_new


    def convert_definition(self,conversion_fun):
        """
        Convert a trajectory definition line-by-line using a conversion function.

        Trajectory definition of type 'direct' and 'interp' can be converted, but
        waveform trajectory definitions cannot.

        Parameters
        ----------
        conversion_fun : function
            Conversion function taking in one waypoint (list) and returning waypoint (list)

        Raises
        ------
        RuntimeError
            If the trajectory definition is not set
        RuntimeError
            If the trajectory type is incompatible (not direct or interp)
        """
        if self.definition is None:
            raise RuntimeError("Trajectory definition has not been set or loaded.")

        if not (self.traj_type in ["direct","interp"]):
            raise RuntimeError("Incompatible trajectory type: %s"%(self.traj_type))
        
        def_new = copy.deepcopy(self.definition)
        setpoints=copy.deep_copy(self.definition['config']['setpoints'])
        setpoints_new={}
        for traj_segment_key in setpoints:

            traj_segment = setpoints[traj_segment_key]

            # If the trajectory segment is empty, pass that along
            if traj_segment is None:
                setpoints_new[traj_segment_key] = None
                continue

            # If the trajectory has lines, convert them
            setpoints_new[traj_segment_key] = []
            for line in traj_segment:
                setpoints_new[traj_segment_key].append(conversion_fun(line))

        setpoints_new['meta'] = {'converted': True}
        def_new['config']['setpoints']=setpoints_new

        self.definition = def_new
        self.build_traj()


    def get_traj_components(self):
        """
        Get trajectory split into compoenents rather than in vector form

        This generates a dictionary with the same trajectory components as 
        a usual trajectory, but the values of each component are dictionaries
        with 'time' and 'values' rather than the usual list of lists.

        Raises
        ------
        RuntimeError
            If the trajectory has not been built
        """
        if self.full_trajectory is None:
            raise RuntimeError("Trajectory has not been built.")

        setpoints = self.full_trajectory['setpoints']
        prefix = self.full_trajectory['prefix']
        suffix = self.full_trajectory['suffix']

        out = {}
        out['setpoints'] = {}
        if setpoints is not None:
            out_traj_all = np.asarray(setpoints)
            out['setpoints']['time'] = out_traj_all[:,0]
            out['setpoints']['values'] = out_traj_all[:,1:]
        else:
            out['setpoints']['time'] = []
            out['setpoints']['values'] = []

        out['prefix'] = {}
        if prefix is not None:
            prefix_arr = np.asarray(prefix)
            out['prefix']['time'] = prefix_arr[:,0].tolist()
            out['prefix']['values'] = prefix_arr[:,1:].tolist()
        else:
            out['prefix']['time'] = []
            out['prefix']['values'] = []

        out['suffix'] = {}
        if suffix is not None:
            suffix_arr = np.asarray(suffix)        
            out['suffix']['time'] = suffix_arr[:,0].tolist()
            out['suffix']['values'] = suffix_arr[:,1:].tolist()
        else:
            out['suffix']['time'] = []
            out['suffix']['values'] = []

        return out


    # Plot the current trajectory 
    def plot_traj(self, fig_kwargs={}, plot_kwargs={}):
        """
        Plot the current trajectory (assuming 1 rep of the main segment)

        Parameters
        ----------
        fig_kwargs : Any
            Keyword args to pass to the matplotlib's figure function
        plot_kwargs : Any
            Keyword args to pass to the matplotlib's plotting function

        Raises
        ------
        RuntimeError
            If the trajectory has not been built
        """
        if self.full_trajectory is None:
            raise RuntimeError("Trajectory has not been built.")
        
        out_traj_all = np.asarray(self.full_trajectory['setpoints'])
        prefix = self.full_trajectory['prefix']
        suffix = self.full_trajectory['suffix']

        if prefix is not None:
            prefix_arr = np.asarray(prefix)
            # Update the times
            out_traj_all[:,0] = out_traj_all[:,0] + prefix_arr[-1,0]

            # Append to the array
            out_traj_all = np.append(prefix_arr,out_traj_all,axis=0);

        if suffix is not None:
            suffix_arr = np.asarray(suffix)        
            suffix_arr[:,0] = suffix_arr[:,0] + out_traj_all[-1,0]
            out_traj_all = np.append(out_traj_all,suffix_arr,axis=0);

        plt.figure(**fig_kwargs)
        plt.plot(out_traj_all[:,0],out_traj_all[:,1:],**plot_kwargs)
        plt.xlabel("Time (s)")
        plt.ylabel("Pressure (psi)")
        plt.show()        
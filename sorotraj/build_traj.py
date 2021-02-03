#!/usr/bin/env python

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import scipy.signal as signal
import matplotlib.pyplot as plt
import yaml
import sys
import os
import copy



traj_folder = "traj_setup"
out_folder  = "traj_built"


class TrajBuilder:
    def __init__(self, graph=True):
        self.filename = None
        self.full_trajectory = None
        self.graph = graph


    # Load the trajectory definition from a file
    def load_traj_def(self, filename):
        filename=filename.replace('.yaml','')

        self.filename = os.path.abspath(filename+'.yaml').replace('.yaml','')

        # Read in the setpoint file
        with open(filename+'.yaml') as f:
            # use safe_load instead of load
            inStuff = yaml.safe_load(f)
            f.close()

        print('Trajectory Loaded: %s'%(filename+'.yaml'))

        self.definition=inStuff
        self.settings = inStuff.get("settings",None)
        self.config = inStuff.get("config",None)
        
        self.traj_type = str(self.settings.get("traj_type"))
        self.subsample_num = self.config.get("subsample_num")

        # Generate the trajectory based on the file definition
        self.go()


    # Save yaml files of trajectory definitions.
    def save_definition(self, filename=None):
        if filename is None:
            if self.filename is None:
                print('You need to get trajectory settings before you can save')
                return
            else:
                filename=self.filename

        # Get rid of file extension if it exists
        basename = os.path.splitext(filename)
        filename = basename[0]

        dirname = os.path.dirname(filename+".yaml")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(filename+".yaml", 'w') as f:
            yaml.dump(self.definition, f, default_flow_style=None)
        
        print('Trajectory Definition Saved: %s'%(filename+".yaml"))


    # Save yaml files of trajectories generated.
    def save_traj(self, filename=None):
        if filename is None:
            if self.filename is None:
                print('You need to get trajectory settings before you can save')
                return
            else:
                filename=self.filename

        # Get rid of file extension if it exists
        basename = os.path.splitext(filename)
        filename = basename[0]

        dirname = os.path.dirname(filename+".traj")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(filename+".traj", 'w') as f:
            yaml.dump(self.full_trajectory, f, default_flow_style=None)
        
        print('Trajectory Saved: %s'%(filename+".traj"))


    # Generate the trajectory based on the file definition
    def go(self):
        if self.traj_type == "waveform":
            success = self.do_waveform()
        elif self.traj_type == "interp":
            success = self.do_interp()
        elif self.traj_type == "direct":
            success = self.do_direct()
        elif self.traj_type == "none":
            success = self.do_none()
        else:
            print('Please give your trajectory a valid type')
            success = False

        if success and self.graph:
            self.plot_traj()
            print("Trajectory has %d lines"%(len(self.full_trajectory['setpoints'] )))



    # Generate a waveform trajectory
    def do_waveform(self):
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
            return False


        #Stick everything together
        out_times = np.vstack(time_samp)
        out_traj  = np.matmul(np.asmatrix(traj).T,np.asmatrix(press_amp))
        press_off_all=np.tile(press_off,(len(traj),1))
        out_traj = press_off_all +out_traj


        out_traj_whole = np.append(out_times,out_traj,axis=1)
        
        self.full_trajectory = {}
        self.full_trajectory['prefix'] = prefix
        self.full_trajectory['suffix'] = suffix
        self.full_trajectory['setpoints']   = out_traj_whole.tolist()

        return True


        
    # Generate a direct trajectory straight from the setpoint list given
    def do_direct(self):
        setpts = self.config.get("setpoints",None)
        traj_setpoints = setpts.get("main",  None)
        prefix = setpts.get("prefix",None)
        suffix = setpts.get("suffix",None)
        interp_type = str(self.config.get("interp_type"))

            
        self.full_trajectory = {}
        self.full_trajectory['prefix'] = prefix
        self.full_trajectory['suffix'] = suffix
        self.full_trajectory['setpoints']   = traj_setpoints

        return True


    # Generate an interpolation trajectory
    def do_interp(self):
        interp_type = str(self.config.get("interp_type"))
        if interp_type == "none":
            return self.do_direct()


        setpts = self.config.get("setpoints",None)
        traj_setpoints = setpts.get("main",  None)
        prefix = setpts.get("prefix",None)
        suffix = setpts.get("suffix",None)
        


        if interp_type == "linear":
            t_step = (traj_setpoints[-1][0]-traj_setpoints[0][0])/self.subsample_num
            # Calculate the longer trajectory
            allOut=[]
            for idx in range(0,len(traj_setpoints)-1):
                seg = self.calculate_lin_segment(traj_setpoints[idx],traj_setpoints[idx+1],t_step)
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
            idx = self.find_nearest(t_intermediate, times)
            t_intermediate[idx] = times

            # Generate a cubic spline
            cs = CubicSpline(times, pres, bc_type = "periodic")
            traj = cs(t_intermediate)
            allOut = np.insert(traj,0, t_intermediate ,axis = 1)
            allOut = allOut.tolist()

            if self.graph:
                plt.plot(t_intermediate,traj)
                plt.plot(times,pres,'ok')
                plt.show()

        else:
            allOut = traj_setpoints
            

        self.full_trajectory = {}
        self.full_trajectory['prefix'] = prefix
        self.full_trajectory['suffix'] = suffix
        self.full_trajectory['setpoints']   = allOut

        return True


    # Do nothing if the trajectory type is not supported
    def do_none(self):
        return True


    # Pass out the built trajectory
    def get_trajectory(self):
        return self.full_trajectory


    # Find the nearest point
    def find_nearest(self, array, values):
        array = np.asarray(array)
        idx=[]
        for val in values:
            idx.append(np.argmin(np.abs(array - val)))
        return idx


    # Calculate a linear trajectory segment
    def calculate_lin_segment(self,start_point,end_point,t_step):

        # Calculate the linear interpolation time vector
        t_intermediate = np.arange(start_point[0],end_point[0],t_step)
        #print(t_step)
        #print(t_intermediate)
        
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
    def convert(self,conversion_fun):
        if self.full_trajectory is None:
            print('No trajectory loaded: Please load a trajectory')
            return False
        
        full_trajectory_new = dict()
        for traj_segment_key in self.full_trajectory:
            # If the trajectory segment key is the metadat tag, skip it.
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
        return True


    # Convert a trajectory definition using a conversion function
    def convert_definition(self,conversion_fun):
        if self.definition is None:
            print('No trajectory loaded: Please load a trajectory')
            return False

        if self.traj_type == "waveform":
            print('Cannot convert trajectory of type: "waveform"')
            return False
        
        def_new = copy.deepcopy(self.definition)
        setpoints=self.definition['config']['setpoints']
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
        return True



    # Plot the current trajectory 
    def plot_traj(self):
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

        plt.plot(out_traj_all[:,0],out_traj_all[:,1:])
        plt.xlabel("Time (s)")
        plt.ylabel("Pressure (psi)")
        plt.show()

        


if __name__ == '__main__':
    if len(sys.argv)==2:

        in_file = os.path.join(traj_folder,sys.argv[1])
        out_file = os.path.join(out_folder,sys.argv[1])
        build = TrajBuilder()
        build.load_traj_def(in_file)
        build.save_traj(out_file)

    else:
        print('make sure you give a filename')
        
        
        
        
        
        
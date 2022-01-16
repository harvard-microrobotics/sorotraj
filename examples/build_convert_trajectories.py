import sorotraj
import os

setup_location = 'traj_setup'
build_location = 'traj_built'

files_to_use = ['waveform_traj_demo','interp_setpoint','setpoint_traj_demo']

# Define a line-by-line conversion function to use
#   This example converts from orthogonal axes to differential actuation.
def linear_conversion(traj_line, weights):
    traj_length=len(traj_line)-1

    traj_line_new = [0]*(traj_length+1)
    traj_line_new[0]=traj_line[0] # Use the same time point

    for idx in range(int(traj_length/2)):
        idx_list = [2*idx+1, 2*idx+2]
        traj_line_new[idx_list[0]] = weights[0]*traj_line[idx_list[0]] + weights[1]*traj_line[idx_list[1]] 
        traj_line_new[idx_list[1]] = weights[0]*traj_line[idx_list[0]] - weights[1]*traj_line[idx_list[1]]

    return traj_line_new


# Set up the specific version of the conversion function to use
weights = [1.0, 0.5]
conversion_fun = lambda line: linear_conversion(line, weights)

# Test the conversion
traj_line_test = [0.00,  5,15,  5,15  ,-10,0,  -10,0]
print(traj_line_test)
print(conversion_fun(traj_line_test))

# Build the trajectories, convert them , and save them
traj = sorotraj.TrajBuilder()
for file in files_to_use:
    traj.load_traj_def(os.path.join(setup_location,file))
    traj.convert_traj(conversion_fun)
    traj.save_traj(os.path.join(build_location,file+'_convert'))

# Convert the definitions if possible
traj = sorotraj.TrajBuilder(graph=False)
for file in files_to_use:
    traj.load_traj_def(os.path.join(setup_location,file))
    traj.convert_definition(conversion_fun)
    traj.save_definition(os.path.join(setup_location,file+'_convert'))
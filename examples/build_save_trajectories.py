import sorotraj
import os

setup_location = 'traj_setup'
build_location = 'traj_built'

files_to_use = ['waveform_traj_demo','interp_setpoint','setpoint_traj_demo']

# Build a trajectory builder
traj = sorotraj.TrajBuilder()
for file in files_to_use:
	# Load, build, and save each trajectory
	traj.load_traj_def(os.path.join(setup_location,file))
	traj.save_traj(os.path.join(build_location,file))
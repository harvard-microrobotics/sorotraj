import sorotraj

file_to_use = 'traj_setup/setpoint_traj_demo'

build = sorotraj.TrajBuilder()
build.load_traj_def(file_to_use)
traj = build.get_trajectory()
for key in traj:
	print(key)
	print(traj[key])
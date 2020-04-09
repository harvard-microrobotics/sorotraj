import sorotraj

file_to_use = 'traj_setup/setpoint_traj_demo'

build = sorotraj.TrajBuilder()
build.load_traj_def(file_to_use)
traj = build.get_trajectory()
for key in traj:
	print(key)
	print(traj[key])

interp = sorotraj.Interpolator(traj)
actuation_fn = interp.get_interp_function(
                num_reps=1,
                speed_factor=2.0,
                invert_direction=False)

print("Interpolation at 2.155")
print(actuation_fn(2.155))
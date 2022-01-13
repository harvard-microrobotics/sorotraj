import sorotraj

#file_to_use = 'traj_setup/setpoint_traj_demo.yaml'
file_to_use = 'traj_setup/waveform_traj_demo.yaml'

builder = sorotraj.TrajBuilder()
builder.load_traj_def(file_to_use)
traj = builder.get_trajectory()
for key in traj:
	print(key)
	print(traj[key])

builder.plot_traj()

interp = sorotraj.Interpolator(traj)
actuation_fn = interp.get_interp_function(
                num_reps=1,
                speed_factor=1.0,
                invert_direction=False)
final_time = interp.get_final_time()
print("Final Interpolation Time: %f"%(final_time))

actuation_fn2 = interp.get_interp_function(
                num_reps=1,
                speed_factor=1.0,
                invert_direction=[1,3])

cycle_fn = interp.get_cycle_function(
                num_reps=1,
                speed_factor=1.0,
                invert_direction=[1,3])

print("Interpolation at 2.155")
print(actuation_fn(4))
print(actuation_fn2(4))
print(actuation_fn(2.155))
print(actuation_fn2(2.155))
print(cycle_fn([0.5, 2.0, 7.0]))
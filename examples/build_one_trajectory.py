import sorotraj
import numpy as np
import matplotlib.pyplot as plt

#file_to_use = 'traj_setup/setpoint_traj_demo.yaml'      # Basic demo
#file_to_use = 'traj_setup/setpoint_traj_demo_err0.yaml' # duplicate time (will throw exception)
#file_to_use = 'traj_setup/setpoint_traj_demo_err1.yaml' # non-monotonic time (will throw exception)
#file_to_use = 'traj_setup/setpoint_traj_demo_0.yaml'    # empty prefix
file_to_use = 'traj_setup/setpoint_traj_demo_1.yaml'    # single line prefix
#file_to_use = 'traj_setup/waveform_traj_demo.yaml'    # single prefix line

# Build the trajectory from the definition file
builder = sorotraj.TrajBuilder()
builder.load_traj_def(file_to_use)
traj = builder.get_trajectory()
for key in traj:
	print(key)
	print(traj[key])

# Plot the trajectory
builder.plot_traj(fig_kwargs={'figsize':(8,6),'dpi':150})

# Make an interpolator from the trajectory
interp = sorotraj.Interpolator(traj)

# Get the actuation function for the specified run parameters
actuation_fn, final_time = interp.get_traj_function(
                num_reps=2,
                speed_factor=1.0,
                invert_direction=[1,3])

print("Final Interpolation Time: %f"%(final_time))

# Get the cycle function for the specified run parameters
cycle_fn = interp.get_cycle_function(
                num_reps=2,
                speed_factor=1.0,
                invert_direction=[1,3])

# Plot the actuation function vs. time
times = np.linspace(-1,20,2000)
vals = actuation_fn(times)

plt.figure(figsize=(8,6),dpi=150)
plt.plot(times, vals)
plt.show()

abs_times = builder.get_absolute_times(num_reps=2, speed_factor=1.0)
abs_vals  = actuation_fn(abs_times)

print("Times")
print(abs_times)

traj_flat = builder.get_flattened_trajectory(num_reps=2, speed_factor=1.0, invert_direction=[1,3])

print(traj_flat)
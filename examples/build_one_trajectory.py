import sorotraj
import numpy as np
import matplotlib.pyplot as plt

#file_to_use = 'traj_setup/setpoint_traj_demo.yaml'      # Basic demo
#file_to_use = 'traj_setup/setpoint_traj_demo_err0.yaml' # duplicate time (will throw exception)
#file_to_use = 'traj_setup/setpoint_traj_demo_err1.yaml' # non-monotonic time (will throw exception)
#file_to_use = 'traj_setup/setpoint_traj_demo_0.yaml'    # empty prefix
file_to_use = 'traj_setup/setpoint_traj_demo_1.yaml'    # single line prefix
#file_to_use = 'traj_setup/waveform_traj_demo.yaml'    # single prefix line

builder = sorotraj.TrajBuilder()
builder.load_traj_def(file_to_use)
traj = builder.get_trajectory()
for key in traj:
	print(key)
	print(traj[key])

#builder.plot_traj()

interp = sorotraj.Interpolator(traj)

actuation_fn, final_time = interp.get_traj_function(
                num_reps=1,
                speed_factor=1.0,
                invert_direction=False)



#actuation_fn = interp.get_interp_function(
#                num_reps=1,
#                speed_factor=1.0,
#                invert_direction=False)
#final_time = interp.get_final_time()

print("Final Interpolation Time: %f"%(final_time))

times = np.linspace(-1,0,2000)
vals = actuation_fn(times)

plt.plot(times, vals)
plt.show()

"""
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
"""
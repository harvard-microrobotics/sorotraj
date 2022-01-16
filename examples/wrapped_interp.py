import sorotraj
import numpy as np
import matplotlib.pyplot as plt

file_to_use = 'traj_setup/setpoint_traj_demo.yaml'

# Build the trajectory from the definition file
builder = sorotraj.TrajBuilder()
builder.load_traj_def(file_to_use)
traj = builder.get_traj_components()

# Make a wrapped interpolator with the looping part
# of the trajectory
interp = sorotraj.interpolator.WrappedInterp1d(
			traj['setpoints']['time'],
			traj['setpoints']['values'],
			axis=0)

interp_fun = interp.get_function()

# Plot the values over a ridiculous range of times
times = np.linspace(-10,20,2000)
vals = interp_fun(times)

plt.plot(times, vals)
plt.show()
import sorotraj
import numpy as np
import matplotlib.pyplot as plt

file_to_use = 'traj_setup/setpoint_traj_demo.yaml'      # Basic demo
#file_to_use = 'traj_setup/setpoint_traj_demo_err0.yaml' # duplicate time (will throw exception)
#file_to_use = 'traj_setup/setpoint_traj_demo_err1.yaml' # non-monotonic time (will throw exception)
#file_to_use = 'traj_setup/setpoint_traj_demo_0.yaml'    # empty prefix
#file_to_use = 'traj_setup/waveform_traj_demo_1.yaml'    # single prefix line

builder = sorotraj.TrajBuilder()
builder.load_traj_def(file_to_use)
traj = builder.get_traj_components()
interp = sorotraj.WrappedInterp1d(traj['setpoints']['time'], traj['setpoints']['values'], axis=0)

interp_fun = interp.get_function()

times = np.linspace(-10,20,2000)
vals = interp_fun(times)

plt.plot(times, vals)
plt.show()
# sorotraj
Generate trajectories for soft robots from yaml files (accompanies the Ctrl-P project)

## Installation
1. Download the package
2. Navigate into the main folder
3. `pip install .`

## Usage

### Minimal Example
``` python
import sorotraj

file_to_use = 'traj_setup/setpoint_traj_demo'

build = sorotraj.TrajBuilder()
build.load_traj_def(file_to_use)
traj = build.get_trajectory()
print(traj)
```
**Check out the _examples_ folder for more detailed usage examples**


## Details:

There are currently two types of trajectories


### Waveform Trajectories

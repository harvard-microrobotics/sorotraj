# sorotraj
Generate trajectories for soft robots from yaml files (accompanies the [Ctrl-P project](https://cbteeple.github.io/pressure_controller_docs))

## Installation
### Install the release version
[This package is on pypi](https://pypi.org/project/sorotraj/), so anyone can install it with pip: `pip install sorotraj`

### Install the most-recent development version
1. Clone the package from the [github repo](https://github.com/harvard-microrobotics/sorotraj)
2. Navigate into the main folder
3. `pip install .`

## Usage

### Minimal Example
``` python
import sorotraj

file_to_use = 'traj_setup/setpoint_traj_demo.yaml'

traj = sorotraj.TrajBuilder()
traj.load_traj_def(file_to_use)
trajectory = traj.get_trajectory()
interp = sorotraj.Interpolator(trajectory)
actuation_fn = interp.get_interp_function(
                num_reps=1,
                speed_factor=2.0,
                invert_direction=False)
print(actuation_fn(2.155))
```
**Check out the _examples_ folder for more detailed usage examples**


## Set Up Trajectories:

Trajectories are made of three parts:
1. **main**: used in a looping trajectory
2. **prefix**: happens once before the main part
3. **suffix**: happens once after the main part

Here's an example of what that might look like defined in a yaml file:
```yaml
config:
    setpoints:
        # [time, finger1, finger2, n/c, n/c]
        main:
            - [0.0,   10, 12, 14,  16]
            - [1.0,    20, 0, 0,  0]
            - [2.0,   0, 20, 0,  0]
            - [3.0,     0, 0, 20, 0]
            - [4.0,     0, 0, 0, 20]
            - [5.0,    10, 12, 14, 16]

        prefix:
            - [0.000,   0, 0, 0,  0]
            - [1.0,    10, 12, 14,  16]

        suffix:
            - [2.000,   10, 12, 14,  16]
            - [3.0,  0, 0, 0,  0]
```


There are currently three types of ways to generate the **main** part of a trajectory:
1. **direct**: You enter waypoints directly
	- Define waypoints as a list of lists of the form: `[time in sec], [a_1], [a_2], ..., [a_n]`
2. **interp**: Interpolate between waypoints
	- Define waypoints as a list of lists of the form: `[time in sec], [a_1], [a_2], ..., [a_n]`
	- Set a few more parameters:
		- **interp_type**: (`string`) The type of interpolation to use. right now types include: `'linear'`, `'cubic'`, and `'none'`
    	- **subsample_num**: (`int`) The total number of subsamples over the whole trajectory
3. **waveform**: Generate waveforms (very basic, still just in-phase waveforms across all channels)
	- Set up the waveform:
		- **waveform_type**: (`string`) Types include: square-sampled, square, sin, cos-up, cos-down, triangle, sawtooth-f, and sawtooth-r
    	- **waveform_freq**: (`float`) Frequency in Hertz
    	- **waveform_max**: (`float`) A list of the maximum values for the waveform, in the form: `[20, 0, 15, 5] `
    	- **waveform_min**: (`float`) A list of the minimum values for the waveform, in the form: `[0, 20, 0, 15]`
	- Set a few more parameters:
    	- **subsample_num**: (`int`) The total number of subsamples over the whole trajectory
    	- **num_cycles**: (`int`) The number of cycles of the waveform
    	- **channels**: (`bool`) Flags to turn channels on and off. A list of the form:  `[1,1,0,0]`

## Convert Trajectories Line-by-Line
Check out the _build_convert_trajectories.py_ example.

1. Set up a conversion function
	- Inputs: one original trajectory line (`list`)
	- Outputs: one new trajectory line (`list`)
2. Load the trajectory like normal
	- `traj.load_traj_def(file_to_use)`
3. Convert the trajectory by passing the conversion function
	- `traj.convert_traj(conversion_function)`
4. This conversion overwrites the original trajectory. Now you can save it like normal
	- `traj.save_traj(file_to_save)`
5. Convert the trajectory definition by passing the conversion function
	- `traj.convert_definition(conversion_function)`
4. This conversion overwrites the original trajectory definition and reguilds the trajectory. Now you can save the definition like normal
	- `traj.save_definition(file_to_save)`




## Build an interpolator
```python
interp = sorotraj.Interpolator(trajectory)
```
- **trajectory**: A trajectory object generated by `sorotraj.TrajBuilder`


```python
actuation_fn, final_time = interp.get_traj_function(
                num_reps=1,
                speed_factor=1.0,
                invert_direction=False)
```
- **num_reps**: (`int`, default=1) Number of times to repeat the main looping trajectory
    - Must be positive, nonzero
- **speed_factor**: (`float`, default=1.0) A speed multiplier that is applied to the main loop (but not the prefix or suffix)
    - Must be positive, nonzero
- **invert_direction**: (`bool`, default=False) Negate the whole trajectory (useful if actuators have different directionalities)
    - (`bool`): Negate all channels
    - (`list of ints`): Choose which channels to negate with a list of channel indices


```python
cycle_fn = interp.get_cycle_function(
                num_reps=1,
                speed_factor=1.0,
                invert_direction=False)
```
- Same inputs as `get_interp_function()`, but returns a cycle function (returns the current cycle as a function of time)
- cycle_fn takes these values:
    - -2 = Prefix
    - -1 = Suffix
    - 0-N = Main loop index

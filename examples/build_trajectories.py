import sorotraj

setup_location = 'traj_setup'
build_location = 'traj_built'

file_to_use = 'waveform_traj_demo.yaml'


traj = sorotraj.TrajBuilder()
build.load_traj_def(os.path.join(setup_location,file_to_use))
build.save_traj(os.path.join(build_location,file_to_use))
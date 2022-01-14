import sorotraj
import os
import pytest

folder_to_use = '../examples/traj_setup'

@pytest.fixture(autouse=True)
def setup_and_teardown():
    print('\nFetching trajectories')
    yield
    print('\nFinished!')



#@pytest.fixture(scope='session')
#def get_trajectories():
#    out_data = []
#    for data in tests_to_do:
#        builder = sorotraj.TrajBuilder()
#        builder.load_traj_def(os.path.join(folder_to_use,data[0]))

    #     out_curr = [builder]
    #     out_curr.extend(data[1:])
    #     out_data.append(out_curr)

    # return out_data
    

# def test_traj_builder(get_trajectories):
#     for data in get_trajectories:
#         one_traj_builder(data[0], data[1])



def pytest_generate_tests(metafunc):
    idlist = []
    argvalues = []
    for scenario in metafunc.cls.scenarios:
        idlist.append(scenario[0])
        items = scenario[1].items()
        argnames = [x[0] for x in items]
        argvalues.append(([x[1] for x in items]))
    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")


def get_traj_builder(filename):
    traj_builder = sorotraj.TrajBuilder()
    traj_builder.load_traj_def(os.path.join(folder_to_use,filename))
    return traj_builder

# Test trajectory bulding capabillities of several relevant trajectories
scenarios_build = [  ('setpoint_empty_prefix', {'filename':'setpoint_traj_demo_0.yaml', 'expected_len':8}),
               ('setpoint_oneline_prefix', {'filename':'setpoint_traj_demo_1.yaml', 'expected_len':9}),
               ('setpoint_twoline_prefix', {'filename':'setpoint_traj_demo.yaml', 'expected_len':10}),
               ('interp_setpoint', {'filename':'interp_setpoint.yaml', 'expected_len':91}),
               ('waveform_traj', {'filename':'waveform_traj_demo.yaml', 'expected_len':96}),
               ]

class TestTrajBuilder:
    scenarios = scenarios_build

    def test_traj_builder(self, filename, expected_len):
        traj_builder = get_traj_builder(filename)
        traj = traj_builder.get_trajectory()
        traj_len =0
        for key in traj:
            traj_seg = traj[key]
            if isinstance(traj_seg, list):
                traj_len += (len(traj_seg))

        print(traj_len, expected_len)
        assert traj_len == expected_len


# Test trajectory building errors for incorrectly defined trajectories
scenarios_errs = [  ('duplicate_time', {'filename':'setpoint_traj_demo_err0.yaml'}),
               ('non_monotonic_time', {'filename':'setpoint_traj_demo_err1.yaml'}),
               ]

class TestTrajBuilderErrs:
    scenarios = scenarios_errs

    def test_traj_builder(self, filename):
        with pytest.raises(Exception) as e_info:
            traj_builder = get_traj_builder(filename)
            #traj = traj_builder.get_trajectory()



# Test interpolation capabillities of several relevant trajectories
scenarios_interp = [('setpoint_empty-prefix_empty-suffix', {'filename':'setpoint_traj_demo_2.yaml', 'expected_time':5}),
               ('setpoint_empty-prefix', {'filename':'setpoint_traj_demo_0.yaml', 'expected_time':8 }),
               ('setpoint_oneline-prefix', {'filename':'setpoint_traj_demo_1.yaml', 'expected_time':8 }),
               ('setpoint_twoline-prefix', {'filename':'setpoint_traj_demo.yaml', 'expected_time':9}),
               ('interp_no-prefix_no-suffix', {'filename':'interp_setpoint.yaml', 'expected_time':5}),
               ('waveform_traj', {'filename':'waveform_traj_demo.yaml', 'expected_time':8}),
               ]

# Ignore warnings related to prefix/suffix exclusion
@pytest.mark.filterwarnings("ignore")
class TestInterpolator:
    scenarios = scenarios_interp

    def test_interpolator(self, filename, expected_time):
        traj_builder = get_traj_builder(filename)
        traj = traj_builder.get_trajectory()
        interp = sorotraj.Interpolator(traj)
        actuation_fn = interp.get_interp_function(
                    num_reps=1,
                    speed_factor=1.0,
                    invert_direction=False)
        final_time = interp.get_final_time()

        assert final_time == expected_time


# Test interpolation warning of several trajectories with missing components
scenarios_warn = [('setpoint_empty-prefix_empty-suffix', {'filename':'setpoint_traj_demo_2.yaml'}),
               ('setpoint_empty-prefix', {'filename':'setpoint_traj_demo_0.yaml'}),
               ('interp_no-prefix_no-suffix', {'filename':'interp_setpoint.yaml'}),
               ]

class TestTrajBuilderWarn:
    scenarios = scenarios_warn

    def test_traj_builder(self, filename):
        with pytest.warns(UserWarning) as e_info:
            traj_builder = get_traj_builder(filename)
            traj = traj_builder.get_trajectory()
            interp = sorotraj.Interpolator(traj)
            actuation_fn = interp.get_interp_function(
                        num_reps=1,
                        speed_factor=1.0,
                        invert_direction=False)
                        

# Test interpolation error if attempting to get a list of interpolators                 
class TestTrajInterpErrors:
    scenarios = [('Testing deprecated inputs',{'error':DeprecationWarning})]
    def test_interp_deprecation(self, error):
        with pytest.raises(error) as e_info:
            filename="setpoint_traj_demo.yaml"
            traj_builder = get_traj_builder(filename)
            traj = traj_builder.get_trajectory()
            interp = sorotraj.Interpolator(traj)
            actuation_fn = interp.get_interp_function(
                        num_reps=1,
                        speed_factor=1.0,
                        invert_direction=False,
                        as_list=True)


#===================================================
# TODO: implement tests for the actual interp function and cycle function.

def notest_interpolator():
    interp = sorotraj.Interpolator(traj)
    actuation_fn = interp.get_interp_function(
                    num_reps=1,
                    speed_factor=2.0,
                    invert_direction=False)
    final_time = interp.get_final_time()
    print("Final Interpolation Time: %f"%(final_time))

    actuation_fn2 = interp.get_interp_function(
                    num_reps=1,
                    speed_factor=2.0,
                    invert_direction=[1,3])

    cycle_fn = interp.get_cycle_function(
                    num_reps=1,
                    speed_factor=2.0,
                    invert_direction=[1,3])

    print("Interpolation at 2.155")
    print(actuation_fn(2.155))
    print(actuation_fn2(2.155))
    print(cycle_fn([0.5, 2.0, 7.0]))
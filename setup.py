from setuptools import setup, find_packages

setup(
    name='sorotraj',
    version='0.14',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='Generate trajectories for soft robots from a file',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=['numpy','scipy','matplotlib','pyyaml'],
    url='https://github.com/cbteeple/soro-traj',
    author='Clark Teeple',
    author_email='cbteeple@gmail.com',
)
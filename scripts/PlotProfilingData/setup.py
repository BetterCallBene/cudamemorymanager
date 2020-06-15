from setuptools import setup, find_packages
from plotprofilingdata.utils import load_version, load_readme

def get_requires(filename):
    requirements = []
    with open(filename, "rt") as req_file:
        for line in req_file.read().splitlines():
            if not line.strip().startswith("#"):
                requirements.append(line)
    return requirements

project_requirements = get_requires("plotprofilingdata/requirements.txt")



setup(
    name='PlotProfilingData',
    version=load_version(),
    author='Benedikt König',
    description='PlotProfilingData read a csv file from trapla profiling and plot it as box plot',
    long_description=load_readme(),
    packages=find_packages(),
    install_requires=project_requirements,
    package_data={
        'plotprofilingdata': ['*.txt', ],
    },
    entry_points={
        'console_scripts': [
            'PlotProfilingData=plotprofilingdata.plot:run',
        ],
    },
    zip_safe=False,
)
# PlotProfilingData

PlotProfilingData read a csv file with follow schema
``` csv
component_start;...component_end;runtime;simulation_reference;
```
calculate the timespan of the columns except the last three (column_two - component_start, ..., component_end - column_last_four) and plot it.

## Getting started

### Requirements

Python Version 3

### Install

``` bash
git clone https://bitbucket.efs-auto.com/scm/avp-se/apsw_mod_trajectory_planning.git
cd scripts
pip3 install PlotProfilingData/
```

### Usage

``` bash
usage: PlotProfilingData [-h] [-v] [-p PATH] [-d DELIMITER]

BoxPlot of trapla profiling points.

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show version
  -p PATH, --path PATH  path to csv
  -d DELIMITER, --delimiter DELIMITER
                        csv delimiter
```

#### Example

PlotProfilingData -p <path_to_csv_file>

## Authors

* **Benedikt Koenig** - *Initial work* - AKKA



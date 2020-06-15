# !/usr/bin/env python3
import argparse
import os
import csv
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from plotprofilingdata.utils import load_version

# for magic numbers
def const(**kwargs):
    for x in kwargs.values():
        return x


def file_to_array(path, delimiter=';'):
    with open(path, 'r') as DataFile:
        read_csv = csv.reader(DataFile, delimiter=delimiter)
        values = dict()
        
        header = {}
        first_line = next(read_csv)
        i = 0
        for name in first_line:
            if name != const(blank_header=''):
                header[name] = i
                i = i + 1
        values = [[] for i in range(0, i)]
      
        for row in read_csv:
            for i in range(0, len(header)):
                try:
                    value = int(row[i])
                except ValueError:
                    continue
                values[i].append(value)
    
    # csv values was reading in transpose form
    values = np.array(values).T

    return header, values


def delete_zero_columns(header, values):
    new_header = {}
    for head in header:
        if np.max(values[:, header[head]]) == 0:
            continue
        new_header[head] = header[head]

    return new_header


def delete_rows_with_zeros(header, values):
    colmns = [header[name] for name in header.keys()]
    
    rows = []
    for row in values[:, colmns]:
        if np.min(row) == 0:
            continue
        rows.append(row.tolist())
    new_values = np.array(rows)

    return new_values


def post_processing(header, values):
    # Remove column and rows with zero/wrong values
    new_header = delete_zero_columns(header, values)
    new_values = delete_rows_with_zeros(new_header, values)
    # compare profiling timepoint, except last ones
    box_plot_values = new_values[:, const(second_col=1):const(second_last_col=-1)] \
                    - new_values[:, const(first_col=0):const(third_last_col=-2)]
    # append runtime column
    new_box_plot_values = np.c_[box_plot_values, new_values[:, new_header.__len__()-1]]
    # From nano to milliseconds
    new_box_plot_values = new_box_plot_values * const(nano_to_milliseconds=1e-6)
    # First row has invalid data
    new_box_plot_values = new_box_plot_values[1:, :]

    return new_header, new_box_plot_values

def print_labels(header):

    labels = []
    headers_sorted = sorted(header.items(), key=lambda x: x[1])
    for right_header, left_header in zip(headers_sorted[0:-2], headers_sorted[1:-1]):
        labels.append("{0} - {1}".format(left_header[0], right_header[0]))
    labels.append("overall runtime")
    return labels


def box_plot(box_plot_values, header, path):
    
    fig = plt.figure(1, figsize=(9, 6))

    ax = fig.add_subplot(111)
    ax.set_title('Timespan of profiling points in TraPla')
    ax.set(ylabel='Milliseconds')
    bp = ax.boxplot(box_plot_values)
    
    xtickNames = plt.setp(ax, xticklabels=print_labels(header))
    plt.setp(xtickNames, rotation=60, fontsize=8)

    head, tail = os.path.split(path)
    png_file = tail.replace("csv", "png")
    png_output_file = os.path.join(head, png_file)
    
    plot_file_output_path = os.path.join(os.getcwd(), const(plot_file_output=png_output_file))
    fig.savefig(plot_file_output_path, bbox_inches='tight')
    print('Print box plot to file: {0}'.format(plot_file_output_path))
    

def run():
    parser = argparse.ArgumentParser(prog='PlotProfilingData', description='BoxPlot of trapla profiling points.')
    
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {0}'.format(load_version()),  
                    help='show version')

    parser.add_argument('-p', '--path', dest='path', 
                    help='path to csv')

    parser.add_argument('-d', '--delimiter', default=';', dest='delimiter', 
                    help='csv delimiter')

    args = parser.parse_args()

    if args.path == None:
        parser.print_help()
        exit()

    if not os.path.isfile(args.path):
        print("File not exists. Exit.")
        exit()

    header, values = file_to_array(args.path, args.delimiter)  
    header, box_plot_values = post_processing(header, values)

    box_plot(box_plot_values, header, args.path)


if __name__ == "__main__":
    run()
    
import numpy as np
import glob
import click
import sys

@click.command()

@click.option('--handle_in', type=str, required=True)
@click.option('--handle_out', type=str, required=True)

def mean_from_mocks(handle_in,
                    handle_out):

    print('\nAveraging mean from mocks for the following arguments:')
    print('handle_in: {}'.format(handle_in))
    print('handle_out: {}'.format(handle_out))

    # loop over all mocks and calculate mean
    mock_files = sorted(glob.glob(handle_in))
    data_list = []

    for mock_file in mock_files:
        data = np.genfromtxt(mock_file)
        data_list.append(data)

    data_list = np.asarray(data_list)
    data_mean = np.mean(data_list, axis=0)
    data_std = np.std(data_list, axis=0)[:,-1]

    print('np.shape(data_list): {}'.format(np.shape(data_list)))
    print('np.shape(data_mean): {}'.format(np.shape(data_mean)))
    print('np.shape(data_std): {}'.format(np.shape(data_std)))

    cout = np.c_[data_mean, data_std]

    fmt = np.shape(cout)[1] * '%15.5f '

    np.savetxt(handle_out, cout, fmt=fmt)

if __name__ == '__main__':
    mean_from_mocks()
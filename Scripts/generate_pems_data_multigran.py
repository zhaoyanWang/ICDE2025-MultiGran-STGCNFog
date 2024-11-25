from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd

def generate_graph_seq2seq_io_data(data, x_offsets, y_offsets):
    """
    Generate samples from data using specified x and y offsets.
    :param data: np.ndarray, the input data array
    :param x_offsets: np.ndarray, the offsets for the input sequence
    :param y_offsets: np.ndarray, the offsets for the output sequence
    :return: tuple of np.ndarray, the input and output sequences
    """
    num_samples, num_nodes, feature_size = data.shape
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = num_samples - abs(max(y_offsets))

    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    if x and y:
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
    else:
        x, y = np.array([]), np.array([])
    return x, y

def generate_all_graph_seq2seq_io_data(data, x_offsets, y_offsets, day_x_offsets, week_x_offsets):
    """
    Generate aligned samples from data using specified x and y offsets for hour, day, and week samples.
    :param data: np.ndarray, the input data array
    :param x_offsets: np.ndarray, the offsets for the input sequence (hourly)
    :param y_offsets: np.ndarray, the offsets for the output sequence
    :param day_x_offsets: np.ndarray, the offsets for the input sequence (daily)
    :param week_x_offsets: np.ndarray, the offsets for the input sequence (weekly)
    :return: tuple of np.ndarray, the aligned input and output sequences for hour, day, and week samples
    """
    num_samples, num_nodes, feature_size = data.shape
    hour_x, day_x, week_x, y = [], [], [], []
    min_t = max(abs(min(x_offsets)), abs(min(day_x_offsets)), abs(min(week_x_offsets)))
    max_t = num_samples - abs(max(y_offsets))

    for t in range(min_t, max_t):
        if t + max(y_offsets) < num_samples:
            hour_x.append(data[t + x_offsets, ...])
            day_x.append(data[t + day_x_offsets, ...])
            week_x.append(data[t + week_x_offsets, ...])
            y.append(data[t + y_offsets, ...])
    if hour_x and day_x and week_x and y:
        hour_x = np.stack(hour_x, axis=0)
        day_x = np.stack(day_x, axis=0)
        week_x = np.stack(week_x, axis=0)
        y = np.stack(y, axis=0)
    else:
        print("No valid samples generated. Check the offsets and input data.")
        hour_x, day_x, week_x, y = np.array([]), np.array([]), np.array([]), np.array([])
    return hour_x, day_x, week_x, y

def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    df = np.load(args.traffic_df_filename)['data']
    df=np.transpose(df,(1,0,2))
    
    print("data shape:", df.shape)

    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    
    points_per_hour = 12
    day_x_offsets = np.sort(np.arange(-(seq_length_x - 1), 1, 1) - (args.num_of_days * 24 -1) * points_per_hour)
    week_x_offsets = np.sort(np.arange(-(seq_length_x - 1), 1, 1) - (args.num_of_weeks * 7 * 24 -1) * points_per_hour)

    print(f"x_offsets: {x_offsets}, y_offsets: {y_offsets}")
    print(f"day_x_offsets: {day_x_offsets}, week_x_offsets: {week_x_offsets}")

    # Generate aligned hour, day, and week samples
    hour_x, day_x, week_x, y = generate_all_graph_seq2seq_io_data(df, x_offsets=x_offsets, y_offsets=y_offsets,
                                                                  day_x_offsets=day_x_offsets, week_x_offsets=week_x_offsets)

    if hour_x.size == 0 or day_x.size == 0 or week_x.size == 0:
        raise ValueError("Generated samples are empty. Check the input data and offsets.")

    # Ensure the node dimensions are the same
    if hour_x.shape[2] != day_x.shape[2] or hour_x.shape[2] != week_x.shape[2]:
        min_nodes = min(hour_x.shape[2], day_x.shape[2], week_x.shape[2])
        hour_x = hour_x[:, :, :min_nodes, :]
        day_x = day_x[:, :, :min_nodes, :]
        week_x = week_x[:, :, :min_nodes, :]

    # Find the minimum number of samples among hour_x, day_x, and week_x
    min_samples = min(hour_x.shape[0], day_x.shape[0], week_x.shape[0])
    
    # Trim the samples to the minimum number of samples
    hour_x = hour_x[:min_samples]
    y = y[:min_samples]
    day_x = day_x[:min_samples]
    week_x = week_x[:min_samples]

    # Concatenate day and week samples with hour samples
    x = np.concatenate([hour_x, day_x, week_x], axis=-1)

    print("x shape: ", x.shape, ", y shape: ", y.shape)

    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.6)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        filename = os.path.join(args.output_dir, f"{cat}_cluster_l3_days{args.num_of_days}_weeks{args.num_of_weeks}.npz")
        np.savez_compressed(
            filename,
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
            day_x_offsets=day_x_offsets.reshape(list(day_x_offsets.shape) + [1]),
            week_x_offsets=week_x_offsets.reshape(list(week_x_offsets.shape) + [1])
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/PEMS08", help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default="data/PEMS08/pems08_original_data_cluster2.npz", help="Raw traffic readings.")
    parser.add_argument("--seq_length_x", type=int, default=12, help="Sequence Length.")
    parser.add_argument("--seq_length_y", type=int, default=12, help="Sequence Length.")
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start")
    parser.add_argument("--num_of_days", type=int, default=1, help="Number of days for generating day samples")
    parser.add_argument("--num_of_weeks", type=int, default=1, help="Number of weeks for generating week samples")
    parser.add_argument("--dow", action='store_true',)

    args = parser.parse_args()
    npz_file = np.load(args.traffic_df_filename)
    print("Keys in the npz file:", npz_file.files)
    if os.path.exists(args.output_dir):
        reply = str(input('%s exists. Do you want to overwrite it? (y/n)' % args.output_dir)).lower().strip()
        if reply[0] != 'y':
            exit
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
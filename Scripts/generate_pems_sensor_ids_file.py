# -*- coding: utf-8 -*-
import os

def generate_numbers_file(filename, max_number):
    with open(filename, 'w') as file:
        for i in range(1, max_number + 1):
            file.write(f'{i}\n')

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'pems')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, 'graph_sensor_ids.txt')
    max_number = 170
    generate_numbers_file(output_filename, max_number)

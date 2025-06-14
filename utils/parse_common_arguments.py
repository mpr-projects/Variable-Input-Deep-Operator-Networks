import os
import time
import json
import torch
import shutil
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Operator Learning.')
    parser.add_argument('settings')
    parser.add_argument('--device', default=None)
    parser.add_argument('--output_dir', default='./outputs')
    parser.add_argument('--no_output', action='store_true')
    parser.add_argument('--comment', default='')
    parser.add_argument('--no_top_level_folder', action='store_true')

    args = parser.parse_args()

    # make sure jobs started at the same time have different output directories
    args.start_time = time.time()

    if not args.no_top_level_folder:
        while True:
            start_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(args.start_time))
            output_dir = os.path.join(args.output_dir, start_time)

            try:
                os.makedirs(output_dir)
                break

            except:
                args.start_time = time.time()

        args.output_dir = output_dir

    if args.settings is not None:
        try:
            shutil.copy(args.settings, args.output_dir)

        except shutil.SameFileError:
            pass

        with open(args.settings) as f:
            settings = json.load(f)

        if 'hooks_file' in settings:
            shutil.copy(settings['hooks_file'], args.output_dir)

    if args.comment != '':
        output_file = os.path.join(args.output_dir, 'comment.txt')

        with open(output_file, 'a') as f:
            f.write(args.comment)

    args.device = args.device or 'cuda' if torch.cuda.is_available() else 'cpu'

    return args

import argparse
import numpy as np


def argParser_block():
    """! Argument parser
    This function reads input argument from command line and returns corresponding program options.

    Returns
    -------
    data_name : dataset name
    num_blk : number of blocks used to separate the whole sample
    num_epoch : number of epoch to run
    """

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-blk", required=False, help="number of blocks")

    ap.add_argument("-bat", required=False, help="batch size at each iteration")

    ap.add_argument("-d", required=False, help="data name")

    ap.add_argument("-ep", required=False, help="number of epoch to run")

    # read arguments
    args = ap.parse_args()

    if args.blk:
        blk = args.blk
    else:
        blk = 32

    if args.bat:
        print("WARNING: This script does not support -bat parameter. will ignore this parameter.")

    if args.d:
        data_name = args.d
    else:
        print("WARNING: data name not selected. will run default data if exists.")
        data_name = None

    if args.ep:
        epoch = args.ep
    else:
        epoch = 300

    return blk, data_name, epoch


def argParser_batch():
    """! Argument parser
    This function reads input argument from command line and returns corresponding program options.

    Returns
    -------
    data_name : dataset name
    num_epoch : number of epoch to run
    batch_size : batch size
    """

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-blk", required=False, help="number of blocks")

    ap.add_argument("-bat", required=False, help="batch size at each iteration")

    ap.add_argument("-d", required=False, help="data name")

    ap.add_argument("-ep", required=False, help="number of epoch to run")

    # read arguments
    args = ap.parse_args()

    if args.blk:
        print("WARNING: This script does not support -blk parameter. will ignore this parameter.")

    if args.bat:
        bat = args.bat
    else:
        bat = 1

    if args.d:
        data_name = args.d
    else:
        print("WARNING: data name not selected. will run default data if exists")
        data_name = None

    if args.ep:
        epoch = args.ep
    else:
        epoch = 3

    return bat, data_name, epoch

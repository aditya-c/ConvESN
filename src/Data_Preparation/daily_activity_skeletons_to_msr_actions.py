import argparse
import numpy as np
import os
from tools.msrda_utils import read_msrda_skeleton_file


def work_on_file(inp_file, out_file, strategy=2):
    """
    load skeleton file which is inp_file
    and saves data into a pandas dataframe
    then the dataframe's x y z and conf are stored into out_file
    """
    print("working on file :: ", inp_file)
    with open(inp_file, "r") as f:
        df, final_frames = read_msrda_skeleton_file(f, strategy)

    # setting conf to 1 (no reason)
    df["conf"] = 1.0

    # Normalization
    # mapping x, y, z to 0:100
    df["x"] = df["x"] * 100
    df["y"] = df["y"] * 100

    df["z"] = ((df["z"] - df["z"].min()) / (df["z"].max() - df["z"].min()) + 1) * 100

    print("writing to :: ", out_file)
    np.savetxt(out_file, df[["x", "y", "z", "conf"]].values, delimiter=' ')
    print("-" * 40)


if __name__ == "__main__":
    """
    takes in a daily activity skeleton data file
    and converts to a daily action format
    ex :::: python <prog_name> daily_skelton.txt test.txt
    can use --folder to give folder as input
    """
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='parse skeleton description')

    # Required positional argument
    parser.add_argument('inp', help='the skeleton data file name')
    parser.add_argument('out', help='the output file name')

    # Optional positional argument
    parser.add_argument('--folder', action='store_true')

    args = parser.parse_args()

    if args.folder:
        for filename in os.listdir(args.inp):
            if filename.endswith(".txt"):
                base_name = os.path.splitext(os.path.basename(filename))[0]
                work_on_file(args.inp + filename, args.out + base_name + "_action.txt")
    else:
        base_name = os.path.splitext(os.path.basename(args.inp))[0]
        work_on_file(args.inp, args.out + base_name + "_action.txt")

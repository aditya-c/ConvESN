import argparse
import numpy as np
import pandas as pd
import os

JOINTS_PER_SKELETON = 20


def read_msr_daily_activity_skeleton_file(filename):
    """
    return a dataframe of the frame data
    """
    global JOINTS_PER_SKELETON

    with open(filename, "r") as f:
        # first line has the number of frames
        line = f.readline().rstrip("\n")
        total_frames = int(line.split(" ")[0])

        # stores the data for each frame
        frame_data = []

        # list of the values for all the frames
        video_data = []
        for frame_count in range(total_frames):
            # first line of each frame data is the number of rows of data for the frame
            datapoints = int(f.readline().rstrip("\n"))
            if datapoints == 0:
                # no skeletons found if 0
                pass
            else:
                # refresh the frame data
                frame_data = []
                # save data in frame_data
                for data_count in range(datapoints):
                    datum = f.readline().rstrip("\n").split(" ")
                    frame_data.append(list(map(float, datum[:4])))
            # consolidate into a video list
            video_data.extend(frame_data[1::2][:JOINTS_PER_SKELETON])
        # return a video dataframe
        _frames = np.array([np.ones(JOINTS_PER_SKELETON) * i for i in range(total_frames)]).flatten()
        video_data = np.array(video_data)
        print("data shape :::", end=" ")
        print(video_data.shape)
        print(_frames.shape)
        df = pd.DataFrame({"frame": _frames, "x": video_data[:, 0],
                           "y": video_data[:, 1], "z": video_data[:, 2], "conf": video_data[:, 3]})
    return df


def work_on_file(inp_file, out_file):
    """
    load skeleton file which is inp_file
    and saves data into a pandas dataframe
    then the dataframe's x y z and conf are stored into out_file
    """
    print("working on file :: ", inp_file)
    df = read_msr_daily_activity_skeleton_file(inp_file)

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

    print(args)

    if args.folder:
        for filename in os.listdir(args.inp):
            if filename.endswith(".txt"):
                base_name = os.path.splitext(os.path.basename(filename))[0]
                work_on_file(args.inp + filename, args.out + base_name + "_action.txt")
    else:
        work_on_file(args.inp, args.out)

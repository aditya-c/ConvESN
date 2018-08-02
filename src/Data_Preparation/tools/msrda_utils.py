import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter
import sys


JOINTS_PER_FRAME = 20
MISSING_FRAME = "MISSING_FRAME"

########################################
# Missing Data Handling Strategies
########################################


def handle_missing_data(video_frames, missing_list, strategy=2):
    global MISSING_FRAME
    print("Missing at Frames :::", missing_list)
    missing_ranges = []

    for _, g in groupby(enumerate(missing_list), lambda x: x[0] - x[1]):
        group = list(map(int, map(itemgetter(1), g)))
        missing_ranges.append((group[0], group[-1]))

    # Strategy 1 :: replicate the old frame
    # we assume that atleast the last "corrupt_frames" frames are corrupt
    if strategy == 1:
        corrupt_frames = 4
        for start, end in missing_ranges:
            replicate_idx = max(start - corrupt_frames, 0)
            for frame_idx in range(replicate_idx + 1, end + 1):
                video_frames[frame_idx] = video_frames[replicate_idx]

    # Strategy 2 :: interpolation
    # we assume that atleast the last "corrupt_frames" frames are corrupt
    elif strategy == 2:
        corrupt_frames = 5
        for start, end in missing_ranges:
            valid_start = max(start - corrupt_frames, 0)
            if end + 1 >= len(video_frames):
                # the video ended with empty frames
                # we will resort to dropping the frames
                video_frames = [value for value in video_frames if value != MISSING_FRAME]
            else:
                # interpolation
                np_start = np.array(video_frames[valid_start])
                np_end = np.array(video_frames[end + 1])
                np_increment = (np_end - np_start) / (end + 1 - valid_start)
                for frame_idx in range(valid_start + 1, end + 1):
                    video_frames[frame_idx] = video_frames[valid_start] + np_increment * (frame_idx - valid_start)
    # Strategy 3 :: fitting a N degree spline (1<= N <=5)
    # we assume that atleast the last "corrupt_frames" number of frames are corrupt
    # we consider the "look_behind" and "look_ahead" number of frames
    # as input to the spline
    elif strategy == 3:
        N = 3
        tck, u = interpolate.splprep([x_sample,y_sample,z_sample], s=2, k=N)
        u_fine = np.linspace(0,1,num_true_pts)
        x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

        pass
    else:
        # drop missing frames
        video_frames = [value for value in video_frames if value != MISSING_FRAME]
    return video_frames

########################################
# File Reader
########################################


def read_msrda_skeleton_file(filename_ptr, strategy=2):
    # return a dataframe of the frame data
    global JOINTS_PER_FRAME, MISSING_FRAME

    missing_at = []

    # first line has the number of frames
    total_frames = int(filename_ptr.readline().rstrip("\n").split(" ")[0])

    # stores the data for each frame
    frame_data = []

    # list of the values for all the frames
    video_data = []
    for frame_count in range(total_frames):
        # first line of each frame data is the number of rows of data for the frame
        datapoints = int(filename_ptr.readline().rstrip("\n"))
        if datapoints == 0:
            # no skeletons found if 0
            missing_at.append(frame_count)
            video_data.append(MISSING_FRAME)
        else:
            # refresh the frame data
            frame_data = []
            # save data in frame_data
            for data_count in range(datapoints):
                datum = filename_ptr.readline().rstrip("\n").split(" ")
                frame_data.append(list(map(float, datum[:4])))
            # consolidate into a video list
            video_data.append(frame_data[1::2][:JOINTS_PER_FRAME])

    # Handling the MISSING_FRAMES
    if missing_at:
        video_data = handle_missing_data(video_data, missing_at, strategy)

    # flattenings
    final_frames = len(video_data)
    _frames = np.array([np.ones(20) * i for i in range(final_frames)]).flatten()
    video_data = [item for sublist in video_data for item in sublist]
    video_data = np.array(video_data)

    # return a video dataframe
    df = None
    try:
        df = pd.DataFrame({"frame": _frames, "x": video_data[:, 0],
                           "y": video_data[:, 1], "z": video_data[:, 2], "conf": video_data[:, 3]})
    except IndexError:
        print("Index Error!!")
        print(video_data)
        sys.exit

    return df, final_frames

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import argparse

from tools.msrda_utils import read_msrda_skeleton_file


def min_max_scaling(dataframe, column_name):
    # min - max a column
    _min = dataframe[[column_name]].min()
    _max = dataframe[[column_name]].max()
    dataframe[[column_name]] = (dataframe[[column_name]] - _min) / (_max - _min)


def get_frames(filename):
    # get the number of frames in the file
    with open(filename, "r") as f:
        return int(f.readline().rstrip("\n").split(" ")[0])

########################################
# Plot Updates
########################################


def draw_line(line_segment, i, j, x, y, z):
    # update the line segment given x, y, z
    # i and j are the two points
    line_segment.set_data([x[i], x[j]], [z[i], z[j]])
    line_segment.set_3d_properties([y[i], y[j]])


def make_bones(data, segments):
    # a determinitic way of adding bones in to the skeleton file
    x, y, z = data.x.values, data.y.values, data.z.values

    draw_line(segments[0], 0, 1, x, y, z)
    draw_line(segments[1], 1, 2, x, y, z)
    draw_line(segments[2], 2, 3, x, y, z)

    # right arm
    draw_line(segments[3], 2, 4, x, y, z)
    draw_line(segments[4], 4, 5, x, y, z)
    draw_line(segments[5], 5, 6, x, y, z)
    draw_line(segments[6], 6, 7, x, y, z)

    # left arm
    draw_line(segments[7], 2, 8, x, y, z)
    draw_line(segments[8], 8, 9, x, y, z)
    draw_line(segments[9], 9, 10, x, y, z)
    draw_line(segments[10], 10, 11, x, y, z)

    # right leg
    draw_line(segments[11], 12, 13, x, y, z)
    draw_line(segments[12], 13, 14, x, y, z)
    draw_line(segments[13], 14, 15, x, y, z)

    # left leg
    draw_line(segments[14], 16, 17, x, y, z)
    draw_line(segments[15], 17, 18, x, y, z)
    draw_line(segments[16], 18, 19, x, y, z)

    # hip
    draw_line(segments[17], 0, 16, x, y, z)
    draw_line(segments[18], 0, 12, x, y, z)


def update_graph(num, df, graph, segments, title):
    # adding the joints at each frame and also the bones
    data = df[df['frame'] == num]
    graph._offsets3d = (data.x, data.z, data.y)
    make_bones(data, segments)
    title.set_text('3D Test, frame={}'.format(num))

########################################
# main runner
########################################


def main():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='parse skeleton description')

    # Required positional argument
    parser.add_argument('filename', help='the skeleton data file name')

    # Optional positional argument
    parser.add_argument('--interval', nargs='?', help='the interval between frames \
        in plot (will not be available in the video)')
    parser.add_argument('--output', nargs='?', help='output file name')

    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()

    out_file = args.filename[:-3] + "_3d_vid.mp4"
    interval = 40
    if args.output:
        out_file = args.output
    if args.interval:
        interval = args.interval

    # skeleton_file = "a16_s10_e02_skeleton.txt"
    # working with the skeleton file
    skeleton_file = args.filename
    with open(skeleton_file, "r") as f:
        df, final_frames = read_msrda_skeleton_file(f, 2)

    # normalize the "z" column
    min_max_scaling(df, "z")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')

    ax.set_zlim(0, 2)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)

    ax.invert_zaxis()

    plt.xlabel('x')
    plt.ylabel('y')

    data = df[df['frame'] == 0]
    graph = ax.scatter(data.x, data.z, data.y)

    num_of_bones = 25
    segments = []
    for _ in range(num_of_bones):
        segments.append(ax.plot([0, 0], [0, 0], [0, 0])[0])

    ani = matplotlib.animation.FuncAnimation(fig, update_graph, final_frames,
                                             fargs=[df, graph, segments, title],
                                             interval=interval, blit=False)

    if args.save:
        ani.save(out_file, fps=30, extra_args=['-vcodec', 'libx264'],
                 metadata={'artist': 'skeleton'})

    plt.show()


if __name__ == "__main__":
    main()

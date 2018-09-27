from MSMC_2_0 import *
from shutil import copyfile, move
import yaml
import sys

if __name__ == "__main__":
    runs = 3
    if len(sys.argv) > 1:
        # loop the execution for "runs" times and save best model in checkpoint_file
        with open(sys.argv[1]) as f:
            args = yaml.safe_load(f)
            checkpoint_file = args["checkpoint_file"]
            dummy = checkpoint_file + "temp"
        best = 0.0
        for _ in range(runs):
            _, acc = MSMC(sys.argv[1])
            if best < acc:
                best = acc
                copyfile(checkpoint_file, dummy)
        move(dummy, checkpoint_file)

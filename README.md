# ConvESN
Implementation of Convolutional Echo State Network for human activity recognition as outlined in https://www.ijcai.org/proceedings/2017/0342.pdf

This now works with MSRDailyActivity dataset.

we saved all the skeleton data files from the MSRDailyActivity dataset at ./data

# data setup
mkdir data && cd data && mkdir DataBackUp MSRDailyAct3D padded modified

the skeleton data files for daily acitivity dataset need to be saved at data/MSRDailyAct3D/

# Code to modify DailyActivityData into the format for ConvESN code

python src/Data_Preparation/daily_activity_skeletons_to_msr_actions.py ./data/MSRDailyAct3D/ ./data/modified/ --folder

python src/Data_Preparation/Load_MSRA3D_real_world_modified.py ./data/modified/
# if you pass --test, the you need to set -output and this is for handling one file

python src/Data_Preparation/Padding.py ./data/DataBackUp ./data/padded/

## deprecated command :: python src/ConvESN_MSMC.py ./data/padded/ 1 --fit_model



# testing code snipets
x="./data/MSRDailyAct3D/a13_s06_e01_skeleton.txt"
python src/Data_Preparation/daily_activity_skeletons_to_msr_actions.py $x ./data/test/

python src/Data_Preparation/plot_dailyactivity3d_skeleton.py $x



# run code with
python src/MSMC_sai.py ./data/padded/ -split_number 1 -checkpoint check_points/test.hdf5 -reservoir reservoir/rs100.pkl --train
# if u dont give --train, it will test
# if u give --test_sample, it will work on a single file

# Tensorboard visulization
tensorboard --logdir ./logs

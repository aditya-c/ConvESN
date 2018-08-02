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

python src/Data_Preparation/Padding.py ./data/DataBackUp ./data/padded/

python src/ConvESN_MSMC.py ./data/padded/ 1 --fit_model



# testing code snipets
x="./data/MSRDailyAct3D/a13_s06_e01_skeleton.txt"
python src/Data_Preparation/daily_activity_skeletons_to_msr_actions.py $x ./data/test/

python src/Data_Preparation/plot_dailyactivity3d_skeleton.py $x

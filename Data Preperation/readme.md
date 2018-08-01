# we saved all the skeleton data files from the MSRDailyActivity dataset at ../data

python daily_activity_skeletons_to_msr_actions.py ../data ./data/ --folder
# modifies data into the format for ConvESN code

python Load_MSRA3D_real_world_modified.py ./data/

python Padding.py

cd .. && python ConvESN_MSMC.py

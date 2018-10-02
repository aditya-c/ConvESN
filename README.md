# ConvESN
Implementation of Convolutional Echo State Network for human activity recognition as outlined in https://www.ijcai.org/proceedings/2017/0342.pdf

This now works with MSRDailyActivity dataset.

### Data Setup

```bash
mkdir data reservoir check_points && cd data && mkdir DataBackUp MSRDailyAct3D padded modified
```
> The skeleton data files can be obtained [here](http://users.eecs.northwestern.edu/~jwa368/my_data.html). The Dataset ought to be saved at data/MSRDailyAct3D/

### Data Pre-processing
```bash
python src/Data_Preparation/daily_activity_skeletons_to_msr_actions.py ./data/MSRDailyAct3D/ ./data/modified/ --folder
```
> this snippet modifies MSRDailyActivityData into a format accepted by the ConvESN code

```bash
python src/Data_Preparation/Load_MSRA3D_real_world_modified.py ./data/modified/
```
> add --test and -output to handle a single file

```bash
python src/Data_Preparation/Padding.py ./data/DataBackUp ./data/padded/
```

### Train network v2.0
```python
python src/MSMC_2_0.py config/train_config.yaml
```


#### Ignore these (testing code snipets)

```bash
x="./data/MSRDailyAct3D/a13_s06_e01_skeleton.txt"
python src/Data_Preparation/daily_activity_skeletons_to_msr_actions.py $x ./data/test/

python src/Data_Preparation/plot_dailyactivity3d_skeleton.py $x

python src/MSMC_sai.py ./data/padded/ -split_number 1 -checkpoint check_points/test.hdf5 -reservoir reservoir/rs100.pkl --train
```
> omit --train, to set mode to test
> add --test_sample, to will work on a single file

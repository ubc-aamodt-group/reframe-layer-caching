device='cuda:0'
batch_size=1
ToYCrCb=False
Demodulate=True
InferToFile=True
Pixelshuffle=True
RandomMask=False
patch_size=240
seq_length=1
num_workers=6
modelPath='./Model/'
scene='SunTemple'
load_dir=''
scale=4
max_infer=20
perceptual_weight=0.2
modelName='Model31_mod_234_finalloss_100'
n_epochs=100
learning_rate=1e-4
scale_dict={
    1.6:'675P',
    2:'540P',
    3:'360P',
    3.75:'288P',
    4:'270P',
    5:'216P',
    6:'180P',
    8:'135P',
    1.7:None
}
train_scale_list=[2,3,4]
test_scale_list=[4]
train_paths=[
    '/data/zjwdataset/Bunker/Bunker_1/',
    '/data/zjwdataset/Bunker/Bunker_2/',
    '/data/zjwdataset/MedievalDocks/MedievalDocks_1/',
    '/data/zjwdataset/MedievalDocks/MedievalDocks_2/',
    '/data/zjwdataset/RedwoodForest/RedwoodForest_1/',
    '/data/zjwdataset/RedwoodForest/RedwoodForest_2/',
    '/data/zjwdataset/WesternTown/WesternTown_1/',
]
test_paths=[
    "./data/"
]

scene_jitter_start_pos={
    'Bunker':4,
    'RedwoodForest':4,
    'MedievalDocks':4,
    'WestTown':4,
    'ShowDown':1,
    'Infiltrator':2
}

# > REFRAME start
use_reframe = True
reframe_cache_policy = "delta-0.25"
# > REFRAME end

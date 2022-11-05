from __future__ import unicode_literals
#=====PARAMETERS OF NETWORK=====#
NUM_CLASS = 4
D_ASCENDING = 'ascending'
D_DESCENDING = 'descending'
D_MIXED = 'mixed'
BEND = 'bend'
HAMM = 'hamm'
NORMAL = 'normal'
PULL = 'pull'
SLIDE = 'slide'
tech_dict = { D_ASCENDING:{BEND:0, HAMM:1, NORMAL:2, SLIDE:3}, 
              D_DESCENDING:{BEND:0, NORMAL:1, PULL:2, SLIDE:3}
            }
inv_tech_dict = { D_ASCENDING: {v: k for k, v in iter(tech_dict[D_ASCENDING].items())},
                  D_DESCENDING: {v: k for k, v in iter(tech_dict[D_DESCENDING].items())}
                }

old_raw_net_opts = {
    'ra_conv_1': {
        'num_filters': 256,
        'filter_size': 512, # like the window size of STFT
        'stride': 128, # like the hop size of STFT
    },
    # 'ra_pool_1': {
    #   'pool_size': 2,
    #   'mode': 'max',
    # },
    # 'ra_conv_1_2': {
    #   'num_filters': 256,
    #   'filter_size': 129, # like the window size of STFT
    #   'stride': 256, # like the hop size of STFT
    # },
    # 'ra_conv_1_3': {
    #   'num_filters': 256,
    #   'filter_size': 257, # like the window size of STFT
    #   'stride': 256, # like the hop size of STFT
    # },
    'ra_conv_2': {
        'num_filters': 64,
        'filter_size': 1, 
        'stride': 1,
    },
    'conv_1': {
        'num_filters': 128,
        'filter_size': 3,
        'stride': 1,
    },
    'pool_1': {
        'pool_size': 2,
        'mode': 'max',
    },
    'conv_2': {
        'num_filters': 128,
        'filter_size': 3,
        'stride': 1,
    },
    'pool_2': {
        'pool_size': 2,
        'mode': 'max',
    },
    'conv_3': {
        'num_filters': 60,
        'filter_size': 3,
        'stride': 1,
    },
    'global_pool_func': 'mean',
    'dropout_p': 0.3,
    'num_class': NUM_CLASS,
}

raw_net_opts = {
    'ra_conv_1': { 'num_filters': 32, 'filter_size': 7, 'stride': 1, },
    'ra_pool_1': { 'pool_size': 4, 'mode': 'max', },
    'ra_conv_2': { 'num_filters': 32, 'filter_size': 7, 'stride': 1, },
    'ra_pool_2': { 'pool_size': 4, 'mode': 'max', },
    'ra_conv_3': { 'num_filters': 64, 'filter_size': 7, 'stride': 1, },
    'ra_pool_3': { 'pool_size': 4, 'mode': 'max', },
    'ra_conv_4': { 'num_filters': 64, 'filter_size': 7, 'stride': 1, },
    'ra_pool_4': { 'pool_size': 2, 'mode': 'max', },

    'conv_1': { 'num_filters': 128, 'filter_size': 3, 'stride': 1, },
    'pool_1': { 'pool_size': 2, 'mode': 'max', },
    'conv_2': { 'num_filters': 128, 'filter_size': 3, 'stride': 1, },
    'pool_2': { 'pool_size': 2, 'mode': 'max', },
    'conv_3': { 'num_filters': 128, 'filter_size': 3, 'stride': 1, },
    # 'dens_1': 256,
    'dens_2': 64,
    'global_pool_func': 'mean',
    'dropout_p': 0.3,
    'num_class': NUM_CLASS,
}

raw_dnn_opts = {
    'ra_conv_1': { 'num_filters': 32, 'filter_size': 7, 'stride': 1, },
    'ra_pool_1': { 'pool_size': 4, 'mode': 'max', },
    'ra_conv_2': { 'num_filters': 32, 'filter_size': 7, 'stride': 1, },
    'ra_pool_2': { 'pool_size': 4, 'mode': 'max', },
    'ra_conv_3': { 'num_filters': 64, 'filter_size': 7, 'stride': 1, },
    'ra_pool_3': { 'pool_size': 4, 'mode': 'max', },
    'ra_conv_4': { 'num_filters': 64, 'filter_size': 7, 'stride': 1, },
    'ra_pool_4': { 'pool_size': 2, 'mode': 'max', },

    'layer_list': [1500, 2000, 2000, 1500],
    'dropout_p': 0.5,
    'num_class': NUM_CLASS,
}

dnn_opts = {
    'layer_list': [1800, 1800, 1800, 900, 900, 900],
    'dropout_p': 0.5,
    'num_class': NUM_CLASS,
}

cnn_opts = {
    'conv_1': { 'num_filters': 256, 'filter_size': 3, 'stride': 1, },
    'pool_1': { 'pool_size': 2, 'mode': 'max', },
    'conv_2': { 'num_filters': 128, 'filter_size': 3, 'stride': 1, },
    'pool_2': { 'pool_size': 2, 'mode': 'max', },
    'conv_3': { 'num_filters': 128, 'filter_size': 3, 'stride': 1, },
    'pool_3': { 'pool_size': 2, 'mode': 'average_exc_pad', },
    'layer_list': [1800, 900],
    'dropout_p': 0.5,
    'num_class': NUM_CLASS,
}

cv_list =  [[5, 17, 32, 50, 55, 57, 58, 65, 70],
            [7, 13, 19, 21, 27, 51, 63, 66],
            [11, 15, 29, 40, 52, 59, 73],
            [3, 4, 25, 30, 42, 53, 60, 71],
            [2, 9, 34, 36, 38, 44, 54, 72]]


#=====PARAMETERS OF MELODY CONTOUR=====#
MC_LENGTH = 25
HOP_LENGTH = 256
SAMPLING_RATE = 44100
frameSize = 2048
guessUnvoiced = True
binResolution = 10
minDuration = 100
harmonicWeight = 0.85
filterIterations = 2
magnitudeThreshold = 20
peakDistributionThreshold = 0.75
minFrequency = 82
maxFrequency = 20000
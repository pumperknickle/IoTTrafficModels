config = {
    "scheduler_config": {
        "gpu": ["0"],
        "config_string_value_maxlen": 1000,
        "result_root_folder": "../results/"
    },

    "global_config": {
        "batch_size": 10,
        "vis_freq": 200,
        "vis_num_sample": 701,
        "d_rounds": 1,
        "g_rounds": 1,
        "num_packing": 2,
        "noise": True,
        "feed_back": False,
        "g_lr": 0.0001,
        "d_lr": 0.0001,
        "d_gp_coe": 10.0,
        "gen_feature_num_layers": 3,
        "gen_feature_num_units": 100,
        "gen_attribute_num_layers": 3,
        "gen_attribute_num_units": 100,
        "disc_num_layers": 5,
        "disc_num_units": 200,
        "initial_state": "random",

        "attr_d_lr": 0.001,
        "attr_d_gp_coe": 10.0,
        "g_attr_d_coe": 1.0,
        "attr_disc_num_layers": 5,
        "attr_disc_num_units": 200,
    },

    "test_config": [
        {
            "dataset": ["iot"],
            "epoch": [10000],
            "run": [0],
            "sample_len": [1],
            "extra_checkpoint_freq": [5],
            "epoch_checkpoint_freq": [1],
            "aux_disc": [False],
            "self_norm": [False]
        }
    ]
}

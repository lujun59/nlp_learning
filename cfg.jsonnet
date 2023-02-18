
local vocab_cfg = {
    "type":"vocab",
    "path": "./wd/vocab.txt"
    };

local str2tensor = {
    "type": "label_sent2tensor",
    "vocab":vocab_cfg,
    "tokenizer":{
        "type": "cjk_char_split"
    },
    "max_len": 80,
    "with_lable":false
};

{
    "type":"lm_trainer",
    "train_dataloader":{
        "type":"dataloader",
        "batch_size":20,
        "dataset":{
            "type": "iter_text_dataset",
            "reader": {
                "type": "text_line_reader",
                "path": "./wd/train.txt"
            },
            "str2tensor_fn": str2tensor
        },
        "collate_fn": {
            "type":"label_sent2batch"
        }
    },
    "valid_dataloader":{
        "type":"dataloader",
        "batch_size":10,
        "dataset":{
            "type": "iter_text_dataset",
            "reader": {
                "type": "text_line_reader",
                "path": "./wd/valid.txt"
            },
            "str2tensor_fn": str2tensor
        },
        "collate_fn": {
            "type":"label_sent2batch"
        }
    },

    "model": {
        "type": "lstm_lm",
        "vocab": vocab_cfg,
        "n_layer": 2
    },

    "out_path":"out",

    "train_params":{
        #"gpus": -1,
        #"auto_select_gpus":true,

        "accelerator": "cpu",
        "max_epochs":1,
        "val_check_interval": 200,
        "accumulate_grad_batches": 4,
        "gradient_clip_val": 5.0
    }
}

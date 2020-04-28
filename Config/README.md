
![](/Resources/Document/IMG_0932.PNG)
# Config

Config [Demo](./Demo.json)


We need make sure that different super-parameter


```json
{
    "instance_id": 0,
    "content": {
        "Net": {
            "DefaultNetwork":true,
            "NetType": "DeepLabV3",
            "BatchSize": 2,
            "BackBone": "None",
            "Optimizer":"SGD",
            "Loss_Function":"MSELoss",
            "learning_rate":0.02,
            "momentum":0.9,
            "weight_decay":1e-4,
            "lr_scheduler":"MultiStepLR",
            "lr_steps":[8,11],
            "lr_gamma":0.1,
            "class_num":2

        },
        "Dataset": {
            "DefaultDataset":false,
            "Transform":[
                    {"ToTensor":"None"},
                    {"Normalize":[[0.485,0.456,0.406],[0.229, 0.224, 0.225]]}
                ]
            ,
            "Type": "Costum_NPY_DataSet",
            "DataRatio":0.8,
            "NPY":"/workspace/SampledDatasetMini.npy",
            "root": "/workspace/WSNets/Data/labelme/demo",
            "train_index_file":"annotation.json",
            "val_index_file":"annotation.json"
        
        },
        "Config": {
            "group_factor":0,
            "DistributedDataParallel":false,
            "Resume":false,
            "multiscale_training": true,
            "logdir": "./root/log",
            "devices":"GPU",
            "gpu_id": "0",
            "epochs": 200,
            "down_pretrain_model": false,
            "checkpoint_path": "/workspace/models",
            "visualization": false,
            "worker_num":1

        }
    },
    "MissionType": "Segmentation"
}

```





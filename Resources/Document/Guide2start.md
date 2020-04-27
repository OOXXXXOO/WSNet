![](./IMG_0932.PNG)
# Get Start:


### 1.Requirement:
**Core:**
* Pytorch 1.3
* cudatoolkit 10.1
* tensorboard (Optional)


Start with conda 

```bash
conda env create -f Resources/Document/py_36_env_config.yaml
source activate stable
```
**Make sure the conda env has been activated;Maybe you need install some requirement package manually;**

Copy the template config json file in `Src/Config`, modify your mission
like:
```json
{
    "instance_id": 0,  // To distinguish different mission
    "content": {
        "Net": {
            "DefaultNetwork":true,   //use default network 
            "NetType": "FasterRCNN",  //if use defaultnetwork this option will not work
            "BatchSize": 1, // modified by your machine GRAM
            "BackBone": "None", // when use the default network allow to use different backbone 
            "Optimizer":"SGD", //Optimizer
            "Loss_Function":"CrossEntropyLoss",//Loss Function
            "learning_rate":0.02,//learning rate
            "momentum":0.9,//momentum
            "weight_decay":1e-4,
            "lr_scheduler":"MultiStepLR",//different lr_scheduler function
            "lr_steps":[8,11],
            "lr_gamma":0.1

        },
        "Dataset": {
            "DefaultDataset":true,//use default dataset or custom dataset
            "Transform":[
                    {"RandomHorizontalFlip":0.5},
                    {"ToTensor":"None"},
                    {"Normalize":[[0.485,0.456,0.406],[0.229, 0.224, 0.225]]}
                ]
            ,
            "Type": "CocoDetection",//default dataset name 
            "root": "/media/winshare/98CA9EE0CA9EB9C8/COCO_Dataset",// root of default dataset
            "train_index_file":"annotations/instances_train2014.json",//train anno file
            "val_index_file":"annotations/instances_val2014.json"// val anno file
        
        },
        "Config": {
            "group_factor":0,
            "DistributedDataParallel":false,//use multi-gpu to train
            "Resume":false,// load checkpoint to train 
            "multiscale_training": true,
            "logdir": "./root/log",//path of output log
            "devices":"GPU",
            "gpu_id": 0,// if use Distributed ,must like [0,1,2]
            "epochs": 200,//epochs
            "down_pretrain_model": true,//just use for default model
            "checkpoint_path": "./root/model",//
            "visualization": true,//plot data & doesn't work in docker env
            "worker_num":4//multithreading - load data

        }
    },
    "MissionType": "Detection"
}



```

****
> You need configure your mission type , dataset path ,dataset type and some option you need change. Then, you could start next step.
****

When you finish your config file . run :
```bash
python Src/instance ./your_config_file_path.json
```
The training flow will be start like:
```bash
Epoch: [0]  [    0/82081]  eta: 1 day, 13:48:55  lr: 0.000040  loss: 0.0775 (0.0775)  loss_classifier: 0.0336 (0.0336)  loss_box_reg: 0.0312 (0.0312)  loss_objectness: 0.0029 (0.0029)  loss_rpn_box_reg: 0.0099 (0.0099)  time: 1.6586  data: 0.5403  max mem: 1192
Epoch: [0]  [   10/82081]  eta: 12:02:13  lr: 0.000240  loss: 0.4530 (0.4466)  loss_classifier: 0.1945 (0.1946)  loss_box_reg: 0.1292 (0.1225)  loss_objectness: 0.0190 (0.0329)  loss_rpn_box_reg: 0.0947 (0.0966)  time: 0.5280  data: 0.0510  max mem: 1585
Epoch: [0]  [   20/82081]  eta: 10:44:32  lr: 0.000440  loss: 0.2011 (0.3080)  loss_classifier: 0.0744 (0.1378)  loss_box_reg: 0.0665 (0.0861)  loss_objectness: 0.0107 (0.0210)  loss_rpn_box_reg: 0.0459 (0.0631)  time: 0.4119  data: 0.0024  max mem: 1669
...
```

monitor board could be run as :
```
tensorboard --log_dir ./root/log  # log path modify by yourself in config file
```

































The requirement yaml content

```yaml
name: stable
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - conda-forge
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - defaults
dependencies:
  - _libgcc_mutex=0.1=main
  - _tflow_select=2.1.0=gpu
  - absl-py=0.8.0=py36_0
  - astor=0.8.0=py36_0
  - attrs=19.3.0=py_0
  - backcall=0.1.0=py_0
  - blas=1.0=mkl
  - bleach=3.1.0=py_0
  - bzip2=1.0.6=3
  - c-ares=1.15.0=h7b6447c_1001
  - ca-certificates=2019.9.11=hecc5488_0
  - cairo=1.14.12=h77bcde2_0
  - cffi=1.10.0=py36_0
  - cudatoolkit=10.1.168=0
  - cudnn=7.6.0=cuda10.1_0
  - cupti=10.1.168=0
  - curl=7.26.0=1
  - dbus=1.13.2=hc3f9b76_0
  - decorator=4.4.1=py_0
  - defusedxml=0.6.0=py_0
  - entrypoints=0.3=py36_1000
  - expat=2.2.6=he6710b0_0
  - fontconfig=2.12.4=h88586e7_1
  - freetype=2.8=hab7d2ae_1
  - freexl=1.0.5=h14c3975_0
  - gast=0.3.2=py_0
  - gdal=2.2.2=py36hc209d97_1
  - geos=3.6.2=heeff764_2
  - giflib=5.1.4=h14c3975_1
  - glib=2.53.6=h5d9569c_2
  - google-pasta=0.1.7=py_0
  - grpcio=1.14.1=py36h9ba97e2_0
  - gst-plugins-base=1.12.4=h33fb286_0
  - gstreamer=1.12.4=hb53b477_0
  - h5py=2.8.0=py36h39dcb92_0
  - hdf4=4.2.13=h3ca952b_2
  - hdf5=1.8.18=h6792536_1
  - icu=58.2=h9c2bf20_1
  - importlib_metadata=0.23=py36_0
  - intel-openmp=2019.4=243
  - ipykernel=5.1.3=py36h5ca1d4c_0
  - ipython=7.9.0=py36h5ca1d4c_0
  - ipython_genutils=0.2.0=py_1
  - jedi=0.15.1=py36_0
  - jinja2=2.10.3=py_0
  - joblib=0.11=py36_0
  - jpeg=9b=0
  - json-c=0.12.1=ha6a3662_2
  - json5=0.8.5=py_0
  - jsonschema=3.2.0=py36_0
  - jupyter_client=5.3.3=py36_1
  - jupyter_core=4.6.1=py36_0
  - jupyterlab=1.2.3=py_0
  - jupyterlab_server=1.0.6=py_0
  - kealib=1.4.7=h5472223_5
  - keras-applications=1.0.8=py_0
  - keras-preprocessing=1.1.0=py_1
  - kiwisolver=1.1.0=py36he6710b0_0
  - libboost=1.67.0=h46d08c1_4
  - libdap4=3.19.0=h8c95237_1
  - libedit=3.1.20181209=hc058e9b_0
  - libffi=3.2.1=1
  - libgcc-ng=9.1.0=hdf63c60_0
  - libgdal=2.2.2=h6bd4d82_1
  - libgfortran-ng=7.3.0=hdf63c60_0
  - libiconv=1.14=0
  - libkml=1.3.0=h590aaf7_4
  - libnetcdf=4.4.1.1=h97d33d9_8
  - libpng=1.6.37=hbc83047_0
  - libpq=9.6.6=h1f21990_0
  - libprotobuf=3.9.2=hd408876_0
  - libsodium=1.0.17=h516909a_0
  - libspatialite=4.3.0a=h72746d6_18
  - libstdcxx-ng=9.1.0=hdf63c60_0
  - libtiff=4.0.10=h2733197_2
  - libxcb=1.12=1
  - libxml2=2.9.4=0
  - markdown=2.6.9=py36_0
  - markupsafe=1.1.1=py36h516909a_0
  - mistune=0.8.4=py36h516909a_1000
  - mkl=2019.4=243
  - mkl-service=2.3.0=py36he904b0f_0
  - mkl_fft=1.0.14=py36ha843d7b_0
  - mkl_random=1.1.0=py36hd6b4f25_0
  - more-itertools=7.2.0=py_0
  - nbconvert=5.6.1=py36_0
  - ncurses=6.1=he6710b0_1
  - ninja=1.7.2=0
  - notebook=5.7.8=py36_1
  - numpy-base=1.17.2=py36hde5b4d6_0
  - olefile=0.44=py36_0
  - opencv=3.4.1=py36h40b0b35_1
  - openjpeg=2.3.0=h05c96fa_1
  - openssl=1.0.2t=h14c3975_0
  - pandas=0.20.3=py36_0
  - pandoc=2.7.3=0
  - parso=0.5.1=py_0
  - pcre=8.39=1
  - pexpect=4.7.0=py36_0
  - pickleshare=0.7.5=py36_1000
  - pillow=5.1.0=py36h3deb7b8_0
  - pixman=0.34.0=0
  - poppler=0.60.1=hc909a00_0
  - poppler-data=0.4.9=0
  - proj4=4.9.3=hc8507d1_7
  - prometheus_client=0.7.1=py_0
  - prompt_toolkit=2.0.10=py_0
  - protobuf=3.9.2=py36he6710b0_0
  - ptyprocess=0.6.0=py_1001
  - pycparser=2.18=py36_0
  - pygments=2.4.2=py_0
  - pyqt=5.6.0=py36h22d08a2_6
  - pyrsistent=0.15.5=py36h516909a_0
  - python=3.6.6=h6e4f718_2
  - pytorch=1.3.0=py3.6_cuda10.1.243_cudnn7.6.3_0
  - pytz=2017.2=py36_0
  - pyzmq=18.1.1=py36h1768529_0
  - qt=5.6.2=h974d657_12
  - readline=7.0=h7b6447c_5
  - scikit-learn=0.21.3=py36hd81dba3_0
  - scipy=1.3.1=py36h7c811a0_0
  - send2trash=1.5.0=py_0
  - sqlite=3.30.0=h7b6447c_0
  - tensorboard=1.14.0=py36hf484d3e_0
  - tensorflow=1.14.0=gpu_py36h3fb9ad6_0
  - tensorflow-base=1.14.0=gpu_py36he45bfe2_0
  - tensorflow-estimator=1.14.0=py_0
  - tensorflow-gpu=1.14.0=h0d30ee6_0
  - termcolor=1.1.0=py36_0
  - terminado=0.8.3=py36_0
  - testpath=0.4.4=py_0
  - tk=8.6.8=hbc83047_0
  - torchvision=0.4.1=py36_cu101
  - tornado=4.5.2=py36_0
  - traitlets=4.3.3=py36_0
  - util-linux=2.21=0
  - wcwidth=0.1.7=py_1
  - werkzeug=0.12.2=py36_0
  - wheel=0.29.0=py36_0
  - wrapt=1.11.2=py36h7b6447c_0
  - xerces-c=3.2.2=h780794e_0
  - xz=5.2.4=h14c3975_4
  - zeromq=4.3.2=he1b5a44_2
  - zipp=0.6.0=py_0
  - zlib=1.2.11=0
  - zstd=1.3.7=h0b5b093_0
  - pip:
    - altgraph==0.16.1
    - astroid==2.3.2
    - certifi==2019.9.11
    - chardet==3.0.4
    - cryptography==2.8
    - cycler==0.10.0
    - cython==0.29.13
    - deprecated==1.2.7
    - gitdb2==2.0.6
    - gitpython==3.0.5
    - idna==2.8
    - imageio==2.6.1
    - ipython-genutils==0.2.0
    - isort==4.3.21
    - lazy-object-proxy==1.4.3
    - matplotlib==3.1.1
    - mccabe==0.6.1
    - nbformat==4.4.0
    - networkx==2.4
    - numpy==1.17.3
    - ogr==0.8.0
    - opencv-python==4.1.1.26
    - packaging==19.2
    - pandocfilters==1.4.2
    - pip==19.3.1
    - psutil==5.6.3
    - pycocotools==2.0.0
    - pygithub==1.44.1
    - pyinstaller==3.5
    - pyjwt==1.7.1
    - pylint==2.4.3
    - pyparsing==2.4.2
    - pyqt5==5.13.2
    - pyqt5-sip==12.7.0
    - pyqtgraph==0.10.0
    - python-dateutil==2.8.0
    - python-gitlab==1.13.0
    - pywavelets==1.1.1
    - pyyaml==5.1.2
    - requests==2.22.0
    - scikit-image==0.16.2
    - seaborn==0.9.0
    - setuptools==41.4.0
    - sip==5.0.0
    - six==1.12.0
    - smmap2==2.0.5
    - tifffile==2019.7.26
    - toml==0.10.0
    - tqdm==4.36.1
    - typed-ast==1.4.0
    - urllib3==1.25.7
    - webencodings==0.5.1
prefix: /home/winshare/anaconda3/envs/stable


```
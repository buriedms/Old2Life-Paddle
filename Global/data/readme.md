# 代码介绍

此部分代码分为三个主要分支。
1. 制作数据大文件(.Bigfile)
2. 制作数据集(dataset)  
3. 定义数据加载器(dataloader)  

# 制作自己的数据集
打开`Create_Bigfile.py`文件，修改其中路径参数，可以实现对自己数据集的制作。  

参数解析：(代码中29-31行)  
`indir`:数据文件存放的主目录路径  
`target_folders`:主目录下存放的照片的文件夹名称  
`out_dir`:输出照片大文件(.Bigfile)路径  

**案例**  

- 目录树：

.
|-- __pycache__
|   `-- fold.cpython-37.pyc
|-- checkpoints
|   |-- domainA_SR_old_photos
|   |-- domainB_old_photos
|   `-- mapping_quality
|-- data
|   |-- Create_Bigfile.py
|   |-- Load_Bigfile.py
|   |-- __init__.py
|   |-- __pycache__
|   |-- base_data_loader.py
|   |-- base_dataset.py
|   |-- custom_dataset_data_loader.py
|   |-- data_loader.py
|   |-- image_folder.py
|   `-- online_dataset_for_old_photos.py
|-- detection.py
|-- detection_models
|   |-- antialiasing.py
|   |-- networks.py
|   `-- sync_batchnorm
|-- detection_util
|   `-- util.py
|-- fold.py
|-- models
|   |-- NonLocal_feature_mapping_model.py
|   |-- __init__.py
|   |-- __pycache__
|   |-- base_model.py
|   |-- initializer.py
|   |-- mapping_model.py
|   |-- models.py
|   |-- networks.py
|   |-- pix2pixHD_model.py
|   `-- pix2pixHD_model_DA.py
|-- options
|   |-- __init__.py
|   |-- __pycache__
|   |-- base_options.py
|   |-- test_options.py
|   `-- train_options.py
|-- run_a.sh
|-- run_b.sh
|-- run_map.sh
|-- temp_old
|   |-- Real_L_old
|   |-- Real_L_old.bigfile
|   |-- Real_RGB_old
|   |-- Real_RGB_old.bigfile
|   |-- VOC
|   `-- VOC.bigfile
|-- test.py
|-- test_Elm.py
|-- test_Sea.py
|-- test_old
|   |-- Real_L_old.bigfile
|   |-- Real_RGB_old.bigfile
|   `-- VOC.bigfile
|-- torchvision_paddle
|   |-- Lambda.py
|   |-- Variable.py
|   |-- __init__.py
|   |-- __pycache__
|   |-- to_pil_image.py
|   `-- utils.py
|-- train_domain_A.py
|-- train_domain_B.py
|-- train_mapping.py
|-- tree.md
`-- util
    |-- FID.py
    |-- LPIPS.py
    |-- PSNR_SSIM.py
    |-- PerceptualVGG.py
    |-- __init__.py
    |-- __pycache__
    |-- image_pool.py
    |-- inception.py
    |-- init.py
    |-- util.py
    `-- visualizer.py

23 directories, 59 files




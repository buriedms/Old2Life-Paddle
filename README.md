

# Old Photo Restoration (Paddle Implementation)

aistudio项目链接地址：https://aistudio.baidu.com/aistudio/projectdetail/2524206?contributionType=1&shared=1  

复现论文为：[Old Photo Restoration via Deep Latent Space Translation](https://paperswithcode.com/paper/old-photo-restoration-via-deep-latent-space)  

官方开源 pytorch 代码：[Bringing-Old-Photos-Back-to-Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life)  

数据集采用VOC数据集，和小部分真实老照片，老照片[数据地址](https://www.hpcbristol.net/photographers)

模型及配置文件存放[网盘链接](https://pan.baidu.com/s/1F-9k4KMPycJuyasotw9Cfw )，提取码：xfmp

可视化效果图[网盘链接](https://pan.baidu.com/s/1uzyHsURuNDw1Kv0vPbrR8Q )，提取码：s32p   

训练数据[网盘链接](https://pan.baidu.com/s/1BRe9tY8Oa5DcAz5wrCoX_Q )，提取码：07oi

训练精度指标达标情况：  

| Performance | PSNR | SSIM | FID | LPIPS |
|:---:|:---:|:---:|:---:|:---:|
| Target| 23.33| 0.69 | 134.35 | 0.25 |
|stage_A/Epoch(20) | 23.929 | 0.749 | 31.928 | 0.302 |
|stage_B/Epoch(20) | 24.269 | 0.834 | 21.873 | 0.189 |
|stage_Map/Epoch(20/A:20,B:20) | 22.930 | 0.709 | 122.859 | 0.321 |

## **训练方式**：

+ 终端在`Old2Life-Paddle`文件目录下执行以下命令：
  ```
  bash train.sh
  ```    
+ 如果想要训练单个阶段，可以通过如下命令执行。  
  1. 训练第一阶段
  ```
  bash Global/run_a.sh
  ```  
  2. 训练第二阶段
  ```
  bash Global/run_b.sh
  ```  
  3. 训练第三阶段
  ```
  bash Global/run_map.sh
  ```  

如果需要更改训练参数，可以在当中进行修改。  

**必选参数解释**  

| 参数 | 说明 | 案例 |
|:---:|:---:|:---:|
| `dataroor` | 存放图片数据的位置。格式为.bigfile。 | `--dataroot /home/aistudio/work/Old2Life/test_old` |
| `output_dir` | 图片输出路径。第一行为扰动照片，第二行为生成的结果，第三行为训练目标。 | `--outputs_dir /home/aistudio/work/Old2Life/output/` |
| `checkpoints_dir` | 保存结果参数和训练日志存放路径。 | `--checkpoints_dir /home/aistudio/work/Old2Life/test_checkpoints` |
| **重要可选参数** | *-* | *-* |
| `batchSize`| 一次训练选用的样本数量 | `--batchSize 64` |
| `gpu_ids`| 选用的gpu序号 | `--gpu_ids 0,1,2,4` |
| `use_vae_which_epoch` | 预训练选用的vae模型的版本 | `--use_vae_which_epoch latest` |
| `which_epoch` | 预训练采用的模型版本 | `--which_epoch latest` |
| `niter` | 学习率不衰减训练的轮数 | `--niter 15` |
| `niter_decay`| 学习率衰减训练的轮数 | `--niter_decay 15` |
| `continue_train`| 是否采用预训练模型进行持续学习，触发生效 | `--continue_train` |
| `debug`| 是否启用debug模式，触发生效| `--debug` |



[comment]: <> (+ `dataroor`:存放图片数据的位置。格式为.bigfile。  )

[comment]: <> (  例如：`/home/aistudio/work/Old2Life/test_old`  )

[comment]: <> (+ `output_dir`:图片输出路径。第一行为扰动照片，第二行为生成的结果，第三行为训练目标。  )

[comment]: <> (  例如：`/home/aistudio/work/Old2Life/output/`)

[comment]: <> (+ `checkpoints_dir`:保存结果参数和训练日志存放路径。  )

[comment]: <> (  例如：`/home/aistudio/work/Old2Life/checkpoints \`)

## **测试方式**： 

**1.查看测试指标**  
终端在`Old2Life-Paddle`目录下执行以下命令：  
```
bash test_Sea.sh
```  
**2.查看图片重建可视化效果**  
终端在`Old2Life-Paddle`目录下执行以下命令：  
```
bash test_Elm.sh
``` 

**必选参数解释**   

| 参数 | 说明 | 案例 |
|:---:|:---:|:---:|
| `load_pretrainA` | 存放A阶段训练模型的路径文件夹。 | `--dataroot /home/aistudio/work/Old2Life/test_old` |
| `load_pretrainB` | 存放B阶段训练模型的路径文件夹。 | `--outputs_dir /home/aistudio/work/Old2Life/output/` |
| `dataroot` | 测试性能指标的图片的存放路径。 | `--checkpoints_dir /home/aistudio/work/Old2Life/test_checkpoints` |
| `checkpoints_dir` | 存放配置信息和模型信息的主文件位置。 | `--checkpoints_dir /home/aistudio/work/Old2Life/test_checkpoints` |
| `test_input` | 测试老照片图片存放的位置。 | `--checkpoints_dir /home/aistudio/work/Old2Life/test_checkpoints` |
| `output_dir` | 转换的图片输出路径。 | `--checkpoints_dir /home/aistudio/work/Old2Life/test_checkpoints` |
| **重要可选参数** | *-* | *-* |
| `batchSize`| 测试选用的样本数量 | `--batchSize 8` |
| `gpu_ids`| 选用的gpu序号 | `--gpu_ids 0` |
| `which_epoch` | 测试采用的模型版本 | `--which_epoch latest` |

[comment]: <> (+ `load_pretrainA`：存放A阶段训练模型的路径文件夹。  )

[comment]: <> (  例如：`/D/Desktop/plan/Old2Life/Global/checkpoints/domainA_SR_old_photos`)

[comment]: <> (+ `load_pretrainB`：存放B阶段训练模型的路径文件夹。  )

[comment]: <> (  例如：`/D/Desktop/plan/Old2Life/Global/checkpoints/domainB_old_photos`)

[comment]: <> (+ `dataroot`：测试性能指标的图片的存放路径。  )

[comment]: <> (  例如：`/home/aistudio/work/Old2Life/test_old`)

[comment]: <> (+ `checkpoints_dir`：存放配置信息和模型信息的主文件位置。  )

[comment]: <> (  例如：`D:\\Desktop\\plan\\Old2Life\\Global\\checkpoints`)

[comment]: <> (+ `test_input`：测试老照片图片存放的位置。  )

[comment]: <> (  例如：`/home/aistudio/work/Old2Life/test_old`  )

[comment]: <> (+ `output_dir`：转换的图片输出路径。  )

[comment]: <> (  例如：`/home/aistudio/work/Old2Life/output/`)
  
## 数据集方面介绍
数据集可以通过两种方式获取：
1. 通过[网盘链接](https://pan.baidu.com/s/1BRe9tY8Oa5DcAz5wrCoX_Q )下载数据集，
   提取码：07oi
2. 通过[自定义数据文件](Global/data/readme.md )进行创建
  
## 效果图展示

上方为测试原图，下方为重建图片  

可以明显发现，通过模型重建后的图片在观感上不论是清晰度还是色彩的饱和度都更加的令人满意，
此效果是模型训练了20个epoch的结果，后续随着训练指标仍旧有所缓慢上升，原论文当中的结果是
进行训练了200个epoch的结果，我们有理由相信，我们所展示的效果不会是最佳效果，随着训练轮数
的上升，重建效果仍旧可以有所提升。

![效果图](imgs/result.png)
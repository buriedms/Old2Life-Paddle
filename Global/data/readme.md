# data部分介绍

此部分代码分为三个主要分支。
1. 制作数据大文件(.Bigfile)
2. 制作数据集(dataset)  
3. 定义数据加载器(dataloader)  

此部分仅介绍如何制作自己的数据大文件(.Bigfile)

# 制作自己的数据集
打开`Create_Bigfile.py`文件，修改其中路径参数，可以实现对自己数据集的制作。  

参数解析：(代码中29-31行)  
`indir`:数据文件存放的主目录路径  
`target_folders`:主目录下存放的照片的文件夹名称  
`out_dir`:输出照片大文件(.Bigfile)路径

**案例**  

- 目录树：
```
|__ Global/temp_old
|   |__ VOC
|   |__ Real_L_old
|   |__ Real_RGB_old
```

- 案例代码：  
```
indir="D:\Desktop\plan\Old2Life\Global\\temp_old"  
target_folders=['VOC','Real_L_old','Real_RGB_old']  
out_dir ="D:\Desktop\plan\Old2Life\Global\\temp_old"  
```

- 输出结果
```
|__ Global/temp_old
|   |__ VOC
|   |__ Real_L_old
|   |__ Real_RGB_old
|   |__ VOC.Bigfile
|   |__ Real_L_old.Bigfile
|   |__ Real_RGB_old.Bigfile
```

**注意**：代码通过 `os.walk(dir)`可遍历访问目录下包括子目录下所有文件
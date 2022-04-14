# PyTorch 实践指南 

## 训练
必须首先启动visdom：

```
python3 -m visdom.server
```

然后使用如下命令启动训练：

```
# 在gpu0上训练,并把可视化结果保存在visdom 的classifier env上
CUDA_VISIBLE_DEVICES=2 python3 train_green.py train --use-gpu --env=a-1
CUDA_VISIBLE_DEVICES=3 python3 train_red.py train --use-gpu --env=a-1

CUDA_VISIBLE_DEVICES=3 python3 train_red_2.py train --use-gpu --env=a-1
CUDA_VISIBLE_DEVICES=3 python3 train_green_2.py train --use-gpu --env=a-1



```


详细的使用命令 可使用
```
python main.py help
```

## 测试

```
CUDA_VISIBLE_DEVICES=0 python3 train_red.py test

# 0080--0.7725--0.7498.pth
CUDA_VISIBLE_DEVICES=0 python.exe train_red_2.py test

# 0079--0.8549--0.8538.pth
CUDA_VISIBLE_DEVICES=0 python.exe train_green_2.py test
```


# 数据标注情况

 红色原图： Image {}_c1.png
 绿色原图： Image {}_c2.png
 
 红色标注： 1.1.png
 绿色标注： 2.1.png
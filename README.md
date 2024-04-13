# Network Slimming (Pytorch)

This repository contains an official pytorch implementation for the following paper  
[Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) (ICCV 2017).  
[Zhuang Liu](https://liuzhuang13.github.io/), [Jianguo Li](https://sites.google.com/site/leeplus/), [Zhiqiang Shen](http://zhiqiangshen.com/), [Gao Huang](http://www.cs.cornell.edu/~gaohuang/), [Shoumeng Yan](https://scholar.google.com/citations?user=f0BtDUQAAAAJ&hl=en), [Changshui Zhang](http://bigeye.au.tsinghua.edu.cn/english/Introduction.html).  

Original implementation: [slimming](https://github.com/liuzhuang13/slimming) in Torch.    
The code is based on [pytorch-slimming](https://github.com/foolwood/pytorch-slimming). We add support for ResNet and DenseNet.  
```


感谢 network slimming的方法，这里魔改了，使用的resnet8x4 继续往下做剪枝，同时加入了教师模型 resnet32x4，并进行知识蒸馏，也就是loss多了个知识蒸馏的知识
resnet8x4 和 resnet32x4 torchvision代码库均可查看源码，也是魔改过的

剪枝代码：

| resnet8x4 原始稀疏性训练 （preactivation）                   | 72.83          | python main.py -sr --s 0.00001 --dataset cifar100 --arch resnet8x4 --depth 8 --save ./experiments/sparisty/cifar8x4_no_dist |
| ------------------------------------------------------------ | -------------- | ------------------------------------------------------------ |
| **resnet8x4 带上 dist 稀疏性训练 (preactivation)**           | **74.73**      | **python main.py -sr --s 0.00001 --dataset cifar100 --arch resnet8x4 --depth 8 --teacher_arch resnet32x4 --teacher_ckpt /home/szy/DIST_KD/classification/ckpt/cifar_ckpts/resnet32x4_vanilla/ckpt_epoch_240.pth --kd dist_t4 --save ./experiments/sparisty/cifar8x4_with_dist --epochs 240 --warmup 120 -dist** |
| **resnet8x4 不带稀疏性训练 (preactivation)** **做dist 知识蒸馏** | **75.19**      | **python train.py -c ./configs/strategies/distill/dist_cifar.yaml --model cifar_resnet8x4_prune --teacher-model cifar_resnet32x4 --teacher-ckpt ./ckpt/cifar_ckpts/resnet32x4_vanilla/ckpt_epoch_240.pth --experiment ./20240411/try_with_channel_selection_nopre/v1** |
| **resnet no preactivation 做稀疏性训练**                     | 效果不好，停了 | **python back_main.py -sr --s 0.00001 --dataset cifar100 --arch backresnet8x4 --depth 8 --teacher_arch resnet32x4 --teacher_ckpt /home/szy/DIST_KD/classification/ckpt/cifar_ckpts/resnet32x4_vanilla/ckpt_epoch_240.pth --kd dist_t4 --save ./experiments/sparisty/cifar8x4_nopreactivation_v1 --epochs 240 --warmup 120 -dist** |



知识蒸馏：

| **resnet8x4 preactivation dist知识蒸馏** | 75.15 |
| ---------------------------------------- | ----- |



做prune

| resnet8x4 没有dist，做完稀疏性直接refine | **python resprune.py --dataset cifar100 --depth 8 --percent {pruned ratio} --model /home/szy/network-slimming/experiments/sparisty/cifar8x4_no_dist/model_best.pth.tar --save ./experiments/pruned_models/cifar8x4_no_dist_{pruned ratio}percent** | cfg     |
| ---------------------------------------- | ------------------------------------------------------------ | ------- |
| **resnet8x4 做了dist稀疏性之后的prune**  | **python resprune.py --dataset cifar100 --depth 8 --percent {pruned ratio} --model /home/szy/network-slimming/experiments/sparisty/cifar8x4_with_dist_v2/model_best.pth.tar --save ./experiments/pruned_models/cifar8x4_dist_with_sparisty_{pruned ratio}percent**               （成功！！！！！！） | **cfg** |
|                                          |                                                              |         |

重头开始训效果在 72.66，剪枝比率为 20%（params: 0.969 M, FLOPs: 0.102 G）

不重头开始训，不管是带不带 dist的finetune，效果都要好

finetune目录树结构如下：

- dist_and_dist_tune（稀疏性为dist，带上dist做finetune）【该模型效果全在  dist项目代码跑的】

  运行代码：

  ```python
  python train.py -c ./configs/strategies/distill/dist_cifar.yaml --model back_resnet8x4_prune --teacher-model cifar_resnet32x4 --teacher-ckpt ./ckpt/cifar_ckpts/resnet32x4_vanilla/ckpt_epoch_240.pth --experiment ./{number of percent}percent --refine /home/szy/network-slimming/experiments/pruned_models/cifar8x4_dist_with_sparisty_{number of percent}percent/pruned.pth.tar --warmup-epochs 10 --epochs 60 --decay-epochs 10
  ```

  - 5percent    （74.16）  60轮
  - 10percent （73.48）60轮
  - 20percent （73.32）240轮
  - 30percent  

- dist_no_dist_tune（稀疏性为dist，不带dist做finetune）  【全在自己的项目代码跑的】

  运行代码：

  ```python
  python main.py --refine /home/szy/network-slimming/experiments/pruned_models/cifar8x4_dist_with_sparisty_{number of percent}percent/pruned.pth.tar --dataset cifar100 --arch resnet8x4 --depth 8 --epochs 60 --warmup 10 --stepsize 10 --save ./experiments/fine_tune/dist_no_dist_tune/{number of percent}percent
  ```

  - 5percent    （73.94）
  - 10percent  （73.36）
  - 20percent   （70.4）
  - 30percent

- no_dist_no_dist_tune （稀疏性为原始，不带dist做finetune）【全在自己的项目代码跑的】

  运行代码：

  ```python
  python main.py --refine /home/szy/network-slimming/experiments/pruned_models/cifar8x4_no_dist_{number of percent}percent/pruned.pth.tar --datase
  t cifar100 --arch resnet8x4 --depth 8 --epochs 60 --warmup 10 --stepsize 10 --save ./experiments/fine_tune/no_dist_no_dist_tune/{number of percent}percent
  ```

  - 5percent      （73.07）
  - 10percent    （72.25）
  - 20percent      （71.33）
  - 30percent

- no_dist_dist_tune（稀疏性为原始，带dist做finetune） （这个就不做了）有时间再做

  - 5percent
  - 10percent
  - 20percent
  - 30percent
python train.py -c ./configs/strategies/distill/dist_cifar.yaml --model cifar_resnet8x4 --teacher-model cifar_resnet32x4 --teacher-ckpt ./ckpt/cifar_ckpts/resnet32x4_vanilla/ckpt_epoch_240.pth --experiment ./test_show



日记 day ？？？ 

不知道多少天了，天天调参，困恼，不清楚自己的方向，不知道自己在干什么，不知道自己的目标

有一个特别的发现，就是剪完模型之后，模型在某些测试集上面效果很好，但是在某些测试集上面效果很差，这是为什么呢，

个人感觉有可能是因为剪枝的时候，剪枝的策略不对，导致剪枝的时候，剪掉了一些重要的信息，导致模型在某些测试集上面效果很差。怎么改进呢？

1. 可以尝试不同的剪枝策略，比如说，剪枝的时候，不要剪掉重要的信息，可以尝试剪枝的时候，剪掉一些不重要的信息，这样可以保证模型的效果不会下降。
2. 可以尝试不同的剪枝率，比如说，剪枝率可以逐渐增大，这样可以保证模型的效果不会下降。
3. 可以尝试不同的剪枝方法，比如说，可以尝试不同的剪枝方法，这样可以保证模型的效果不会下降。



剪枝代码：

| resnet8x4 原始稀疏性训练 （preactivation）                   | 72.83          | python main.py -sr --s 0.00001 --dataset cifar100 --arch resnet8x4 --depth 8 --save ./experiments/sparisty/cifar8x4_no_dist |
| ------------------------------------------------------------ | -------------- | ------------------------------------------------------------ |
| **resnet8x4 带上 dist 稀疏性训练 (preactivation)**           | **74.73**      | **python main.py -sr --s 0.00001 --dataset cifar100 --arch resnet8x4 --depth 8 --teacher_arch resnet32x4 --teacher_ckpt /home/szy/PCKD/classification/ckpt/cifar_ckpts/resnet32x4_vanilla/ckpt_epoch_240.pth --kd dist_t4 --save ./experiments/sparisty/cifar8x4_with_dist --epochs 240 --warmup 120 -dist** |
| **resnet8x4 不带稀疏性训练 (preactivation)** **做dist 知识蒸馏** | **75.19**      | **python train.py -c ./configs/strategies/distill/dist_cifar.yaml --model cifar_resnet8x4_prune --teacher-model cifar_resnet32x4 --teacher-ckpt ./ckpt/cifar_ckpts/resnet32x4_vanilla/ckpt_epoch_240.pth --experiment ./20240411/try_with_channel_selection_nopre/v1** |
| **resnet no preactivation 做稀疏性训练**                     | 效果不好，停了 | **python back_main.py -sr --s 0.00001 --dataset cifar100 --arch backresnet8x4 --depth 8 --teacher_arch resnet32x4 --teacher_ckpt /home/szy/PCKD/classification/ckpt/cifar_ckpts/resnet32x4_vanilla/ckpt_epoch_240.pth --kd dist_t4 --save ./experiments/sparisty/cifar8x4_nopreactivation_v1 --epochs 240 --warmup 120 -dist** |



知识蒸馏：

| **resnet8x4 preactivation dist知识蒸馏** | 75.15 |
| ---------------------------------------- | ----- |



做prune

| resnet8x4 没有dist，做完稀疏性直接refine | **python resprune.py --dataset cifar100 --depth 8 --percent {pruned ratio} --model /home/szy/network-slimming/experiments/sparisty/cifar8x4_no_dist/model_best.pth.tar --save ./experiments/pruned_models/cifar8x4_no_dist_{pruned ratio}percent** | cfg     |
| ---------------------------------------- | ------------------------------------------------------------ | ------- |
| **resnet8x4 做了dist稀疏性之后的prune**  | **python resprune.py --dataset cifar100 --depth 8 --percent {pruned ratio} --model /home/szy/network-slimming/experiments/sparisty/cifar8x4_with_dist_v2/model_best.pth.tar --save ./experiments/pruned_models/cifar8x4_dist_with_sparisty_{pruned ratio}percent**               （成功！！！！！！） | **cfg** |
|                                          |                                                              |         |





目前第一个想法： resnet preactivation dist 在dist上面跑的效果是 75.15，那么我把参数转移过去看看，直接resprune即可

第二个想法： 把dist没有preactivation的模型做参数转移，也就是对应的 back_resnet，做转移，试一试效果，也是直接resprune即可，不过要在 dist先训好





同时 fine tune 也有几个策略：

1. 带dist稀疏性训练，不带dist进行finetune
2. 带dist稀疏性训练，不带dist进行finetune
3. 不带dist进行稀疏性训练，不带dist进行finetune



fine tune：

| **不带dist进行稀疏性训练后不带dist进行finetune** | **python main.py --refine /home/szy/network-slimming/experiments/pruned_models/cifar8x4_no_dist_20percent/pruned.pth.tar --dataset cifar100 --arch resnet8x4 --depth 8 --epochs 160 --warmup 80 --save ./experiments/fine_tune/cifar8x4_no_dist_fine** | **71.4** |
| ------------------------------------------------ | ------------------------------------------------------------ | -------- |
| **不带dist进行稀疏性训练后带dist进行finetune**   | **python main.py --refine /home/szy/network-slimming/experiments/pruned_models/cifar8x4_no_dist_20percent/pruned.pth.tar --dataset cifar100 --arch resnet8x4 --depth 8 --epochs 240 --warmup 120  --teacher_arch resnet32x4 --teacher_ckpt /home/szy/PCKD/classification/ckpt/cifar_ckpts/resnet32x4_vanilla/ckpt_epoch_240.pth --kd dist_t4 --save ./experiments/fine_tune/cifar_no_dist_fine_with_dist -dist** | 72.5     |
| **带dist稀疏性训练，不带dist进行finetune**       | **python main.py --refine /home/szy/network-slimming/experiments/pruned_models/cifar8x4_dist_with_sparisty_20percent/pruned.pth.tar --dataset cifar100 --arch resnet8x4 --depth 8 --epochs 160 --warmup 80 --save ./experiments/fine_tune/cifar_dist_no_fine_dist **  (10percent) | 73.33    |
| **带dist稀疏性训练，带dist进行finetune**         | **python main.py --refine /home/szy/network-slimming/experiments/pruned_models/cifar8x4_dist_with_sparisty_20percent/pruned.pth.tar --dataset cifar100 --arch resnet8x4 --depth 8 --epochs 240 --warmup 120 --teacher_arch resnet32x4 --teacher_ckpt /home/szy/PCKD/classification/ckpt/cifar_ckpts/resnet32x4_vanilla/ckpt_epoch_240.pth --kd dist_t4 --save ./experiments/fine_tune/cifar_dist_fine_sparisty -dist**   （10 percent） | 73.8     |
| baseline                                         |                                                              | 72.5     |

所有的带dist训练，epoch要多，收敛得慢很多，所以打算明天尝试对于带dist进行finetune 的模型，多一点epoch

而对于不带dist进行finetune，其收敛快很多很多

python train.py -c ./configs/strategies/distill/dist_cifar.yaml --model cifar_resnet8x4_prune --teacher-model cifar_resnet32x4 --teacher-ckpt ./ckpt/cifar_ckpts/resnet32x4_vanilla/ckpt_epoch_240.pth --experiment ./cifar8x4_dist_fine_with_dist --refine /home/szy/network-slimming/experiments/pruned_models/cifar8x4_dist_with_sparisty_20percent/pruned.pth.tar



发现对于这种已经是很小的模型而言，其剪枝的比率不能太高，否则其对应的准确率要降低，甚至不如模型本身以及模型本身没有dist的剪枝后的finetune结果



重头开始训效果在 72.66，剪枝比率为 20%（params: 0.969 M, FLOPs: 0.102 G）

不重头开始训，不管是带不带 dist的finetune，效果都要好

finetune目录树结构如下：

- dist_and_dist_tune（稀疏性为dist，带上dist做finetune）【该模型效果全在  dist项目代码跑的】

  运行代码：

  (分为重头开始训和不重头开始训)

  ```python
  python train.py -c ./configs/strategies/distill/dist_cifar.yaml --model back_resnet8x4_prune --teacher-model cifar_resnet32x4 --teacher-ckpt ./ckpt/cifar_ckpts/resnet32x4_vanilla/ckpt_epoch_240.pth --experiment ./{number of percent}percent --refine /home/szy/network-slimming/experiments/pruned_models/cifar8x4_dist_with_sparisty_{number of percent}percent/pruned.pth.tar
  ```

  ```python
  python train.py -c ./configs/strategies/distill/dist_cifar.yaml --model back_resnet8x4_prune --teacher-model cifar_resnet32x4 --teacher-ckpt ./ckpt/cifar_ckpts/resnet32x4_vanilla/ckpt_epoch_240.pth --experiment ./{number of percent}percent --refine /home/szy/network-slimming/experiments/pruned_models/cifar8x4_dist_with_sparisty_{number of percent}percent/pruned.pth.tar --warmup-epochs 10 --epochs 60 --decay-epochs 10
  ```

  

  - 5percent（74.16）  
  - 10percent（74.08）
  - 15percent （72.95）
  - 20percent（73.32）
  - 25percent（72.43）



| percent%           | acc(top-1)% | FLOPS  | params |
| ------------------ | ----------- | ------ | ------ |
| 5                  | 75.34       | 1.195M | 0.144G |
| 10                 | 74.94       | 1.141M | 0.123G |
| 15                 | 74.19       | 1.061M | 0.111G |
| 20                 | 73.32       | 0.969M | 0.102G |
| 25                 | 72.55       | 0.886M | 0.091G |
| baseline(sparisty) | 75.42       | 1.234M | 0.177G |
| baseline           | 76.45       | 1.234M | 0.177G |



- dist_no_dist_tune（稀疏性为dist，不带dist做finetune）  【全在自己的项目代码跑的】

  运行代码：

  ```python
  python main.py --refine /home/szy/network-slimming/experiments/pruned_models/cifar8x4_dist_with_sparisty_{number of percent}percent/pruned.pth.tar --dataset cifar100 --arch resnet8x4 --depth 8 --epochs 60 --warmup 10 --stepsize 10 --save ./experiments/fine_tune/dist_no_dist_tune/{number of percent}percent
  ```

  - 5percent（73.94） （均为60轮）
  - 10percent（73.36）
  - 15percent（72.35）
  - 20percent（70.40）
  - 25percent（68.70）

- no_dist_no_dist_tune （稀疏性为原始，不带dist做finetune）【全在自己的项目代码跑的】

  运行代码：

  ```python
  python main.py --refine /home/szy/network-slimming/experiments/pruned_models/cifar8x4_no_dist_{number of percent}percent/pruned.pth.tar --datase
  t cifar100 --arch resnet8x4 --depth 8 --epochs 60 --warmup 10 --stepsize 10 --save ./experiments/fine_tune/no_dist_no_dist_tune/{number of percent}percent
  ```

  - 5percent（73.07）（均为60轮）
  - 10percent（72.25）
  - 15percent（71.99）
  - 20percent（71.33）
  - 25percent（70.65）

- no_dist_dist_tune（稀疏性为原始，带dist做finetune） （这个就不做了）有时间再做

  - 5percent
  - 10percent
  - 15percent
  - 20percent
  - 25percent

- random

```python
python main.py --random /home/szy/network-slimming/experiments/pruned_models/cifar8x4_no_dist_5percent/pruned.pth.tar --dataset cifar100 --arch resnet8x4 --depth 8 --epochs 60 --warmup 10 --stepsize 10 --save ./experiments/fine_tune/random_prune/5percent --prto 0.1
```

直接画三线表



画柱状图，

单画自己方法的准去率图，一共是四张图



结构图，蒸馏结构图

剪枝图，剪枝结构图+蒸馏方法

随机剪枝效果对比三线表



蒸馏效果对比三线表

 

目前的实验结果证明：

Training From Scratch 确实要好很多，而且性能要超过自己的baseline，这从侧面说明这样剪枝出来的模型是更合理的，或者说在考虑了知识的同时，也考虑到了模型的架构设计问题
1205

## 1.network：卷积改为tr-卷积
评估结果见eva_psnr_result-000.xlsx

**结果**：和未改之前一样，残差网络预测值为0

## 分析原因
这种现象有可能是梯度消失。梯度在反向传播过程中逐渐减小，导致模型无法有效更新权重。
因为network用的是relu激活函数，而ReLU在负区间的导数为0，会导致死亡relu问题。
神经元输出为零，导致梯度消失。
尤其是我训练的是残差网络，模型输出0.00x最佳的这种网络，容易出现某些层输出接近于零。


## 优化尝试方向：
1.网络结构：死亡relu和步长

2.归一化，去掉或者maxmin改为zeroscore

3.学习率

4.残差网络变为常规预测的网络


## 2.去掉归一化

**结果**：效果很差  PSNR个位数

## 3.学习率
- 此时发现之前的模型初始化代码有误，写成了nn.conv层，应该是ME.MinkowskiConvolution

        # MinkowskiConvolution 使用 He 初始化
        # 对于 BatchNorm 层，通常初始化权重为 1，偏置为 0
- 鉴于网络结构深且复杂，学习率减小至 lr = 0.0001

**结果**：预测0

## 4.Leaky ReLU-解决死亡relu

尝试使用Leaky ReLU替代relu，修改network

$$
f(x) = 
\begin{cases} 
x & \text{if } x > 0 \\
\text{negative_slope} \times x & \text{if } x \leq 0 
\end{cases}
$$

**结果**：预测0

## 5.残差网络变为常规预测的网络

**结果**：预测0


## 6.残差网络变为常规预测的网络- 去除归一化

**结果**：效果很差  PSNR个位数



## 没头绪

## 7.针对4实验基础，步长改为2，以up的为主

**结果**：


## 8.针对实验1-baseline，修改归一化为Z-score

**结果**：效果一般，PSNR下降至二三十

## 9.针对实验8，Z-score+ 步长改为2

**结果**：效果一般，PSNR持续下降


## 10.针对实验9，Z-score+ 步长改为2 + 非残差

**结果**：效果一般，PSNR持续下降
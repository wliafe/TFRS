# TFRS

#### 介绍

毕业设计：基于深度学习的印刷字体识别系统设计与实现

#### 数据集

个人自建数据集，包含 10 种字体，每种字体 1000 个样本，共 10000 个样本。

数据集下载链接：https://pan.baidu.com/s/1bE2N7AyGpEuYPDILAuem2A?pwd=6t7n

模型文件下载链接：https://pan.baidu.com/s/1-CFPdk8A7AT6Sw2ByePHCg?pwd=i82x

#### 安装依赖

```
pip install -r requirements.txt
```

#### 使用说明

1. 下载数据集并解压到 TFRS/data 目录下；
2. 下载模型文件并解压到 TFRS/models 目录下；
3. 运行 `python TFRS_predict.py`，即可开始预测。

#### 开发进度

2024.6.10 开发完成
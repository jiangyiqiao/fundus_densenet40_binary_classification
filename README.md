# fundus_binary_classification
### 眼底图二分类——densenet 40层实现

## 图像对比
* 正样本/负样本

<img src="https://github.com/jiangyiqiao/fundus_binary_classification/blob/master/results/Figure_good.jpeg" width="360" height="360"/> <img src="https://github.com/jiangyiqiao/fundus_binary_classification/blob/master/results/Figure_bad.jpeg" width="360" height="360" /> 


## Dependencies
* keras

 
其中，训练集正负样本各约3600(data/train),验证集各约900(data/validation)。输入图像大小32×32×3

训练：

    python densenet_40.py



## result
    Validation accuracy on random sampled 100 examples = 83.2%
* 训练准确率、损失

<img src="https://github.com/jiangyiqiao/fundus_densenet40_binary_classification/blob/master/results/keras40_valacc.png" width="1000" height="400"/> <img src="https://github.com/jiangyiqiao/fundus_densenet40_binary_classification/blob/master/results/keras40_loss.png" width="1000" height="400"/> 

* 验证准确率、损失

<img src="https://github.com/jiangyiqiao/fundus_densenet40_binary_classification/blob/master/results/keras40_valacc.png" width="1000" height="400"/> <img src="https://github.com/jiangyiqiao/fundus_densenet40_binary_classification/blob/master/results/keras40_valloss.png" width="1000" height="400"/> 



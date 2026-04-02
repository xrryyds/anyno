# MATH
## 2048
### train （7496）（hints-785）
model       ｜ -@pass 1 ｜ -@pass 8
DS_7B       ｜ 3693     ｜ 2511 
                          880
DS_32B      | 3513     
### test 



## 4096
### train （7496）
model       ｜ -@pass 1 ｜ -@pass 8
DS_7B       ｜ 2031        ｜ 1141（928）
### test 




实验一（表）： 各大数据集训练后的不同模型在test上的（7B/32 B）acc （2048（原始 + sira），16k（原始））
# DS_7b
## MATH


实验二（表）： DS 7b Math 在2048/4096/8192/16k 下的性能

实验三（图）： MATH上7b 2048，train/test ｜ anchor/mode b 每一个epoch的acc在训练中的变化 

实验四（消融）： 无序 + mode b 算法 + with anchor

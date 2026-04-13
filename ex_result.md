# MATH
## 2048
### train （7496）（hints-785）
model       ｜ -@pass 1 ｜ -@pass 8
DS_1.5B     | 4245      | 2627
1.5B+risa   ｜3014       ｜842
DS_7B       ｜ 3693     ｜ 2511 
7B+risa     | 2500      |  834
DS_32B（771）| 3513      | 2335（2207）
32B+sira.    |2241。    ｜ 817
DS_14B.     | 3488.    | 2337
14B+sira.   | 2366.    | 852
### test 
DS_7B       | 2433.    | 1569
7B+risa      |1672     | 508
DS_32B      | 2212.    | 1467
32b+risa。  ｜1499。   ｜508
DS_14B      | 2279。   ｜ 1470
14b+risa    |1691      | 540
        

## 4096
### train （7496）🌞
model       ｜ -@pass 1 ｜ -@pass 8
DS_7B       ｜ 2031        ｜ 1141（928）
### test 






# GSM8k（7473+1319）
## 2048
model       ｜ -@pass 1 ｜ -@pass 8
DS_1.5B.    | 2174 + 465 | 551 + 137
            | 2337       |  
DS_7B       ｜1214 + 251｜ 372 + 86
7B+risa     | 1389 + 270｜ 282 + 62
DS_14B.     | 986 +  176 | 361 + 61



DS_32B      | 875 + 154  ｜ 293 + 53
32B+risa.   | 1055 + 144 |  246 + 31

## 16384
model       ｜ -@pass 1 ｜ -@pass 8
DS_1.5B      | 2118  ｜ 465





























实验一（表）： 各大数据集训练后的不同模型在test上的（7B/32 B）acc （2048（原始 + sira），16k（原始））
# DS_7b
## MATH


实验二（表）： DS 7b Math 在2048/4096/8192/16k（原始/2048下训练后/各精度下训练后） 下的性能

实验三（图）： MATH上7b 2048，train/test ｜ anchor/mode b 每一个epoch的acc在训练中的变化 

实验四（消融）： 无序 + mode b 算法 + with anchor


实验5： MATH_DS_7B_多伦下的极限


hintd的位置加图
提升优化



DS_2048_with_sira
---------------4096-------------
4096_train_-pass@1: 2302
4096_train_-pass@8: 674
4096_test_pass@1:  1432
4096_test_pass@8:  407

without sira
4096_train_-pass@1:2022 
4096_train_-pass@8: 1134
4096_test_pass@1: 1261  
4096_test_pass@8: 654
---------------8192---------------
8192_train_-pass@1: 2227
8192_train_-pass@8: 
8192_test_pass@1: 1444
8192_test_pass@8:  364

without sira
8192_train_-pass@1: 1388
8192_train_-pass@8: 617
8192_test_pass@1: 846
8192_test_pass@8: 364
---------------16384---------------
16384_train_-pass@1: 
16384_train_-pass@8: 
16384_test_pass@1:  1069
16384_test_pass@8:  309

without sira
16384_train_-pass@1: 
16384_train_-pass@8: 
16384_test_pass@1: 751
16384_test_pass@8: 302





#16384
## 1.5B 
with：
@pass 1: 1910
@pass 8：556
witout：
@pass 1: 1729
@pass 8： 384
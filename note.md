[Done] ./data/data_sym0: train=131, test=15
[Done] ./data/data_sym1: train=140, test=16
[Done] ./data/data_sym2: train=140, test=16
[Done] ./data/data_sym3: train=140, test=16
[Done] ./data/data_sym4: train=140, test=16
[Done] ./data/data_sym5: train=119, test=14
[Done] ./data/data_sym6: train=131, test=15
[Done] ./data/data_sym7: train=142, test=16
[Done] ./data/data_sym8: train=140, test=16
[Done] ./data/data_sym9: train=142, test=16
Train / Test split (copy) and summary finished.

tmp1：加上了很多量（成交额）价的特征，penalty = 100，早停150

tmp2：早停改为200



## 分sym、参数搜索pipeline
1. bash run.sh：修改想要搜索的参数组合
2. python select_from_output.py：从output日志中选出最优参数组合，生成model_config.json
3. bash run3.sh：根据config.txt中的参数组合，逐个训练不同sym的数据集，并保存模型到对应目录
4. 将生成的所有model_symi文件夹中的模型文件提取出来（可直接使用下面的命令行指令）、model_config.json放入mmpc文件夹
    for i in {0..9}; do
        # 检查文件夹是否存在
        if [ -d "models_sym$i" ]; then
            # 将文件夹内的所有内容移动到当前目录
            mv models_sym$i/* .
            # 删除已经变空的文件夹
            rmdir models_sym$i
        fi
    done
5. zip -r yyh.zip mmpc

tmp1：penalty = 100
model_20_all_20260102_210830.json

tmp2：penalty = 10
model_20_all_20260102_210527.json

tmp3: penalty = 10，加上了三个量的特征
model_20_all_20260102_213347

tmp4: penalty = 10，加上了三个量的特征，且量的特征做了线性回归
model_20_all_20260102_235359
tmp4_：早停改为100

tmp5: penalty = 100，加上了三个量的特征，且量的特征做了线性回归
model_20_all_20260102_235444
tmp5_：早停改为100

tmp6: penalty = 1000，加上了三个量的特征，且量的特征做了线性回归
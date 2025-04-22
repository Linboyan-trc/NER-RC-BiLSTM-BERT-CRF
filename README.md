# NER-RC
### 数据集
- `data`目录下:
    - `train.json`: 训练集
    - `dev.json`: 验证集
    - `test.json`: 测试集
### BiLSTM + CRF 完成实体抽取任务
- 源代码: `bilstm_crf_model.py`
- 源文件中包含对数据的处理
- 可以选择: 
    1. 训练(生成模型文件)
    2. 实体抽取(基于模型文件)
    3. 训练并进行实体抽取(不生成模型文件)

### Bert微调 + CRF 完成实体抽取任务
- 源代码: `bert_crf_model.py`
- 源文件中包含对数据的处理
- 可以选择: 
    1. 训练(生成模型文件)
    2. 实体抽取(基于模型文件)
    3. 训练并进行实体抽取(不生成模型文件)

### Softmax完成关系分类任务
- 源代码: `softmax_classification_model.py`
- 源文件中包含对数据的处理
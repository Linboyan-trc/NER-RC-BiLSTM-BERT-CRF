# NER-RC
### 此版本仅用于提交
- 仅包含以下内容:
  - 三个源文件:
    - `BiLSTM + CRF`实体抽取源文件: `bilstm_crf_model.py`
    - `Bert + CRF`实体抽取源文件: `bert_crf_model.py`
    - `Softmax`关系分类源文件: `softmax_classification_model.py`
  - 三份训练日志, 三份结果日志:
    - `log/bilstm`
    - `log/bert`
    - `log/softmax`
  - 三份抽取/分类结果:
    - `predicted_entities_bilstm.txt`
    - `predicted_entities_bert.txt `
    - `predicted_entity_relations.txt`
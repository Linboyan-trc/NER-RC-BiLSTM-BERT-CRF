# 作者: toryn
# 时间: 2025/4/19
# 作者: toryn
# 时间: 2025/4/19
# 作者: toryn
# 时间: 2025/4/15

import torch
import torch.nn as nn
from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel
from seqeval.metrics import classification_report, f1_score
from tqdm import tqdm
from torch.optim import AdamW
import json


####################################################################################################
# 1. 处理数据
def convert_to_BIO(sentences, ner):
    # 1. 每个句子是一个列表，列表的元素是字符串，代表一个个单独的单词
    # 1.1 多个句子的全部单词都在all_tokens里
    # 1.2 因为ner中，每个句子的ner中的实体的[start, end, label]的起止为止是在多个句子中的token位置
    # 1.2 所以必须先把多个句子的token处理成一个列表，以符合ner中的定位
    all_tokens = []
    for sentence in sentences:
        for token in sentence:
            all_tokens.append(token)

    # 2. 创建一个列表，长度和all_tokens单词数一样
    # 2. 所有元素都先初始化为'O'
    token_labels = ['O'] * len(all_tokens)

    # 3. ner有三个元素，每个元素对应一个句子
    # 3. 每个元素内有多个元素，一个元素对应一个实体，标明了第一句话中实体的起始位置和类型
    for single_sentence_ner in ner:
        for start, end, label in single_sentence_ner:
            # 3.1 修改默认的BIO标注列表，将起始位置改成B-类型
            token_labels[start] = f'B-{label}'

            # 3.2 修改默认的BIO标注列表，将后续位置改成I-类型
            for i in range(start + 1, end + 1):
                token_labels[i] = f'I-{label}'

    # 4. 遍历单个句子
    # 4.1 index用于遍历全局的token_lables
    index = 0
    sentences_result = []
    for sentence in sentences:
        # 4.1 单个句子的BIO结果
        sentence_single_result = []
        for token in sentence:
            sentence_single_result.append((token, token_labels[index]))
            index += 1

        sentences_result.append(sentence_single_result)

    # 5. 最终三个句子，在sentences_result中三个元素
    # 5.1 每个元素是对应句子的BIO序列
    return sentences_result


def get_all_files_bios(filename):
    # 1. 读取json文件
    with open(filename) as f:
        datas = [json.loads(line) for line in f]

    # 2. BIO序列
    # 2.1 一个句子对应这个列表中的一个元素，每个元素都是BIO序列
    all_file_bios = []

    # 3. 遍历json
    # 3.1 每行内容是{"clusters": ..., "sentences": ..., "ner": ... "relations": ...,"doc_key": ...}
    # 3.2 在将json转化为BIO的过程中，只需要用到"sentences","ner"
    for data_line in datas:
        # 3.1 sentences + ner转换为BIO序列
        bio = convert_to_BIO(data_line["sentences"], data_line["ner"])

        # 3.2 序列加入列表中
        # 3.2 得到的bio是一个列表，当中有多个句子的BIO序列，每个句子是BIO序列是bio中一个元素
        # 3.2 需要把这些元素一个个取出，也就是拿到每个句子的BIO序列，直接加入到all_file_bios中，让all_file_bios使用一层遍历就可以取出文件中每个单句的BIO序列
        all_file_bios.extend(bio)

    return all_file_bios


####################################################################################################
# 1. 构建标签表
# 1.1 使得用BIO的标签可以映射到唯一的数字
# 1.2 使得用数字可以映射到对应的标签
def build_tag_map(all_file_bios):
    # 1.1 标签集合
    tag_set = set()

    # 1.2 遍历BIO序列中每个句子的每个二元组，以获取所有tag，形成一个tag集合
    for sentence in all_file_bios:
        for token, tag in sentence:
            tag_set.add(tag)

    # 1.3 将tag集合转换为列表
    tag_list = sorted(list(tag_set))

    # 1.4 建立tag2idx，idx2tag的映射
    tag2idx = {label: idx for idx, label in enumerate(tag_list)}
    idx2tag = {idx: label for label, idx in tag2idx.items()}
    return tag2idx, idx2tag


# 3. NERDataset
# 3.1 这个数据集的样本数是整个train.json的句子数量，单个样本是指单个句子
class NERDataset(Dataset):
    # 3.1 self.data就是一个列表，每个元素对应一个句子的BIO序列
    # 3.2 同时给定了token工具、标签表
    # 3.3 max_len是单个句子最大长度，训练集中单个句子最长101个token，因此max_len设置为128
    def __init__(self, data, tokenizer, tag2idx, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.tag2idx = tag2idx
        self.max_len = max_len

    # 3.1 样本数
    # 3.2 就是train.json的句子总数
    def __len__(self):
        return len(self.data)

    # 3.1 获取单个样本
    # 3.2 根据下标来获取到单个样本
    def __getitem__(self, idx):
        # 3.1 获取单个句子的所有单词和每个单词的标签
        sentence = self.data[idx]

        # 3.2 获取到的是单词字符串序列
        words = []
        for token, tag in sentence:
            words.append(token)

        # 3.2 获取到的是tag对应的索引序列
        tags = []
        for token, tag in sentence:
            tags.append(self.tag2idx[tag])

        encoding = self.tokenizer(words,
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len,
                                  return_tensors='pt')
        print(encoding)

        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(tags[word_id])

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids)
        }

####################################################################################################
# 1. 主程序
def main():
    # 1.1 token工具
    # 1.1 用于将一段文字转化为token_id, 分隔句子，句子掩码
    # 1.1 比如output = tokenizer("我爱北京天安门")
    # 1.1 则output =
    # {
    #   1. input_ids: 101-标志开始, 2769-我, 4263-爱, 1266-北, 776-京, 1921-天, 2128-安, 7305-门, 102-标志结束
    #   2. token_type_ids: 00000,11111，代表第一句话，第二句话
    #   3. attention_mask: 11111111111，注意力掩码，表示有效的token
    # }
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    # 2. 获取BIO数据
    train_all_file_bios = get_all_files_bios("data/train.json")
    dev_all_file_bios = get_all_files_bios("data/dev.json")

    # with open('data/train-bio.txt', 'w') as f:
    #     for single_sentence_bio in train_all_file_bios:
    #         for pair in single_sentence_bio:
    #             print(pair, file=f)
    #         print("",file=f)

    # 3. 构建BIO-label索引表
    # 3.1 在bert+crf的任务中，不需要构建word索引表，只需要构建label索引表，因为word->id的映射tokenizer已经有这个作用
    tag2idx, idx2tag = build_tag_map(train_all_file_bios)

    # with open('data/train-tag-set.txt', 'w') as f:
    #     print(json.dumps(tag2idx), file=f)
    #     print(json.dumps(idx2tag), file=f)
    #     print(len(tag2idx))

    # 4.加载为Dataset
    # 4.1 对齐接口
    train_dataset = NERDataset(train_all_file_bios, tokenizer, tag2idx)



if __name__ == "__main__":
    main()

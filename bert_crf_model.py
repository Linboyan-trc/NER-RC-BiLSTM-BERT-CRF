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
import time
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

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


def get_words(filename):
    # 1. 读取json文件
    with open(filename) as f:
        datas = [json.loads(line) for line in f]

    # 2. all_words
    all_words = []
    for data_line in datas:
        for sentence in data_line["sentences"]:
            all_words.append(sentence)

    return all_words

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


# 2. 模型
class BERT_CRF(nn.Module):
    def __init__(self, tagset_size):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.fc = nn.Linear(self.bert.config.hidden_size, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    # 2.1 模型计算
    def forward(self, x, tags=None, mask=None):
        outputs = self.bert(input_ids=x, attention_mask=mask)
        emissions = self.fc(outputs.last_hidden_state)

        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask.bool(), reduction='mean')
            return loss
        else:
            pred = self.crf.decode(emissions, mask=mask.bool())
            return pred


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

        # 3.3 对一个句子的token进行向量化
        # 3.3 就和tokenizer("我爱北京天安门")一样
        # 3.3 会得到encoding['input_ids']，并且由于return_tensors='pt'，所以已经转化成二维的tensor张量
        # 3.3 为了bert可以直接处理这些张量，这里张量都是二维，而不是一维
        # 3.3 会得到encoding['token_type_ids']，由于是单句，所以肯定都是tensor([[0,0,0,...,0]])
        # 3.4 由于max_length=self.max_len, padding='max_length'表示单句最大长度为128，并且不足的会通过补齐padding到128
        # 3.4 使得'input_ids', 'token_type_ids', 'attention_mask'都会是128维的二维张量
        # 3.4 对于超出128的会进行截断，但最大单句句长101，所以只会发生padding
        # 3.5 还会有encoding['offset_mapping']用于记录每个token在各自单词字符串中的起止为止，[[  [0,3],  [0,5],...  ]]什么的
        # 3.6 最后得到3个1*128的二维张量，分别是input_ids，是单词索引，token_type_ids，约束单句，attention_mask，带padding的掩码
        # 3.6 最后得到1个1*128*2的三维张量，offset_mapping
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors='pt',

            max_length=self.max_len,
            padding='max_length',
            truncation=True,

            return_offsets_mapping=True
        )

        # 3.7 就是将input_ids每个token的索引对应到在words中对饮单词的下标，对于[101]-起始位置, [102]-结束为止, [0]-填充，则对应None
        # 3.7 所以word_ids = [None, 0, 1, 2, ..., None, None, None]
        word_ids = encoding.word_ids(batch_index=0)

        # 3.8 再按照word_ids的顺序，对于在原始words中存在的token，找到这个token对应的标签对应的标签索引
        # 3.8 对于None则设置为-100
        # 3.8 最终tag_ids = [-100, 1~13, 1~13, 1~13,..., -100]
        tag_ids = []
        for word_id in word_ids:
            if word_id is None:
                tag_ids.append(0)
            else:
                tag_ids.append(tags[word_id])

        # 3.9 最后对于单个句子
        # 3.9 返回128的一维input_ids，squeeze()会将所有1维的维度除去，也就是一个1*1*1*128的张量squeeze()之后会变成一个一维的128张量
        # 3.9 返回128的一维tag_ids
        # 3.9 返回128的一维attention_mask
        # 3.9 也就是单词索引 (101, 146,   2185,..., 102,   0,   0,     0  )
        # 3.9 也就是标签索引 (-100, 1~13, 1~13,..., -100, -100, -100, -100)
        # 3.9 也就是掩码    (0,    1,    1,...,     0,    0,   0,     0  )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'tag_ids': torch.tensor(tag_ids),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


####################################################################################################
# 1. 训练函数
def training(model, dataloader, optimizer):
    # 1.1 对每个批训练
    # 1.1.1 加载1861条数据后，设置batch_size=32，train_loader中共有59批数据，训练59批，在这个过程中已经在进行参数更新
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc='Training'):
        # 1.3 获取每个批次的数据
        # 1.3.1 每个批次共3个张量，每个张量都有32个向量，每个向量128维
        # 1.3.2 第一个张量是32个词向量，第二个张量是32个标签向量，第三个张量是32个掩码，每个张量32行，128列
        # 1.3.3 因此input_data是一个32*128的张量，对应32个句子的词向量，target是一个32*128的张量，对应32个句子的标签向量
        optimizer.zero_grad()
        input_data = batch['input_ids']
        target = batch['tag_ids']
        mask = batch['attention_mask']

        # 1.4 就是调用BERT_CRF的forword方法
        # 1.4 对32个句子进行bert + linear计算，最终crf计算损
        loss = model(input_data, tags=target, mask=mask)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


# 2. 验证函数
def evaluate(model, dataloader, idx2tag):
    # 2.1 训练完59批，进行评估
    model.eval()

    # 2.2 加载验证集的数据，共50条，2批
    dev_predictions = []
    dev_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # 2.2.1 input_data, target, mask都是32*128的张量
            # 2.2.1 其中target已经转换为数组
            input_data = batch['input_ids']
            target = batch['tag_ids'].numpy()
            mask = batch['attention_mask']

            # 2.2.2 得到32个句子的tags预测结果，是一个有32个元素的列表，每个元素是对应单句的tags预测结果
            bio_tags_predicted = model(input_data, mask=mask)
            dev_predictions += bio_tags_predicted
            
            # 2.2.3 这里bio_tags_predicted, target, mask都是二维列表，每个元素是句子的预测标签，句子的真实标签，句子的掩码
            # 6.3.3 遍历32个句子，将预测的tags和真实的tags汇成两个列表
            for i in range(len(bio_tags_predicted)):
                # 6.3.4 单个句子的真实标签索引，这里会有padding，需要结合mask去掉padding
                single_sentence_true_tag_unchunked = target[i]
                single_sentence_mask_unchunked = mask[i].bool().numpy()
                single_sentence_true_tag = []
                for j in range(len(single_sentence_mask_unchunked)):
                    if single_sentence_mask_unchunked[j]:
                        single_sentence_true_tag.append(single_sentence_true_tag_unchunked[j].item())
                dev_labels.append(single_sentence_true_tag)

    dev_labels_str = []
    for sentence in dev_labels:
        sentence_labels = []
        for label in sentence:
            sentence_labels.append(idx2tag[label])
        dev_labels_str.append(sentence_labels)

    dev_predictions_str = []
    for sentence in dev_predictions:
        sentence_labels = []
        for label in sentence:
            sentence_labels.append(idx2tag[label])
        dev_predictions_str.append(sentence_labels)

    with open("output/bert/dev_labels.txt", "w") as f:
        for sentence in dev_labels_str:
            print(sentence, file=f)
    with open("output/bert/dev_predictions.txt", "w") as f:
        for sentence in dev_predictions_str:
            print(sentence, file=f)

    precision = precision_score(dev_labels_str, dev_predictions_str, zero_division=1)
    recall = recall_score(dev_labels_str, dev_predictions_str, zero_division=1)
    f1 = f1_score(dev_labels_str, dev_predictions_str, zero_division=1)
    report = classification_report(dev_labels_str, dev_predictions_str, zero_division=1)
    
    return precision, recall, f1, report

####################################################################################################
# 1. 训练
def train():
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
    # 2.1 train_all_file_bios是一个列表，每个元素是一个句子的BIO序列，就是一堆(token, BIO-label)二元组组成的序列
    train_all_file_bios = get_all_files_bios("data/train.json")
    dev_all_file_bios = get_all_files_bios("data/dev.json")

    # 3. 构建BIO-label索引表
    # 3.1 在bert+crf的任务中，不需要构建word索引表，只需要构建label索引表，因为word->id的映射tokenizer已经有这个作用
    # 3.2 共13类标签，不含PAD
    tag2idx, idx2tag = build_tag_map(train_all_file_bios)

    # 4.加载为Dataset
    # 4.1 对齐接口
    train_dataset = NERDataset(train_all_file_bios, tokenizer, tag2idx)
    dev_dataset = NERDataset(dev_all_file_bios, tokenizer, tag2idx)

    # 5.传入数据
    # 5.1 训练集中共1861条数据，每次处理32个句子，batch_size设置为32
    # 5.2 因此每个epoch会分为1861/32 = 59次来批处理数据，每个epoch开始前会对1861个数据进行打乱，一共训练10个epoch
    # 5.3 而损失函数和优化器在每次批处理后就工作，去更新模型参数，因此损失函数和优化器一共工作5 * 59 = 295次
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=True)

    # 6. 定义模型
    # 6.1 优化器用于调整模型参数，减少误差
    # 6.2 在优化器中设置了学习率lr = 0.00001
    model = BERT_CRF(tagset_size=len(tag2idx))
    optimizer = AdamW(model.parameters(), lr=3e-5)

    # 7. 训练
    epochs = 3
    best_dev_f1 = 0
    best_model_state = None
    with open("log/bert/log_train.txt", "w") as f:
        pass
    for epoch in range(epochs):
        with open("log/bert/log_train.txt", "a") as f:
            print(f"####################################################################################################",file=f)
            print(f"Epoch {epoch + 1}:", file=f)

            # 7.1 对本epoch的117批训练
            start_time = time.time()
            train_loss = training(model, train_loader, optimizer)
            epoch_time = time.time() - start_time
            print(f"  - Avg Training Loss: {train_loss:.4f}", file=f)

            # 7.2 基于dev集进行评估
            precision, recall, f1, report = evaluate(model, dev_loader, idx2tag)
            print(f"  - Dev Precision: {precision:.4f}", file=f)
            print(f"  - Dev Recall: {recall:.4f}", file=f)
            print(f"  - Dev F1 Score: {f1:.4f}", file=f)
            print(f"  - Training Time: {epoch_time:.2f} seconds", file=f)
            print(report, file=f)

            if f1 > best_dev_f1:
                best_dev_f1 = f1
                best_model_state = model.state_dict()

    return best_model_state


# 2. 推理
def infer(use_saved_model=True, best_model_state=None):
    # 1. 加载数据
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    train_all_file_bios = get_all_files_bios("data/train.json")
    test_all_file_bios = get_all_files_bios("data/test.json")
    test_sent_words = get_words("data/test.json")  # 原始单词列表

    # 2. 构建标签映射
    tag2idx, idx2tag = build_tag_map(train_all_file_bios)

    # 3. 构建测试集 Dataset 和 DataLoader
    test_dataset = NERDataset(test_all_file_bios, tokenizer, tag2idx)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 4. 加载模型
    model = BERT_CRF(tagset_size=len(tag2idx))

    # 5. 加载权重
    if use_saved_model:
        model.load_state_dict(torch.load("best_model_bert.pth"))
    else:
        model.load_state_dict(best_model_state)

    # 6. 推理
    model.eval()
    all_preds = []
    all_labels = []
    all_entities = []
    all_words = []

    sent_idx = 0  
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            # 2.2.1 input_data, target, mask都是32*128的张量
            input_data = batch['input_ids']
            target = batch['tag_ids']
            mask = batch['attention_mask'].bool()

            # 2.2.2 一个列表，每个元素也是列表，是一个句子的预测标签
            pred_tags = model(input_data, mask=mask)

             # 5.2 遍历列表，取出每个句子
            for i in range(len(pred_tags)):
                gold_seq = target[i]
                pred_seq = pred_tags[i]

                # 用 get_words() 中提取的句子替代原 raw_words
                raw_words = test_sent_words[sent_idx]

                encoding = tokenizer(raw_words,
                                     is_split_into_words=True,
                                     return_offsets_mapping=False,
                                     return_tensors='pt',
                                     truncation=True,
                                     padding='max_length',
                                     max_length=128)

                word_ids = encoding.word_ids(batch_index=0)

                sent_words, sent_gold_tags, sent_pred_tags, sent_entities = [], [], [], []
                previous_word_idx = None
                for j, word_idx in enumerate(word_ids):
                    if word_idx is None or word_idx == previous_word_idx:
                        continue
                    previous_word_idx = word_idx

                    word = raw_words[word_idx]
                    gold_label = idx2tag[gold_seq[j].item()]
                    pred_label = idx2tag[pred_seq[j]]

                    sent_words.append(word)
                    sent_gold_tags.append(gold_label)
                    sent_pred_tags.append(pred_label)

                    if pred_label != "O":
                        sent_entities.append(f"{word} : {pred_label}")

                all_words.append(sent_words)
                all_labels.append(sent_gold_tags)
                all_preds.append(sent_pred_tags)
                all_entities.append(sent_entities)

                sent_idx += 1

    # 5. 提取实体短语
    entity_phrases = set()
    for words, tags in zip(all_words, all_preds):
        i = 0
        while i < len(tags):
            tag = tags[i]
            if tag.startswith("B-"):
                entity_type = tag[2:]
                entity_tokens = [words[i]]
                i += 1
                while i < len(tags) and tags[i] == f"I-{entity_type}":
                    entity_tokens.append(words[i])
                    i += 1
                phrase = " ".join(entity_tokens)
                entity_phrases.add(f"{phrase} : {entity_type}")
            else:
                i += 1

    # 写入文件
    with open("output/predicted_entities_bert.txt", "w", encoding="utf-8") as f:
        for item in sorted(entity_phrases):
            f.write(item + "\n")

    # 打印评估报告
    precision = precision_score(all_labels, all_preds, zero_division=1)
    recall = recall_score(all_labels, all_preds, zero_division=1)
    f1 = f1_score(all_labels, all_preds, zero_division=1)
    report = classification_report(all_labels, all_preds, zero_division=1)

    with open("log/bert/log_infer.txt", "w") as f:
        print("===== Evaluation on Test Set =====", file=f)
        print(f"Precision: {precision:.4f}", file=f)
        print(f"Recall:    {recall:.4f}", file=f)
        print(f"F1 Score:  {f1:.4f}", file=f)
        print(report, file=f)

    with open("output/bert/test_labels.txt", "w") as f:
        for sentence in all_labels:
            print(sentence, file=f)
    with open("output/bert/test_predictions.txt", "w") as f:
        for sentence in all_preds:
            print(sentence, file=f)
    
####################################################################################################
# 1. 主代码
if __name__ == "__main__":
    print("请输入数字以选择操作:")
    print("1: train进行训练(生成模型文件)")
    print("2: infer进行实体抽取(基于模型文件)")
    print("3: train+infer进行实体抽取(不生成模型文件)")
    choice = input("你的选择是：").strip()

    if choice == "1":
        best_model_state = train()
        torch.save(best_model_state, "best_model_bert.pth")
    elif choice == "2":
        infer(use_saved_model=True)
    elif choice == "3":
        best_model_state = train()
        infer(use_saved_model=False, best_model_state=best_model_state)
    else:
        print("无效输入，请输入 1、2 或 3。")

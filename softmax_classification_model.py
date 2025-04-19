# 作者: toryn
# 时间: 2025/4/19
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from tqdm import tqdm
import time
import logging
from sklearn.metrics import f1_score
import numpy as np


####################################################################################################
# 1. 数据集
def load_all_relations(filename):
    # 1. 存储三元组数据，是一个一维列表
    all_relations = []

    # 2. 读数据
    # 2.1 sentences是一个列表，每个元素是一个单词列表
    # 2.2 ner是一个列表，每个元素是一个实体标注列表，表示一个句子中的所有实体
    # 2.3 relations是一个列表，每个元素是一个关系列表，表示一个句子中的所有关系，用[start1, end1, start2, end2, relation]表示
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            sentences = doc['sentences']
            ner = doc['ner']
            relations = doc['relations']

            # 3.all_tokens存储了多条句子的所有单词，是一维列表
            # 3.1 sentences_start_index记录所有句子的开头单词在all_tokens中的下标
            all_tokens = []
            sentences_start_index = []
            index = 0
            for sentence in sentences:
                all_tokens.extend(sentence)
                sentences_start_index.append(index)
                index += len(sentence)

            # 4. 遍历relations中每个[start1, end1, start2, end2, relation]，并同时获取元素下标
            for rel_index, sentence_relation in enumerate(relations):
                # 4.1 获取第0个句子的开头token在all_token中的下标
                sentence_start_index = sentences_start_index[rel_index]

                # 4.2 获取第0个句子的实体列表
                sentence_entities = ner[rel_index]  # 当前句子的实体列表

                # 4.3 遍历当前句子的每一个关系
                for entity_relation in sentence_relation:
                    # 4.4 获取[start1, end1, start2, end2, relation]
                    h_start, h_end, t_start, t_end, rel_kind = entity_relation

                    # 4.5 取出h和t对应的单词
                    h_tokens = all_tokens[h_start:h_end + 1]
                    t_tokens = all_tokens[t_start:t_end + 1]
                    h_entity = ' '.join(h_tokens)
                    t_entity = ' '.join(t_tokens)

                    all_relations.append((h_entity, t_entity, rel_kind))

    return all_relations


# 2. 获取关系表
def build_tag2idx(relations):
    all_tags = set(r for _, _, r in relations)
    return {tag: idx for idx, tag in enumerate(sorted(all_tags))}


def prepare_sentences_with_entity_merge(filename):
    merged_sentences = []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)

            # 1. 句子列表
            sentences = doc['sentences']
            all_sentences = []
            for sentence in sentences:
                all_sentences.extend(sentence)

            # 2. 句子实体列表
            ner = doc['ner']
            all_ner = []
            for single_ner in ner:
                all_ner.extend(single_ner)

            # 3. sentence_tokens是单词列表，entities是[pos1, pos2, _]列表
            # 3.1 当前句子
            merged = []

            # 3.2 [23,23,_]
            idx = 0
            for ent in all_ner:
                start, end, _ = ent  # 实体范围是 [start, end]
                if idx < start:
                    merged.extend(all_sentences[idx:start])  # 非实体 token 加入
                entity_text = ' '.join(all_sentences[start:end+1])  # 实体合并为一个词
                merged.append(entity_text)
                idx = end + 1

            # 剩下的非实体 token 加入
            if idx < len(all_sentences):
                merged.extend(all_sentences[idx:])

            merged_sentences.append(merged)

    return merged_sentences



# 3. 获取Word2Vec模型
def get_word2vec_model(sentences, vector_size=100, window=5, min_count=1, workers=4):
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    return model


####################################################################################################
# 1. 转换为向量关系
def convert_to_vector_relation(relations, word2vec_model):
    # 1.1 存储数据
    vector_relation = []

    # 1.2 遍历关系
    for h, t, r in relations:
        h_vec = torch.zeros(100)
        t_vec = torch.zeros(100)

        if h in word2vec_model.wv:
            h_vec = word2vec_model.wv[h]
        if t in word2vec_model.wv:
            t_vec = word2vec_model.wv[t]

        vector_relation.append((h_vec, t_vec, r))

    # 1.3 返回向量关系
    return vector_relation


# 2. 定义模型
class RelationClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RelationClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h_vec, t_vec):
        x = torch.cat([h_vec, t_vec], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


# 3. 数据集
class RelationDataset(Dataset):
    def __init__(self, relations, tag2idx):
        self.relations = relations
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.relations)

    def safe_tensor(self, vec):
        if isinstance(vec, torch.Tensor):
            return vec.clone().detach().float()
        elif isinstance(vec, np.ndarray):
            return torch.from_numpy(vec.copy()).float()
        else:
            raise TypeError(f"Unsupported type: {type(vec)}")

    def __getitem__(self, idx):
        h_vec, t_vec, r = self.relations[idx]
        return (
            self.safe_tensor(h_vec),
            self.safe_tensor(t_vec),
            torch.tensor(self.tag2idx[r], dtype=torch.long)
        )


####################################################################################################
# 1. 训练
def train(train_loader, dev_loader, model, criterion, optimizer, epochs=10):
    # 1.1 训练过程，输出每个epoch的损失值和F1分数
    logging.basicConfig(filename='log/softmax/log_train.txt', level=logging.INFO)

    # 1.2 最佳模型
    best_f1 = 0
    best_model = None
    for epoch in range(epochs):
        # 1.3 时间戳
        start_time = time.time()
        epoch_loss = 0.0

        # 1.4 h_ver是32*100二维张量， t_vec是32*100二维张量，r是32*1一维张量
        # 1.4 共3219条数据，一批32个，共3219/32 = 101批
        for batch in tqdm(train_loader, desc='Training'):
            h_vec, t_vec, r = batch
            h_vec = h_vec.float()
            t_vec = t_vec.float()
            r = r.long()

            optimizer.zero_grad()
            outputs = model(h_vec, t_vec)
            loss = criterion(outputs, r)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 1.5 评估
        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - start_time

        # 1.6 评估
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc='Evaluating'):
                h_vec, t_vec, r = batch
                h_vec = h_vec.float()
                t_vec = t_vec.float()
                r = r.long()

                outputs = model(h_vec, t_vec)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.numpy())
                all_labels.extend(r.numpy())

        f1 = f1_score(all_labels, all_preds, average='macro')

        logging.info(f'Epoch {epoch + 1}: Loss={avg_loss:.4f}, F1={f1:.4f}, Time={epoch_time:.2f}s')

        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            best_model = model.state_dict()

    return best_model


# 2. 推理
def infer(test_loader, all_test_relations, tag2idx, model, best_model_state):
    # 2.1 加载权重
    model.load_state_dict(best_model_state)

    # 2.2 推理
    model.eval()
    idx2tag = {v: k for k, v in tag2idx.items()}
    all_preds = []
    all_labels = []
    with open('log/softmax/log_infer.txt', 'w', encoding='utf-8') as log_file, open('output/predicted_entity_relations.txt', 'w', encoding='utf-8') as output_file:
        with torch.no_grad():
            # 2.3 获取测试集中的实体向量
            for idx, (h_vec, t_vec, r) in enumerate(test_loader):
                h_vec = h_vec.float()
                t_vec = t_vec.float()

                outputs = model(h_vec, t_vec)
                preds = torch.argmax(outputs, dim=1)

                batch_start = idx * test_loader.batch_size
                batch_end = batch_start + len(preds)

                # 获取原始的字符串对
                for i, pred in enumerate(preds):
                    h_str = all_test_relations[batch_start + i][0]  # 第一个实体
                    t_str = all_test_relations[batch_start + i][1]  # 第二个实体
                    r_str = idx2tag[pred.item()]  # 预测的关系
                    true_label = all_test_relations[batch_start + i][2]

                    all_preds.append(r_str)  # 存储预测标签
                    all_labels.append(true_label)  # 存储真实标签

                    # 输出到 predicted_entity_relations.txt
                    output_file.write(f"{h_str}\t{t_str}\t{r_str}\n")

        # 计算总体F1分数
        f1 = f1_score(all_labels, all_preds, average='macro')  # 使用宏平均来计算总体 F1 分数
        print(f"Overall F1 score: {f1:.4f}")

        # 将总体 F1 分数写入到日志文件
        with open('log/softmax/log_infer.txt', 'a') as f:
            f.write(f"Overall F1 score: {f1:.4f}\n")

####################################################################################################
# 1. 主程序
def main():
    # 1. 处理数据集
    all_train_relations = load_all_relations('data/train.json')
    all_dev_relations = load_all_relations('data/dev.json')
    all_test_relations = load_all_relations('data/test.json')

    # 2. 获取关系表
    tag2idx = build_tag2idx(all_train_relations)

    # 2. 创建Word2Vec模型
    # 2.1 获取所有实体对
    # 2.2 创建Word2Vec模型
    all_sentences = prepare_sentences_with_entity_merge('data/train.json')
    word2vec_model = get_word2vec_model(all_sentences)

    # 3. 字符串关系转换为向量关系
    all_train_vector_relations = convert_to_vector_relation(all_train_relations, word2vec_model)
    all_dev_vector_relations = convert_to_vector_relation(all_dev_relations, word2vec_model)
    all_test_vector_relations = convert_to_vector_relation(all_test_relations, word2vec_model)

    # 3. 加载数据集
    train_dataset = RelationDataset(all_train_vector_relations, tag2idx)
    dev_dataset = RelationDataset(all_dev_vector_relations, tag2idx)
    test_dataset = RelationDataset(all_test_vector_relations, tag2idx)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 4. 定义模型
    # 4.1 损失函数
    # 4.2 优化器
    input_size = 200
    hidden_size = 128
    target_size = len(tag2idx)
    model = RelationClassifier(input_size, hidden_size, target_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. 训练
    best_model_state = train(
        train_loader,dev_loader,
        model, criterion, optimizer,
        epochs=10
    )

    # 6. 推理
    infer(test_loader, all_test_relations, tag2idx, model, best_model_state)

####################################################################################################
# 1. 程序入口
if __name__ == '__main__':
    main()


# 作者: toryn
# 时间: 2025/4/19
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
import time
import logging


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


# 2. 获取所有实体对
def get_all_entity_pairs(relations):
    all_entity_pairs = []
    for relation in relations:
        all_entity_pairs.append([relation[0], relation[1]])
    return all_entity_pairs


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

    def __getitem__(self, idx):
        h_vec, t_vec, r = self.relations[idx]
        return (
            torch.tensor(h_vec, dtype=torch.float32),
            torch.tensor(t_vec, dtype=torch.float32),
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
    for epoch in range(1):
        # 1.3 时间戳
        start_time = time.time()
        epoch_loss = 0.0

        # 1.4 h_ver是32*100二维张量， t_vec是32*100二维张量，r是32*1一维张量
        for h_vec, t_vec, r in train_loader:
            h_vec = h_vec.float()
            t_vec = t_vec.float()
            r = r.long()

        #     optimizer.zero_grad()
        #     outputs = model(h_vec, t_vec)
        #     loss = criterion(outputs, r)
        #     loss.backward()
        #     optimizer.step()
        #
        #     epoch_loss += loss.item()
        #
        # end_time = time.time()
        # logging.info(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Time: {end_time - start_time:.2f}s')
        # print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Time: {end_time - start_time:.2f}s')

####################################################################################################
# 1. 主程序
def main():
    # 1. 处理数据集
    all_train_relations = load_all_relations('data/train.json')
    all_dev_relations = load_all_relations('data/dev.json')

    # max_len_entity = 0
    # max_len_rel = 0
    # with open ("data/train-relations.txt","w") as f:
    #     for rel in all_train_relations:
    #         print(rel,file=f)
    #         if len(rel[0]) > max_len_entity or len(rel[1]) > max_len_entity:
    #             max_len_entity = max(len(rel[0]),len(rel[1]))
    #         if len(rel[2]) > max_len_rel:
    #             max_len_rel = len(rel[2])
    # print(f"关系总数: {len(all_train_relations)}")
    # print(f"最大单个实体长度: {max_len_entity}")
    # print(f"最大关系长度: {max_len_rel}")

    # 2. 获取关系表
    tag2idx = build_tag2idx(all_train_relations)

    # 2. 创建Word2Vec模型
    # 2.1 获取所有实体对
    # 2.2 创建Word2Vec模型
    entity_pairs = get_all_entity_pairs(all_train_relations)
    word2vec_model = get_word2vec_model(entity_pairs)

    # 3. 字符串关系转换为向量关系
    all_train_vector_relations = convert_to_vector_relation(all_train_relations, word2vec_model)
    all_dev_vector_relations = convert_to_vector_relation(all_dev_relations, word2vec_model)

    # with open ("data/train-vector-relations.txt","w") as f:
    #     for rel in all_train_vector_relations:
    #         print(rel,file=f)
    # print(f"关系总数: {len(all_train_vector_relations)}")

    # 3. 加载数据集
    train_dataset = RelationDataset(all_train_vector_relations, tag2idx)
    dev_dataset = RelationDataset(all_dev_vector_relations, tag2idx)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

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
    best_model = train(
        train_loader, dev_loader,
        model, criterion, optimizer,
        epochs=10
    )



####################################################################################################
# 1. 程序入口
if __name__ == '__main__':
    main()
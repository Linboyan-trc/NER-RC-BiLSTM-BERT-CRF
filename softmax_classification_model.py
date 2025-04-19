# 作者: 21375077-杨荣津
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


def load_all_relations(filename):
    all_relations = []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            sentences = doc['sentences']
            ner = doc['ner']
            relations = doc['relations']

            all_tokens = []
            sentences_start_index = []
            index = 0
            for sentence in sentences:
                all_tokens.extend(sentence)
                sentences_start_index.append(index)
                index += len(sentence)

            for rel_index, sentence_relation in enumerate(relations):
                sentence_start_index = sentences_start_index[rel_index]

                sentence_entities = ner[rel_index]  # 当前句子的实体列表

                for entity_relation in sentence_relation:
                    h_start, h_end, t_start, t_end, rel_kind = entity_relation

                    h_tokens = all_tokens[h_start:h_end + 1]
                    t_tokens = all_tokens[t_start:t_end + 1]
                    h_entity = ' '.join(h_tokens)
                    t_entity = ' '.join(t_tokens)

                    all_relations.append((h_entity, t_entity, rel_kind))

    return all_relations



def build_tag2idx(relations):
    all_tags = set(r for _, _, r in relations)
    return {tag: idx for idx, tag in enumerate(sorted(all_tags))}


def prepare_sentences_with_entity_merge(filename):
    merged_sentences = []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)


            sentences = doc['sentences']
            all_sentences = []
            for sentence in sentences:
                all_sentences.extend(sentence)


            ner = doc['ner']
            all_ner = []
            for single_ner in ner:
                all_ner.extend(single_ner)


            merged = []


            idx = 0
            for ent in all_ner:
                start, end, _ = ent
                if idx < start:
                    merged.extend(all_sentences[idx:start])
                entity_text = ' '.join(all_sentences[start:end+1])
                merged.append(entity_text)
                idx = end + 1


            if idx < len(all_sentences):
                merged.extend(all_sentences[idx:])

            merged_sentences.append(merged)

    return merged_sentences



def get_word2vec_model(sentences, vector_size=100, window=5, min_count=1, workers=4):
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    return model



def convert_to_vector_relation(relations, word2vec_model):

    vector_relation = []


    for h, t, r in relations:
        h_vec = torch.zeros(100)
        t_vec = torch.zeros(100)

        if h in word2vec_model.wv:
            h_vec = word2vec_model.wv[h]
        if t in word2vec_model.wv:
            t_vec = word2vec_model.wv[t]

        vector_relation.append((h_vec, t_vec, r))


    return vector_relation



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



def train(train_loader, dev_loader, model, criterion, optimizer, epochs=10):

    logging.basicConfig(filename='log/softmax/log_train.txt', level=logging.INFO)


    best_f1 = 0
    best_model = None
    for epoch in range(epochs):

        start_time = time.time()
        epoch_loss = 0.0


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


        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - start_time


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


        if f1 > best_f1:
            best_f1 = f1
            best_model = model.state_dict()

    return best_model



def infer(test_loader, all_test_relations, tag2idx, model, best_model_state):

    model.load_state_dict(best_model_state)


    model.eval()
    idx2tag = {v: k for k, v in tag2idx.items()}
    all_preds = []
    all_labels = []
    with open('log/softmax/log_infer.txt', 'w', encoding='utf-8') as log_file, open('output/predicted_entity_relations.txt', 'w', encoding='utf-8') as output_file:
        with torch.no_grad():

            for idx, (h_vec, t_vec, r) in enumerate(test_loader):
                h_vec = h_vec.float()
                t_vec = t_vec.float()

                outputs = model(h_vec, t_vec)
                preds = torch.argmax(outputs, dim=1)

                batch_start = idx * test_loader.batch_size
                batch_end = batch_start + len(preds)


                for i, pred in enumerate(preds):
                    h_str = all_test_relations[batch_start + i][0]
                    t_str = all_test_relations[batch_start + i][1]
                    r_str = idx2tag[pred.item()]
                    true_label = all_test_relations[batch_start + i][2]

                    all_preds.append(r_str)
                    all_labels.append(true_label)


                    output_file.write(f"{h_str}\t{t_str}\t{r_str}\n")


        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Overall F1 score: {f1:.4f}")

        # 将总体 F1 分数写入到日志文件
        with open('log/softmax/log_infer.txt', 'a') as f:
            f.write(f"Overall F1 score: {f1:.4f}\n")


def main():

    all_train_relations = load_all_relations('data/train.json')
    all_dev_relations = load_all_relations('data/dev.json')
    all_test_relations = load_all_relations('data/test.json')


    tag2idx = build_tag2idx(all_train_relations)


    all_sentences = prepare_sentences_with_entity_merge('data/train.json')
    word2vec_model = get_word2vec_model(all_sentences)


    all_train_vector_relations = convert_to_vector_relation(all_train_relations, word2vec_model)
    all_dev_vector_relations = convert_to_vector_relation(all_dev_relations, word2vec_model)
    all_test_vector_relations = convert_to_vector_relation(all_test_relations, word2vec_model)


    train_dataset = RelationDataset(all_train_vector_relations, tag2idx)
    dev_dataset = RelationDataset(all_dev_vector_relations, tag2idx)
    test_dataset = RelationDataset(all_test_vector_relations, tag2idx)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = 200
    hidden_size = 128
    target_size = len(tag2idx)
    model = RelationClassifier(input_size, hidden_size, target_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    best_model_state = train(
        train_loader,dev_loader,
        model, criterion, optimizer,
        epochs=10
    )


    infer(test_loader, all_test_relations, tag2idx, model, best_model_state)


# 1. 程序入口
if __name__ == '__main__':
    main()


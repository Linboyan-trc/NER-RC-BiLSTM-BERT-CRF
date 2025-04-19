# 作者: 21375077-杨荣津
# 时间: 2025/4/18

import torch
import torch.nn as nn
from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel
from tqdm import tqdm
from torch.optim import AdamW
import json
import time
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

def convert_to_BIO(sentences, ner):
    all_tokens = []
    for sentence in sentences:
        for token in sentence:
            all_tokens.append(token)

    token_labels = ['O'] * len(all_tokens)

    for single_sentence_ner in ner:
        for start, end, label in single_sentence_ner:
            token_labels[start] = f'B-{label}'
            for i in range(start + 1, end + 1):
                token_labels[i] = f'I-{label}'

    index = 0
    sentences_result = []
    for sentence in sentences:
        sentence_single_result = []
        for token in sentence:
            sentence_single_result.append((token, token_labels[index]))
            index += 1

        sentences_result.append(sentence_single_result)

    return sentences_result


def get_all_files_bios(filename):
    with open(filename) as f:
        datas = [json.loads(line) for line in f]

    all_file_bios = []

    for data_line in datas:
        bio = convert_to_BIO(data_line["sentences"], data_line["ner"])
        all_file_bios.extend(bio)

    return all_file_bios


def get_words(filename):
    with open(filename) as f:
        datas = [json.loads(line) for line in f]

    all_words = []
    for data_line in datas:
        for sentence in data_line["sentences"]:
            all_words.append(sentence)

    return all_words


def build_tag_map(all_file_bios):
    tag_set = set()

    for sentence in all_file_bios:
        for token, tag in sentence:
            tag_set.add(tag)

    tag_list = sorted(list(tag_set))

    tag2idx = {label: idx for idx, label in enumerate(tag_list)}
    idx2tag = {idx: label for label, idx in tag2idx.items()}
    return tag2idx, idx2tag


class BERT_CRF(nn.Module):
    def __init__(self, tagset_size):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.fc = nn.Linear(self.bert.config.hidden_size, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, x, tags=None, mask=None):
        outputs = self.bert(input_ids=x, attention_mask=mask)
        emissions = self.fc(outputs.last_hidden_state)

        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask.bool(), reduction='mean')
            return loss
        else:
            pred = self.crf.decode(emissions, mask=mask.bool())
            return pred



class NERDataset(Dataset):
    def __init__(self, data, tokenizer, tag2idx, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.tag2idx = tag2idx
        self.max_len = max_len


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        sentence = self.data[idx]

        words = []
        for token, tag in sentence:
            words.append(token)

        tags = []
        for token, tag in sentence:
            tags.append(self.tag2idx[tag])

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors='pt',

            max_length=self.max_len,
            padding='max_length',
            truncation=True,

            return_offsets_mapping=True
        )


        word_ids = encoding.word_ids(batch_index=0)


        tag_ids = []
        for word_id in word_ids:
            if word_id is None:
                tag_ids.append(0)
            else:
                tag_ids.append(tags[word_id])

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'tag_ids': torch.tensor(tag_ids),
            'attention_mask': encoding['attention_mask'].squeeze()
        }



def training(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc='Training'):
        optimizer.zero_grad()
        input_data = batch['input_ids']
        target = batch['tag_ids']
        mask = batch['attention_mask']

        loss = model(input_data, tags=target, mask=mask)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, idx2tag):
    model.eval()

    dev_predictions = []
    dev_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_data = batch['input_ids']
            target = batch['tag_ids'].numpy()
            mask = batch['attention_mask']

            bio_tags_predicted = model(input_data, mask=mask)
            dev_predictions += bio_tags_predicted

            for i in range(len(bio_tags_predicted)):
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


def train():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    train_all_file_bios = get_all_files_bios("data/train.json")
    dev_all_file_bios = get_all_files_bios("data/dev.json")

    tag2idx, idx2tag = build_tag_map(train_all_file_bios)

    train_dataset = NERDataset(train_all_file_bios, tokenizer, tag2idx)
    dev_dataset = NERDataset(dev_all_file_bios, tokenizer, tag2idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=True)

    model = BERT_CRF(tagset_size=len(tag2idx))
    optimizer = AdamW(model.parameters(), lr=3e-5)

    epochs = 10
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

    with open("output/predicted_entities_bert.txt", "w", encoding="utf-8") as f:
        for item in sorted(entity_phrases):
            f.write(item + "\n")

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

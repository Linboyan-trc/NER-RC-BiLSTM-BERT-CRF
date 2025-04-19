# # 作者: 21375077-杨荣津
# 时间: 2025/4/15
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn, optim
from torchcrf import CRF
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



def build_vocab(all_file_bios):
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    tag2idx = {'<PAD>': 0}
    for single_sentence_bios in all_file_bios:
        for word, tag in single_sentence_bios:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
            if tag not in tag2idx:
                tag2idx[tag] = len(tag2idx)
    return word2idx, tag2idx



class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=256):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(hidden_dim, tagset_size)

        self.crf = CRF(tagset_size, batch_first=True)


    def forward(self, x, tags=None, mask=None):

        embeds = self.embedding(x)


        lstm_out, _ = self.lstm(embeds)


        emissions = self.fc(lstm_out)


        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss


        else:
            return self.crf.decode(emissions, mask=mask)



class NERDataset(Dataset):
    def __init__(self, data, word2idx, tag2idx, max_len=128):
        self.data = data
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_len = max_len


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        words, tags = zip(*self.data[idx])

        words = list(words)[:self.max_len]
        tags = list(tags)[:self.max_len]

        word_ids = []
        for w in words:
            if w in self.word2idx:
                word_ids.append(self.word2idx[w])
            else:
                word_ids.append(self.word2idx['<UNK>'])

        tag_ids = []
        for t in tags:
            tag_ids.append(self.tag2idx[t])

        tail_length = self.max_len - len(word_ids)
        word_ids += [self.word2idx['<PAD>']] * tail_length
        tag_ids += [self.tag2idx['<PAD>']] * tail_length

        mask = [True] * len(words) + [False] * tail_length

        return torch.tensor(word_ids), torch.tensor(tag_ids), torch.tensor(mask)



def train():
    train_all_file_bios = get_all_files_bios("data/train.json")
    dev_all_file_bios = get_all_files_bios("data/dev.json")

    word2idx, tag2idx = build_vocab(train_all_file_bios)

    model = BiLSTM_CRF(vocab_size=len(word2idx), tagset_size=len(tag2idx))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = NERDataset(train_all_file_bios, word2idx, tag2idx)
    dev_dataset = NERDataset(dev_all_file_bios, word2idx, tag2idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

    best_dev_f1 = 0
    best_model_state = None

    epochs = 20
    with open("log/bilstm/log_train.txt", "w") as f:
        pass
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        for batch in train_loader:
            # 6.3 获取每个批次的数据
            # 6.3.1 每个批次共3个张量，每个张量都有32个向量，每个向量128维
            # 6.3.2 第一个张量是32个词向量，第二个张量是32个标签向量，第三个张量是32个掩码，每个张量32行，128列
            # 6.3.3 因此input_data是一个32*128的张量，对应32个句子的词向量，target是一个32*128的张量，对应32个句子的标签向量
            optimizer.zero_grad()
            input_data, target, mask = batch

            # 6.4 就是调用BiLSTM_CRF的forword方法
            # 6.4 对32个句子进行embedding + lstm + linear计算，最终crf计算损
            loss = model(input_data, tags=target, mask=mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time

        model.eval()

        dev_predictions = []
        dev_labels = []
        with torch.no_grad():
            for batch in dev_loader:
                input_data, target, mask = batch

                bio_tags_predicted = model(input_data, mask=mask)
                dev_predictions += bio_tags_predicted

                for i in range(len(bio_tags_predicted)):
                    single_sentence_true_tag_unchunked = target[i]
                    single_sentence_mask_unchunked = mask[i]
                    single_sentence_true_tag = []
                    for j in range(len(single_sentence_mask_unchunked)):
                        if single_sentence_mask_unchunked[j]:
                            single_sentence_true_tag.append(single_sentence_true_tag_unchunked[j].item())
                    dev_labels.append(single_sentence_true_tag)

        idx2tag = {v: k for k, v in tag2idx.items()}
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

        with open("output/bilstm/dev_labels.txt", "w") as f:
            for sentence in dev_labels_str:
                print(sentence, file=f)
        with open("output/bilstm/dev_predictions.txt", "w") as f:
            for sentence in dev_predictions_str:
                print(sentence, file=f)

        precision = precision_score(dev_labels_str, dev_predictions_str, zero_division=1)
        recall = recall_score(dev_labels_str, dev_predictions_str, zero_division=1)
        f1 = f1_score(dev_labels_str, dev_predictions_str, zero_division=1)
        report = classification_report(dev_labels_str, dev_predictions_str, zero_division=1)

        with open("log/bilstm/log_train.txt", "a") as f:
            print(f"####################################################################################################",file=f)
            print(f"Epoch {epoch + 1}:", file=f)
            print(f"  - Avg Training Loss: {avg_loss:.4f}", file=f)
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

    train_all_file_bios = get_all_files_bios("data/train.json")
    test_all_file_bios = get_all_files_bios("data/test.json")
    test_words = get_words("data/test.json")


    word2idx, tag2idx = build_vocab(train_all_file_bios)
    idx2tag = {v: k for k, v in tag2idx.items()}
    idx2word = {v: k for k, v in word2idx.items()}


    test_dataset = NERDataset(test_all_file_bios, word2idx, tag2idx)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    model = BiLSTM_CRF(vocab_size=len(word2idx), tagset_size=len(tag2idx))


    if use_saved_model:
        model.load_state_dict(torch.load("best_model_bilstm.pth"))
    else:
        model.load_state_dict(best_model_state)


    model.eval()
    all_preds = []
    all_labels = []
    all_entities = []
    all_words = []

    cnt = 0
    with torch.no_grad():
        for batch in test_loader:
            input_data, target, mask = batch


            pred_tags = model(input_data, mask=mask)


            for i in range(len(pred_tags)):

                word_seq = input_data[i]
                gold_seq = target[i]
                mask_seq = mask[i]
                pred_seq = pred_tags[i]

                sent_words = []
                sent_gold_tags = []
                sent_pred_tags = []

                sent_entities = []


                for j in range(len(mask_seq)):

                    if mask_seq[j]:
                        word = test_words[i + 32 * cnt][j]
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

            cnt += 1


    entity_phrases = set()

    for entity_list in all_preds:
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


    with open("output/predicted_entities_bilstm.txt", "w", encoding="utf-8") as f:
        for item in sorted(entity_phrases):
            f.write(item + "\n")


    precision = precision_score(all_labels, all_preds, zero_division=1)
    recall = recall_score(all_labels, all_preds, zero_division=1)
    f1 = f1_score(all_labels, all_preds, zero_division=1)
    report = classification_report(all_labels, all_preds, zero_division=1)

    with open("log/bilstm/log_infer.txt", "w") as f:
        print("===== Evaluation on Test Set =====", file=f)
        print(f"Precision: {precision:.4f}", file=f)
        print(f"Recall:    {recall:.4f}", file=f)
        print(f"F1 Score:  {f1:.4f}", file=f)
        print(report, file=f)


    with open("output/bilstm/test_labels.txt", "w") as f:
        for sentence in all_labels:
            print(sentence, file=f)
    with open("output/bilstm/test_predictions.txt", "w") as f:
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
        torch.save(best_model_state, "best_model_bilstm.pth")
    elif choice == "2":
        infer(use_saved_model=True)
    elif choice == "3":
        best_model_state = train()
        infer(use_saved_model=False, best_model_state=best_model_state)
    else:
        print("无效输入，请输入 1、2 或 3。")
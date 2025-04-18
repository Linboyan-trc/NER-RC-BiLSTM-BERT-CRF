# 作者: toryn
# 时间: 2025/4/18
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn, optim
from torchcrf import CRF
import time
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report


################################################################################################################################################################
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
    token_labels = ['O'] * len(all_tokens)  # 默认每个token都是 O

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


################################################################################################################################################################
# 1. 构建单词表和标签表
# 1.1 使得用token可以映射到唯一的数字
# 1.2 使得用BIO的标签可以映射到唯一的数字
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


# 2. 定义模型
class BiLSTM_CRF(nn.Module):
    # 2.1 初始化模型，需要单词表大小和标签表大小
    # 2.2 训练集单词表大小为6000作用，词嵌入维度设置为100即可
    # 2.2 词嵌入维度就是一个单词用(1,2,...100)一个有100个元素的向量表示
    # 2.3 词嵌入维度100，只表示单词的初始表示
    # 2.3 隐藏层维度256，表示一个单词结合上下文之后，在记忆中占用的真实向量的大小
    # 2.3 并且在BiLSTM中，从前往后读和从后往前读并行执行，前者维护每个单词隐藏层维度的前128个元素，后者维护每个单词隐藏层维度的后128个元素，互不干扰，最终一个单词的隐藏层维度是256
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=256):
        super().__init__()
        # 2.1 self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)本质是实例化了一个nn.Embedding类
        # 2.1 这个类通过初始化，规定了这个实例中包含一个6000*100的张量，通过单词的索引去获得张量中具体的某一行作为单词的词向量
        # 2.1 并且通过padding_idx = 0, 指定词向量(0,0,0,...0)对应的单词索引是0，也就是对于单词索引为0的单词，也就是word2idx中的<PAD>:0，默认词向量为(0,0,0,...,0)
        # 2.1 这6000个100维的词向量在一开始就以及初始化好，是随机的6000个向量值
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 2.2 self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)实例化了nn.LSTM类
        # 2.2 初始化规定这个实例在计算的时候，输入必须是100维
        # 2.2 初始化规定这个实例在计算的时候，输出是256维，启用双向LSTM
        # 2.2 初始化规定这个实例在计算的时候，默认本来是将输入的32*128*100的张量当作128*32*100来处理，现在当作32*128*100来处理
        # 2.2 因为规定第一维叫作seq_len，第二维叫作batch_size，第三维叫作input_size; 默认以输入的第二维作为batch_size
        # 2.2 但实际在我们的数据中，第一维才是batch_size，因此batch_first=True可以让第一维和第二维交换名字，使得LSTM识别32为batch_size
        # 2.3 最终返回一个32*128*256的向量
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)

        # 2.4 一个线性变换层
        # 2.4 输入维度是256，输出维度是14
        # 2.4 对于一个线性变换层，输入维度只需要和输入张量的最后一个维度的大小对齐
        # 2.4 比如对于输入的张量32*128*256，和32*32*32*128*256，self.fc = nn.Linear(256,14)对于这两者都可以处理
        # 2.5 而计算过程，就是对32*128*256中每个256维的柱向量进行加权计算
        # 2.5 得到的14维向量的第0个元素y0 = 256个w0系数 去和 256维柱向量的每个元素做加权，然后 + b0，w[0][0]~w[0][256]各不相同
        # 2.5 得到的14维向量的第1个元素y1 = 256个w1系数 去和 256维柱向量的每个元素做加权，然后 + b1，w[1][0]~w[1][256]各不相同
        self.fc = nn.Linear(hidden_dim, tagset_size)

        # 2.6 标签数量14，输入的emissions的32*128*14，batch_size为第一维数值，后两维为seq_len句子长度，num_tags标签数量
        self.crf = CRF(tagset_size, batch_first=True)

    # 2.1 模型计算
    # 2.2 x是32*128的词向量张量，tags是32*128的标签向量张量，mask是32*128的掩码
    def forward(self, x, tags=None, mask=None):
        # 2.1 embeds = self.embedding(x)的时候
        # 2.1 这个实例会调用默认的方法去计算x，具体计算操作就是根据32*128每一个元素的值，也就是一个整数，是单词在单词表中的索引
        # 2.1 去找到6000*100中下标对应的那一行词向量，最后返回一个32*128*100的张量，包含了32*128个单词及其对应的词向量
        # 2.2 也就是说最后变成了一个32*128*100的立方体，每一个柱子有100个小方格，每个小方格有一个数值，这100个小方格组成了一个单词的词向量
        embeds = self.embedding(x)

        # 2.2 对32*128*100的embeds计算，得出一个32*128*256的张量
        # 2.2 此时已经具备了初步的记忆
        lstm_out, _ = self.lstm(embeds)

        # 2.3 对32*128*256的lstm_out计算，得出一个32*128*14的张量
        emissions = self.fc(lstm_out)

        # 2.4 计算损失
        # 2.4 根据最后线性层的输出32*128*14，以及本批次的标签张量32*128，本批次的掩码32*128，计算损失
        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss

        # 2.5 推理的时候用
        else:
            return self.crf.decode(emissions, mask=mask)


# 3. NERDataset
# 3.1 这个数据集的样本数是整个train.json的句子数量，单个样本是指单个句子
class NERDataset(Dataset):
    # 3.1 self.data就是一个列表，每个元素对应一个句子的BIO序列
    # 3.2 同时给定了训练集的单词表、标签表
    # 3.3 max_len是单个句子最大长度，训练集中单个句子最长101个token，因此max_len设置为128
    def __init__(self, data, word2idx, tag2idx, max_len=128):
        self.data = data
        self.word2idx = word2idx
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
        # 3.1.1 先执行data[0]，取出列表[("A","I"),("B","I"),("C","I")]
        # 3.1.2 然后*解包成三个单独的元素("A","I"),("B","I"),("C","I")
        # 3.1.3 zip对每个单词的元素分开，所有元素的第一个位置形成新的列表，所有元素的第二个位置形成新的列表，最终得到指定句子的所有单词，和所有标签
        words, tags = zip(*self.data[idx])

        # 3.2 限制句子的单词数量<=128，单词对应标签数量<=128
        words = list(words)[:self.max_len]
        tags = list(tags)[:self.max_len]

        # 3.3 将每个单词转化为单词表中的索引
        word_ids = []
        for w in words:
            if w in self.word2idx:
                word_ids.append(self.word2idx[w])
            else:
                word_ids.append(self.word2idx['<UNK>'])

        # 3.4 将每个标签转化为标签表中的索引
        tag_ids = []
        for t in tags:
            tag_ids.append(self.tag2idx[t])

        # 3.5 对句子做补齐，使得单个句子的单词表、标签表大小最终都为128
        tail_length = self.max_len - len(word_ids)
        word_ids += [self.word2idx['<PAD>']] * tail_length
        tag_ids += [self.tag2idx['<PAD>']] * tail_length

        # 3.6 制作掩码，标记单词表和标签表的128个元素中，哪些元素有效
        mask = [True] * len(words) + [False] * tail_length

        # 3.7
        # 返回三个向量
        # 这三个向量都有128个元素，分别是一个句子中每个单词在单词表中索引，每个标签在标签表中的索引，和一个128维的掩码
        return torch.tensor(word_ids), torch.tensor(tag_ids), torch.tensor(mask)


####################################################################################################
# 1. 训练
def train():
    # 1. 处理数据
    train_all_file_bios = get_all_files_bios("data/train.json")
    dev_all_file_bios = get_all_files_bios("data/dev.json")

    # 2. 根据训练集构建单词表和标签表
    word2idx, tag2idx = build_vocab(train_all_file_bios)

    # 3. 定义模型
    # 3.1 损失函数用于评估当前误差，使用交叉熵
    # 3.2 优化器用于调整模型参数，减少误差
    # 3.3 在优化器中设置了学习率lr = 0.001
    model = BiLSTM_CRF(vocab_size=len(word2idx), tagset_size=len(tag2idx))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4.加载为Dataset
    # 4.1 对齐接口
    train_dataset = NERDataset(train_all_file_bios, word2idx, tag2idx)
    dev_dataset = NERDataset(dev_all_file_bios, word2idx, tag2idx)

    # 5.传入数据
    # 5.1 训练集中共1861条数据，每次处理32个句子，batch_size设置为32
    # 5.2 因此每个epoch会分为1861/32 = 59次来批处理数据，每个epoch开始前会对1861个数据进行打乱，一共训练5个epoch
    # 5.3 而损失函数和优化器在每次批处理后就工作，去更新模型参数，因此损失函数和优化器一共工作5 * 59 = 295次
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

    # 训练过程
    best_dev_f1 = 0
    best_model_state = None

    # 6. 训练
    # 6.1 对1861条数据训练5次，每次分59批训练，一批有32个句子
    epochs = 20
    with open("log/bilstm/log_train.txt", "w") as f:
        pass
    for epoch in range(epochs):
        # 6.1 对每个批训练
        # 6.1.1 加载1861条数据后，设置batch_size=32，train_loader中共有59批数据，训练59批，在这个过程中已经在进行参数更新
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

        # 6.2 训练完59批，进行评估
        model.eval()

        # 6.3 加载验证集的数据，共50条，2批
        dev_predictions = []
        dev_labels = []
        with torch.no_grad():
            for batch in dev_loader:
                # 6.3.1 input_data, target, mask都是32*128的张量
                input_data, target, mask = batch

                # 6.3.2 得到32个句子的tags预测结果，是一个有32个元素的列表，每个元素是对应单句的tags预测结果
                bio_tags_predicted = model(input_data, mask=mask)
                dev_predictions += bio_tags_predicted

                # 6.3.3 遍历32个句子，将预测的tags和真实的tags汇成两个列表
                for i in range(len(bio_tags_predicted)):
                    # 6.3.4 单个句子的真实标签索引，这里会有padding，需要结合mask去掉padding
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


# 2. 推理
def infer(use_saved_model=True, best_model_state=None):
    # 1. 加载数据
    train_all_file_bios = get_all_files_bios("data/train.json")
    test_all_file_bios = get_all_files_bios("data/test.json")
    test_words = get_words("data/test.json")

    # 2. 构建词表和标签表
    word2idx, tag2idx = build_vocab(train_all_file_bios)
    idx2tag = {v: k for k, v in tag2idx.items()}
    idx2word = {v: k for k, v in word2idx.items()}

    # 3. 构建测试集 Dataset 和 DataLoader
    test_dataset = NERDataset(test_all_file_bios, word2idx, tag2idx)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 4. 创建模型
    model = BiLSTM_CRF(vocab_size=len(word2idx), tagset_size=len(tag2idx))

    # 5. 加载权重
    if use_saved_model:
        model.load_state_dict(torch.load("best_model_bilstm.pth"))
    else:
        model.load_state_dict(best_model_state)

    # 6. 推理
    model.eval()
    all_preds = []
    all_labels = []
    all_entities = []
    all_words = []

    cnt = 0
    with torch.no_grad():
        for batch in test_loader:
            input_data, target, mask = batch

            # 5.1 一个列表，每个元素也是列表，是一个句子的预测标签
            pred_tags = model(input_data, mask=mask)

            # 5.2 遍历列表，取出每个句子
            for i in range(len(pred_tags)):
                # 5.2.1 word_seq = [tensor(1),...]
                # 5.2.1 gold_seq = [tensor(1),...]
                # 5.2.1 mask = [True,...]
                # 5.2.1 pred_seq = [1,2,4,...]
                word_seq = input_data[i]
                gold_seq = target[i]
                mask_seq = mask[i]
                pred_seq = pred_tags[i]

                sent_words = []
                sent_gold_tags = []
                sent_pred_tags = []

                sent_entities = []

                # 5.2.2 遍历单个句子的mask
                for j in range(len(mask_seq)):
                    # 5.2.3 单词有效
                    # 5.2.3 获取单词索引，然后索引转换成字符串
                    # 5.2.3 单词真实标签，转换为字符串
                    # 5.2.3 单词预测标签，转换为字符串
                    # 5.2.4 然后添加到句子的单词，句子的真实标签，句子的预测标签中
                    # 5.2.5 如果单词的标签不是"O"，就加入到句子实体
                    if mask_seq[j]:
                        word = test_words[i + 32 * cnt][j]
                        gold_label = idx2tag[gold_seq[j].item()]
                        pred_label = idx2tag[pred_seq[j]]

                        sent_words.append(word)
                        sent_gold_tags.append(gold_label)
                        sent_pred_tags.append(pred_label)

                        if pred_label != "O":
                            sent_entities.append(f"{word} : {pred_label}")

                # 5.2.6 将单个句子的单词，真实标签，预测标签，实体对，加入到全局列表中
                all_words.append(sent_words)
                all_labels.append(sent_gold_tags)
                all_preds.append(sent_pred_tags)
                all_entities.append(sent_entities)

            cnt += 1

    # 保存实体提取结果
    # 保存实体短语的集合，避免重复
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

    # 写入文件
    with open("output/predicted_entities_bilstm.txt", "w", encoding="utf-8") as f:
        for item in sorted(entity_phrases):
            f.write(item + "\n")

    # 打印评估报告
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

    # 8. 保存预测结果
    with open("output/bilstm/test_labels.txt", "w") as f:
        for sentence in all_labels:
            print(sentence, file=f)
    with open("output/bilstm/test_predictions.txt", "w") as f:
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
        torch.save(best_model_state, "best_model_bilstm.pth")
    elif choice == "2":
        infer(use_saved_model=True)
    elif choice == "3":
        best_model_state = train()
        infer(use_saved_model=False, best_model_state=best_model_state)
    else:
        print("无效输入，请输入 1、2 或 3。")
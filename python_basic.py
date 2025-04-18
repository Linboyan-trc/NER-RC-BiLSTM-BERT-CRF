# # 1. zip和*的作用
# # 1.1 先执行data[0]，取出列表[("A","I"),("B","I"),("C","I")]
# # 1.2 然后*解包成三个单独的元素("A","I"),("B","I"),("C","I")
# # 1.3 zip对每个单词的元素分开，所有元素的第一个位置形成新的列表，所有元素的第二个位置形成新的列表
# data = [
#     [("A","I"),("B","I"),("C","I")],
#     [("D","I"),("E","I"),("F","I")]
# ]
# words, tags = zip(*data[0])
# print(words, tags)

# 2. 查看一个batch_size = 32中，inputs, data = batch具体是什么
import json
from collections import defaultdict
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torchcrf import CRF
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import classification_report

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
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, x, tags=None, mask=None):
        print(x.shape)

        embeds = self.embedding(x)
        print(embeds.shape)

        lstm_out, _ = self.lstm(embeds)
        print(lstm_out.shape)

        emissions = self.fc(lstm_out)
        print(emissions)
        print(emissions.shape)

        if tags is not None:
            # 训练时计算 loss
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            # 推理时返回预测路径
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

################################################################################################################################################################
# 1. 处理数据
train_all_file_bios = get_all_files_bios("data/train.json")
# dev_all_file_bios = get_all_files_bios("dev.json")
# test_all_file_bios = get_all_files_bios("test.json")

# 2. 根据训练集构建单词表和标签表
word2idx, tag2idx = build_vocab(train_all_file_bios)

# 3. 定义模型
# 3.1 损失函数用于评估当前误差，使用交叉熵
# 3.2 优化器用于调整模型参数，减少误差
# 3.3 在优化器中设置了学习率lr = 0.001
model = BiLSTM_CRF(vocab_size=len(word2idx), tagset_size=len(tag2idx))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4.加载为Dataset
# 4.1 对齐接口
train_dataset = NERDataset(train_all_file_bios, word2idx, tag2idx)
# dev_dataset = NERDataset(dev_data, word2idx, tag2idx)
# test_dataset = NERDataset(test_data, word2idx, tag2idx)

# 5.传入数据
# 5.1 训练集中共1861条数据，每次处理32个句子，batch_size设置为32
# 5.2 因此每个epoch会分为1861/32 = 59次来批处理数据，每个epoch开始前会对1861个数据进行打乱，一共训练5个epoch
# 5.3 而损失函数和优化器在每次批处理后就工作，去更新模型参数，因此损失函数和优化器一共工作5 * 59 = 295次
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# print(f"加载1861条数据后，设置batch_size=32，train_loader中共有{len(train_loader)}批数据")
for batch in train_loader:
    # print(f"每个批次共{len(batch)}个张量，每个张量都有32个向量，每个向量128维")
    # print("第一个张量是32个词向量，第二个张量是32个标签向量，第三个张量是32个掩码")
    # print("每个张量32行，128列")
    # for item in batch:
    #     print(item)

    input_data, target, mask = batch
    print(target)

    loss = model(input_data, tags=target, mask=mask)

    for bio_tags_predicted in model(input_data, mask=mask):
        print(bio_tags_predicted)
    break


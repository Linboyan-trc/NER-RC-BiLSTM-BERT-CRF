# 作者: toryn
# 时间: 2025/4/16

# 1. tensor
# import torch
#
# target = torch.tensor([
#     [1,2],
#     [3,4]
# ])
# labels = []
# labels.extend(target[0].tolist())
# labels.extend(target[1].tolist())
# print(labels)

# 2. dict
# tag2idx = {"<PAD>":1,"B-Material":2}
# valid_labels = [tag for tag in tag2idx.keys() if tag != "<PAD>"]
# valid_labels_idx = [idx for tag, idx in tag2idx.items() if tag != "<PAD>"]
# print(valid_labels)
# print(valid_labels_idx)

# 3. convert tensor
# import torch
# a_tensor = torch.tensor([1, 2, 3])
# a_list = a_tensor.tolist()
# print(f"a_tensor: {a_tensor}")
# print(f"a_tensor[0]: {a_tensor[0]}")
# print(f"a_list: {a_list}")
# print(f"a_list[0]: {a_list[0]}")
#
# b_tensor = torch.tensor([[1, 2, 3],[4, 5, 6]])
# print(f"b_tensor: {b_tensor}")
# print(f"b_tensor[0]: {b_tensor[0]}")

# 4. AdamW
# from transformers import AdamW
# print(AdamW)

# 5. tokenizer
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
tokens = tokenizer(
            ["I","Love","You", "I", "Beijing", "Main", "Square","."],
            is_split_into_words=True,
            return_tensors='pt',

            max_length=16,
            padding='max_length',
            truncation=True,

            return_offsets_mapping=True
        )
for i in tokens:
    print(i,end=" ")
    print(tokens[i])
print(tokens.word_ids(batch_index=0))
print(tokens['input_ids'].squeeze())


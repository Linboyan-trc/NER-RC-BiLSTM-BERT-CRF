# 作者: toryn
# 时间: 2025/4/19
import json


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
####################################################################################################
# 1. 主程序
def main():
    # 2. 创建Word2Vec模型
    # 2.1 获取所有实体对
    # 2.2 创建Word2Vec模型
    all_sentences = prepare_sentences_with_entity_merge('data/train.json')
    with open ("data/train-relation-sentences.txt", "w") as f:
        for sentence in all_sentences:
            print(sentence, file=f)

####################################################################################################
# 1. 程序入口
if __name__ == '__main__':
    main()
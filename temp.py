# 作者: toryn
# 时间: 2025/4/19

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
            mask = batch['attention_mask'].numpy()

            # 2.2.2 得到32个句子的tags预测结果，是一个有32个元素的列表，每个元素是对应单句的tags预测结果
            bio_tags_predicted = model(input_data, mask=mask)
            dev_predictions += bio_tags_predicted

            # 2.2.3 这里bio_tags_predicted, target, mask都是二维列表，每个元素是句子的预测标签，句子的真实标签，句子的掩码
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
    epochs = 10
    best_dev_f1 = 0
    best_model_state = None
    with open("log/bert/log_train.txt", "w") as f:
        pass
    for epoch in range(epochs):
        with open("log/bilstm/log_train.txt", "a") as f:
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
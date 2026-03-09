import json
import torch
import torch.nn as nn
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from collections import defaultdict
import re
import subprocess
import sys


# --------------------------
# 检查并安装必要依赖
# --------------------------
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# 检查并安装accelerate
try:
    import accelerate

    assert accelerate.__version__ >= "1.1.0"
except (ImportError, AssertionError):
    print("正在安装accelerate>=1.1.0...")
    install_package("accelerate>=1.1.0")
    import accelerate

# 确保transformers版本兼容
try:
    from transformers import __version__ as transformers_version
    from packaging import version

    assert version.parse(transformers_version) >= version.parse("4.20.0")
except:
    print("正在更新transformers...")
    install_package("transformers>=4.20.0")


# --------------------------
# 修复字体问题 - 自动适配系统字体
# --------------------------
def setup_matplotlib_font():
    try:
        font_list = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        chinese_fonts = [f for f in font_list if any(c in f.lower() for c in ['hei', 'song', 'kai', 'ming'])]
        if chinese_fonts:
            font_name = font_manager.FontProperties(fname=chinese_fonts[0]).get_name()
            plt.rcParams["font.family"] = font_name
        else:
            plt.rcParams["font.family"] = "DejaVu Sans"
    except Exception:
        plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


setup_matplotlib_font()

# --------------------------
# 1. 配置参数（简化版，仅保留核心）
# --------------------------
MODEL_PATH = "google-bert/bert-base-chinese"
DATA_PATH = "merged_can_dataset3.json"
NUM_ATTACK_CLASSES = 4  # normal, fuzzy, dos, spoofing_gear
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3  # 修改为3轮训练
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.1
DROPOUT_RATE = 0.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 标签映射
LABEL2ID = {"normal": 0, "fuzzy": 1, "dos": 2, "spoofing_gear": 3}
ID2LABEL = {0: "normal", 1: "fuzzy", 2: "dos", 3: "spoofing_gear"}
CLASS_NAMES = list(LABEL2ID.keys())
ATTACK_TYPES = ["fuzzy", "dos", "spoofing_gear"]  # 异常攻击类型


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"使用随机种子: {seed}")


set_seed()


# --------------------------
# 数据增强：仅适配CAN数据特征
# --------------------------
def augment_text(text, aug_prob=0.1):
    """仅对CAN数据文本特征进行随机扰动"""
    if random.random() < aug_prob:
        # 随机替换CAN相关数字
        def random_replace_num(match):
            num = match.group()
            if '.' in num:
                return f"{random.uniform(1478195728, 1478195800):.6f}"
            elif num.startswith('0x'):
                return f"0x{random.randint(0, 4095):x}"
            else:
                if num.isdigit() and 0 <= int(num) <= 8:
                    return str(random.randint(0, 8))
                elif num.isdigit() and 0 <= int(num) <= 255:
                    return str(random.randint(0, 255))
                else:
                    return num

        text = re.sub(r'0x[0-9a-fA-F]+|\d+\.?\d*', random_replace_num, text)

    # 随机删除部分字段（低概率）
    if random.random() < 0.05:
        parts = text.split(", ")
        if len(parts) > 3:
            del_idx = random.randint(0, len(parts) - 1)
            parts.pop(del_idx)
            text = ", ".join(parts)

    return text


# --------------------------
# 自定义数据收集器
# --------------------------
def custom_data_collator(features):
    input_ids = torch.stack([f["input_ids"] for f in features])
    attention_mask = torch.stack([f["attention_mask"] for f in features])
    labels = torch.tensor([f["labels"].item() for f in features], dtype=torch.long)
    can_ids = [f["can_id"] for f in features]  # 保留CAN ID
    raw_items = [f.get("raw_item", {}) for f in features]  # 保留原始数据项
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "can_ids": can_ids,
        "raw_items": raw_items
    }


# --------------------------
# 简化版MoE模型（移除源定位相关）
# --------------------------
class MoEAttackDetector(nn.Module):
    def __init__(self, bert_model_path, num_classes=4, dropout_rate=0.4):
        super().__init__()
        # 加载BERT模型 - 增加ModelScope兼容
        try:
            self.bert = BertModel.from_pretrained(
                bert_model_path,
                ignore_mismatched_sizes=True
            )
        except Exception as e:
            print(f"尝试兼容ModelScope加载模型: {e}")
            self.bert = BertModel.from_pretrained(
                bert_model_path.replace("google-bert/", ""),
                ignore_mismatched_sizes=True
            )
        self.bert_dim = self.bert.config.hidden_size

        # 冻结部分BERT层
        for param in list(self.bert.parameters())[:-8]:
            param.requires_grad = False

        # 门控网络（输出3个攻击检测专家的权重）
        self.gate_network = nn.Sequential(
            nn.Linear(self.bert_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 3),  # 仅保留3个攻击检测专家
            nn.Softmax(dim=-1)
        )

        # 3个攻击检测专家模块
        self.expert_dos = nn.Sequential(
            nn.Linear(self.bert_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        self.expert_fuzzy = nn.Sequential(
            nn.Linear(self.bert_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        self.expert_gear = nn.Sequential(
            nn.Linear(self.bert_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask, labels=None):
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        # BERT特征提取
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_feat = bert_output.last_hidden_state[:, 0, :]
        cls_feat = self.dropout(cls_feat)

        # 3个专家推理
        gate_weights = self.gate_network(cls_feat)
        expert_dos_out = self.expert_dos(cls_feat)
        expert_fuzzy_out = self.expert_fuzzy(cls_feat)
        expert_gear_out = self.expert_gear(cls_feat)

        # 攻击类型预测（加权融合）
        attack_logits = (gate_weights[:, 0:1] * expert_dos_out +
                         gate_weights[:, 1:2] * expert_fuzzy_out +
                         gate_weights[:, 2:3] * expert_gear_out)

        # 计算损失
        loss = None
        if labels is not None:
            labels = labels.to(DEVICE)
            attack_loss = nn.CrossEntropyLoss()(attack_logits, labels)

            # L2正则化
            l2_reg = torch.tensor(0.).to(DEVICE)
            for param in self.parameters():
                if param.requires_grad:
                    l2_reg += torch.norm(param)
            loss = attack_loss + 0.001 * l2_reg

        return {
            "loss": loss,
            "logits": attack_logits,
            "gate_weights": gate_weights
        }


# --------------------------
# 自定义数据集类（简化版，保留CAN ID）
# --------------------------
class AttackDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, is_train=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def preprocess_text(self, item):
        """纯CAN数据转文本特征"""
        text_parts = [
            f"timestamp={item['timestamp']}",
            f"can_id={item['can_id']}",
            f"data_length={item['data_length']}",
            f"attack_type={item['attack_type']}",
            f"type={item['type']}",
            f"label={item['label']}",
            f"data={','.join(map(str, item['data']))}"
        ]
        text = ", ".join(text_parts)

        # 训练时添加文本增强
        if self.is_train:
            text = augment_text(text)

        return text

    def __getitem__(self, idx):
        item = self.data[idx]
        attack_type = item["attack_type"].lower()

        # 标签容错
        if attack_type not in LABEL2ID:
            attack_type = "normal"

        # 训练时添加标签噪声
        if self.is_train and random.random() < 0.1:
            attack_type = random.choice([k for k in LABEL2ID.keys() if k != attack_type])

        label = LABEL2ID[attack_type]

        # BERT编码
        encoding = self.tokenizer(
            self.preprocess_text(item),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
            "can_id": item.get("can_id", "unknown"),  # 直接保留原始CAN ID
            "raw_item": item  # 保留原始数据项
        }


# --------------------------
# 加载并划分数据集
# --------------------------
def load_and_split_data(data_path, test_size=0.25):
    if not os.path.exists(data_path):
        print(f"警告：数据文件 {data_path} 不存在，生成纯CAN示例数据")
        # 生成模拟纯CAN数据集
        sample_data = []
        attack_types = ["normal", "fuzzy", "dos", "spoofing_gear"]
        can_ids = [f"0x{hex(i)[2:]}" for i in range(0x100, 0x200)] + ["0x0", "0x03f6"]  # 加入你的示例CAN ID
        for i in range(2000):
            attack_type = random.choice(attack_types)
            # 15%标签噪声
            if random.random() < 0.15:
                attack_type = random.choice([t for t in attack_types if t != attack_type])
            sample_data.append({
                "timestamp": f"{random.uniform(1478195728, 1478195800):.6f}",
                "can_id": random.choice(can_ids),
                "data_length": random.randint(0, 8),
                "data": [random.randint(0, 255) for _ in range(8)],
                "attack_type": attack_type,
                "type": "CAN",
                "label": LABEL2ID[attack_type]
            })
        all_data = sample_data
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)

    # 数据清洗
    valid_data = []
    for item in all_data:
        try:
            item["attack_type"] = item["attack_type"].lower()
            # 补全缺失字段
            if "data" not in item:
                item["data"] = [0] * 8
            if "type" not in item:
                item["type"] = "CAN"
            if item["attack_type"] not in LABEL2ID:
                item["attack_type"] = "normal"
            # 移除IP相关字段
            for k in ["src_ip", "dst_ip"]:
                item.pop(k, None)
            valid_data.append(item)
        except KeyError as e:
            print(f"数据清洗：跳过缺失{e}字段的样本")
            continue

    print(f"加载纯CAN数据 {len(valid_data)} 条")

    # 随机划分训练/测试集
    train_data, test_data = train_test_split(
        valid_data,
        test_size=test_size,
        random_state=42,
        shuffle=True
    )

    random.shuffle(train_data)
    random.shuffle(test_data)

    print(f"训练集 {len(train_data)} 条，测试集 {len(test_data)} 条")
    return train_data, test_data


# --------------------------
# 评估指标计算
# --------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, dict):
        logits = logits["logits"]
    logits = np.array(logits)
    labels = np.array(labels).flatten()

    predictions = np.argmax(logits, axis=1)
    accuracy = np.mean(predictions == labels)

    report = classification_report(
        labels, predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    metrics = {"accuracy": accuracy}
    for cls in CLASS_NAMES:
        metrics[f"{cls}_precision"] = report[cls]["precision"]
        metrics[f"{cls}_recall"] = report[cls]["recall"]
        metrics[f"{cls}_f1"] = report[cls]["f1-score"]

    return metrics


# --------------------------
# 自定义Trainer（修复参数兼容问题）
# --------------------------
class AttackTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        # 学习率调度
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer if optimizer is not None else self.optimizer,
            num_warmup_steps=int(num_training_steps * 0.1),
            num_training_steps=num_training_steps,
            last_epoch=-1
        )
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        修复核心问题：添加num_items_in_batch参数兼容新版本transformers
        """
        # 弹出不需要参与模型计算的字段
        can_ids = inputs.pop("can_ids", None)
        raw_items = inputs.pop("raw_items", None)
        labels = inputs.pop("labels")

        # 前向传播
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels
        )

        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        ignore_keys = ignore_keys or []
        # 弹出不需要的字段
        can_ids = inputs.pop("can_ids", None)
        raw_items = inputs.pop("raw_items", None)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs.get("labels")
            )

            loss = outputs["loss"].detach().cpu() if outputs["loss"] is not None else None
            logits = outputs["logits"].detach().cpu()
            labels = inputs.get("labels").detach().cpu() if inputs.get("labels") is not None else None

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)


# --------------------------
# 检测异常攻击并提取CAN ID
# --------------------------
def detect_attack_can_ids(model, tokenizer, test_dataset):
    """检测所有异常攻击并提取对应的CAN ID"""
    model.eval()
    attack_can_ids = defaultdict(set)  # 按攻击类型存储CAN ID
    all_attack_records = []  # 存储所有攻击记录

    print("\n开始检测异常攻击并提取CAN ID...")

    for idx in range(len(test_dataset)):
        data_item = test_dataset[idx]
        input_ids = data_item["input_ids"].unsqueeze(0).to(DEVICE)
        attention_mask = data_item["attention_mask"].unsqueeze(0).to(DEVICE)
        can_id = data_item["can_id"]
        raw_item = data_item["raw_item"]

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # 预测攻击类型
        pred_label = torch.argmax(outputs["logits"], dim=-1).item()
        pred_attack_type = ID2LABEL[pred_label]

        # 如果是异常攻击，记录CAN ID
        if pred_attack_type in ATTACK_TYPES:
            attack_can_ids[pred_attack_type].add(can_id)
            all_attack_records.append({
                "can_id": can_id,
                "predicted_attack_type": pred_attack_type,
                "actual_attack_type": raw_item["attack_type"],
                "timestamp": raw_item["timestamp"],
                "data_length": raw_item["data_length"]
            })

    return attack_can_ids, all_attack_records


# --------------------------
# 主流程
# --------------------------
def main():
    # 加载数据
    train_data, test_data = load_and_split_data(DATA_PATH)

    # 加载分词器 - 增加ModelScope兼容
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    except:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH.replace("google-bert/", ""))

    # 初始化模型
    model = MoEAttackDetector(
        bert_model_path=MODEL_PATH,
        num_classes=NUM_ATTACK_CLASSES,
        dropout_rate=DROPOUT_RATE
    ).to(DEVICE)
    print(f"模型加载完成，使用设备：{DEVICE}")
    print(f"保留原BERT模型：{MODEL_PATH}")
    print(f"3个攻击检测专家模块：DoS/Fuzzy/Gear欺骗")
    print(f"训练轮数：{EPOCHS}")

    # 创建数据集
    train_dataset = AttackDataset(train_data, tokenizer, MAX_LEN, is_train=True)
    test_dataset = AttackDataset(test_data, tokenizer, MAX_LEN, is_train=False)

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./moe_attack_results",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        weight_decay=WEIGHT_DECAY,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
        fp16=False,
        gradient_accumulation_steps=2,
        disable_tqdm=False,
        remove_unused_columns=False,
        lr_scheduler_type="linear",
        no_cuda=not torch.cuda.is_available(),
        seed=42,
        data_seed=42,
    )

    # 训练模型
    trainer = AttackTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.01)],
        data_collator=custom_data_collator
    )
    print("开始训练3专家MoE CAN攻击检测模型...")
    trainer.train()

    # 测试集评估
    print("\n开始评估测试集...")
    predict_results = trainer.predict(test_dataset)
    predictions = predict_results.predictions if isinstance(predict_results.predictions,
                                                            np.ndarray) else predict_results.predictions["logits"]
    predictions = np.array(predictions)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.array(predict_results.label_ids).flatten()

    print(f"\n测试集准确率：{np.mean(pred_labels == true_labels):.4f}")
    print("\n详细分类报告（4类CAN攻击）：")
    print(classification_report(
        true_labels,
        pred_labels,
        target_names=CLASS_NAMES,
        digits=4
    ))

    # 保存模型
    final_model_path = "./moe_attack_final"
    os.makedirs(final_model_path, exist_ok=True)
    model.bert.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    torch.save(model.state_dict(), os.path.join(final_model_path, "moe_model.bin"))
    print(f"\n模型保存至：{final_model_path}")

    # 检测异常攻击并提取CAN ID
    attack_can_ids, all_attack_records = detect_attack_can_ids(model, tokenizer, test_dataset)

    # 输出所有异常攻击的CAN ID
    print("\n" + "=" * 60)
    print("              异常攻击CAN ID检测结果")
    print("=" * 60)

    total_attack_can_ids = set()
    for attack_type, can_ids in attack_can_ids.items():
        print(f"\n【{attack_type.upper()}攻击】检测到的CAN ID数量：{len(can_ids)}")
        print(f"CAN ID列表：{sorted(list(can_ids))}")
        total_attack_can_ids.update(can_ids)

    print(f"\n【总计】异常攻击CAN ID数量：{len(total_attack_can_ids)}")
    print(f"所有异常攻击CAN ID：{sorted(list(total_attack_can_ids))}")

    # 保存攻击检测结果到文件
    result_file = "attack_can_ids_results.json"
    results = {
        "attack_type_can_ids": {k: list(v) for k, v in attack_can_ids.items()},
        "total_attack_can_ids": sorted(list(total_attack_can_ids)),
        "all_attack_records": all_attack_records
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n详细检测结果已保存至：{result_file}")


if __name__ == "__main__":
    main()
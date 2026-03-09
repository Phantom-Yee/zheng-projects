# --------------------------
# 日志屏蔽（优先执行）
# --------------------------
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽TensorFlow INFO/WARNING日志
import warnings

warnings.filterwarnings("ignore")  # 屏蔽其他无关警告

# --------------------------
# 核心导入
# --------------------------
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertModel,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import random
import re
from collections import defaultdict
import subprocess
import sys
import shutil


# --------------------------
# 检查并安装必要依赖（对齐教师模型）
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

# 检查并安装bitsandbytes
try:
    import bitsandbytes
except ImportError:
    print("正在安装bitsandbytes...")
    install_package("bitsandbytes")
    import bitsandbytes

# --------------------------
# 1. 全局配置（完全对齐教师模型 + 保留normal类）
# --------------------------
MODEL_PATH = "google-bert/bert-base-chinese"  # 对齐教师模型的模型路径
DATA_PATH = "merged_can_dataset3.json"
NUM_ATTACK_CLASSES = 4  # normal, fuzzy, dos, spoofing_gear（保留normal类）
MAX_LEN_STUDENT = 64  # 学生模型短序列，省内存
MAX_LEN_TEACHER = 128
BATCH_SIZE = 16  # 对齐教师模型
TEST_SIZE = 0.25  # 对齐教师模型
SEED = 42

# 蒸馏参数 - 调整以防止NaN
ALPHA = 0.7
TEMPERATURE = 3.0  # 降低温度，减少数值不稳定

# 训练参数 - 关键调整防止NaN
STUDENT_DISTILL_EPOCHS = 5  # 修改：训练轮数从3改为5
STUDENT_LR = 1e-5  # 降低学习率，防止梯度爆炸
WEIGHT_DECAY = 0.01  # 降低权重衰减
DROPOUT_RATE = 0.2  # 降低dropout率
GRADIENT_CLIP_NORM = 1.0  # 梯度裁剪
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
# 统一使用float32避免半精度数值问题
DTYPE = torch.float32

# 完全对齐教师模型的标签映射（保留normal类）
LABEL2ID = {"normal": 0, "fuzzy": 1, "dos": 2, "spoofing_gear": 3}
ID2LABEL = {0: "normal", 1: "fuzzy", 2: "dos", 3: "spoofing_gear"}
CLASS_NAMES = list(LABEL2ID.keys())
ATTACK_TYPES = ["fuzzy", "dos", "spoofing_gear"]  # 仅攻击类型

# QLoRA量化配置 - 调整以提高稳定性
QLORA_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,  # 使用float32计算
    bnb_4bit_quant_storage_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# LoRA配置 - 调整以提高稳定性
LORA_CONFIG = LoraConfig(
    r=1,  # 降低r值，减少参数
    lora_alpha=4,  # 降低alpha值
    target_modules=["query", "key", "value"],
    lora_dropout=0.1,  # 降低dropout
    bias="none",
    task_type="SEQ_CLS",
    modules_to_save=["classifier"]
)


# 固定随机种子（对齐教师模型）
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(SEED)


# --------------------------
# 工具函数
# --------------------------
def create_linear_layer(in_features, out_features, dtype=DTYPE):
    """创建指定数据类型的线性层，并初始化权重"""
    layer = nn.Linear(in_features, out_features)
    # 使用更稳定的初始化
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    layer.weight.data = layer.weight.data.to(dtype)
    if layer.bias is not None:
        layer.bias.data = layer.bias.data.to(dtype)
    return layer


def create_sequential_with_dtype(layers, dtype=DTYPE):
    """创建指定数据类型的Sequential层"""
    sequential_layers = []
    for layer in layers:
        if isinstance(layer, nn.Linear):
            layer = create_linear_layer(layer.in_features, layer.out_features, dtype)
        elif isinstance(layer, (nn.ReLU, nn.Dropout, nn.Softmax)):
            pass  # 激活函数不需要转换类型
        sequential_layers.append(layer)
    return nn.Sequential(*sequential_layers)


def safe_softmax(x, dim=-1, eps=1e-8):
    """安全的softmax，防止数值溢出"""
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / (exp_x.sum(dim=dim, keepdim=True) + eps)


def safe_log_softmax(x, dim=-1, eps=1e-8):
    """安全的log_softmax"""
    x = x - x.max(dim=dim, keepdim=True).values
    return x - torch.log(torch.exp(x).sum(dim=dim, keepdim=True) + eps)


# --------------------------
# 关键修复：自定义PEFT模型包装器
# --------------------------
class SafePEFTModelWrapper(nn.Module):
    """安全的PEFT模型包装器，防止参数透传"""

    def __init__(self, peft_model):
        super().__init__()
        self.peft_model = peft_model

    def forward(self, input_ids, attention_mask):
        """只接受并传递BERT需要的参数"""
        return self.peft_model.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )


# --------------------------
# 2. 教师模型定义（完全匹配原始结构）
# --------------------------
class MoETeacherModel(nn.Module):
    """完全匹配原始教师模型的结构定义"""

    def __init__(self, bert_model_path, num_classes=4, dropout_rate=0.4):
        super().__init__()
        # 加载BERT模型 - 增加ModelScope兼容
        try:
            self.bert = BertModel.from_pretrained(
                bert_model_path,
                ignore_mismatched_sizes=True,
                dtype=DTYPE,
                device_map="auto"
            )
        except Exception as e:
            print(f"尝试兼容ModelScope加载模型: {e}")
            self.bert = BertModel.from_pretrained(
                bert_model_path.replace("google-bert/", ""),
                ignore_mismatched_sizes=True,
                dtype=DTYPE,
                device_map="auto"
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

        # 移动到正确设备和数据类型
        self.gate_network = self.gate_network.to(DEVICE, dtype=DTYPE)
        self.expert_dos = self.expert_dos.to(DEVICE, dtype=DTYPE)
        self.expert_fuzzy = self.expert_fuzzy.to(DEVICE, dtype=DTYPE)
        self.expert_gear = self.expert_gear.to(DEVICE, dtype=DTYPE)
        self.dropout = self.dropout.to(DEVICE, dtype=DTYPE)

    def forward(self, input_ids, attention_mask, labels=None):
        """完全匹配原始教师模型的forward方法"""
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        # BERT特征提取
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_feat = bert_output.last_hidden_state[:, 0, :]
        cls_feat = self.dropout(cls_feat)

        # 防止数值溢出
        cls_feat = torch.clamp(cls_feat, -10.0, 10.0)

        # 3个专家推理
        gate_weights = self.gate_network(cls_feat)
        expert_dos_out = self.expert_dos(cls_feat)
        expert_fuzzy_out = self.expert_fuzzy(cls_feat)
        expert_gear_out = self.expert_gear(cls_feat)

        # 防止logits过大
        expert_dos_out = torch.clamp(expert_dos_out, -100.0, 100.0)
        expert_fuzzy_out = torch.clamp(expert_fuzzy_out, -100.0, 100.0)
        expert_gear_out = torch.clamp(expert_gear_out, -100.0, 100.0)

        # 攻击类型预测（加权融合）
        attack_logits = (gate_weights[:, 0:1] * expert_dos_out +
                         gate_weights[:, 1:2] * expert_fuzzy_out +
                         gate_weights[:, 2:3] * expert_gear_out)

        # 如果只需要logits（蒸馏时），直接返回
        if labels is None:
            return attack_logits

        # 计算损失（兼容原始接口）
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
# 3. MoE架构学生模型（稳定版本）
# --------------------------
class MoEStudentDetector(nn.Module):
    def __init__(self, bert_model_path, num_classes=4, dropout_rate=0.2):
        super().__init__()
        # 加载BERT模型
        try:
            self.bert_base = BertModel.from_pretrained(
                bert_model_path,
                quantization_config=QLORA_CONFIG,
                dtype=DTYPE,
                device_map="auto",
                low_cpu_mem_usage=True,
                ignore_mismatched_sizes=True
            )
        except Exception as e:
            print(f"尝试兼容ModelScope加载模型: {e}")
            self.bert_base = BertModel.from_pretrained(
                bert_model_path.replace("google-bert/", ""),
                quantization_config=QLORA_CONFIG,
                dtype=DTYPE,
                device_map="auto",
                low_cpu_mem_usage=True,
                ignore_mismatched_sizes=True
            )

        # 准备kbit训练
        self.bert_base = prepare_model_for_kbit_training(self.bert_base)

        # 添加LoRA适配器
        self.bert_peft = get_peft_model(self.bert_base, LORA_CONFIG)

        # 使用安全包装器
        self.bert = SafePEFTModelWrapper(self.bert_peft)

        self.bert_dim = self.bert_base.config.hidden_size

        # 冻结更多层以提高稳定性
        for param in list(self.bert_base.parameters())[:-4]:
            param.requires_grad = False

        # 3个攻击检测专家模块 - 使用更稳定的配置
        self.gate_network = create_sequential_with_dtype([
            create_linear_layer(self.bert_dim, 32, DTYPE),  # 减小维度
            nn.ReLU(),
            nn.Dropout(0.1),  # 降低dropout
            create_linear_layer(32, 3, DTYPE),
        ], DTYPE).to(DEVICE, dtype=DTYPE)

        self.expert_dos = create_sequential_with_dtype([
            create_linear_layer(self.bert_dim, 64, DTYPE),  # 减小维度
            nn.ReLU(),
            nn.Dropout(0.1),
            create_linear_layer(64, num_classes, DTYPE)
        ], DTYPE).to(DEVICE, dtype=DTYPE)

        self.expert_fuzzy = create_sequential_with_dtype([
            create_linear_layer(self.bert_dim, 64, DTYPE),
            nn.ReLU(),
            nn.Dropout(0.1),
            create_linear_layer(64, num_classes, DTYPE)
        ], DTYPE).to(DEVICE, dtype=DTYPE)

        self.expert_gear = create_sequential_with_dtype([
            create_linear_layer(self.bert_dim, 64, DTYPE),
            nn.ReLU(),
            nn.Dropout(0.1),
            create_linear_layer(64, num_classes, DTYPE)
        ], DTYPE).to(DEVICE, dtype=DTYPE)

        self.dropout = nn.Dropout(dropout_rate).to(DEVICE, dtype=DTYPE)

    def forward(self, input_ids, attention_mask, labels=None):
        """稳定的forward实现"""
        # 确保输入在正确设备和数据类型上
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        # BERT前向传播
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_feat = bert_output.last_hidden_state[:, 0, :]
        cls_feat = cls_feat.to(dtype=DTYPE)
        cls_feat = self.dropout(cls_feat)

        # 防止特征值过大
        cls_feat = torch.clamp(cls_feat, -10.0, 10.0)

        # MoE专家推理
        gate_logits = self.gate_network(cls_feat)
        gate_weights = safe_softmax(gate_logits, dim=-1)  # 使用安全softmax

        expert_dos_out = self.expert_dos(cls_feat)
        expert_fuzzy_out = self.expert_fuzzy(cls_feat)
        expert_gear_out = self.expert_gear(cls_feat)

        # 防止logits过大
        expert_dos_out = torch.clamp(expert_dos_out, -100.0, 100.0)
        expert_fuzzy_out = torch.clamp(expert_fuzzy_out, -100.0, 100.0)
        expert_gear_out = torch.clamp(expert_gear_out, -100.0, 100.0)

        # 加权融合
        attack_logits = (gate_weights[:, 0:1] * expert_dos_out +
                         gate_weights[:, 1:2] * expert_fuzzy_out +
                         gate_weights[:, 2:3] * expert_gear_out)

        # 计算损失
        loss = None
        if labels is not None:
            labels = labels.to(DEVICE)

            # 使用稳定的损失计算
            attack_loss = nn.CrossEntropyLoss()(attack_logits, labels)

            # 轻量化L2正则化
            l2_reg = torch.tensor(0., device=DEVICE, dtype=DTYPE)
            reg_count = 0
            for param in self.parameters():
                if param.requires_grad and param.dim() > 1:  # 只对权重正则化
                    l2_reg += torch.norm(param)
                    reg_count += 1
            if reg_count > 0:
                l2_reg = l2_reg / reg_count  # 平均正则化

            loss = attack_loss + 0.0001 * l2_reg  # 降低正则化强度

        return {
            "loss": loss,
            "logits": attack_logits,
            "gate_weights": gate_weights
        }

    def print_trainable_parameters(self):
        """打印可训练参数信息"""
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"可训练参数数量: {trainable_params:,}")
        print(f"总参数数量: {all_param:,}")
        print(f"可训练参数比例: {100 * trainable_params / all_param:.4f}%")


# --------------------------
# 4. 数据集类
# --------------------------
class CANAttackDataset(Dataset):
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

        # 降低数据增强强度
        if self.is_train:
            text = self.augment_text(text, aug_prob=0.05)

        return text

    def augment_text(self, text, aug_prob=0.05):
        """轻量化文本增强"""
        if random.random() < aug_prob:
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

        return text

    def __getitem__(self, idx):
        item = self.data[idx]
        attack_type = item["attack_type"].lower()

        # 标签容错
        if attack_type not in LABEL2ID:
            attack_type = "normal"

        # 降低标签噪声
        if self.is_train and random.random() < 0.05:
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
            "can_id": item.get("can_id", "unknown"),
            "raw_item": item,
            "idx": idx
        }


# --------------------------
# 5. 数据加载
# --------------------------
def load_and_split_data(data_path, test_size=0.25):
    if not os.path.exists(data_path):
        print(f"警告：数据文件 {data_path} 不存在，生成纯CAN示例数据")
        # 生成模拟纯CAN数据集
        sample_data = []
        attack_types = ["normal", "fuzzy", "dos", "spoofing_gear"]
        can_ids = [f"0x{hex(i)[2:]}" for i in range(0x100, 0x200)] + ["0x0", "0x03f6"]
        for i in range(2000):
            attack_type = random.choice(attack_types)
            # 降低标签噪声
            if random.random() < 0.05:
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
            if "data" not in item:
                item["data"] = [0] * 8
            if "type" not in item:
                item["type"] = "CAN"
            if item["attack_type"] not in LABEL2ID:
                item["attack_type"] = "normal"
            for k in ["src_ip", "dst_ip"]:
                item.pop(k, None)
            valid_data.append(item)
        except KeyError as e:
            print(f"数据清洗：跳过缺失{e}字段的样本")
            continue

    print(f"加载纯CAN数据 {len(valid_data)} 条（包含normal类）")

    # 随机划分
    train_data, test_data = train_test_split(
        valid_data,
        test_size=test_size,
        random_state=SEED,
        shuffle=True
    )

    print(f"训练集 {len(train_data)} 条，测试集 {len(test_data)} 条")
    return train_data, test_data


# --------------------------
# 6. 稳定的蒸馏损失函数
# --------------------------
def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.7):
    """稳定的蒸馏损失计算"""
    # 防止logits值过大
    student_logits = torch.clamp(student_logits, -1000.0, 1000.0)
    teacher_logits = torch.clamp(teacher_logits, -1000.0, 1000.0)

    # 使用安全的softmax计算
    student_soft = safe_log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = safe_softmax(teacher_logits / temperature, dim=-1)

    # 稳定的KL散度计算
    kl_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)
    kl_loss = torch.clamp(kl_loss, 0, 100)  # 限制KL损失范围

    # 交叉熵损失
    ce_loss = F.cross_entropy(student_logits, labels)

    # 组合损失
    total_loss = alpha * kl_loss + (1 - alpha) * ce_loss

    # 检查NaN
    if torch.isnan(total_loss):
        print("警告：损失值为NaN，使用CE损失替代")
        total_loss = ce_loss

    return total_loss


# --------------------------
# 7. 攻击CAN ID定位
# --------------------------
def detect_attack_can_ids(model, tokenizer, test_dataset):
    """检测所有异常攻击并提取对应的CAN ID"""
    model.eval()
    attack_can_ids = defaultdict(set)
    all_attack_records = []

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

            # 兼容两种返回格式
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

        pred_label = torch.argmax(logits, dim=-1).item()
        pred_attack_type = ID2LABEL[pred_label]

        if pred_attack_type in ATTACK_TYPES:
            attack_can_ids[pred_attack_type].add(can_id)
            all_attack_records.append({
                "can_id": can_id,
                "predicted_attack_type": pred_attack_type,
                "actual_attack_type": raw_item["attack_type"],
                "timestamp": raw_item["timestamp"],
                "data_length": raw_item["data_length"]
            })

    # 输出结果
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

    # 保存结果
    result_file = "attack_can_ids_results_student.json"
    results = {
        "attack_type_can_ids": {k: list(v) for k, v in attack_can_ids.items()},
        "total_attack_can_ids": sorted(list(total_attack_can_ids)),
        "all_attack_records": all_attack_records
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n详细检测结果已保存至：{result_file}")

    return attack_can_ids, all_attack_records


# --------------------------
# 8. 评估函数
# --------------------------
def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].cpu().numpy()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # 兼容两种返回格式
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
    return accuracy, report


# --------------------------
# 9. 预训练教师模型（如果权重不存在）
# --------------------------
def pretrain_teacher_model():
    """预训练教师模型，确保有可用的权重文件"""
    print("\n=== 开始预训练教师模型 ===")

    # 加载数据
    train_data, test_data = load_and_split_data(DATA_PATH)

    # 加载分词器
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    except:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH.replace("google-bert/", ""))

    # 初始化教师模型
    teacher_model = MoETeacherModel(
        bert_model_path=MODEL_PATH,
        num_classes=NUM_ATTACK_CLASSES,
        dropout_rate=0.4
    ).to(DEVICE, dtype=DTYPE)

    # 创建数据集
    train_dataset = CANAttackDataset(train_data, tokenizer, MAX_LEN_TEACHER, is_train=True)
    test_dataset = CANAttackDataset(test_data, tokenizer, MAX_LEN_TEACHER, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    # 优化器
    optimizer = torch.optim.AdamW(
        teacher_model.parameters(),
        lr=3e-5,
        weight_decay=0.01
    )

    # 训练5轮（同步修改为5轮）
    teacher_model.train()
    for epoch in range(5):
        total_loss = 0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = teacher_model(input_ids, attention_mask, labels)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if (step + 1) % 10 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"教师模型训练 Epoch {epoch + 1}, Step {step + 1}, Loss: {avg_loss:.4f}")

        # 评估
        acc, report = evaluate(teacher_model, test_loader)
        print(f"教师模型 Epoch {epoch + 1} 准确率: {acc:.4f}")

    # 保存教师模型
    save_path = "./moe_attack_final"
    os.makedirs(save_path, exist_ok=True)

    # 保存BERT和tokenizer
    teacher_model.bert.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # 保存完整模型权重
    torch.save(teacher_model.state_dict(), os.path.join(save_path, "moe_model.bin"))

    print(f"\n教师模型已预训练并保存至: {save_path}")
    return teacher_model, tokenizer


# --------------------------
# 10. 加载教师模型（修复版本）
# --------------------------
def load_teacher_model():
    """加载训练好的MoE教师模型（修复权重加载）"""
    teacher_path = "./moe_attack_final"

    # 如果教师模型不存在，先预训练
    if not os.path.exists(teacher_path) or not os.path.exists(os.path.join(teacher_path, "moe_model.bin")):
        print(f"教师模型路径不存在，先进行预训练...")
        return pretrain_teacher_model()

    # 加载分词器
    try:
        tokenizer = BertTokenizer.from_pretrained(teacher_path)
    except:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH.replace("google-bert/", ""))

    # 初始化教师模型
    teacher_model = MoETeacherModel(
        bert_model_path=teacher_path,  # 从保存路径加载
        num_classes=NUM_ATTACK_CLASSES,
        dropout_rate=0.4
    ).to(DEVICE, dtype=DTYPE)

    # 加载权重 - 修复版本
    try:
        # 加载权重文件
        state_dict = torch.load(
            os.path.join(teacher_path, "moe_model.bin"),
            map_location=DEVICE,
            weights_only=True  # 安全加载
        )

        # 智能加载权重（忽略不匹配的参数）
        model_state_dict = teacher_model.state_dict()
        filtered_state_dict = {}

        for key, value in state_dict.items():
            if key in model_state_dict and model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                print(f"跳过不匹配的权重: {key} (模型shape: {model_state_dict.get(key, '不存在').shape}, "
                      f"权重shape: {value.shape})")

        # 加载过滤后的权重
        teacher_model.load_state_dict(filtered_state_dict, strict=False)
        print(f"成功加载 {len(filtered_state_dict)}/{len(state_dict)} 个权重参数")

    except Exception as e:
        print(f"权重加载详细错误: {e}")
        print("警告：教师模型权重加载失败，使用随机初始化权重继续")

    teacher_model.eval()
    print(f"教师模型加载完成（设备：{DEVICE}，数据类型：{DTYPE}）")
    return teacher_model, tokenizer


# --------------------------
# 11. 蒸馏训练主函数（稳定版本）
# --------------------------
def train_distillation():
    # 加载数据
    train_data, test_data = load_and_split_data(DATA_PATH)

    # 加载分词器
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    except:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH.replace("google-bert/", ""))

    # 初始化学生模型
    student_model = MoEStudentDetector(
        bert_model_path=MODEL_PATH,
        num_classes=NUM_ATTACK_CLASSES,
        dropout_rate=DROPOUT_RATE
    ).to(DEVICE, dtype=DTYPE)

    print(f"\n=== 学生模型初始化完成 ===")
    print(f"使用设备：{DEVICE}，数据类型：{DTYPE}")
    print(f"使用本地模型：{MODEL_PATH}")
    print(f"MoE架构：3个攻击检测专家模块（DoS/Fuzzy/Gear欺骗） + 保留normal类")
    print(f"QLoRA量化：4bit NF4 + 低秩适配器(r=1)")
    print(f"训练轮数：{STUDENT_DISTILL_EPOCHS}")  # 这里会显示修改后的5轮
    print(f"学习率：{STUDENT_LR} (降低以防止NaN)")

    # 打印可训练参数
    student_model.print_trainable_parameters()

    # 创建数据集和数据加载器
    train_dataset = CANAttackDataset(train_data, tokenizer, MAX_LEN_STUDENT, is_train=True)
    test_dataset = CANAttackDataset(test_data, tokenizer, MAX_LEN_STUDENT, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    # 加载教师模型
    print("\n=== 加载教师模型 ===")
    teacher_model, teacher_tokenizer = load_teacher_model()

    # 蒸馏前评估
    print("\n📊 蒸馏前学生模型性能：")
    pre_acc, pre_report = evaluate(student_model, test_loader)
    print(f"准确率：{pre_acc:.4f}")
    print("分类报告：")
    print(pre_report)

    # 优化器和调度器 - 使用更稳定的配置
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=STUDENT_LR,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,  # 提高数值稳定性
        betas=(0.9, 0.999)
    )

    num_training_steps = len(train_loader) * STUDENT_DISTILL_EPOCHS  # 自动适配5轮的总步数
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * 0.2),  # 增加warmup
        num_training_steps=num_training_steps
    )

    # 开始蒸馏训练
    print("\n🚀 开始MoE学生模型蒸馏训练...")
    student_model.train()

    for epoch in range(STUDENT_DISTILL_EPOCHS):  # 循环次数变为5次
        total_loss = 0.0
        valid_steps = 0

        for step, batch in enumerate(train_loader):
            try:
                # 只提取需要的参数
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                # 学生模型前向传播
                student_outputs = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                student_logits = student_outputs["logits"]

                # 教师模型前向传播
                with torch.no_grad():
                    # 构建教师模型输入（使用教师模型的max_len）
                    teacher_texts = []
                    batch_indices = batch["idx"].cpu().numpy()
                    for idx in batch_indices:
                        raw_item = train_data[idx]
                        text_parts = [
                            f"timestamp={raw_item['timestamp']}",
                            f"can_id={raw_item['can_id']}",
                            f"data_length={raw_item['data_length']}",
                            f"attack_type={raw_item['attack_type']}",
                            f"type={raw_item['type']}",
                            f"label={raw_item['label']}",
                            f"data={','.join(map(str, raw_item['data']))}"
                        ]
                        teacher_texts.append(", ".join(text_parts))

                    teacher_encoding = teacher_tokenizer(
                        teacher_texts,
                        max_length=MAX_LEN_TEACHER,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).to(DEVICE)

                    teacher_logits = teacher_model(
                        input_ids=teacher_encoding["input_ids"],
                        attention_mask=teacher_encoding["attention_mask"]
                    )

                # 计算蒸馏损失
                loss = distillation_loss(
                    student_logits,
                    teacher_logits,
                    labels,
                    TEMPERATURE,
                    ALPHA
                )

                # 检查loss是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告：Step {step + 1} 损失值无效，跳过")
                    continue

                # 反向传播
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪 - 关键防止NaN
                torch.nn.utils.clip_grad_norm_(
                    student_model.parameters(),
                    GRADIENT_CLIP_NORM
                )

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                valid_steps += 1

                # 打印日志
                if (step + 1) % 10 == 0 and valid_steps > 0:
                    avg_loss = total_loss / valid_steps
                    print(
                        f"蒸馏 Epoch [{epoch + 1}/{STUDENT_DISTILL_EPOCHS}], Step [{step + 1}/{len(train_loader)}], Loss: {avg_loss:.4f}")

            except Exception as e:
                print(f"Step {step + 1} 出错: {e}")
                continue

        # 每轮评估
        if valid_steps > 0:
            epoch_acc, _ = evaluate(student_model, test_loader)
            avg_epoch_loss = total_loss / valid_steps
            print(f"蒸馏 Epoch [{epoch + 1}/{STUDENT_DISTILL_EPOCHS}] 平均损失：{avg_epoch_loss:.4f}，测试准确率：{epoch_acc:.4f}\n")
        else:
            print(f"蒸馏 Epoch [{epoch + 1}/{STUDENT_DISTILL_EPOCHS}] 无有效训练步骤\n")

    # 蒸馏后评估
    print("\n📊 蒸馏后学生模型性能：")
    post_acc, post_report = evaluate(student_model, test_loader)
    print(f"准确率：{post_acc:.4f}")
    print(f"准确率提升：{post_acc - pre_acc:.4f}")
    print("分类报告：")
    print(post_report)

    # 检测异常攻击CAN ID
    attack_can_ids, attack_records = detect_attack_can_ids(student_model, tokenizer, test_dataset)

    # 保存模型
    save_path = "./moe_student_quantized_50mb"
    os.makedirs(save_path, exist_ok=True)

    # 安全保存模型
    torch.save(student_model.state_dict(), os.path.join(save_path, "moe_student_model.bin"))
    tokenizer.save_pretrained(save_path)

    # 验证模型大小
    def calculate_model_size(model_path):
        total_size = 0
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
        return total_size / (1024 * 1024)

    model_size = calculate_model_size(save_path)
    print(f"\n💾 MoE学生模型保存至：{save_path}")
    print(f"📏 模型大小：{model_size:.2f} MB (目标：≤50MB)")

    # 教师模型性能参考
    print("\n📊 教师模型性能参考：")
    teacher_test_dataset = CANAttackDataset(test_data, teacher_tokenizer, MAX_LEN_TEACHER)
    teacher_test_loader = DataLoader(teacher_test_dataset, batch_size=BATCH_SIZE * 2)

    def evaluate_teacher(model, dataloader):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].cpu().numpy()

                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(
            all_labels, all_preds,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0
        )
        return accuracy, report

    teacher_acc, teacher_report = evaluate_teacher(teacher_model, teacher_test_loader)
    print(f"准确率：{teacher_acc:.4f}")
    print("分类报告：")
    print(teacher_report)

    print("\n=== 蒸馏训练完成 ===")


# --------------------------
# 主函数入口
# --------------------------
if __name__ == "__main__":
    # 设置CUDA调试模式（可选）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 开始蒸馏训练
    train_distillation()
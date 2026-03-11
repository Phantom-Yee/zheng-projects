"""
teacher.py — CAN总线攻击检测教师模型

【特征设计】
  ✅ 只用真实物理特征：timestamp / can_id / dlc / data_bytes
  ❌ 严禁包含：attack_type / label / type 等任何语义标签

  物理特征的区分能力：
    - spoofing_gear : 固定伪造特定 can_id（如 0x03f6），载荷字节固定模式
    - dos           : 同一 can_id 高频重复，时间戳连续且间隔极短
    - fuzzy         : can_id 随机分布（0x000~0x7ff 均匀），载荷随机
    - normal        : can_id 在固定范围内，载荷具有固定格式

【输出】
  ./teacher_model/
    ├── bert_weights/     ← BERT权重（供学生迁移）
    ├── teacher_full.bin  ← 完整模型权重
    ├── config.json       ← BERT配置
    ├── vocab.txt         ← tokenizer
    └── meta.json         ← 训练信息
"""

import json, os, random, sys, subprocess, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings; warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertModel,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from collections import defaultdict


# ── 依赖安装 ──────────────────────────────────────────────────
def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:
    import accelerate
except ImportError:
    _pip("accelerate>=1.1.0"); import accelerate


# ================================================================
# 1. 全局配置
# ================================================================
BERT_PATH  = "google-bert/bert-base-chinese"   # ModelScope 兼容路径
DATA_PATH  = "merged_can_dataset3.json"
SAVE_DIR   = "./teacher_model"

NUM_CLASSES  = 4
BATCH_SIZE   = 16
MAX_LEN      = 128
TEST_SIZE    = 0.2
SEED         = 42

LR           = 2e-5
EPOCHS       = 3
DROPOUT      = 0.3
WEIGHT_DECAY = 0.01
GRAD_CLIP    = 1.0
WARMUP_RATIO = 0.06

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL2ID     = {"normal": 0, "fuzzy": 1, "dos": 2, "spoofing_gear": 3}
ID2LABEL     = {v: k for k, v in LABEL2ID.items()}
CLASS_NAMES  = list(LABEL2ID.keys())
ATTACK_TYPES = ["fuzzy", "dos", "spoofing_gear"]

print(f"运行设备: {DEVICE}")


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)


# ================================================================
# 2. 特征函数 —— 纯物理特征，零标签泄露
# ================================================================
def item_to_text(item):
    """
    CAN帧 → 纯物理特征文本。

    格式示例：
      ts 1478195756.122400 id 0x03f6 dlc 8 payload 01 23 45 67 89 ab cd ef

    绝对不包含: attack_type / label / type 等任何语义信息
    """
    try:
        raw_data = item["data"]
        if isinstance(raw_data, list):
            payload = " ".join(f"{b & 0xFF:02x}" for b in raw_data)
        else:
            payload = "00 " * 8
    except Exception:
        payload = "00 " * 8

    can_id    = str(item.get("can_id", "0x000")).strip()
    dlc       = str(item.get("data_length", 8))
    timestamp = str(item.get("timestamp", "0")).strip()

    return f"ts {timestamp} id {can_id} dlc {dlc} payload {payload}"


# ================================================================
# 3. 数据集
# ================================================================
class CANDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, is_train=False):
        self.data     = data
        self.tok      = tokenizer
        self.max_len  = max_len
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item  = self.data[idx]
        at    = item["attack_type"].lower()
        if at not in LABEL2ID:
            at = "normal"
        label = LABEL2ID[at]

        text = item_to_text(item)

        enc = self.tok(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label, dtype=torch.long),
            "can_id":         item.get("can_id", "unknown"),
            "raw_item":       item,
        }


# ================================================================
# 4. 数据加载与分割
# ================================================================
def load_and_split(path=DATA_PATH):
    if not os.path.exists(path):
        print(f"⚠️  {path} 不存在，生成模拟数据")
        data = []
        # 模拟真实数据集的物理规律
        normal_ids  = [f"0x{i:03x}" for i in range(0x100, 0x180)]
        dos_ids     = ["0x100", "0x101", "0x102"]   # DoS 集中少数ID
        fuzzy_ids   = [f"0x{i:03x}" for i in range(0x000, 0x7ff)]  # 随机全范围
        spoof_ids   = ["0x03f6"]                     # spoofing 固定ID

        specs = [
            ("normal",        normal_ids, 6000),
            ("dos",           dos_ids,    1000),
            ("fuzzy",         fuzzy_ids,  1000),
            ("spoofing_gear", spoof_ids,  1000),
        ]
        ts = 1478195728.0
        for at, ids, n in specs:
            for _ in range(n):
                ts += random.uniform(0.001, 0.01)
                data.append({
                    "timestamp":   f"{ts:.6f}",
                    "can_id":      random.choice(ids),
                    "data_length": random.randint(1, 8),
                    "data":        [random.randint(0, 255) for _ in range(8)],
                    "attack_type": at,
                    "label":       LABEL2ID[at],
                    "type":        "CAN",
                })
        random.shuffle(data)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    valid = []
    for item in data:
        try:
            item["attack_type"] = item["attack_type"].lower()
            if "data"  not in item or not isinstance(item["data"], list):
                item["data"] = [0] * 8
            if "data_length" not in item:
                item["data_length"] = len(item["data"])
            if "type" not in item:
                item["type"] = "CAN"
            if item["attack_type"] not in LABEL2ID:
                item["attack_type"] = "normal"
            # 确保 label 字段存在（用于兼容但不放入特征文本）
            item["label"] = LABEL2ID[item["attack_type"]]
            for k in ["src_ip", "dst_ip"]: item.pop(k, None)
            valid.append(item)
        except Exception:
            continue

    print(f"有效数据: {len(valid)} 条")

    # 分层抽样保证训练/测试分布一致
    train_data, test_data = train_test_split(
        valid, test_size=TEST_SIZE, random_state=SEED, shuffle=True,
        stratify=[d["attack_type"] for d in valid],
    )
    print(f"训练集: {len(train_data)} | 测试集: {len(test_data)}")

    # 打印类别分布
    def _dist(d):
        cnt = defaultdict(int)
        for x in d: cnt[x["attack_type"]] += 1
        return dict(cnt)
    print(f"  训练集分布: {_dist(train_data)}")
    print(f"  测试集分布: {_dist(test_data)}")
    return train_data, test_data


# ================================================================
# 5. 教师模型：MoE + 12层BERT
#    门控 + 3专家（DoS/Fuzzy/Gear）结构与原教师代码一致
#    但特征输入已去除所有标签信息
# ================================================================
class MoETeacherModel(nn.Module):
    def __init__(self, bert_path, num_classes=4, dropout=0.3):
        super().__init__()
        # 加载 BERT
        try:
            self.bert = BertModel.from_pretrained(
                bert_path, ignore_mismatched_sizes=True)
        except Exception:
            self.bert = BertModel.from_pretrained(
                bert_path.replace("google-bert/", ""),
                ignore_mismatched_sizes=True)
        self.bert_dim = self.bert.config.hidden_size  # 768

        # 解冻后 4 层（encoder.layer[8~11]），冻结前 8 层
        # 这样既保留预训练表示，又能针对CAN特征微调
        all_params = list(self.bert.parameters())
        n_freeze   = int(len(all_params) * 0.75)
        for p in all_params[:n_freeze]:
            p.requires_grad = False

        # 门控网络：输出3个专家权重
        self.gate = nn.Sequential(
            nn.Linear(self.bert_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1),
        )

        # 3个专家：分别针对 DoS / Fuzzy / Spoofing_Gear 特征
        def _expert():
            return nn.Sequential(
                nn.Linear(self.bert_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(128, num_classes),
            )
        self.expert_dos   = _expert()
        self.expert_fuzzy = _expert()
        self.expert_gear  = _expert()

        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, labels=None):
        out  = self.bert(
            input_ids=input_ids.to(DEVICE),
            attention_mask=attention_mask.to(DEVICE),
        )
        feat = self.drop(out.last_hidden_state[:, 0, :])

        gw     = self.gate(feat)
        logits = (
            gw[:, 0:1] * self.expert_dos(feat) +
            gw[:, 1:2] * self.expert_fuzzy(feat) +
            gw[:, 2:3] * self.expert_gear(feat)
        )

        if labels is None:
            return logits   # 直接返回 Tensor，供蒸馏使用

        labels = labels.to(DEVICE)
        ce     = F.cross_entropy(logits, labels)
        # 轻量 L2 正则
        l2 = sum(torch.norm(p) for p in self.parameters() if p.requires_grad)
        loss = ce + 1e-4 * l2
        return {"loss": loss, "logits": logits}


# ================================================================
# 6. 评估
# ================================================================
def evaluate(model, loader, desc="测试集"):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            out  = model(input_ids=ids, attention_mask=mask)
            logits = out["logits"] if isinstance(out, dict) else out
            preds_all.extend(torch.argmax(logits, -1).cpu().tolist())
            labels_all.extend(batch["labels"].tolist())

    acc = accuracy_score(labels_all, preds_all)
    rep = classification_report(
        labels_all, preds_all,
        target_names=CLASS_NAMES, digits=4, zero_division=0,
    )
    return acc, rep


# ================================================================
# 7. 检测攻击CAN ID
# ================================================================
def detect_attack_ids(model, tokenizer, test_data):
    print("\n开始检测异常攻击CAN ID...")
    ds = CANDataset(test_data, tokenizer, MAX_LEN)
    model.eval()
    attack_ids = defaultdict(set)
    records    = []

    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            out  = model(
                input_ids=item["input_ids"].unsqueeze(0).to(DEVICE),
                attention_mask=item["attention_mask"].unsqueeze(0).to(DEVICE),
            )
            pred = ID2LABEL[torch.argmax(out if not isinstance(out, dict)
                                         else out["logits"], -1).item()]
            if pred in ATTACK_TYPES:
                attack_ids[pred].add(item["can_id"])
                records.append({
                    "can_id":               item["can_id"],
                    "predicted_attack":     pred,
                    "actual_attack":        item["raw_item"]["attack_type"],
                    "timestamp":            item["raw_item"]["timestamp"],
                })

    print("=" * 60)
    print("              异常攻击CAN ID检测结果（教师）")
    print("=" * 60)
    total = set()
    for at, cids in sorted(attack_ids.items()):
        print(f"\n【{at.upper()}】{len(cids)} 个CAN ID")
        print(f"  {sorted(cids)[:20]}{'...' if len(cids) > 20 else ''}")
        total |= cids
    print(f"\n【总计】{len(total)} 个异常CAN ID")

    with open("attack_can_ids_teacher.json", "w", encoding="utf-8") as f:
        json.dump({
            "attack_type_can_ids":  {k: sorted(v) for k, v in attack_ids.items()},
            "total_attack_can_ids": sorted(total),
            "records":              records,
        }, f, ensure_ascii=False, indent=2)
    print("已保存至 attack_can_ids_teacher.json")
    return attack_ids


# ================================================================
# 8. 主训练流程
# ================================================================
def main():
    print("=" * 60)
    print("  CAN总线攻击检测 — 教师模型训练（纯物理特征）")
    print("=" * 60)
    print("  特征: timestamp / can_id / dlc / payload（无标签信息）")
    print("=" * 60)

    train_data, test_data = load_and_split(DATA_PATH)

    try:
        tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    except Exception:
        tokenizer = BertTokenizer.from_pretrained(
            BERT_PATH.replace("google-bert/", ""))

    # 验证特征样例
    print(f"\n特征样例（前2条）：")
    for i in range(min(2, len(train_data))):
        print(f"  [{train_data[i]['attack_type']}] {item_to_text(train_data[i])}")

    model = MoETeacherModel(BERT_PATH, NUM_CLASSES, DROPOUT).to(DEVICE)

    # 统计参数量
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数: 总计 {total:,}  |  可训练 {trainable:,} "
          f"({100*trainable/total:.1f}%)")

    # 数据集
    tr_ds = CANDataset(train_data, tokenizer, MAX_LEN, is_train=True)
    te_ds = CANDataset(test_data,  tokenizer, MAX_LEN)
    tr_ld = DataLoader(tr_ds, BATCH_SIZE,   shuffle=True,  num_workers=0)
    te_ld = DataLoader(te_ds, BATCH_SIZE*2, shuffle=False, num_workers=0)

    # 优化器
    opt   = torch.optim.AdamW(
        model.parameters(), lr=LR,
        weight_decay=WEIGHT_DECAY, eps=1e-8)
    total_steps = len(tr_ld) * EPOCHS
    sched = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps)

    best_acc, best_state = 0.0, None
    epoch_times, epoch_accs = [], []

    print(f"\n🚀 开始训练 (epochs={EPOCHS}, lr={LR}, batch={BATCH_SIZE})")

    for epoch in range(EPOCHS):
        model.train()
        t0           = time.time()
        tloss, steps = 0.0, 0

        for step, batch in enumerate(tr_ld):
            ids    = batch["input_ids"].to(DEVICE)
            mask   = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            out  = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = out["loss"]

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step(); sched.step()

            tloss += loss.item(); steps += 1
            if (step + 1) % 100 == 0:
                print(f"  Epoch[{epoch+1}/{EPOCHS}] "
                      f"Step[{step+1}/{len(tr_ld)}] "
                      f"Loss={tloss/steps:.4f}")

        elapsed = (time.time() - t0) / 60.0
        epoch_times.append(elapsed)
        acc, _ = evaluate(model, te_ld)
        epoch_accs.append(acc)
        print(f"\n  ✅ Epoch[{epoch+1}/{EPOCHS}] "
              f"Loss={tloss/steps:.4f} | Acc={acc:.4f} | "
              f"时间={elapsed:.2f}min")

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  ⭐ 新最优: {best_acc:.4f}")
        print()

    # 恢复最优权重
    model.load_state_dict(best_state)
    print(f"✅ 已恢复最优权重  Acc={best_acc:.4f}")

    # 最终评估
    print("\n📊 教师模型最终性能：")
    final_acc, final_rep = evaluate(model, te_ld)
    print(f"准确率: {final_acc:.4f}")
    print(final_rep)

    # 检测攻击CAN ID
    detect_attack_ids(model, tokenizer, test_data)

    # ── 保存 ─────────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. 完整模型权重
    torch.save(best_state,
               os.path.join(SAVE_DIR, "teacher_full.bin"))

    # 2. BERT权重（供学生迁移）
    bert_dir = os.path.join(SAVE_DIR, "bert_weights")
    os.makedirs(bert_dir, exist_ok=True)
    model.bert.save_pretrained(bert_dir)

    # 3. tokenizer
    tokenizer.save_pretrained(SAVE_DIR)

    # 4. 元信息
    with open(os.path.join(SAVE_DIR, "meta.json"), "w") as f:
        json.dump({
            "bert_path":        BERT_PATH,
            "num_classes":      NUM_CLASSES,
            "max_len":          MAX_LEN,
            "best_accuracy":    best_acc,
            "final_accuracy":   final_acc,
            "label2id":         LABEL2ID,
            "id2label":         ID2LABEL,
            "feature_fields":   ["timestamp", "can_id", "dlc", "payload"],
            "excluded_fields":  ["attack_type", "label", "type"],
        }, f, indent=2)

    print(f"\n💾 教师模型已保存至: {SAVE_DIR}/")
    print(f"   teacher_full.bin  — 完整权重")
    print(f"   bert_weights/     — BERT权重（学生迁移用）")

    # 汇总
    print("\n" + "=" * 60)
    print("             训练汇总")
    print("=" * 60)
    print(f"  特征设计:  纯物理特征（无标签泄露）")
    print(f"  最优准确率: {best_acc:.4f}")
    for i, (t, a) in enumerate(zip(epoch_times, epoch_accs)):
        print(f"  Epoch {i+1}: {t:.2f}min | Acc={a:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
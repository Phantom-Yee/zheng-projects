"""
ablation_no_distill.py — CAN总线攻击检测 消融实验：去除知识蒸馏

【消融实验说明】
  本文件为 dis2.py 的消融对照版本，去除了知识蒸馏（KL软标签损失），
  改用纯CE损失直接监督训练，其余架构（QLoRA + LoRA + MoE）完全保留。

【完整流程（消融版）】
  Step 1  加载BERT（QLoRA：4-bit NF4 / INT8动态量化）+ LoRA适配器
  Step 2  构建MoE头（gate + 3专家，与dis2.py完全一致）
  Step 3  纯CE监督训练（无教师、无KL、无软标签）
  Step 4  保存模型 → 打印消融汇总报告

【与 dis2.py 的差异点】
  - 无 load_teacher() 调用，不依赖 teacher_model/ 目录
  - 无 t_cache 软标签预计算
  - 无 distill_loss()，训练损失 = CE + BALANCE_LOSS_W * balance_loss
  - 无 ALPHA / TEMPERATURE 超参
  - 保存路径改为 ./student_model_ablation
  - detect_attack_ids() 输出改为 attack_can_ids_results_ablation.json
  - meta.json 中 "distill": null, "ablation": "no_distillation"
  - 训练结束后打印完整消融实验汇总报告

【修复 RuntimeError: Unexpected key(s) 报错】
  根因：量化模型 state_dict 含 quant_state/absmax/quant_map 等bitsandbytes特殊buffer，
        用普通 load_state_dict 恢复时这些key被视为"Unexpected"而报错。
  方案：引入 get_saveable_state() / load_saveable_state() 方法，
        best_state 只保存 LoRA参数 + MoE头参数 的普通float tensor快照，
        恢复时只更新这两部分，BERT量化结构完全不动。

【特征（与teacher.py完全一致）】
  只用: timestamp / can_id / dlc / data_bytes（纯物理特征）
  不含: attack_type / label / type
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


# ── 依赖检查与安装 ──────────────────────────────────────────
def _pip(*pkgs):
    for pkg in pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:
    import accelerate
except ImportError:
    _pip("accelerate>=1.1.0")
    import accelerate

BNB_AVAILABLE = False
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
    print("✅ bitsandbytes 可用（支持4-bit量化）")
except ImportError:
    print("⚙️  安装 bitsandbytes...")
    _pip("bitsandbytes")
    try:
        import bitsandbytes as bnb
        BNB_AVAILABLE = True
        print("✅ bitsandbytes 安装成功")
    except Exception:
        print("⚠️  bitsandbytes 不可用，退化为 INT8 量化")

PEFT_AVAILABLE = False
try:
    from peft import (
        LoraConfig, get_peft_model, TaskType,
        prepare_model_for_kbit_training,
    )
    PEFT_AVAILABLE = True
    print("✅ PEFT/LoRA 可用")
except ImportError:
    print("⚙️  安装 peft...")
    _pip("peft")
    try:
        from peft import (
            LoraConfig, get_peft_model, TaskType,
            prepare_model_for_kbit_training,
        )
        PEFT_AVAILABLE = True
        print("✅ PEFT/LoRA 安装成功")
    except Exception as e:
        print(f"⚠️  PEFT 不可用: {e}")


# ================================================================
# 1. 全局配置（与 dis2.py 保持一致，方便对比）
# ================================================================
BERT_BASE  = "google-bert/bert-base-chinese"
DATA_PATH  = "merged_can_dataset3.json"
SAVE_PATH  = "./student_model_ablation"   # 消融版保存路径

NUM_CLASSES = 4
BATCH_SIZE  = 16
TEST_SIZE   = 0.2
SEED        = 42

STUDENT_EPOCHS = 3
STUDENT_LR     = 2e-4
MAX_LEN        = 128
DROPOUT        = 0.15
WEIGHT_DECAY   = 0.01
GRAD_CLIP      = 1.0

# QLoRA配置
LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
# 扩展到FFN层，增强CAN十六进制模式特征提取
LORA_TARGET  = ["query", "value", "intermediate.dense", "output.dense"]

# 消融版：无 ALPHA 和 TEMPERATURE（不需要蒸馏超参）
BALANCE_LOSS_W = 0.01   # MoE负载均衡损失权重（保留）

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL2ID     = {"normal": 0, "fuzzy": 1, "dos": 2, "spoofing_gear": 3}
ID2LABEL     = {v: k for k, v in LABEL2ID.items()}
CLASS_NAMES  = list(LABEL2ID.keys())
ATTACK_TYPES = ["fuzzy", "dos", "spoofing_gear"]

print(f"运行设备: {DEVICE}")
print(f"QLoRA模式: {'4-bit NF4 (GPU)' if (BNB_AVAILABLE and torch.cuda.is_available()) else 'INT8动态量化 (CPU)'}")


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)


# ================================================================
# 2. 工具函数
# ================================================================
def make_fc(in_f, out_f):
    l = nn.Linear(in_f, out_f)
    nn.init.xavier_uniform_(l.weight)
    nn.init.zeros_(l.bias)
    return l


def dir_mb(path):
    return sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(path) for f in fs
    ) / 1024 ** 2


def balance_loss(gate_weights):
    """
    MoE负载均衡损失：惩罚专家使用率不均匀。
    gate_weights: [B, 3]，已经过softmax。
    强制3个专家被均等使用，防止退化成只用1个专家。
    """
    mean_usage = gate_weights.mean(dim=0)           # [3]
    target     = torch.ones_like(mean_usage) / 3.0
    return F.mse_loss(mean_usage, target)


# ================================================================
# 3. 特征函数（与 teacher.py 完全一致）
# ================================================================
def item_to_text(item):
    """纯物理特征，绝对不含任何标签信息"""
    try:
        raw_data = item["data"]
        payload  = (" ".join(f"{b & 0xFF:02x}" for b in raw_data)
                    if isinstance(raw_data, list) else "00 " * 8)
    except Exception:
        payload = "00 " * 8
    return (f"ts {str(item.get('timestamp', '0')).strip()} "
            f"id {str(item.get('can_id', '0x000')).strip()} "
            f"dlc {str(item.get('data_length', 8))} "
            f"payload {payload}")


# ================================================================
# 4. 数据集
# ================================================================
class CANDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=MAX_LEN, is_train=False):
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
        enc   = self.tok(
            item_to_text(item),
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
            "idx":            idx,
        }


# ================================================================
# 5. 数据加载（与 teacher.py 完全一致）
# ================================================================
def load_and_split(path=DATA_PATH):
    if not os.path.exists(path):
        print(f"⚠️  {path} 不存在，生成模拟数据")
        data  = []
        specs = [
            ("normal",        [f"0x{i:03x}" for i in range(0x100, 0x180)], 6000),
            ("dos",           ["0x100", "0x101", "0x102"],                  1000),
            ("fuzzy",         [f"0x{i:03x}" for i in range(0x000, 0x7ff)], 1000),
            ("spoofing_gear", ["0x03f6"],                                   1000),
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
            if "data" not in item or not isinstance(item["data"], list):
                item["data"] = [0] * 8
            if "data_length" not in item:
                item["data_length"] = len(item["data"])
            if "type" not in item:
                item["type"] = "CAN"
            if item["attack_type"] not in LABEL2ID:
                item["attack_type"] = "normal"
            item["label"] = LABEL2ID[item["attack_type"]]
            for k in ["src_ip", "dst_ip"]: item.pop(k, None)
            valid.append(item)
        except Exception:
            continue

    print(f"有效数据: {len(valid)} 条")
    train_data, test_data = train_test_split(
        valid, test_size=TEST_SIZE, random_state=SEED, shuffle=True,
        stratify=[d["attack_type"] for d in valid],
    )
    print(f"训练集: {len(train_data)} | 测试集: {len(test_data)}")

    def _dist(d):
        cnt = defaultdict(int)
        for x in d: cnt[x["attack_type"]] += 1
        return dict(cnt)

    print(f"  训练集分布: {_dist(train_data)}")
    print(f"  测试集分布: {_dist(test_data)}")
    return train_data, test_data


# ================================================================
# 6. 学生模型：QLoRA（不冻结）→ MoE头
#    （消融版：架构与 dis2.py 完全相同，无教师权重复制）
# ================================================================
class MoEStudentModel(nn.Module):
    """
    【修复 RuntimeError: Unexpected key(s) 的核心设计】

    量化模型的 state_dict() 包含 bitsandbytes 特有的 buffer：
      bert.encoder.layer.0.attention.self.query.base_layer.weight.absmax
      bert.encoder.layer.0.attention.self.query.base_layer.weight.quant_state.bitsandbytes__nf4
      ...（数百个）

    这些key不在普通 BertModel 的 state_dict 里，直接调用
    model.load_state_dict(best_state) 就会触发 "Unexpected key(s)" 报错。

    解决方案：
      - get_saveable_state()：只提取 LoRA参数 + MoE头参数 的普通tensor
      - load_saveable_state()：按名称精确更新对应参数，完全绕过 state_dict
      - BERT量化结构（含quant_state等buffer）始终不动，也不需要动
    """

    def __init__(self, bert_path, num_classes=4, dropout=DROPOUT):
        super().__init__()
        self.bert_dim    = 768
        self.num_classes = num_classes
        self._weight_dir = bert_path

        # ── 构建QLoRA BERT ───────────────────────────────────
        print("\n  [QLoRA] 加载12层BERT，应用QLoRA...")
        self.bert = self._build_qlora_bert(bert_path)

        # ── 构建MoE头（与 dis2.py 完全一致）────────────────
        print("\n  [MoE]  构建 gate + 3专家（结构与dis2.py一致）...")
        d = self.bert_dim
        self.gate = nn.Sequential(
            make_fc(d, 128), nn.GELU(), nn.Dropout(0.1),
            make_fc(128, 3), nn.Softmax(dim=-1),
        )

        def _expert():
            return nn.Sequential(
                make_fc(d, 256),   nn.GELU(), nn.Dropout(0.1),
                make_fc(256, 128), nn.GELU(), nn.Dropout(0.05),
                make_fc(128, num_classes),
            )

        self.expert_dos   = _expert()
        self.expert_fuzzy = _expert()
        self.expert_gear  = _expert()
        self.drop         = nn.Dropout(dropout)
        print("  ✅ MoE头: gate(768→128→3) + 3×专家(768→256→128→4)")

    # ── 构建QLoRA BERT ───────────────────────────────────────
    def _build_qlora_bert(self, weight_dir):
        use_4bit = BNB_AVAILABLE and PEFT_AVAILABLE and torch.cuda.is_available()

        if use_4bit:
            print("    模式: 4-bit NF4量化 + LoRA（GPU）")
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            bert = None
            for path in [weight_dir, BERT_BASE, BERT_BASE.replace("google-bert/", "")]:
                try:
                    bert = BertModel.from_pretrained(
                        path, quantization_config=bnb_cfg,
                        ignore_mismatched_sizes=True)
                    print(f"    ✅ 4-bit量化加载成功，来源: {path}")
                    break
                except Exception as e:
                    print(f"    ⚠️  {path}: {e}")
            if bert is None:
                raise RuntimeError("4-bit量化加载失败，请检查GPU和bitsandbytes版本")
            bert = prepare_model_for_kbit_training(
                bert, use_gradient_checkpointing=True)
        else:
            print("    模式: 普通加载 + LoRA + INT8量化（CPU）")
            bert = None
            for path in [weight_dir, BERT_BASE, BERT_BASE.replace("google-bert/", "")]:
                try:
                    bert = BertModel.from_pretrained(path, ignore_mismatched_sizes=True)
                    print(f"    ✅ BERT加载成功，来源: {path}")
                    break
                except Exception:
                    continue
            if bert is None:
                raise RuntimeError("BERT加载失败")

        # 添加LoRA
        if PEFT_AVAILABLE:
            # 过滤实际存在的target模块
            all_module_names = [n for n, _ in bert.named_modules()]
            valid_targets = []
            for t in LORA_TARGET:
                if any(t in name for name in all_module_names):
                    valid_targets.append(t)
                else:
                    print(f"    ⚠️  跳过不存在的模块: {t}")
            if not valid_targets:
                valid_targets = ["query", "value"]
                print(f"    退化到默认target: {valid_targets}")

            lora_cfg = LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                target_modules=valid_targets,
                lora_dropout=LORA_DROPOUT,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            bert = get_peft_model(bert, lora_cfg)
            trainable = sum(p.numel() for p in bert.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in bert.parameters())
            print(f"    ✅ LoRA添加完成 (r={LORA_R}, target={valid_targets})")
            print(f"    LoRA可训练: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")

            if not use_4bit:
                # CPU路径：LoRA加完后再INT8量化（压缩非LoRA层体积）
                bert = torch.quantization.quantize_dynamic(
                    bert, {nn.Linear}, dtype=torch.qint8, inplace=False)
                print("    ✅ INT8动态量化完成")
        else:
            print("    ⚠️  PEFT不可用，仅INT8量化")
            if not use_4bit:
                bert = torch.quantization.quantize_dynamic(
                    bert, {nn.Linear}, dtype=torch.qint8, inplace=False)

        return bert

    # ── 前向传播 ─────────────────────────────────────────────
    def forward(self, input_ids, attention_mask, labels=None,
                return_gate_weights=False):
        out  = self.bert(
            input_ids=input_ids.to(DEVICE),
            attention_mask=attention_mask.to(DEVICE),
        )
        feat = self.drop(out.last_hidden_state[:, 0, :].float())
        feat = torch.clamp(feat, -10, 10)

        gw     = self.gate(feat)
        e_dos  = torch.clamp(self.expert_dos(feat),   -50, 50)
        e_fuz  = torch.clamp(self.expert_fuzzy(feat), -50, 50)
        e_gear = torch.clamp(self.expert_gear(feat),  -50, 50)
        logits = gw[:, 0:1] * e_dos + gw[:, 1:2] * e_fuz + gw[:, 2:3] * e_gear

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.to(DEVICE))

        if return_gate_weights:
            return {"loss": loss, "logits": logits, "gate_weights": gw}
        return {"loss": loss, "logits": logits}

    # ── 核心修复：安全保存/恢复（绕开quant_state问题）───────
    def get_saveable_state(self):
        """
        只保存 LoRA参数 + MoE头参数 的普通float tensor。
        不碰量化相关的任何buffer（quant_state/absmax/quant_map等）。
        """
        state = {}
        # LoRA参数（requires_grad=True且名称含lora_）
        for name, param in self.bert.named_parameters():
            if param.requires_grad and "lora_" in name:
                try:
                    state[f"bert_lora::{name}"] = (
                        param.detach().cpu().float().clone())
                except Exception:
                    pass
        # MoE头全部参数
        for moe_attr in ["gate", "expert_dos", "expert_fuzzy", "expert_gear"]:
            module = getattr(self, moe_attr, None)
            if module is None:
                continue
            for pname, param in module.named_parameters():
                state[f"moe::{moe_attr}::{pname}"] = (
                    param.detach().cpu().float().clone())
        return state

    def load_saveable_state(self, state):
        """
        精确恢复 LoRA参数 + MoE头参数，不触碰BERT量化结构。
        """
        # 恢复LoRA参数
        bert_named = dict(self.bert.named_parameters())
        for key, val in state.items():
            if not key.startswith("bert_lora::"):
                continue
            pname = key[len("bert_lora::"):]
            if pname in bert_named and bert_named[pname].requires_grad:
                try:
                    bert_named[pname].data.copy_(
                        val.to(bert_named[pname].device))
                except Exception as e:
                    print(f"    ⚠️  恢复LoRA参数失败 {pname}: {e}")

        # 恢复MoE头参数
        for moe_attr in ["gate", "expert_dos", "expert_fuzzy", "expert_gear"]:
            module = getattr(self, moe_attr, None)
            if module is None:
                continue
            module_named = dict(module.named_parameters())
            for pname, param in module_named.items():
                key = f"moe::{moe_attr}::{pname}"
                if key in state:
                    try:
                        param.data.copy_(state[key].to(param.device))
                    except Exception as e:
                        print(f"    ⚠️  恢复MoE参数失败 {key}: {e}")

    # ── 参数统计 ─────────────────────────────────────────────
    def print_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora_p    = sum(p.numel() for n, p in self.named_parameters()
                        if p.requires_grad and "lora_" in n)
        moe_p     = sum(p.numel() for n, p in self.named_parameters()
                        if p.requires_grad and any(x in n for x in
                           ["gate", "expert_dos", "expert_fuzzy", "expert_gear"]))
        print(f"  总参数量:       {total:>12,}")
        print(f"  可训练参数:     {trainable:>12,}  ({100*trainable/max(total,1):.2f}%)")
        print(f"  ├─ LoRA参数:    {lora_p:>12,}")
        print(f"  └─ MoE头参数:   {moe_p:>12,}")
        print(f"  预估fp16大小:   ~{total*2/1024/1024:.1f} MB（未含量化压缩）")


# ================================================================
# 7. 评估
# ================================================================
def evaluate(model, loader):
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
# 8. 检测攻击CAN ID（消融版：输出文件名区分）
# ================================================================
def detect_attack_ids(model, tokenizer, test_data):
    print("\n开始检测异常攻击CAN ID（消融版）...")
    ds         = CANDataset(test_data, tokenizer, MAX_LEN)
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
            logits = out["logits"] if isinstance(out, dict) else out
            pred   = ID2LABEL[torch.argmax(logits, -1).item()]
            if pred in ATTACK_TYPES:
                attack_ids[pred].add(item["can_id"])
                records.append({
                    "can_id":                item["can_id"],
                    "predicted_attack_type": pred,
                    "actual_attack_type":    item["raw_item"]["attack_type"],
                    "timestamp":             item["raw_item"]["timestamp"],
                    "data_length":           item["raw_item"]["data_length"],
                })

    print("\n" + "=" * 60)
    print("       异常攻击CAN ID检测结果（消融版：无蒸馏）")
    print("=" * 60)
    total_ids = set()
    for at, cids in sorted(attack_ids.items()):
        print(f"\n【{at.upper()}攻击】CAN ID数量: {len(cids)}")
        print(f"  {sorted(cids)[:20]}{'...' if len(cids) > 20 else ''}")
        total_ids |= cids
    print(f"\n【总计】{len(total_ids)} 个异常CAN ID")

    out_file = "attack_can_ids_results_ablation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "attack_type_can_ids":  {k: sorted(v) for k, v in attack_ids.items()},
            "total_attack_can_ids": sorted(total_ids),
            "all_attack_records":   records,
        }, f, ensure_ascii=False, indent=2)
    print(f"已保存至: {out_file}")
    return attack_ids


# ================================================================
# 9. 保存模型（fp16，≤50MB）
#    消融版：路径改为 ./student_model_ablation，meta.json 标注消融信息
# ================================================================
def save_student(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    print(f"\n💾 保存消融版模型 (fp16) → {path}")

    fp16_state = {}

    # 尝试合并LoRA权重到BERT（使权重可独立推理）
    merged_bert = None
    try:
        if hasattr(model.bert, "merge_and_unload"):
            merged_bert = model.bert.merge_and_unload()
            print("  ✅ LoRA权重已合并到BERT")
            for n, p in merged_bert.named_parameters():
                try:
                    fp16_state[f"bert_merged.{n}"] = (
                        p.detach().cpu().to(torch.float16))
                except Exception:
                    pass
    except Exception as e:
        print(f"  ⚠️  LoRA合并失败({e})，保存LoRA参数")

    # 保存LoRA参数（若未合并）
    if merged_bert is None:
        for n, p in model.bert.named_parameters():
            if "lora_" in n:
                try:
                    fp16_state[f"bert_lora.{n}"] = (
                        p.detach().cpu().to(torch.float16))
                except Exception:
                    pass

    # 保存MoE头（fp16）
    for attr in ["gate", "expert_dos", "expert_fuzzy", "expert_gear"]:
        module = getattr(model, attr, None)
        if module is None:
            continue
        for pname, param in module.named_parameters():
            try:
                fp16_state[f"{attr}.{pname}"] = (
                    param.detach().cpu().to(torch.float16))
            except Exception:
                fp16_state[f"{attr}.{pname}"] = param.detach().cpu().float()

    torch.save(fp16_state, os.path.join(path, "student_fp16.bin"))

    # 保存BERT config
    try:
        cfg = (model.bert.config if hasattr(model.bert, "config") else
               model.bert.base_model.model.config)
        cfg.save_pretrained(path)
    except Exception:
        pass

    # 保存meta（消融版：distill=null，ablation="no_distillation"）
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump({
            "architecture":  "MoEStudent_QLoRA_Ablation",
            "bert_layers":   12,
            "bert_hidden":   768,
            "qlora": {
                "quantization": ("4bit-NF4" if (BNB_AVAILABLE and
                                                torch.cuda.is_available())
                                 else "INT8-dynamic"),
                "lora_r":      LORA_R,
                "lora_alpha":  LORA_ALPHA,
                "lora_target": LORA_TARGET,
            },
            "moe": {
                "num_experts":       3,
                "expert_arch":       "768→256→128→4",
                "gate_arch":         "768→128→3",
                "balance_loss_w":    BALANCE_LOSS_W,
            },
            "distill":   None,                     # 消融：无蒸馏
            "ablation":  "no_distillation",        # 消融标注
            "max_len":        MAX_LEN,
            "num_classes":    NUM_CLASSES,
            "label2id":       LABEL2ID,
            "id2label":       ID2LABEL,
            "feature_fields": ["timestamp", "can_id", "dlc", "payload"],
        }, f, indent=2)

    tokenizer.save_pretrained(path)

    bin_mb = os.path.getsize(os.path.join(path, "student_fp16.bin")) / 1024 ** 2
    tot_mb = dir_mb(path)
    print(f"  权重(fp16): {bin_mb:.2f} MB  |  目录总计: {tot_mb:.2f} MB")

    if tot_mb > 50:
        for fname in ["tokenizer.json", "vocab.txt"]:
            fp = os.path.join(path, fname)
            if os.path.exists(fp):
                sz = os.path.getsize(fp) / 1024 ** 2
                os.remove(fp)
                tot_mb -= sz
                print(f"  移除 {fname} ({sz:.2f}MB) → 剩余 {tot_mb:.2f}MB")

    flag = "✅ ≤50MB" if tot_mb <= 50 else "❌ >50MB"
    print(f"  最终大小: {tot_mb:.2f} MB  {flag}")
    return bin_mb, tot_mb


# ================================================================
# 10. 主流程（消融版：3步，无教师，无KL）
# ================================================================
def main():
    print("=" * 65)
    print("  CAN攻击检测 — 消融实验（QLoRA + MoE，去除蒸馏）")
    print("  实验条件：纯CE监督训练，无教师模型，无KL软标签")
    print("=" * 65)

    train_data, test_data = load_and_split(DATA_PATH)

    try:
        tokenizer = BertTokenizer.from_pretrained(BERT_BASE)
    except Exception:
        tokenizer = BertTokenizer.from_pretrained(
            BERT_BASE.replace("google-bert/", ""))

    # ── Step 1: 构建模型（QLoRA + LoRA + MoE）───────────────
    print("\n" + "=" * 55)
    print("  Step 1/3  构建模型（QLoRA + LoRA + MoE）")
    print("           [消融：直接从BERT初始化，无教师权重复制]")
    print("=" * 55)
    student = MoEStudentModel(BERT_BASE, NUM_CLASSES, DROPOUT).to(DEVICE)
    print("\n消融版模型参数统计:")
    student.print_params()

    # ── Step 2: 数据准备 ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Step 2/3  数据准备")
    print("=" * 55)
    tr_ds = CANDataset(train_data, tokenizer, MAX_LEN, is_train=True)
    te_ds = CANDataset(test_data,  tokenizer, MAX_LEN)
    tr_ld = DataLoader(tr_ds, BATCH_SIZE,     shuffle=True,  num_workers=0)
    te_ld = DataLoader(te_ds, BATCH_SIZE * 2, shuffle=False, num_workers=0)

    print("\n📊 训练前基线（随机初始化后直接评估）:")
    pre_acc, pre_rep = evaluate(student, te_ld)
    print(f"  基线准确率: {pre_acc:.4f} ({pre_acc*100:.2f}%)")
    print(pre_rep)

    # ── Step 3: 纯CE训练（无教师、无KL、无软标签）───────────
    print("\n" + "=" * 55)
    print("  Step 3/3  纯CE监督训练（消融：无蒸馏）")
    print("=" * 55)

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"  实际可训练参数: {n_trainable:,}")
    print(f"  训练配置: Epochs={STUDENT_EPOCHS}, LR={STUDENT_LR}, "
          f"Batch={BATCH_SIZE}, 损失=CE+Balance")

    if not trainable_params:
        print("  ⚠️  无可训练参数，自动解冻MoE头...")
        for n, p in student.named_parameters():
            if any(x in n for x in
                   ["gate", "expert_dos", "expert_fuzzy", "expert_gear"]):
                p.requires_grad = True
        trainable_params = [p for p in student.parameters() if p.requires_grad]
        print(f"  已解冻MoE头: {sum(p.numel() for p in trainable_params):,} 参数")

    opt         = torch.optim.AdamW(
        trainable_params, lr=STUDENT_LR,
        weight_decay=WEIGHT_DECAY, eps=1e-8, betas=(0.9, 0.999),
    )
    total_steps = len(tr_ld) * STUDENT_EPOCHS
    sched       = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    print(f"\n🚀 开始CE训练 ({STUDENT_EPOCHS} epochs)...")
    epoch_times, epoch_accs = [], []
    best_acc   = 0.0
    best_state = None  # 只存 LoRA + MoE 的普通tensor快照

    for epoch in range(STUDENT_EPOCHS):
        student.train()
        t0                = time.time()
        tloss, bal, steps = 0.0, 0.0, 0

        for step, batch in enumerate(tr_ld):
            try:
                s_ids  = batch["input_ids"].to(DEVICE)
                s_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                out      = student(input_ids=s_ids, attention_mask=s_mask,
                                   labels=labels, return_gate_weights=True)
                s_logits = out["logits"]
                gw       = out["gate_weights"]

                # 消融版：纯CE + 均衡损失（无KL蒸馏项）
                ce_loss = F.cross_entropy(s_logits, labels)
                b_loss  = balance_loss(gw)
                loss    = ce_loss + BALANCE_LOSS_W * b_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
                opt.step()
                sched.step()

                tloss += ce_loss.item()
                bal   += b_loss.item()
                steps += 1

                if (step + 1) % 100 == 0:
                    print(f"  Epoch[{epoch+1}/{STUDENT_EPOCHS}] "
                          f"Step[{step+1}/{len(tr_ld)}] "
                          f"CE={tloss/steps:.4f}  "
                          f"Balance={bal/steps:.4f}")
            except Exception as e:
                print(f"  ⚠️  Step {step+1}: {e}")
                continue

        elapsed = (time.time() - t0) / 60.0
        epoch_times.append(elapsed)

        if steps > 0:
            acc, _ = evaluate(student, te_ld)
            epoch_accs.append(acc)
            print(f"\n  Epoch[{epoch+1}/{STUDENT_EPOCHS}] "
                  f"CE={tloss/steps:.4f} | Balance={bal/steps:.4f} | "
                  f"Acc={acc:.4f} ({acc*100:.2f}%) | 时间={elapsed:.2f}min")
            if acc > best_acc:
                best_acc   = acc
                # ★ 只保存普通tensor快照，完全避免quant_state key冲突
                best_state = student.get_saveable_state()
                print(f"  ⭐ 新最优: {best_acc:.4f}")
        else:
            epoch_accs.append(0.0)
        print()

    # ★ 用专用方法恢复，不调用 load_state_dict
    if best_state is not None:
        student.load_saveable_state(best_state)
        print(f"✅ 已恢复最优参数  Acc={best_acc:.4f}")

    print("\n📊 CE训练后最终性能:")
    post_acc, post_rep = evaluate(student, te_ld)
    print(f"准确率: {post_acc:.4f} ({post_acc*100:.2f}%)  "
          f"(相对基线 {post_acc-pre_acc:+.4f})")
    print(post_rep)

    # 推理速度测量（热身3个batch后统计100个batch）
    print("⏱️  测量推理速度（热身3批 + 统计100批）...")
    student.eval()
    ms_list, warmup = [], 0
    with torch.no_grad():
        for bench in te_ld:
            ids  = bench["input_ids"].to(DEVICE)
            mask = bench["attention_mask"].to(DEVICE)
            if warmup < 3:
                student(input_ids=ids, attention_mask=mask)
                warmup += 1; continue
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            student(input_ids=ids, attention_mask=mask)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            ms_list.append((time.perf_counter() - t0) * 1000)
            if len(ms_list) >= 100: break
    avg_ms = float(np.mean(ms_list)) if ms_list else 0.0
    std_ms = float(np.std(ms_list))  if ms_list else 0.0
    per_ms = avg_ms / (BATCH_SIZE * 2)

    detect_attack_ids(student, tokenizer, test_data)
    bin_mb, tot_mb = save_student(student, tokenizer, SAVE_PATH)

    # ────────────────────────────────────────────────────────────
    # 消融实验汇总报告
    # ────────────────────────────────────────────────────────────
    quant_mode = "4-bit NF4" if (BNB_AVAILABLE and torch.cuda.is_available()) else "INT8动态量化"
    print()
    print("=================================================================")
    print("            消融实验报告：去除蒸馏（QLoRA + MoE，无蒸馏）")
    print("=================================================================")
    print(f"  实验条件：纯CE监督训练，无教师模型，无KL软标签")
    print()
    print("  【模型架构（消融组）】")
    print(f"  BERT:         12层 bert-base-chinese")
    print(f"  量化:         {quant_mode}")
    print(f"  LoRA:         r={LORA_R}, alpha={LORA_ALPHA}, target=[query/value/FFN]")
    print(f"  MoE:          gate(768→128→3) + 3×专家(768→256→128→4)")
    print(f"  蒸馏:         ❌ 已去除（消融对照）")
    print()
    print("  【准确率 Acc (%)】")
    print(f"  训练前基线:   {pre_acc*100:.2f}%")
    for i, acc in enumerate(epoch_accs):
        print(f"  Epoch {i+1}:      {acc*100:.2f}%")
    print(f"  最终(最优):   {post_acc*100:.2f}%")
    print()
    print("  【模型大小】")
    print(f"  权重文件:     {bin_mb:.2f} MB  (student_fp16.bin)")
    print(f"  目录总计:     {tot_mb:.2f} MB")
    print()
    print("  【每Epoch训练时间】")
    for i, t in enumerate(epoch_times):
        print(f"  Epoch {i+1}:      {t:.2f} min")
    if epoch_times:
        print(f"  均值:         {np.mean(epoch_times):.2f} min/epoch")
    print()
    print("  【每Batch推理时间】")
    print(f"  平均:         {avg_ms:.2f} ± {std_ms:.2f} ms  (batch={BATCH_SIZE*2})")
    print(f"  单样本:       {per_ms:.3f} ms/sample")
    print()
    print("  【与蒸馏版对比说明】")
    print("  本文件为消融实验，去除了KD软标签损失。")
    print("  对比 dis2.py（含蒸馏）可评估蒸馏带来的准确率提升。")
    print("=================================================================")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()

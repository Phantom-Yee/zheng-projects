import csv
import json
import random


def parse_can_csv(csv_file_path, attack_type):
    """
    解析单个CAN CSV文件，返回结构化数据列表
    :param csv_file_path: CSV文件路径
    :param attack_type: 攻击类型（dos/fuzzy/spoofing_gear/normal）
    :return: 解析后的字典列表
    """
    data_list = []
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader):
                # 跳过空行/格式异常行
                if len(row) < 12:
                    print(f"警告：第{row_idx + 1}行格式异常，跳过")
                    continue

                # 解析核心字段
                timestamp = row[0]
                can_id_hex = row[1]
                data_length = int(row[2])
                # 16进制字节转十进制整数列表（固定8位）
                data_dec_list = [int(byte, 16) for byte in row[3:11]]
                flag = row[11]  # T=攻击/注入, R=正常

                # 确定label和最终attack_type（R标记强制为normal）
                label = 1 if flag == "T" else 0
                final_attack_type = "normal" if flag == "R" else attack_type

                # 构建符合要求的JSON结构
                item = {
                    "timestamp": timestamp,
                    "can_id": f"0x{can_id_hex}",
                    "data_length": data_length,
                    "data": data_dec_list,
                    "attack_type": final_attack_type,
                    "type": "CAN",
                    "label": label
                }
                data_list.append(item)
        print(f"✅ {csv_file_path} 解析完成，共{len(data_list)}条数据")
        return data_list
    except FileNotFoundError:
        print(f"❌ 错误：未找到文件 {csv_file_path}")
        return []


def merge_and_shuffle_datasets(output_file):
    """
    合并所有数据集并打乱，输出最终JSON文件
    :param output_file: 输出JSON文件路径
    """
    # 1. 定义数据集映射（文件路径 + 攻击类型）
    datasets = [
        ("D:\save\distillation\DoS_dataset.csv", "dos"),
        ("D:\save\distillation\Fuzzy_dataset.csv", "fuzzy"),
        ("D:\save\distillation\gear_dataset.csv", "spoofing_gear")
        # 若有独立的normal数据集，可添加：("normal_dataset.csv", "normal")
    ]

    # 2. 解析所有数据集
    all_data = []
    for csv_path, attack_type in datasets:
        all_data.extend(parse_can_csv(csv_path, attack_type))

    # 3. 随机打乱数据（实验用）
    random.seed(42)  # 固定随机种子，保证实验可复现；如需每次不同，注释此行
    random.shuffle(all_data)
    print(f"\n📊 合并后总数据量：{len(all_data)} 条")

    # 4. 统计各类攻击数量（验证）
    attack_count = {}
    for item in all_data:
        at = item["attack_type"]
        attack_count[at] = attack_count.get(at, 0) + 1
    print("📈 数据分布：", attack_count)

    # 5. 写入最终JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 最终文件已生成：{output_file}")


# ---------------------- 执行主函数 ----------------------
if __name__ == "__main__":
    # 输出文件路径（可自定义）
    OUTPUT_JSON = "merged_can_dataset.json"
    merge_and_shuffle_datasets(OUTPUT_JSON)
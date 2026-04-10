import os
import json
import torch
import gc
import psutil
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# 配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

MODEL_PATH = r"D:\2026projects\MedIntellect\models\qwen2.5-1.5b-instruct"
DATA_PATH = r"D:\2026projects\MedIntellect\data\finetune_data.jsonl"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "qwen2.5-1.5b-instruct-medical")


def print_memory_usage():
    """打印当前内存使用情况（仅CPU）"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    print(f"[内存] RSS: {mem.rss / 1024 / 1024:.1f} MB, VMS: {mem.vms / 1024 / 1024:.1f} MB")
    print(f"[内存] 系统总内存: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB, 可用: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")


def load_data(file_path):
    """加载医疗问答数据（处理每行是包含单个对象的数组格式）"""
    data = []

    print(f"正在加载数据: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        line_count = 0
        success_count = 0

        for line in f:
            line = line.strip()
            if not line:
                continue

            line_count += 1
            try:
                # 解析这一行的 JSON
                parsed = json.loads(line)

                # 处理不同格式
                if isinstance(parsed, list) and len(parsed) == 1:
                    # 格式: [{"system": "...", "prompt": "...", "response": "..."}]
                    item = parsed[0]
                elif isinstance(parsed, dict):
                    # 标准格式: {"system": "...", "prompt": "...", "response": "..."}
                    item = parsed
                else:
                    print(f"警告: 第 {line_count} 行格式不支持，跳过")
                    continue

                # 提取字段
                system = item.get('system',
                                  '你是一个专业的医疗健康助手，擅长用通俗易懂的语言解答患者的常见健康问题，并提供科学、安全的建议。')
                prompt = item.get('prompt')
                response = item.get('response')

                if not prompt or not response:
                    print(f"警告: 第 {line_count} 行缺少 prompt 或 response 字段")
                    print(f"可用字段: {list(item.keys())}")
                    continue

                # 构造对话格式（Qwen 格式）
                text = f"""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>"""

                data.append({"text": text})
                success_count += 1

                # 显示进度
                if line_count % 10 == 0:
                    print(f"已处理 {line_count} 行，成功加载 {success_count} 条...")
                    print_memory_usage()

            except json.JSONDecodeError as e:
                print(f"错误: 第 {line_count} 行 JSON 解析失败: {e}")
                continue

    print(f"成功加载 {success_count} 条有效数据（共处理 {line_count} 行）")
    print_memory_usage()

    if success_count == 0:
        raise ValueError("没有加载到任何有效数据，请检查文件格式")

    return Dataset.from_list(data)


def tokenize_function(examples, tokenizer, max_length=256):  # 调低max_length以节省内存
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )


def main():
    # 初始内存状态
    print("初始内存状态：")
    print_memory_usage()

    # 检查模型路径
    if not os.path.exists(MODEL_PATH):
        print(f"错误：模型路径不存在: {MODEL_PATH}")
        return

    print(f"从本地加载模型: {MODEL_PATH}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True
    )

    # 设置填充 token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 强制使用float32，避免CPU不支持bfloat16
    torch_dtype = torch.float32
    print(f"使用数据类型: {torch_dtype}")

    # 加载模型（CPU环境，device_map会自动选择CPU）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto"
    )

    # 启用梯度检查点（大幅减少激活内存）
    model.gradient_checkpointing_enable()
    print("已启用梯度检查点")

    print("模型加载成功！")
    print_memory_usage()

    # 检查数据路径
    if not os.path.exists(DATA_PATH):
        print(f"错误：数据路径不存在: {DATA_PATH}")
        return

    # 加载数据集
    dataset = load_data(DATA_PATH)

    # 分词处理
    print("开始分词...")
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=256),  # 明确传入max_length
        batched=True,
        remove_columns=["text"]
    )

    print(f"数据集大小: {len(dataset)} 条")
    print_memory_usage()

    # 训练参数 - 针对CPU内存优化
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,                # 批次保持最小
        gradient_accumulation_steps=8,                 # 累积步数保持有效批次
        num_train_epochs=3,
        learning_rate=2e-5,
        fp16=False,                                    # CPU不支持fp16
        bf16=False,                                    # CPU不支持bf16
        # 关键：禁用中间保存，训练结束后手动保存
        save_strategy="no",                             # 不保存中间检查点
        save_total_limit=0,                             # 不保留任何检查点
        logging_dir=os.path.join(PROJECT_ROOT, "logs"),
        report_to="none",
        remove_unused_columns=False,
        logging_steps=10,
        # CPU内存优化设置
        dataloader_num_workers=0,                       # 禁止多进程加载
        dataloader_pin_memory=False,                    # 禁止锁页内存
        optim="adamw_torch",                             # 标准AdamW（可尝试其他优化器）
    )

    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("开始训练...")
    print(f"训练数据数量: {len(dataset)}")
    print(f"批次大小: {training_args.per_device_train_batch_size}")
    print(f"梯度累积步数: {training_args.gradient_accumulation_steps}")
    print(f"有效批次大小: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print_memory_usage()

    # 开始训练
    trainer.train()

    # 训练完成后手动清理内存并保存
    print("训练完成，准备保存模型...")
    gc.collect()  # 强制垃圾回收
    print_memory_usage()

    # 保存模型
    print(f"保存模型到: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"微调完成，模型已保存至 {OUTPUT_DIR}")
    print_memory_usage()


if __name__ == "__main__":
    main()
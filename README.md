# 🏥 KidMed - 儿科医疗智能体

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.0-green.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/license-MIT-red.svg)](LICENSE)

> 基于 **LangGraph + Qwen2.5-1.5B (INT4量化) + RAG + MySQL** 的儿科医疗对话系统。  
> 用户通过自然语言描述症状，系统自动分析病情、推荐科室、查询真实号源并支持多轮预约。

**演示视频**：[点击观看](https://www.bilibili.com/video/BV1RXQcB4EP9?vd_source=7983cd7c6f2a7fa6386e82374e1aa190) 

---

## 📌 项目背景

KidMed 是一款面向儿科常见症状的智能问诊助手。项目旨在利用大语言模型和检索增强生成技术，帮助家长快速获取病情分析和就医指导，同时对接医院号源系统完成预约挂号。系统采用轻量级量化模型，可在普通笔记本电脑上运行。

---

## ✨ 核心功能

| 模块               | 功能描述                                                     |
| ------------------ | ------------------------------------------------------------ |
| **症状分诊**       | 输入“孩子发烧咳嗽”，LLM 生成 30~100 字病情分析，自动推荐科室（内科/儿科/眼科/皮肤科/口腔科等） |
| **号源查询与预约** | 连接 MySQL 查询未来 7 天空闲号源，支持用户回复序号原子锁定号源，防止超卖 |
| **医疗知识问答**   | 基于 RAG 检索用药指南（20000+ 条）和疾病手册，回答药物用法、疾病预防等问题 |
| **多轮记忆**       | 使用 `memory_store` 维护会话历史和预约状态，支持上下文连贯对话 |
| **闲聊与退出**     | 处理非医疗话题并引导回正题，支持“再见”“谢谢”等退出词         |

---

## 🧠 技术栈

- **状态机编排**：LangGraph（定义路由、分诊、预约、问答、计划节点）
- **大语言模型**：Qwen2.5-1.5B-Instruct（4‑bit NF4 量化，显存 ~1.1GB）
- **检索增强生成**：Chroma + HuggingFace Embeddings（all‑MiniLM‑L6‑v2）
- **数据库**：MySQL（科室、医生、排班、预约记录表，支持行锁）
- **后端框架**：Flask + CORS
- **前端界面**：Vue 2 + Axios（响应式聊天界面）
- **其他**：bitsandbytes、transformers、torch、sentence‑transformers

---

## 🚀 快速开始

### 环境要求
- Python 3.10+
- MySQL 5.7+
- 8 GB RAM（推荐 16 GB）
- 支持 CUDA 的 GPU 可选（CPU 亦可运行）

### 1. 克隆仓库
```bash
git clone https://github.com/aqiqiqiu/KidMed1_0.git
cd KidMed1_0
```

### 2. 创建并激活虚拟环境

bash

```
python -m venv agent.env
# Windows
agent.env\Scripts\activate
# Linux/Mac
source agent.env/bin/activate
```

### 3. 安装依赖

bash

```
pip install -r requirements.txt
```



### 4. 配置 MySQL

- 创建数据库 `med_intellect`
- 修改 `db_manager.py` 中的 `DB_CONFIG` 为你的数据库账号密码
- 首次运行会自动建表并插入测试数据（科室、医生、排班）

### 5. 下载模型

将以下模型放入 `models/` 目录（或让代码自动下载）：

text

```
models/
├── qwen2.5-1.5b-instruct/
└── all-MiniLM-L6-v2/
```

### 6. 启动服务

bash

```
python app.py
```

访问 `http://localhost:5000` 即可使用。

------

## 📁 项目目录结构

text

```
KidMed/
├── agent.env/                     # Python 虚拟环境
├── data/                          # 数据目录
│   ├── knowladge/                 # 知识库文档（PDF/TXT）
│   └── finetune_data.jsonl        # 微调数据（可选）
├── models/                        # 本地模型文件
│   ├── qwen2.5-1.5b-instruct/    # 量化后的 Qwen 模型
│   └── all-MiniLM-L6-v2/         # Embedding 模型
├── modules/                       # 核心模块
│   ├── __init__.py
│   ├── agents.py                  # LangGraph 状态图与节点
│   ├── finetune.py                # 微调脚本
│   ├── memory_store.py            # 会话记忆与预约状态缓存
│   └── rag_retriever.py           # RAG 检索与向量库管理
├── template/                      # 前端界面
│   └── index.html                 # Vue 聊天页面
├── vectorstore/                   # Chroma 向量数据库持久化目录
├── .env                           # 环境变量（数据库密码等，不提交）
├── .gitignore                     # Git 忽略规则
├── app.py                         # Flask 主入口
├── database.py                    # 数据库辅助函数
├── db_manager.py                  # MySQL 操作（建表、查询、预约）
├── appointments.txt               # 预约记录备份（文本文件）
├── requirements.txt               # Python 依赖清单
└── README.md                      # 项目说明文档
```



> 注：`agent.env` 为虚拟环境目录，实际使用时请勿提交到仓库；`models/` 和 `vectorstore/` 已在 `.gitignore` 中忽略。

------

## 📊 测试用例与表现





| 测试场景  | 示例消息                 | 预期结果                                   | 通过率       |
| --------- | ------------------------ | ------------------------------------------ | ------------ |
| 分诊+号源 | “我头疼发烧，还有点咳嗽” | 分析头痛可能原因 → 推荐内科 → 显示号源列表 | 100% (20/20) |
| 预约流程  | 回复“1”                  | 预约成功，锁定号源                         | 100% (15/15) |
| 药物问答  | “布洛芬能退烧吗”         | 基于用药指南给出剂量和注意事项             | 90% (18/20)  |
| 疾病科普  | “什么是糖尿病”           | 给出定义和核心注意事项                     | 85% (17/20)  |

详细测试用例及录屏请见 [演示视频](https://www.bilibili.com/video/BV1RXQcB4EP9?vd_source=7983cd7c6f2a7fa6386e82374e1aa190)。

------

## 🧪 项目亮点

- **轻量量化**：4‑bit 量化使模型可在普通笔记本运行，显存仅 1.1GB，推理速度较快。
- **原子预约**：MySQL `FOR UPDATE` 行锁防止号源超卖，保证并发安全。
- **RAG 增强**：用药问答准确率从 50% 提升至 90%（通过扩充 20000+ 条结构化用药指南）。
- **模块化设计**：LangGraph 节点可插拔，便于扩展新功能（如电子病历、随访提醒）。
- **完整演示**：提供可交互前端、真实数据库查询、多轮记忆，开箱即用。

------

## 📈 后续优化方向

- 扩充 RAG 知识库至 10 万条，覆盖更多罕见病和药物相互作用。
- 接入真实医院挂号 API（如健康浙江、微医等）。
- 增加语音输入支持，提升家长使用便捷性。
- 使用 LoRA 微调模型，使其更贴合儿科问诊风格与语气。

------

## 🤝 贡献

本项目为个人课程设计/求职作品，欢迎提 Issue 或 Pull Request。

## 📄 许可证

MIT © [aqiqiqiu](https://github.com/aqiqiqiu)

------

**如果这个项目对你有帮助，欢迎给个 ⭐️ Star！**

```
**使用说明**：
1. 复制上面的全部内容。
2. 在项目根目录创建 `README.md` 文件并粘贴。
3. 将 `https://your-video-link.com` 替换为你的实际演示视频地址。
4. 提交到 GitHub：
   ```bash
   git add README.md
   git commit -m "Add comprehensive README"
   git push
```
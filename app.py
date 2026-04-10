import os
import sys
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline

# 导入你的自定义模块
from modules.agents import build_agent_graph, set_global_instances
from modules.rag_retriever import MedicalRAG
from modules.memory_store import memory_store

load_dotenv()

# ================= 1. 智能检测网页文件夹 =================
possible_folders = ['templete', 'template', 'html', 'static']
static_folder_name = 'templete'
for folder in possible_folders:
    if os.path.exists(folder):
        static_folder_name = folder
        break

print(f"📂 [系统] 检测到网页文件夹: ./{static_folder_name}")

app = Flask(__name__, static_folder=static_folder_name, static_url_path='')
CORS(app)

print("🚀 [系统] 正在初始化医疗智能体架构 (LangGraph + INT4 量化极速版)...")

model_path = "./models/qwen2.5-1.5b-instruct"
llm_chain = None
rag_instance = None
compiled_graph = None

try:
    # ================= 2. 加载 Tokenizer =================
    print(f"⏳ 正在加载 Tokenizer: {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    print("✅ Tokenizer 加载完成")

    # ================= 3. 配置 4-bit 量化 =================
    print("⚡ 正在配置 4-bit 量化参数...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ================= 4. 加载量化模型 =================
    print(f"⏳ 正在加载 4-bit 量化模型: {model_path} ...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    print("✅ 模型权重加载完成 (INT4 模式)")

    memory_allocated = torch.cuda.memory_allocated(0) / 1024 ** 2
    print(f"💾 当前显存占用: {memory_allocated:.2f} MB")

    # ================= 5. ⭐ 关键修复：构建 Pipeline ⭐ =================
    # 移除所有可能引起冲突的参数，只保留最核心的
    # 不在这里设置 max_length，避免与 max_new_tokens 冲突
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        # 生成参数
        max_new_tokens=512,  # 只控制新生成的长度
        temperature=0.3,
        top_p=0.8,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        # 重要：添加 stop_strings 防止模型啰嗦 (可选)
        # stop_strings=["Human:", "Assistant:"]
    )

    llm_chain = HuggingFacePipeline(pipeline=pipe)
    print("✅ LLM 管道构建成功 (无参数冲突)")

    # ================= 6. 加载 RAG =================
    rag_instance = MedicalRAG()
    rag_instance.load_vectorstore()
    print("✅ RAG 就绪")

    # ================= 7. 注入依赖 =================
    set_global_instances(llm_chain, rag_instance)

    # ================= 8. 编译 LangGraph =================
    compiled_graph = build_agent_graph()
    print("✅ LangGraph 架构编译完成")

    # ================= 9. ⭐ 关键修复：简化预热逻辑 ⭐ =================
    # 原代码跑完整的 Graph (包含 RAG 检索 + LLM 生成)，太慢且容易卡死。
    # 新策略：只测试模型能否生成一个字，不跑完整业务逻辑。
    print("⚡ 正在进行轻量级显存预热 (仅测试模型生成)...")

    try:
        # 直接调用底层 pipeline 进行一次极短的生成，绕过 LangGraph 和 RAG
        # 这样即使 RAG 有问题，也不会卡住启动
        test_input = "你好"
        test_output = pipe(test_input, max_new_tokens=5)
        print(f"✅ 预热成功！模型测试输出: '{test_output}'")
        print("✅ 系统现已就绪，无需等待完整流程预热。")
    except Exception as e:
        print(f"⚠️ 轻量预热失败: {e}")
        print("✅ 系统将跳过预热，首次请求时自动初始化。")

except Exception as e:
    print(f"❌ 初始化失败: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)


# ================= 10. 路由配置 =================

@app.route('/')
def serve_index():
    try:
        return send_from_directory(app.static_folder, 'index.html')
    except FileNotFoundError:
        return f"<h3>404: 未找到 index.html</h3><p>检查文件夹: ./{static_folder_name}</p>", 404


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

#
# @app.route('/api/chat', methods=['POST'])
# def chat():
#     data = request.json
#     user_msg = data.get('message', '')
#     patient_id = data.get('patient_id', 'guest')
#
#     if not user_msg or not compiled_graph:
#         return jsonify({"response": "系统未就绪", "status": "error"}), 503
#
#     initial_state = {
#         "messages": [{"role": "user", "content": user_msg}],
#         "loop_count": 0,
#         "user_intent": "",
#         "retrieved_docs": [],
#         "patient_id": patient_id,
#         "appointment_info": {},
#         "plan": ""
#     }
#
#     try:
#         final_state = None
#         # 流式执行，收集所有事件
#         for event in compiled_graph.stream(initial_state):
#             # event 结构通常是 {"node_name": {...}}，我们需要合并它们拿到最终状态
#             # LangGraph 的 stream 默认返回的是每个节点的输出，我们需要最后的完整状态
#             # 简单做法：直接取最后一个 event 的值，或者遍历合并
#             if isinstance(event, dict):
#                 # 如果是 {"__end__": {...}} 或者 {"qa": {...}} 这种形式
#                 # 我们尝试更新 final_state，确保拿到最新的 messages
#                 if final_state is None:
#                     final_state = {}
#                 # 合并状态 (简单粗暴版：保留所有键的最新值)
#                 for key, value in event.items():
#                     if isinstance(value, dict):
#                         final_state.update(value)
#                     else:
#                         final_state[key] = value
#
#         # 【防御性编程】检查 final_state 是否有 messages
#         if final_state and "messages" in final_state:
#             msgs = final_state["messages"]
#             # 倒序查找最后一条助手消息
#             for msg in reversed(msgs):
#                 if msg.get("role") == "assistant":
#                     content = msg.get("content", "")
#                     # 清洗可能的残留前缀
#                     if "Human:" in content:
#                         content = content.split("Human:")[-1].split("Assistant:")[-1].strip()
#
#                     return jsonify({
#                         "response": content,
#                         "status": "success",
#                         "intent": final_state.get("user_intent", "unknown"),
#                         "loops": final_state.get("loop_count", 0)
#                     })
#
#             # 如果没找到 assistant 消息（极端情况）
#             return jsonify({"response": "处理完成但未生成回复", "status": "success"}), 200
#
#         return jsonify({"response": "未生成回复", "status": "error", "debug": str(final_state)}), 500
#
#     except Exception as e:
#         print(f"⚠️ 运行时错误：{e}")
#         import traceback
#         traceback.print_exc()  # 打印详细错误堆栈，方便调试
#         return jsonify({"response": f"系统内部错误：{str(e)}", "status": "error"}), 500
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_msg = data.get('message', '')
    patient_id = data.get('patient_id', 'guest')

    if not user_msg or not compiled_graph:
        return jsonify({"response": "系统未就绪", "status": "error"}), 503

    from modules.memory_store import memory_store
    # 获取历史消息（最近4条）
    history = memory_store.get_recent_history(patient_id, limit=4)
    # 获取预约状态（包含 available_slots）
    appt_state = memory_store.get_appointment_state(patient_id)

    messages = history + [{"role": "user", "content": user_msg}]

    initial_state = {
        "messages": messages,
        "loop_count": 0,
        "user_intent": "",
        "retrieved_docs": [],
        "patient_id": patient_id,
        "appointment_info": {},
        "plan": "",
        "recommended_dept": appt_state.get("recommended_dept", ""),
        "available_slots": appt_state.get("available_slots", []),
        "selected_schedule_id": appt_state.get("selected_schedule_id"),
        "symptom_analysis": appt_state.get("symptom_analysis", ""),
    }

    try:
        final_state = None
        for event in compiled_graph.stream(initial_state, stream_mode="values"):
            final_state = event

        if final_state and "messages" in final_state:
            new_messages = final_state["messages"]
            # 保存新增消息
            if len(new_messages) > len(history):
                for msg in new_messages[len(history):]:
                    memory_store.add_message(patient_id, msg["role"], msg["content"])

            # 保存预约状态（关键：保存 available_slots）
            new_appt_state = {
                "recommended_dept": final_state.get("recommended_dept", ""),
                "available_slots": final_state.get("available_slots", []),
                "selected_schedule_id": final_state.get("selected_schedule_id"),
                "symptom_analysis": final_state.get("symptom_analysis", ""),
            }
            memory_store.set_appointment_state(patient_id, new_appt_state)

            # 获取最后一条助手消息
            for msg in reversed(new_messages):
                if msg["role"] == "assistant":
                    return jsonify({
                        "response": msg["content"],
                        "status": "success",
                        "intent": final_state.get("user_intent", "unknown")
                    })
            return jsonify({"response": "处理完成但未生成回复", "status": "success"}), 200
        else:
            return jsonify({"response": "未生成回复", "status": "error"}), 500
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"response": f"系统错误: {str(e)}", "status": "error"}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ready",
        "architecture": "LangGraph-Agent-INT4",
        "model": model_path,
        "quantization": "4-bit NF4"
    })


if __name__ == '__main__':
    # 只有当上面所有步骤顺利执行完，才会打印这里
    print("\n" + "=" * 60)
    print("🏥 医疗智能体服务启动成功!")
    print(f"📂 网页目录：./{static_folder_name}")
    print("🌐 访问地址: http://127.0.0.1:5000/")
    print("🔌 API 地址: http://127.0.0.1:5000/api/chat")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=6006, debug=False, threaded=True)
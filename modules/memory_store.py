# import os
# import uuid
# import traceback
# from dotenv import load_dotenv
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
#
# load_dotenv()
#
#
# class MemoryStore:
#     def __init__(self, persist_directory="./memory_store"):
#         self.persist_directory = persist_directory
#         self.model_path = "./models/all-MiniLM-L6-v2"  # 本地模型路径
#         self._embeddings = None
#         self._vectorstore = None
#         self._pending_persist = False
#
#     @property
#     def embeddings(self):
#         if self._embeddings is None:
#             print("   [MemoryStore] 首次使用，从本地加载Embedding模型...")
#             self._embeddings = HuggingFaceEmbeddings(
#                 model_name=self.model_path,
#                 model_kwargs={'device': 'cpu'},
#                 encode_kwargs={'normalize_embeddings': True}
#             )
#             print("   [MemoryStore] Embedding模型加载完成。")
#         return self._embeddings
#
#     @property
#     def vectorstore(self):
#         if self._vectorstore is None:
#             print("   [MemoryStore] 首次使用，连接向量数据库...")
#             try:
#                 self._vectorstore = Chroma(
#                     persist_directory="./memory_store",  # 注意目录
#                     embedding_function=self.embeddings,
#                     collection_name="chat_memories"
#                 )
#             except Exception as e:
#                 print("=" * 50)
#                 print("ChromaDB 初始化失败，详细错误：")
#                 traceback.print_exc()
#                 print("=" * 50)
#                 # 如果初始化失败，重新创建（通常是因为数据库损坏）
#                 print("尝试重新创建向量库...")
#                 self._vectorstore = Chroma.from_documents(
#                     [],  # 空文档集合
#                     self.embeddings,
#                     persist_directory="./memory_store",
#                     collection_name="chat_memories"
#                 )
#                 self._vectorstore.persist()
#         return self._vectorstore
#
#     def add_memory(self, user_input, assistant_response, metadata=None):
#         """添加记忆（不立即持久化）"""
#         text = f"用户：{user_input}\n助手：{assistant_response}"
#         doc_id = str(uuid.uuid4())
#         self.vectorstore.add_texts([text], metadatas=[metadata or {}], ids=[doc_id])
#         self._pending_persist = True
#
#     def persist_if_needed(self):
#         """如果存在未持久化的记忆，则写入磁盘"""
#         if self._pending_persist and self._vectorstore is not None:
#             try:
#                 self.vectorstore.persist()
#                 self._pending_persist = False
#                 print("   [MemoryStore] 记忆已持久化。")
#             except Exception as e:
#                 print(f"   [MemoryStore] 持久化失败: {e}")
#
#     def retrieve_memories(self, query, k=3):
#         """检索相似记忆"""
#         try:
#             docs = self.vectorstore.similarity_search(query, k=k)
#             return [doc.page_content for doc in docs]
#         except Exception as e:
#             print(f"   [MemoryStore] 检索异常: {e}")
#             return []

"""
记忆存储模块 - 管理会话历史和预约状态
"""


class MemoryStore:
    def __init__(self):
        self.conversations = {}
        self.appointment_states = {}
        self.max_history = 20

    def get_history(self, patient_id: str) -> list:
        """获取患者会话历史"""
        return self.conversations.get(patient_id, [])

    def add_message(self, patient_id: str, role: str, content: str):
        """添加消息到历史"""
        if patient_id not in self.conversations:
            self.conversations[patient_id] = []

        self.conversations[patient_id].append({
            "role": role,
            "content": content
        })

        # 清理过长历史
        if len(self.conversations[patient_id]) > self.max_history:
            self.conversations[patient_id] = self.conversations[patient_id][-self.max_history:]

    def clear_history(self, patient_id: str):
        """清除患者历史"""
        if patient_id in self.conversations:
            self.conversations[patient_id] = []
        if patient_id in self.appointment_states:
            del self.appointment_states[patient_id]

    def get_recent_history(self, patient_id: str, limit: int = 6) -> list:
        """获取最近N轮对话"""
        history = self.get_history(patient_id)
        return history[-limit:] if history else []

    # ============ 预约状态管理 ============
    def set_appointment_state(self, patient_id: str, state: dict):
        """设置预约状态"""
        self.appointment_states[patient_id] = state

    def get_appointment_state(self, patient_id: str) -> dict:
        """获取预约状态"""
        return self.appointment_states.get(patient_id, {})

    def clear_appointment_state(self, patient_id: str):
        """清除预约状态"""
        if patient_id in self.appointment_states:
            del self.appointment_states[patient_id]


# 单例模式
memory_store = MemoryStore()
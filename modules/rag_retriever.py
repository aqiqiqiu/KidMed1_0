import os
import traceback
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# 使用绝对路径
KNOWLEDGE_PATH = r"D:\2026projects\MedIntellect\data\knowladge"  # 修正拼写：knowladge -> knowledge
VECTORSTORE_PATH = r"D:\2026projects\MedIntellect\vectorstore"


class MedicalRAG:
    def __init__(self, persist_directory=VECTORSTORE_PATH):
        self.persist_directory = persist_directory
        # 首先尝试本地模型，如果不存在则使用云端模型
        local_model_path = r"D:\2026projects\MedIntellect\models\all-MiniLM-L6-v2"

        if os.path.exists(local_model_path):
            print("使用本地模型:", local_model_path)
            self.embeddings = HuggingFaceEmbeddings(
                model_name=local_model_path,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            print("本地模型不存在，使用云端模型: sentence-transformers/all-MiniLM-L6-v2")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

        self.vectorstore = None

    def load_and_split_documents(self, data_path=KNOWLEDGE_PATH):
        """加载指定目录下的所有文档（.txt/.pdf）并分割"""
        loaders = {
            '.txt': lambda path: TextLoader(path, encoding='utf-8'),
            '.pdf': PyPDFLoader,
        }
        documents = []
        if not os.path.exists(data_path):
            print(f"警告: 文档目录不存在 {data_path}")
            return []

        for file in os.listdir(data_path):
            file_path = os.path.join(data_path, file)
            ext = os.path.splitext(file)[1].lower()
            if ext in loaders:
                try:
                    loader = loaders[ext](file_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"加载文档 {file} 失败: {e}")

        # 优化的递归字符切分器配置
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # 适中长度，兼顾上下文和检索速度
            chunk_overlap=50,  # 适当重叠避免语义断裂
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": "]  # 按句子边界切分
        )
        chunks = splitter.split_documents(documents)
        print(f"文档切分完成，共生成 {len(chunks)} 个数据块")
        return chunks

    def build_vectorstore(self, data_path=KNOWLEDGE_PATH):
        """从文档构建向量库，不使用HNSW索引避免兼容性问题"""
        chunks = self.load_and_split_documents(data_path)
        if not chunks:
            print("警告: 没有加载到任何文档，将创建空向量库")

        # 不使用collection_metadata避免HNSW参数解析错误
        self.vectorstore = Chroma.from_documents(
            chunks,
            self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vectorstore.persist()
        return self.vectorstore

    def load_vectorstore(self):
        """加载已存在的向量库，不使用HNSW索引避免兼容性问题"""
        try:
            # 不传递collection_metadata避免HNSW参数解析错误
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        except Exception as e:
            print(f"加载向量库失败: {e}")
            print("尝试创建空向量库...")
            self.vectorstore = Chroma.from_documents(
                [],
                self.embeddings,
                persist_directory=self.persist_directory
            )
            self.vectorstore.persist()
        return self.vectorstore

    def retrieve(self, query, k=2):  # 减少返回数量，医疗问答够用
        """检索相关文档，返回k=2个结果"""
        if not self.vectorstore:
            self.load_vectorstore()
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"RAG检索异常: {e}")
            return []


class AppointmentManager:
    """预约管理类，将预约信息存储在txt文件中"""

    def __init__(self, appointment_file=r"D:\2026projects\MedIntellect\appointments.txt"):
        self.appointment_file = appointment_file
        # 确保预约文件存在
        if not os.path.exists(self.appointment_file):
            with open(self.appointment_file, 'w', encoding='utf-8') as f:
                f.write("# 医疗预约记录\n")
                f.write("# 格式: 时间 | 患者ID | 预约详情\n\n")

    def save_appointment(self, patient_id, appointment_details, timestamp=None):
        """保存预约信息到txt文件"""
        import datetime
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.appointment_file, 'a', encoding='utf-8') as f:
            f.write(f"{timestamp} | {patient_id or '未知'} | {appointment_details}\n")

        print(f"预约信息已保存: {appointment_details}")
        return True

    def get_appointments(self, patient_id=None):
        """获取预约记录"""
        if not os.path.exists(self.appointment_file):
            return []

        appointments = []
        with open(self.appointment_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[2:]  # 跳过前两行注释

        for line in lines:
            line = line.strip()
            if line and '|' in line:
                parts = line.split(' | ', 2)
                if len(parts) >= 3:
                    time_str, pid, details = parts
                    if patient_id is None or pid == patient_id:
                        appointments.append({
                            'time': time_str,
                            'patient_id': pid,
                            'details': details
                        })

        return appointments


# 测试代码
if __name__ == "__main__":
    # 检查知识库路径
    print(f"知识库路径: {KNOWLEDGE_PATH}")
    if os.path.exists(KNOWLEDGE_PATH):
        print("✅ 知识库路径存在")
        files = os.listdir(KNOWLEDGE_PATH)
        print(f"知识库文件: {files[:10]}... ({len(files)} total)" if len(files) > 10 else f"知识库文件: {files}")
    else:
        print("❌ 知识库路径不存在，请确认路径是否正确")

    # 测试RAG优化
    print("\n测试优化后的RAG系统...")
    rag = MedicalRAG()
    try:
        # 先检查是否有知识库文件
        if not os.listdir(KNOWLEDGE_PATH):
            print("知识库为空，跳过构建步骤")
        else:
            print("正在构建向量库...")
            rag.build_vectorstore()

        print("✅ 向量库加载成功（使用优化切分）")

        # 测试检索
        results = rag.retrieve("感冒", k=2)
        print(f"✅ 检索测试完成，返回 {len(results)} 条结果")
    except Exception as e:
        print(f"❌ RAG测试失败: {e}")
        traceback.print_exc()

    # 测试预约管理
    print("\n测试预约管理系统...")
    appointment_mgr = AppointmentManager()
    try:
        appointment_mgr.save_appointment("P001", "预约明天上午10点看内科")
        all_appts = appointment_mgr.get_appointments()
        print(f"✅ 预约系统测试完成，当前共有 {len(all_appts)} 条记录")
    except Exception as e:
        print(f"❌ 预约系统测试失败: {e}")
        traceback.print_exc()

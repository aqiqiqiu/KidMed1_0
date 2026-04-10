# D:\2026projects\MedIntellect\db_manager.py
import pymysql
import os
import datetime
import logging
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager




# --- 配置部分 ---
# 优先从环境变量读取，如果没有则使用默认值 (方便部署到服务器)
DB_CONFIG = {
    'host': '127.0.0.1',          # 改这里
    'port': 3307,                 # 加这一行
    'user': os.getenv('DB_USER', '2656211880'),  # 改默认用户
    'password': os.getenv('DB_PASSWORD', '20040915'),
    'database': os.getenv('DB_NAME', 'med_intellect'),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
    'autocommit': False
}

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [DB] %(message)s')
logger = logging.getLogger(__name__)


# --- 连接管理 ---

@contextmanager
def get_db_connection():
    """
    上下文管理器：自动处理连接的获取、提交/回滚和关闭。
    用法: with get_db_connection() as conn: ...
    """
    conn = None
    try:
        conn = pymysql.connect(**DB_CONFIG)
        logger.debug("数据库连接成功")
        yield conn
        # 如果没有异常，自动提交事务
        conn.commit()
        logger.debug("事务已提交")
    except Exception as e:
        if conn:
            conn.rollback()
            logger.error(f"发生错误，事务已回滚：{e}")
        raise e
    finally:
        if conn:
            conn.close()
            logger.debug("数据库连接已关闭")


# --- 初始化与种子数据 ---

def init_db_tables():
    """
    初始化数据库表结构并插入种子数据。
    如果表已存在则跳过，如果数据已存在则不重复插入。
    """
    logger.info("开始初始化数据库...")

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # 1. 创建科室表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS departments (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(50) NOT NULL UNIQUE,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # 2. 创建医生表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS doctors (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(50) NOT NULL,
                    department_id INT NOT NULL,
                    title VARCHAR(50),
                    specialty VARCHAR(100),
                    FOREIGN KEY(department_id) REFERENCES departments(id) ON DELETE CASCADE,
                    INDEX idx_dept (department_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # 3. 创建排班表 (核心业务表)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schedules (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    doctor_id INT NOT NULL,
                    date DATE NOT NULL,
                    time_slot VARCHAR(50) NOT NULL,
                    is_available TINYINT DEFAULT 1, -- 1:空闲, 0:已约
                    FOREIGN KEY(doctor_id) REFERENCES doctors(id) ON DELETE CASCADE,
                    UNIQUE KEY unique_slot (doctor_id, date, time_slot),
                    INDEX idx_avail (date, is_available)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # 4. 创建预约记录表 (历史流水)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS appointments (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    patient_name VARCHAR(50) NOT NULL,
                    patient_phone VARCHAR(20),
                    doctor_id INT NOT NULL,
                    schedule_id INT NOT NULL,
                    appointment_date DATE NOT NULL,
                    time_slot VARCHAR(50) NOT NULL,
                    status ENUM('confirmed', 'cancelled', 'completed') DEFAULT 'confirmed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(doctor_id) REFERENCES doctors(id),
                    FOREIGN KEY(schedule_id) REFERENCES schedules(id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            logger.info("表结构检查/创建完成。")

            # --- 插入种子数据 (如果为空) ---
            cursor.execute("SELECT COUNT(*) as cnt FROM departments")
            if cursor.fetchone()['cnt'] == 0:
                logger.info("检测到空数据库，正在插入种子数据...")

                # A. 插入科室
                depts = [
                    ('内科', '处理感冒、发烧、肠胃炎等常见内科疾病'),
                    ('外科', '处理外伤、骨折、阑尾炎等外科手术需求'),
                    ('儿科', '0-14岁儿童常见疾病诊疗'),
                    ('皮肤科', '湿疹、痤疮、过敏等皮肤问题'),
                    ('眼科', '视力检查、近视手术、眼部疾病'),
                    ('口腔科', '拔牙、补牙、牙齿矫正')
                ]
                cursor.executemany(
                    "INSERT INTO departments (name, description) VALUES (%s, %s)",
                    depts
                )
                logger.info(f"已插入 {len(depts)} 个科室。")

                # B. 插入医生
                doctors_data = [
                    ('张伟', 1, '主任医师', '心血管专家'),
                    ('李娜', 1, '主治医师', '呼吸科专家'),
                    ('王强', 2, '副主任医师', '骨科专家'),
                    ('赵敏', 2, '医师', '普外专家'),
                    ('孙杰', 3, '主任医师', '儿科专家'),
                    ('周婷', 4, '主治医师', '美容皮肤专家'),
                    ('吴刚', 5, '主任医师', '眼底病专家'),
                    ('郑丽', 6, '医师', '正畸专家')
                ]
                cursor.executemany(
                    "INSERT INTO doctors (name, department_id, title, specialty) VALUES (%s, %s, %s, %s)",
                    doctors_data
                )
                logger.info(f"已插入 {len(doctors_data)} 位医生。")

                # C. 生成未来 7 天的排班
                base_date = datetime.date.today()
                time_slots = ["08:30-09:30", "09:30-10:30", "10:30-11:30", "14:00-15:00", "15:00-16:00", "16:00-17:00"]
                schedule_batch = []

                for doctor_id in range(1, len(doctors_data) + 1):
                    for day_offset in range(0, 7):  # 今天起未来 7 天
                        current_date = base_date + datetime.timedelta(days=day_offset)

                        # 模拟周末部分医生休息 (简单逻辑：周六日医生ID为偶数的休息)
                        is_weekend = current_date.weekday() >= 5
                        if is_weekend and doctor_id % 2 == 0:
                            continue

                        for slot in time_slots:
                            # 随机让 20% 的号源不可用 (模拟已被占或休息)
                            is_avail = 1 if (doctor_id + day_offset) % 5 != 0 else 0
                            schedule_batch.append((doctor_id, current_date, slot, is_avail))

                cursor.executemany(
                    "INSERT INTO schedules (doctor_id, date, time_slot, is_available) VALUES (%s, %s, %s, %s)",
                    schedule_batch
                )
                logger.info(f"已生成 {len(schedule_batch)} 条排班记录。")

            else:
                logger.info("数据库已有数据，跳过种子数据插入。")

        logger.info("✅ 数据库初始化全部完成！")

    except Exception as e:
        logger.error(f"❌ 数据库初始化失败：{e}")
        raise e


# --- 业务逻辑函数 ---

def query_available_slots(dept_name: str, limit: int = 5) -> List[Dict]:
    """
    查询指定科室的空闲号源。
    返回：列表，包含医生姓名、职称、日期、时间段、排班ID。
    """
    sql = """
        SELECT 
            d.id as doctor_id, d.name as doctor_name, d.title, d.specialty,
            s.date, s.time_slot, s.id as schedule_id
        FROM doctors d
        JOIN schedules s ON d.id = s.doctor_id
        JOIN departments dep ON d.department_id = dep.id
        WHERE dep.name = %s AND s.is_available = 1 AND s.date >= CURDATE()
        ORDER BY s.date ASC, s.time_slot ASC
        LIMIT %s
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (dept_name, limit))
            results = cursor.fetchall()
            logger.info(f"查询科室 [{dept_name}] 获得 {len(results)} 个空闲号源。")
            return results
    except Exception as e:
        logger.error(f"查询失败：{e}")
        return []


def confirm_appointment(schedule_id: int, patient_name: str, patient_phone: str = "未提供") -> Tuple[bool, str]:
    """
    确认预约（原子操作，防止超卖）。
    流程：
    1. 锁定并检查该排班是否可用。
    2. 如果可用，标记为不可用。
    3. 写入预约记录表。
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # 1. 查询并锁定行 (FOR UPDATE 防止并发冲突)
            # 注意：MySQL InnoDB 需要在事务中使用 FOR UPDATE
            cursor.execute("""
                SELECT doctor_id, date, time_slot, is_available 
                FROM schedules 
                WHERE id = %s 
                FOR UPDATE
            """, (schedule_id,))

            slot_info = cursor.fetchone()

            if not slot_info:
                return False, "号源不存在或已被删除。"

            if slot_info['is_available'] == 0:
                return False, "抱歉，该号源刚刚被其他人抢走了，请刷新重试。"

            doctor_id = slot_info['doctor_id']
            appt_date = slot_info['date']
            time_slot = slot_info['time_slot']

            # 2. 更新状态为不可用
            update_sql = "UPDATE schedules SET is_available = 0 WHERE id = %s AND is_available = 1"
            cursor.execute(update_sql, (schedule_id,))

            if cursor.rowcount == 0:
                # 理论上上面查到了，这里应该能更新，除非极端并发
                return False, "系统繁忙，号源状态变更失败。"

            # 3. 插入预约记录
            insert_sql = """
                INSERT INTO appointments (patient_name, patient_phone, doctor_id, schedule_id, appointment_date, time_slot, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'confirmed')
            """
            cursor.execute(insert_sql, (patient_name, patient_phone, doctor_id, schedule_id, appt_date, time_slot))

            logger.info(f"✅ 预约成功：患者 {patient_name} 预约了 {appt_date} {time_slot} 的号。")
            return True, f"预约成功！\n医生：{slot_info.get('doctor_id')} (ID)\n时间：{appt_date} {time_slot}\n请准时就诊。"

    except Exception as e:
        logger.error(f"预约过程出错：{e}")
        # 上下文管理器会自动 rollback
        return False, f"系统错误：{str(e)}"


def cancel_appointment(appointment_id: int) -> Tuple[bool, str]:
    """
    取消预约：
    1. 修改预约记录状态为 cancelled。
    2. 恢复排班表的 is_available 为 1。
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # 查找预约
            cursor.execute("SELECT schedule_id, status FROM appointments WHERE id = %s", (appointment_id,))
            appt = cursor.fetchone()

            if not appt:
                return False, "未找到该预约记录。"
            if appt['status'] != 'confirmed':
                return False, f"该预约状态为 [{appt['status']}]，无法取消。"

            schedule_id = appt['schedule_id']

            # 1. 更新预约状态
            cursor.execute("UPDATE appointments SET status = 'cancelled' WHERE id = %s", (appointment_id,))

            # 2. 恢复号源
            cursor.execute("UPDATE schedules SET is_available = 1 WHERE id = %s", (schedule_id,))

            logger.info(f"🚫 预约已取消：ID {appointment_id}")
            return True, "取消成功。"

    except Exception as e:
        logger.error(f"取消失败：{e}")
        return False, str(e)


def get_patient_history(patient_name: str, limit: int = 10) -> List[Dict]:
    """查询患者的预约历史"""
    sql = """
        SELECT a.id, a.appointment_date, a.time_slot, a.status, a.created_at,
               d.name as doctor_name, dep.name as dept_name
        FROM appointments a
        JOIN doctors d ON a.doctor_id = d.id
        JOIN departments dep ON d.department_id = dep.id
        WHERE a.patient_name = %s
        ORDER BY a.created_at DESC
        LIMIT %s
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (patient_name, limit))
            return cursor.fetchall()
    except Exception as e:
        logger.error(f"查询历史失败：{e}")
        return []


# --- 入口测试 ---

if __name__ == "__main__":
    print("🚀 正在启动数据库管理模块测试...")

    # 1. 初始化
    init_db_tables()

    # 2. 测试查询
    print("\n🔍 测试：查询【内科】空闲号源")
    slots = query_available_slots("内科", limit=3)
    for s in slots:
        print(f"   - {s['date']} {s['time_slot']} | {s['doctor_name']} ({s['title']}) [ID:{s['schedule_id']}]")

    if slots:
        # 3. 测试预约
        test_slot_id = slots[0]['schedule_id']
        print(f"\n📝 测试：预约 ID 为 {test_slot_id} 的号源...")
        success, msg = confirm_appointment(test_slot_id, "测试用户_张三", "13800138000")
        print(f"   结果：{'✅ ' if success else '❌ '}{msg}")

        # 4. 测试重复预约 (应该失败)
        print(f"\n🔄 测试：再次预约同一个 ID (应失败)...")
        success2, msg2 = confirm_appointment(test_slot_id, "测试用户_李四", "13900139000")
        print(f"   结果：{'✅ ' if success2 else '❌ '}{msg2}")

    print("\n✅ 所有测试完成。")
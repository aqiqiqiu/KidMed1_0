"""
数据库模块 - MySQL 连接和预约操作
"""

import pymysql
from pymysql.cursors import DictCursor
from dbutils.pooled_db import PooledDB
import os
from dotenv import load_dotenv

load_dotenv()

class Database:
    def __init__(self):
        try:
            self.pool = PooledDB(
                creator=pymysql,
                maxconnections=10,
                mincached=2,
                host=os.getenv('MYSQL_HOST', '127.0.0.1'),
                port=int(os.getenv('MYSQL_PORT', 3307)),
                user=os.getenv('MYSQL_USER', '2656211880'),
                password=os.getenv('MYSQL_PASSWORD', '20040915'),
                database=os.getenv('MYSQL_DATABASE', 'med_intellect'),
                charset='utf8mb4',
                cursorclass=DictCursor
            )
            self.connected = True
            print("✅ 数据库连接池初始化完成")
        except Exception as e:
            print(f"⚠️ 数据库连接失败：{e}")
            self.connected = False
            self.pool = None

    def get_connection(self):
        if not self.connected or not self.pool:
            raise Exception("数据库未连接")
        return self.pool.connection()

    def get_departments(self):
        if not self.connected:
            return [
                {"id": 1, "name": "内科", "description": "诊治常见内科疾病"},
                {"id": 2, "name": "外科", "description": "诊治外伤及外科疾病"},
                {"id": 3, "name": "儿科", "description": "诊治儿童疾病"},
                {"id": 4, "name": "皮肤科", "description": "诊治皮肤相关疾病"},
                {"id": 5, "name": "眼科", "description": "诊治眼部疾病"},
                {"id": 6, "name": "口腔科", "description": "诊治口腔疾病"},
            ]
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, name, description FROM departments WHERE status=1")
                return cursor.fetchall()
        except Exception as e:
            print(f"查询科室失败：{e}")
            return []
        finally:
            conn.close()

    def get_doctors_by_dept(self, dept_id):
        if not self.connected:
            return [{"id": 1, "name": "张医生", "title": "主治医师"}]
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, name, title FROM doctors WHERE dept_id=%s AND status=1",
                    (dept_id,)
                )
                return cursor.fetchall()
        except Exception as e:
            print(f"查询医生失败：{e}")
            return []
        finally:
            conn.close()

    def get_available_slots(self, dept_name=None, date=None):
        if not self.connected:
            return [
                {"id": 1, "work_date": "2026-03-16", "time_slot": "09:00-10:00",
                 "total_slots": 30, "booked_slots": 5, "doctor_name": "张医生", "dept_name": "内科"},
                {"id": 2, "work_date": "2026-03-16", "time_slot": "10:00-11:00",
                 "total_slots": 30, "booked_slots": 10, "doctor_name": "李医生", "dept_name": "内科"},
                {"id": 3, "work_date": "2026-03-16", "time_slot": "14:00-15:00",
                 "total_slots": 30, "booked_slots": 8, "doctor_name": "张医生", "dept_name": "内科"},
            ]
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                if dept_name:
                    cursor.execute("""
                        SELECT s.id, s.work_date, s.time_slot, s.total_slots, s.booked_slots,
                               d.name as doctor_name, dp.name as dept_name
                        FROM schedules s
                        JOIN doctors d ON s.doctor_id = d.id
                        JOIN departments dp ON d.dept_id = dp.id
                        WHERE s.status=1 AND s.booked_slots < s.total_slots
                        AND dp.name = %s
                        ORDER BY s.work_date, s.time_slot
                        LIMIT 5
                    """, (dept_name,))
                else:
                    cursor.execute("""
                        SELECT s.id, s.work_date, s.time_slot, s.total_slots, s.booked_slots,
                               d.name as doctor_name, dp.name as dept_name
                        FROM schedules s
                        JOIN doctors d ON s.doctor_id = d.id
                        JOIN departments dp ON d.dept_id = dp.id
                        WHERE s.status=1 AND s.booked_slots < s.total_slots
                        ORDER BY s.work_date, s.time_slot
                        LIMIT 5
                    """)
                return cursor.fetchall()
        except Exception as e:
            print(f"查询号源失败：{e}")
            return []
        finally:
            conn.close()

    def create_appointment(self, patient_id, patient_name, patient_phone, schedule_id,
                           dept_id, doctor_id, appointment_date, time_slot):
        if not self.connected:
            print(f"✅ 模拟创建预约：patient_id={patient_id}, schedule_id={schedule_id}")
            return 1001
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO appointments 
                    (patient_id, patient_name, patient_phone, schedule_id, dept_id, doctor_id, appointment_date, time_slot, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 1)
                """, (patient_id, patient_name, patient_phone, schedule_id, dept_id, doctor_id, appointment_date, time_slot))
                cursor.execute("UPDATE schedules SET booked_slots = booked_slots + 1 WHERE id = %s", (schedule_id,))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            conn.rollback()
            print(f"创建预约失败：{e}")
            raise e
        finally:
            conn.close()

    def get_appointment_by_id(self, appointment_id):
        if not self.connected:
            return {"id": appointment_id, "dept_name": "内科", "doctor_name": "张医生",
                    "appointment_date": "2026-03-16", "time_slot": "09:00-10:00", "status": 1}
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT a.*, d.name as doctor_name, dp.name as dept_name
                    FROM appointments a
                    JOIN doctors d ON a.doctor_id = d.id
                    JOIN departments dp ON a.dept_id = dp.id
                    WHERE a.id = %s
                """, (appointment_id,))
                return cursor.fetchone()
        except Exception as e:
            print(f"查询预约失败：{e}")
            return None
        finally:
            conn.close()

    def get_patient_appointments(self, patient_id):
        if not self.connected:
            return []
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT a.*, d.name as doctor_name, dp.name as dept_name
                    FROM appointments a
                    JOIN doctors d ON a.doctor_id = d.id
                    JOIN departments dp ON a.dept_id = dp.id
                    WHERE a.patient_id = %s
                    ORDER BY a.created_at DESC
                """, (patient_id,))
                return cursor.fetchall()
        except Exception as e:
            print(f"查询患者预约失败：{e}")
            return []
        finally:
            conn.close()

    def cancel_appointment(self, appointment_id):
        if not self.connected:
            print(f"✅ 模拟取消预约：id={appointment_id}")
            return True
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT schedule_id FROM appointments WHERE id = %s", (appointment_id,))
                appointment = cursor.fetchone()
                if appointment:
                    cursor.execute("UPDATE appointments SET status = 3 WHERE id = %s", (appointment_id,))
                    cursor.execute("UPDATE schedules SET booked_slots = booked_slots - 1 WHERE id = %s",
                                   (appointment['schedule_id'],))
                    conn.commit()
                    return True
                return False
        except Exception as e:
            conn.rollback()
            print(f"取消预约失败：{e}")
            raise e
        finally:
            conn.close()

db = Database()
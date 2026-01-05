# app.py
# GUI CustomTkinter + YOLOv8 + FaceNet
# - Live camera trong GUI
# - Tự động chụp 1 ảnh mỗi N giây
# - Auto-train sau khi chụp xong
# - ⭐ NÂNG CẤP: Dùng Cosine Distance để so sánh embeddings ⭐
# - ⭐ NÂNG CẤP VIP: Chế độ Tối (Dark Mode) tự động khi không có người ⭐

import os
import time
import threading
import pickle
import shutil
from datetime import datetime

import customtkinter as ctk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cosine # ⭐ THÊM SCIPY.COSINE ⭐

from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
# MediaPipe aligner (local)
from aligner import align_face_mediapipe, center_crop_resize

# ---------------- CONFIG ----------------
YOLO_WEIGHTS = "C:/Users/admin/Downloads/Face-recognition-using-YoloV8-and-Facenet-main/Face-recognition-using-YoloV8-and-Facenet-main/detection/weights/best02m.pt"
KNOWN_FACES_DIR = "known_faces"         # mỗi người một thư mục
EMBEDDINGS_FILE = "known_embeddings.pkl"
ATTENDANCE_FILE = "attendance.csv"
CAPTURE_LOG_DIR = "captured_logs"

LAST_LOG_TIME = {} # Thời gian lần cuối chấm công của mỗi người
COOLDOWN_SECONDS = 10 # Thời gian chờ giữa 2 lần chấm công liên tiếp (có thể thay đổi)

# Default auto-capture settings (có thể thay)
AUTO_CAPTURE_INTERVAL = 3.0    # giây giữa 2 lần chụp
AUTO_CAPTURE_COUNT = 10        # số ảnh cần chụp cho 1 người

# ⭐ CẤU HÌNH VIP: DYNAMIC ACTIVATION ⭐
DARK_MODE_TIMEOUT = 8 # Số giây không có người sẽ chuyển sang nền đen/tối
last_detection_time = time.time() # Thời điểm phát hiện khuôn mặt gần nhất

# Ngưỡng Cosine Distance cho độ chính xác cao
EMBEDDING_THRESHOLD = 0.20

# ---------------- Make dirs ----------------
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(CAPTURE_LOG_DIR, exist_ok=True)

# ---------------- Load models (1 lần) ----------------
print("[INFO] Loading models (YOLOv8 + MTCNN + FaceNet). Please wait...")
yolo_model = YOLO(YOLO_WEIGHTS) 
#mtcnn = MTCNN(image_size=160, margin=0, keep_all=False)
resnet = InceptionResnetV1(pretrained="vggface2").eval()
print("[INFO] Models loaded.")

# Assuming this setup is done elsewhere:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

resnet = resnet.to(device)

# ---------------- Known embeddings persistence ----------------
def save_known_embeddings(d):
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(d, f)

def load_known_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}

known_embeddings = load_known_embeddings()

# ---------------- Notification and Helper Functions ----------------

# Global notification label (defined below in GUI Init)
notification_label = None
lbl_train_status = None

def create_dark_placeholder(width, height):
    """Tạo ảnh nền đen với thông báo placeholder."""
    img = Image.new('RGB', (width, height), color = 'black')
    d = ImageDraw.Draw(img)
    # Thêm text "HỆ THỐNG ĐANG NGHỈ"
    try:
        # Tên font tùy thuộc vào hệ thống, dùng font mặc định nếu Segoe UI không có
        font_size = 30
        try:
            from customtkinter.windows.widgets.core_widget_classes import CTkFont
            font = CTkFont(family="Segoe UI", size=font_size, weight="bold")
        except ImportError:
            font = None
            
        d.text((width/2, height/2), "HỆ THỐNG ĐANG NGHỈ", 
               fill=(255, 255, 255), anchor="mm", font=font)
    except Exception:
        d.text((width/2, height/2), "HỆ THỐNG ĐANG NGHỈ", 
               fill=(255, 255, 255), anchor="mm")
        
    return img


def show_notification(message, color="green", duration=2000):
    """Hiển thị overlay thông báo lớn trên GUI."""
    if notification_label:
        notification_label.configure(
            text=message, 
            fg_color=color
        )
        # Sử dụng place để đặt label chồng lên các phần tử khác và căn giữa
        notification_label.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.6, relheight=0.15)
        
        # Lên lịch ẩn thông báo sau duration milliseconds
        app.after(duration, hide_notification)

def hide_notification():
    """Ẩn overlay thông báo."""
    if notification_label:
        notification_label.place_forget()

# ---------------- Attendance logging ----------------
def load_logged_today():
    today = datetime.now().strftime("%Y-%m-%d")
    s = set()
    if os.path.exists(ATTENDANCE_FILE):
        try:
            df = pd.read_csv(ATTENDANCE_FILE)
            if "Time" in df.columns and "Name" in df.columns:
                df_today = df[df["Time"].astype(str).str.startswith(today)]
                s = set(df_today["Name"].tolist())
        except Exception as e:
            print("[WARN] Could not read attendance.csv:", e)
    return s

attendance_today = load_logged_today()

def log_attendance(name, frame):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    today_str = now.strftime("%Y-%m-%d")

    current_logs = []
    if os.path.exists(ATTENDANCE_FILE):
        try:
            df = pd.read_csv(ATTENDANCE_FILE)
            # Lọc ra tất cả các lần chấm công của người này trong ngày hôm nay
            df_today_person = df[(df["Name"] == name) & (df["Time"].astype(str).str.startswith(today_str))]
            current_logs = df_today_person["Type"].tolist()
        except Exception as e:
            print("[WARN] Could not inspect attendance.csv:", e)

    # Logic mới: Chỉ cho phép tối đa 1 lần 'Vào' và 1 lần 'Ra'
    log_type = None
    if "Vào" not in current_logs:
        log_type = "Vào"
    elif "Vào" in current_logs and "Ra" not in current_logs:
        log_type = "Ra"
    else:
        # Đã có cả 'Vào' và 'Ra'
        print(f"[INFO] {name} đã điểm danh cả Vào và Ra hôm nay. Bỏ qua.")
        return # Thoát, không chấm công nữa

    # Nếu xác định được loại chấm công (Vào hoặc Ra)
    if log_type:
        df_new = pd.DataFrame([[name, timestamp, log_type]], columns=["Name", "Time", "Type"])
        df_new.to_csv(ATTENDANCE_FILE, mode='a', index=False, header=not os.path.exists(ATTENDANCE_FILE))

        # Ghi log ảnh (không thay đổi)
        fname = f"{CAPTURE_LOG_DIR}/{name}_{log_type}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.jpg" # Thêm loại chấm công vào tên file
        os.makedirs(CAPTURE_LOG_DIR, exist_ok=True)
        cv2.imwrite(fname, frame)
        print(f"[INFO] Logged {name} ({log_type}) at {timestamp}, saved image {fname}")
        
        #  HIỂN THỊ THÔNG BÁO TRỰC QUAN 
        log_message = f"{log_type.upper()} THÀNH CÔNG: {name}"
        # Sử dụng màu xanh lá cho Vào, màu xanh dương cho Ra
        color = "green" if log_type == "Vào" else "blue" # Đã sửa lỗi chính tả string
        # Gọi show_notification trong luồng chính của GUI
        app.after(0, lambda: show_notification(log_message, color))


# ---------------- Embedding compare ----------------
def compare_embedding(emb):
    """
    So sánh embedding sử dụng Cosine Distance.
    """
    if not known_embeddings:
        return "Unknown", float("inf")
    min_dist = float("inf")
    best = "Unknown"
    
    for name, known_emb in known_embeddings.items():
        #  THAY THẾ np.linalg.norm bằng scipy.spatial.distance.cosine 
        dist = cosine(emb, known_emb) 
        
        if dist < min_dist:
            min_dist = dist
            best = name if dist < EMBEDDING_THRESHOLD else "Unknown"
            
    return best, min_dist

# ---------------- GUI init ----------------
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("green")

app = ctk.CTk()
app.title("Điểm danh khuôn mặt") # Đổi tên cho phù hợp
app.geometry("1100x700")

# Frames
top_frame = ctk.CTkFrame(master=app)
top_frame.pack(padx=10, pady=8, fill="both", expand=False)

left_frame = ctk.CTkFrame(master=app)
left_frame.pack(side="left", padx=10, pady=10, fill="both", expand=True)

right_frame = ctk.CTkFrame(master=app, width=300)
right_frame.pack(side="right", padx=10, pady=10, fill="y")

# Title
title = ctk.CTkLabel(master=top_frame, text="📷 Hệ thống điểm danh bằng khuôn mặt ", font=("Segoe UI", 20, "bold"))
title.pack(pady=6)

# Video label
video_label = ctk.CTkLabel(master=left_frame, text="")
video_label.pack(padx=10, pady=6)

# Log textbox
log_box = ctk.CTkTextbox(master=left_frame, width=760, height=140)
log_box.pack(padx=10, pady=6)

# ⭐ TRAINING STATUS LABEL (NEW) ⭐
lbl_train_status = ctk.CTkLabel(master=left_frame, text="", font=("Segoe UI", 14, "italic"), anchor="w")
lbl_train_status.pack(padx=10, pady=(2, 0), fill="x")

# Progress bar
progress = ctk.CTkProgressBar(master=left_frame, width=600)
progress.set(0)
progress.pack(padx=10, pady=4)

# ⭐ NOTIFICATION OVERLAY LABEL (NEW) ⭐
notification_label = ctk.CTkLabel(
    master=app,
    text="",
    fg_color="green", # Default color
    text_color="white",
    font=("Segoe UI", 36, "bold"),
    corner_radius=10
)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    log_box.insert("end", f"{ts} - {msg}\n")
    log_box.see("end")
    print(ts, "-", msg)

# Right side controls
lbl_detect = ctk.CTkLabel(master=right_frame, text="👤 Nhận diện: None", font=("Segoe UI", 14, "bold"))
lbl_detect.pack(pady=6)


# MSSV entry
ctk.CTkLabel(master=right_frame, text="MSSV:").pack(pady=(6,2))
entry_mssv = ctk.CTkEntry(master=right_frame, width=220)
entry_mssv.pack(pady=4)

# name entry for register
ctk.CTkLabel(master=right_frame, text="Tên (đăng ký):").pack(pady=(6,2))
entry_name = ctk.CTkEntry(master=right_frame, width=220)
entry_name.pack(pady=4)

# auto capture settings
ctk.CTkLabel(master=right_frame, text="Cài đặt chụp tự động:").pack(pady=(8,2))
interval_var = ctk.DoubleVar(value=AUTO_CAPTURE_INTERVAL)
count_var = ctk.IntVar(value=AUTO_CAPTURE_COUNT)
ctk.CTkLabel(master=right_frame, text="Khoảng (giây):").pack(pady=(4,0))
interval_entry = ctk.CTkEntry(master=right_frame, textvariable=interval_var, width=120)
interval_entry.pack(pady=2)
ctk.CTkLabel(master=right_frame, text="Số ảnh:").pack(pady=(4,0))
count_entry = ctk.CTkEntry(master=right_frame, textvariable=count_var, width=120)
count_entry.pack(pady=2)

# Buttons
btn_start = ctk.CTkButton(master=right_frame, text="▶️ Bắt đầu", width=240) # Đổi tên nút
btn_stop = ctk.CTkButton(master=right_frame, text="⏹️ Dừng", width=240)
btn_auto_capture = ctk.CTkButton(master=right_frame, text="📸 Chụp tự động", width=240)
btn_reload = ctk.CTkButton(master=right_frame, text="🔁 Tải lại embeddings", width=240)
btn_stats = ctk.CTkButton(master=right_frame, text="📊 Thống kê hôm nay", width=240)
btn_export = ctk.CTkButton(master=right_frame, text="📥 Xuất Excel", width=240)
btn_declec = ctk.CTkButton(master=right_frame, text="🗑️ Xóa khuôn mặt",width=240)
btn_exit = ctk.CTkButton(master=right_frame, text="❌ Thoát", width=240)

btn_start.pack(pady=6)
btn_stop.pack(pady=6)
btn_auto_capture.pack(pady=6)
btn_reload.pack(pady=6)
btn_stats.pack(pady=6)
btn_export.pack(pady=6)
btn_declec.pack(pady=6)
btn_exit.pack(pady=6)

# ---------------- Camera & state ----------------
video_capture = None
running = False
stop_auto_capture_flag = False
attendance_today = load_logged_today()

# ---------------- Video processing (recognize) ----------------
def process_video(label_widget):
    global video_capture, running, attendance_today, last_detection_time
    import threading, time
    from queue import Queue

    FRAME_SKIP = 0
    FRAME_SIZE = (640, 480)
    running = True
    log("▶️ Bắt đầu chấm công")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo_model.to(device)
    resnet.to(device)
    resnet.eval()
    
    # Kích thước cố định của video label để tạo placeholder
    display_w = 640
    display_h = 640

    frame_queue = Queue(maxsize=2)
    result_queue = Queue(maxsize=1)

    # 🎥 Thread đọc camera (Luôn chạy để lấy khung hình)
    def capture_thread():
        global video_capture
        video_capture = cv2.VideoCapture(0, cv2.CAP_MSMF)
        video_capture.set(cv2.CAP_PROP_FPS, 30)
        if not video_capture.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở camera.")
            return

        while running:
            ret, frame = video_capture.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1) # Lật ngang
            if not frame_queue.full():
                frame_queue.put(frame)

        video_capture.release()
        log("🎥 Camera stopped")

    # 🧠 Thread xử lý YOLO + FaceNet (Luôn chạy để kiểm tra sự hiện diện)
    def inference_thread():
        frame_count = 0
        prev_time = time.time()

        while running:
            if frame_queue.empty():
                time.sleep(0.01)
                continue

            frame = frame_queue.get()
            frame_count += 1
            if FRAME_SKIP > 0 and frame_count % FRAME_SKIP != 0:
                continue
            
            # --- Bắt đầu đo thời gian YOLO ---
            t0 = time.time()
            try:
                small = cv2.resize(frame, FRAME_SIZE)
                results = yolo_model.predict(small, verbose=False)
            except Exception as e:
                print("[WARN] YOLO error:", e)
                continue
            infer_time = (time.time() - t0) * 1000  # ms

            detections = []
            is_face_detected = False
            
            if len(results) and len(results[0].boxes) > 0:
                is_face_detected = True
                
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    scale_x = frame.shape[1] / FRAME_SIZE[0]
                    scale_y = frame.shape[0] / FRAME_SIZE[1]
                    x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
                    x2, y2 = int(x2 * scale_x), int(y2 * scale_y)

                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    # Use MediaPipe aligner first (align on the original frame + bbox)
                    try:
                        aligned = align_face_mediapipe(frame, (x1, y1, x2, y2), output_size=160)
                    except Exception as e:
                        aligned = None

                    if aligned is None:
                        # fallback to center-crop + resize (returns RGB)
                        aligned = center_crop_resize(frame, (x1, y1, x2, y2), output_size=160)

                    if aligned is None:
                        continue

                    face_tensor = torch.tensor(aligned.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
                    face_tensor = face_tensor.to(device)
                    with torch.no_grad():
                        emb = resnet(face_tensor).detach().cpu().numpy().flatten()

                    detections.append((x1, y1, x2, y2, emb))
                    
            # --- Tính FPS ---
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time

            # --- Cập nhật thời gian phát hiện cuối cùng nếu có khuôn mặt ---
            if is_face_detected:
                global last_detection_time
                last_detection_time = current_time
            
            # --- Gửi frame + detect + FPS sang GUI ---
            result_queue.put((frame, detections, fps))

    # 🖼️ Thread cập nhật GUI (Chỉ cập nhật khi đang Active)
    def gui_thread():
        frame_count = 0
        while running:
            if result_queue.empty():
                time.sleep(0.01)
                continue

            frame, detections, fps = result_queue.get()
            
            #  LOGIC CHUYỂN CHẾ ĐỘ 
            time_since_detection = time.time() - last_detection_time
            is_active_mode = time_since_detection < DARK_MODE_TIMEOUT
            
            # Xử lý chấm công và vẽ box (chỉ khi có detections)
            for (x1, y1, x2, y2, emb) in detections:
                if is_active_mode:
                    name, dist = compare_embedding(emb) # Dùng hàm compare_embedding mới (Cosine)
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{name}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frame, f"{dist:.2f}", (x1, y2 + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

                    if name != "Unknown":
                        # Kiểm tra cooldown trước khi chấm công
                        if name not in LAST_LOG_TIME or (time.time() - LAST_LOG_TIME[name]) > COOLDOWN_SECONDS:
                            log_attendance(name, frame)
                            LAST_LOG_TIME[name] = time.time() # Cập nhật thời gian chấm công
                            log(f"✅ Đã điểm danh: {name}")

                    lbl_detect.configure(text=f"👤 Nhận diện: {name}")
                else:
                    # Nếu đang trong chế độ tối mà vẫn detect được, không cần vẽ box 
                    # vì nó sẽ chuyển sang Active ở frame tiếp theo
                    pass 


            # --- HIỂN THỊ TRONG ACTIVE MODE ---
            if is_active_mode:
                # Hiển thị FPS
                color_fps = (0, 255, 0) if fps > 25 else ((0, 255, 255) if fps > 15 else (0, 0, 255))
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_fps, 2)
                
                # Cập nhật video label
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                pil_img = pil_img.resize((display_w, display_h))
                
                # Cập nhật nhãn nhận diện (nếu không có ai, sẽ giữ lại tên cuối cùng)
                if not detections:
                     lbl_detect.configure(text=f"👤 Nhận diện: None")
                     
            # --- HIỂN THỊ TRONG DARK MODE ---
            else: 
                # Chuyển sang ảnh nền đen
                pil_img = create_dark_placeholder(display_w, display_h)
                lbl_detect.configure(text=f"👤 Nhận diện: Chế độ Tối")

            ctk_img = ctk.CTkImage(light_image=pil_img, size=pil_img.size)
            label_widget.configure(image=ctk_img)
            label_widget.image = ctk_img


        log("🖼️ GUI stopped")

    # 🚀 Khởi động đa luồng
    t1 = threading.Thread(target=capture_thread, daemon=True)
    t2 = threading.Thread(target=inference_thread, daemon=True)
    t3 = threading.Thread(target=gui_thread, daemon=True)
    t1.start()
    t2.start()
    t3.start()

    log(f"✅ Dynamic Mode đang chạy. Chuyển sang Tối sau {DARK_MODE_TIMEOUT} không phát hiện.")
    
import pandas as pd
from datetime import datetime

def export_attendance_to_excel():
    if not os.path.exists(ATTENDANCE_FILE):
        messagebox.showwarning("Thông báo", "Chưa có dữ liệu điểm danh để xuất.")
        return
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        if "Name" not in df.columns or "Time" not in df.columns or "Type" not in df.columns:
            messagebox.showerror("Lỗi", "File điểm danh bị thiếu cột cần thiết (Name, Time, Type).")
            return

        # Lọc dữ liệu hôm nay
        today_str = datetime.now().strftime("%Y-%m-%d")
        df_today = df[df["Time"].astype(str).str.startswith(today_str)].copy()

        if df_today.empty:
            messagebox.showinfo("Thông báo", "Hôm nay chưa có dữ liệu điểm danh.")
            return

        # Tách MSSV và Tên từ cột Name (định dạng MSSV_Tên)
        df_today[["MSSV", "Tên"]] = df_today["Name"].str.split("_", n=1, expand=True)
        #df_today["Tên"] = df_today["Name"]
        df_today["Ngày"] = pd.to_datetime(df_today["Time"]).dt.strftime("%Y-%m-%d")
        df_today["Giờ"] = pd.to_datetime(df_today["Time"]).dt.strftime("%H:%M:%S")
        df_today["Trạng thái"] = df_today["Type"]

        # Chọn cột cần xuất
        df_export = df_today[["MSSV", "Tên", "Ngày", "Giờ", "Trạng thái"]]

        out_file = f"attendance_today_{today_str}.xlsx"
        df_export.to_excel(out_file, index=False)
        messagebox.showinfo("Thành công", f"Đã xuất dữ liệu điểm danh hôm nay sang {out_file}")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể xuất file Excel:\n{e}")


def start_recognition():
    global running, last_detection_time
    if running:
        return
    # Khởi tạo thời gian phát hiện ngay lập tức để bắt đầu ở chế độ Active
    last_detection_time = time.time() 
    running = True
    threading.Thread(target=process_video, args=(video_label,), daemon=True).start()

def stop_recognition():
    global running
    running = False

# ---------------- Auto-capture action ----------------
def auto_capture_action():
    mssv = entry_mssv.get().strip()
    name = entry_name.get().strip()
    if not mssv or not name:
        messagebox.showwarning("Cảnh báo", "Vui lòng nhập MSSV và Tên.")
        return

    try:
        interval_str = interval_var.get()
        interval = float(interval_str) if interval_str else 1.0  # mặc định 1 giây
    except ValueError:
        messagebox.showwarning("Cảnh báo", "Khoảng thời gian không hợp lệ!")
        return

    try:
        count_str = count_var.get()
        count = int(count_str) if count_str else 5  # mặc định 5 ảnh
    except ValueError:
        messagebox.showwarning("Cảnh báo", "Số ảnh không hợp lệ!")
        return

    person_id = f"{mssv}_{name}"

    threading.Thread(
        target=auto_capture_and_train,
        args=(person_id, name, interval, count),
        daemon=True
    ).start()


# ---------------- Auto-capture routine ----------------
# ---------------- Auto-capture nâng cấp ----------------
def auto_capture_and_train(person_id, display_name, interval_sec, total_count):
    """
    Chụp tự động nhưng bỏ qua ảnh kém chất lượng:
    - Mờ (blur)
    - Quá tối (dark)
    - Khuôn mặt quá nhỏ
    """
    global stop_auto_capture_flag, video_capture

    person_dir = os.path.join(KNOWN_FACES_DIR, person_id)
    os.makedirs(person_dir, exist_ok=True)

    stop_auto_capture_flag = False
    saved = 0
    last_time = time.time() - interval_sec

    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không thể mở camera để chụp.")
        return

    log(f"📸 Bắt đầu chụp tự động: {total_count} ảnh, mỗi {interval_sec}s cho '{display_name}'")
    progress.set(0)

    # ------------------ HÀM LỌC CHẤT LƯỢNG ------------------
    def is_blurry(img, threshold=100.0):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

    def is_too_dark(img, threshold=50):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) < threshold

    def is_face_too_small(box, min_width=50, min_height=50):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        return w < min_width or h < min_height

    while saved < total_count and not stop_auto_capture_flag:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        now = time.time()

        # Hiển thị live video
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        display_w = 760
        display_h = int(display_w * pil_img.height / pil_img.width)
        pil_img = pil_img.resize((display_w, display_h))
        ctk_img = ctk.CTkImage(light_image=pil_img, size=pil_img.size)
        video_label.configure(image=ctk_img)
        video_label.image = ctk_img

        if now - last_time >= interval_sec:
            try:
                results = yolo_model(frame)
                boxes = results[0].boxes
            except Exception as e:
                boxes = []
                print("[WARN] YOLO error during capture:", e)

            if boxes is None or len(boxes) == 0:
                log("⚠️ Không phát hiện khuôn mặt lúc chụp, thử lại sau.")
                last_time = now
                time.sleep(0.2)
                continue

            box = boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face = frame[y1:y2, x1:x2]

            # ------------------ CHECK QUALITY ------------------
            if face.size == 0:
                last_time = now
                continue
            if is_blurry(face):
                log("⚠️ Ảnh mờ, bỏ qua")
                last_time = now
                continue
            if is_too_dark(face):
                log("⚠️ Ảnh tối, bỏ qua")
                last_time = now
                continue
            if is_face_too_small((x1, y1, x2, y2)):
                log("⚠️ Khuôn mặt quá nhỏ, bỏ qua")
                last_time = now
                continue

            # ------------------ LƯU ẢNH ------------------
            fname = os.path.join(person_dir, f"{person_id}_{int(time.time())}_{saved}.jpg")
            cv2.imwrite(fname, face)
            saved += 1
            last_time = now
            progress.set(saved / total_count)
            log(f"💾 Đã lưu {saved}/{total_count}: {fname}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    progress.set(0)

    if stop_auto_capture_flag:
        log("⏸️ Chụp tự động đã bị hủy.")
        return

    if saved >= total_count:
        log(f"✅ Hoàn tất chụp {saved} ảnh cho '{display_name}'. Bắt đầu huấn luyện...")
        threading.Thread(target=train_single_person, args=(person_id, display_name), daemon=True).start()
    else:
        log("⚠️ Không đủ ảnh được lưu, hủy auto-capture.")


# ---------------- Train single person ----------------
def train_single_person(person_id, display_name):
    """
    - person_id: MSSV_Tên (folder & embeddings)
    - display_name: tên hiển thị/log
    """
    global known_embeddings
    person_path = os.path.join(KNOWN_FACES_DIR, person_id)
    if not os.path.isdir(person_path):
        log(f"❌ Thư mục không tồn tại: {person_path}")
        return

    emb_list = []
    for fname in os.listdir(person_path):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        p = os.path.join(person_path, fname)
        img = cv2.imread(p)
        if img is None:
            continue
        try:
            results = yolo_model(img)
            boxes = results[0].boxes
        except Exception as e:
            boxes = []

        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue

            try:
                aligned = align_face_mediapipe(img, (x1, y1, x2, y2), output_size=160)
            except Exception:
                aligned = None
            if aligned is None:
                aligned = center_crop_resize(img, (x1, y1, x2, y2), output_size=160)
            if aligned is None:
                continue

            face_tensor = torch.tensor(aligned.transpose(2,0,1)).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                face_tensor = face_tensor.to(device)
                emb = resnet(face_tensor).detach().cpu().numpy().flatten()
            emb_list.append(emb)

    if emb_list:
        known_embeddings[person_id] = np.mean(emb_list, axis=0)
        save_known_embeddings(known_embeddings)
        log(f"🎯 Đã huấn luyện & cập nhật embedding cho: {display_name} ({person_id})")
        messagebox.showinfo("Thành công", f"Người '{display_name}' đã được thêm vào hệ thống.")
    else:
        log(f"⚠️ Không tạo được embedding cho {display_name}. Kiểm tra ảnh trong {person_path}.")


def train_all_embeddings():
    log("🔧 Bắt đầu huấn luyện toàn bộ embeddings từ known_faces/ ...")
    
    #  CẬP NHẬT TRẠNG THÁI KHỞI TẠO 
    app.after(0, lambda: lbl_train_status.configure(text="Đang xử lý: Khởi tạo..."))
    
    persons = [d for d in os.listdir(KNOWN_FACES_DIR) if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))]
    total = len(persons)
    i = 0
    new_embeddings = {}
    for person in persons:
        i += 1
        #  CẬP NHẬT TÊN NGƯỜI ĐANG XỬ LÝ 
        # Sử dụng lambda với tham số mặc định để tránh vấn đề closure trong Python
        app.after(0, lambda p=person: lbl_train_status.configure(text=f"Đang xử lý: {p} ({i}/{total})"))
        
        # reuse train_single_person logic per-person (but to avoid repeated saving/IO, inline similar ops)
        person_path = os.path.join(KNOWN_FACES_DIR, person)
        emb_list = []
        for fname in os.listdir(person_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            p = os.path.join(person_path, fname)
            img = cv2.imread(p)
            if img is None:
                continue
            try:
                results = yolo_model(img)
                boxes = results[0].boxes
            except Exception as e:
                boxes = []
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                face = img[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                # Align using MediaPipe first, fallback to center-crop
                try:
                    aligned = align_face_mediapipe(img, (x1, y1, x2, y2), output_size=160)
                except Exception:
                    aligned = None

                if aligned is None:
                    aligned = center_crop_resize(img, (x1, y1, x2, y2), output_size=160)
                if aligned is None:
                    continue

                face_tensor = torch.tensor(aligned.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    face_tensor = face_tensor.to(device)
                    emb = resnet(face_tensor).detach().cpu().numpy().flatten()
                emb_list.append(emb)
        if emb_list:
            new_embeddings[person] = np.mean(emb_list, axis=0)
            log(f"✅ Huấn luyện: {person} ({len(emb_list)} ảnh)")
        else:
            log(f"⚠️ Không có ảnh hợp lệ cho: {person}")
        progress.set(i/max(1,total))
    # save
    save_known_embeddings(new_embeddings)
    global known_embeddings
    known_embeddings = new_embeddings
    
    #  XÓA TRẠNG THÁT SAU KHI HOÀN TẤT 
    app.after(0, lambda: lbl_train_status.configure(text="")) 
    
    progress.set(0)
    log("🎯 Huấn luyện toàn bộ hoàn tất.")

# ---------------- UI Actions ----------------
def delete_face():
    name = simpledialog.askstring("Xóa khuôn mặt", "Nhập tên người cần xóa:")
    if not name:
        return

    person_folder = os.path.join("known_faces", name)

    if os.path.exists(person_folder):
        try:
            shutil.rmtree(person_folder)
            
            # Cập nhật lại embeddings ngay lập tức
            if name in known_embeddings:
                del known_embeddings[name]
                save_known_embeddings(known_embeddings)
                
            messagebox.showinfo("Thành công", f"Đã xóa toàn bộ dữ liệu khuôn mặt của {name}.")
            print(f"[INFO] Đã xóa thư mục: {person_folder}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xóa: {e}")
    else:
        messagebox.showwarning("Không tìm thấy", f"Không tìm thấy người tên {name} trong known_faces.")

def start_recognition_action():
    start_recognition()

def stop_recognition_action():
    stop_recognition()


def reload_embeddings_action():
    threading.Thread(target=train_all_embeddings, daemon=True).start()

def show_stats_action():
    """
    Hiển thị cửa sổ thống kê chi tiết cho ngày hôm nay.
    - Tính toán thời gian làm việc (Earliest Vào đến Latest Ra).
    - Hiển thị trạng thái (Hoàn tất/Đã vào/Chưa vào).
    """
    if not os.path.exists(ATTENDANCE_FILE):
        messagebox.showinfo("Thông báo", "Chưa có dữ liệu điểm danh.")
        return
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        if "Time" not in df.columns or "Name" not in df.columns or "Type" not in df.columns:
            messagebox.showerror("Lỗi", "File điểm danh bị thiếu cột cần thiết (Name, Time, Type).")
            return
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể đọc file điểm danh: {e}")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    # Lọc dữ liệu hôm nay và tạo bản sao để tránh SettingWithCopyWarning
    df_today = df[df["Time"].astype(str).str.startswith(today)].copy()

    if df_today.empty:
        # show in simple window (Top Level)
        top = ctk.CTkToplevel(app)
        top.geometry("700x500")
        top.title("📊 Thống kê điểm danh hôm nay")
        txt = ctk.CTkTextbox(master=top, width=660, height=420)
        txt.pack(padx=10, pady=10)
        txt.insert("0.0", "Không có dữ liệu điểm danh hôm nay.")
        txt.configure(state="disabled")
        return

    # Convert Time column to datetime objects
    df_today.loc[:, 'Time'] = pd.to_datetime(df_today['Time'])

    summary = []
    
    # Aggregate data by person
    for name, group in df_today.groupby('Name'):
        logs_in = group[group['Type'] == 'Vào']
        logs_out = group[group['Type'] == 'Ra']

        first_in = logs_in['Time'].min() if not logs_in.empty else None
        last_out = logs_out['Time'].max() if not logs_out.empty else None
        
        working_time_str = "N/A"
        status = "❌ Chưa vào"

        if first_in:
            if last_out and last_out > first_in:
                # Calculate working time (timedelta)
                time_delta = last_out - first_in
                
                # Format timedelta to HH:MM:SS
                total_seconds = int(time_delta.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                
                working_time_str = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"
                status = "✅ Hoàn tất (Vào/Ra)"
            else:
                status = "🟠 Đã vào (Chưa ra)"
        
        # Format times for display
        time_in_str = first_in.strftime("%H:%M:%S") if first_in else "N/A"
        time_out_str = last_out.strftime("%H:%M:%S") if last_out else "N/A"

        summary.append({
            "Tên": name,
            "Vào (Earliest)": time_in_str,
            "Ra (Latest)": time_out_str,
            "Thời gian ": working_time_str,
            "Trạng thái": status
        })

    # Create a Pandas DataFrame for better formatting
    df_summary = pd.DataFrame(summary)

    # Prepare display text
    header = "📊 THỐNG KÊ ĐIỂM DANH HÔM NAY\n"
    header += f"Ngày: {today}\n"
    
    # Use to_string for nice alignment
    report_text = df_summary.to_string(index=False)
    final_text = header + "\n" + report_text

    # Show the results
    top = ctk.CTkToplevel(app)
    top.geometry("800x600") # Increase size for better view
    top.title("📊 Thống kê điểm danh hôm nay")
    # Sử dụng font Courier để đảm bảo các cột được căn chỉnh đều
    txt = ctk.CTkTextbox(master=top, width=780, height=520, font=("Courier", 12)) 
    txt.pack(padx=10, pady=10)
    txt.insert("0.0", final_text)
    txt.configure(state="disabled")

# Buttons binding
btn_start.configure(command=start_recognition_action)
btn_stop.configure(command=stop_recognition_action)
btn_auto_capture.configure(command=auto_capture_action)
btn_reload.configure(command=reload_embeddings_action)
btn_stats.configure(command=show_stats_action)
btn_export.configure(command=export_attendance_to_excel)
btn_declec.configure(command=delete_face)
btn_exit.configure(command=lambda: (stop_recognition(), app.destroy()))

# ---------------- Start app ----------------
if __name__ == "__main__":
    log("Ứng dụng sẵn sàng. Nhập tên, điều chỉnh thời gian và số ảnh, bấm 'Chụp tự độngs' hoặc 'Bắt đầu nhận diện'.")
    app.mainloop()  
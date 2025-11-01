# =========================
# Stage: Runtime (Python)
# =========================
FROM python:3.11-slim

# Libs ضرورية لـ OpenCV والويبكام
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgtk-3-0 \
    v4l-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# نثبّت المتطلبات
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ننسخّ المشروع
COPY . /app

# Env defaults (يمكن تغيّرهم فـ compose)
ENV DB_HOST=mysqldb \
    DB_USER=faceid_user \
    DB_PASSWORD=faceid_pass \
    DB_NAME=faceid_db \
    PYTHONUNBUFFERED=1 \
    OPENCV_LOG_LEVEL=ERROR

# تأكيد المجلدات
RUN mkdir -p data/dataset sql

# ندخل bash بشكل افتراضي؛ غادي نختار السكريبتات بـ docker exec
CMD ["bash"]

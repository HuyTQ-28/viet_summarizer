
FROM python:3.10-slim

# Thiết lập thư mục làm việc bên trong container
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Cài đặt các thư viện cần thiết
COPY requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn của dự án vào container
COPY . .

# Mở cổng 8000 để bên ngoài có thể truy cập vào ứng dụng
EXPOSE 8000

# Lệnh để chạy ứng dụng khi container khởi động
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:8000"]

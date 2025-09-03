# 使用支持多平台构建的基础镜像
FROM --platform=linux/amd64 python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖项（用于matplotlib和其他依赖）
# 临时禁用代理以解决连接问题
RUN export http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= && \
    apt-get update && apt-get install -y \
    libfontconfig1 \
    libfreetype6-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖清单并安装项目依赖
COPY requirements.txt .
RUN export http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= && \
    pip install --no-cache-dir -r requirements.txt

# 复制项目文件到容器中
COPY . .

# 创建上传目录
RUN mkdir -p uploads

# 暴露应用端口
EXPOSE 5000

# 设置环境变量
ENV FLASK_APP=main_web.py
ENV PYTHONUNBUFFERED=1

# 启动应用
CMD ["python", "main_web.py"]

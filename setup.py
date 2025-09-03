import os
import sys
from pathlib import Path

def resource_path(relative_path):
    """获取资源的绝对路径，适用于开发环境和PyInstaller打包后的环境"""
    try:
        # PyInstaller创建临时文件夹，将路径存储在_MEIPASS中
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def setup_app():
    """应用程序启动前的设置"""
    try:
        # 获取应用程序根目录或临时目录
        base_dir = getattr(sys, '_MEIPASS', os.path.abspath("."))
        # 确保uploads文件夹存在（使用绝对路径）
        uploads_dir = os.path.join(base_dir, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        print(f"上传文件夹位置: {uploads_dir}")
    except Exception as e:
        print(f"创建上传文件夹时出错: {str(e)}")
        
    # 返回字体文件路径（相对于应用根目录）
    return resource_path('AlibabaPuHuiTi-3-55-Regular.ttf')

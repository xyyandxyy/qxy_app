import os
import sys
from pathlib import Path

def resource_path(relative_path):
    """获取资源的绝对路径，适用于开发环境和PyInstaller打包后的环境"""
    try:
        # PyInstaller创建临时文件夹，将路径存储在_MEIPASS中
        base_path = sys._MEIPASS
        print(f"PyInstaller环境: MEIPASS={base_path}")
    except Exception:
        base_path = os.path.abspath(".")
        print(f"标准环境: 当前目录={base_path}")
    
    # 处理Windows路径
    result_path = os.path.normpath(os.path.join(base_path, relative_path))
    print(f"资源路径: {relative_path} -> {result_path}")
    return result_path

def setup_app():
    """应用程序启动前的设置"""
    try:
        # 获取应用程序根目录或临时目录
        base_dir = getattr(sys, '_MEIPASS', os.path.abspath("."))
        print(f"应用根目录: {base_dir}")
        
        # 由于现在使用内存处理，不再需要上传目录
        print("使用内存处理模式，无需创建上传目录")
        
        # 只返回字体文件路径
        return resource_path('AlibabaPuHuiTi-3-55-Regular.ttf'), None
    except Exception as e:
        print(f"设置应用时出错: {str(e)}")
        # 出错时仍然返回字体文件路径
        return resource_path('AlibabaPuHuiTi-3-55-Regular.ttf'), None

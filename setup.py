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
        
        # 确保uploads文件夹存在（使用绝对路径）
        uploads_dir = os.path.normpath(os.path.join(base_dir, 'uploads'))
        print(f"尝试创建上传文件夹: {uploads_dir}")
        
        # 处理可能的权限问题
        try:
            os.makedirs(uploads_dir, exist_ok=True)
            print(f"成功创建/确认上传文件夹: {uploads_dir}")
        except PermissionError:
            # 如果无权限在应用目录创建文件夹，尝试在用户临时目录创建
            import tempfile
            temp_dir = tempfile.gettempdir()
            uploads_dir = os.path.normpath(os.path.join(temp_dir, 'qxy_app_uploads'))
            os.makedirs(uploads_dir, exist_ok=True)
            print(f"在临时目录创建上传文件夹: {uploads_dir}")
            
        # 验证目录是否可写
        try:
            test_file = os.path.join(uploads_dir, 'test_write.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"上传目录可写: {uploads_dir}")
        except Exception as e:
            print(f"警告: 上传目录可能不可写: {str(e)}")
            
        # 返回创建的uploads目录路径，供后续使用
        return resource_path('AlibabaPuHuiTi-3-55-Regular.ttf'), uploads_dir
    except Exception as e:
        print(f"创建上传文件夹时出错: {str(e)}")
        # 出错时仍然返回字体文件路径
        return resource_path('AlibabaPuHuiTi-3-55-Regular.ttf'), None

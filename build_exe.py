import os
import sys
import shutil
import subprocess
import platform

def main():
    """构建Windows可执行文件"""
    print("开始构建Windows可执行文件...")
    
    # 确保我们在正确的目录中
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 清理上一次构建的残留文件
    build_dirs = ['build', 'dist']
    for dir_name in build_dirs:
        if os.path.exists(dir_name):
            print(f"清理目录: {dir_name}")
            shutil.rmtree(dir_name)
    
    # 使用PyInstaller执行打包
    try:
        print("开始PyInstaller打包过程...")
        # 使用预定义的spec文件
        result = subprocess.run(
            ["pyinstaller", "--clean", "qxy_app.spec"],
            check=True,
            text=True,
            capture_output=True
        )
        print("PyInstaller完成:")
        print(result.stdout)
        
        # 构建成功
        dist_path = os.path.join(script_dir, "dist")
        if os.path.exists(dist_path):
            print(f"\n打包成功! 可执行文件位于: {dist_path}")
            print("文件列表:")
            for file in os.listdir(dist_path):
                file_path = os.path.join(dist_path, file)
                if os.path.isfile(file_path) and file.endswith('.exe'):
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f" - {file} ({size_mb:.2f}MB)")
        else:
            print("警告: 找不到dist目录，打包可能失败")
            
    except subprocess.CalledProcessError as e:
        print("打包失败:")
        print(e.stdout)
        print(e.stderr)
        return 1
    
    print("\n打包操作完成!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

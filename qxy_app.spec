# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main_web.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('AlibabaPuHuiTi-3-55-Regular.ttf', '.'),
        ('templates', 'templates'),
        ('favicon.ico', '.'),
    ],
    hiddenimports=[
        'pandas',
        'matplotlib',
        'seaborn',
        'numpy',
        'flask',
        'werkzeug',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='智能数据分析系统',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='x86',
    codesign_identity=None,
    entitlements_file=None,
    icon='favicon.ico',
    # 减少临时文件使用
    onefile=True,
    # 不创建临时目录，直接解压到内存
    tempdir=None,
)

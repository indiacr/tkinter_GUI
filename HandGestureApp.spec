# -*- mode: python ; coding: utf-8 -*-

# Define block_cipher
block_cipher = None

a = Analysis(
    ['sound with Text_App.py'],
    pathex=['c:\\Pycham_files\\ASL_trans\\v8\\Sign_Text\\tkinter'],  # Make sure this path is correct
    binaries=[],
    datas=[('model.p', 'model.p')],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,  # This references the block_cipher defined above
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HandGestureApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False  # Set to True if you want a console window
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HandGestureApp'
) 

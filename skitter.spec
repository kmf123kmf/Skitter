# PyInstaller build of the app: a folder with Skitter.exe and everything it needs.
#
#     pyinstaller skitter.spec
#
# The result is dist/Skitter/ (run dist/Skitter/Skitter.exe). Build on the
# platform it is for: PyInstaller doesn't cross-compile.

from pathlib import Path

src = Path(SPECPATH) / "src"

a = Analysis(
    [str(src / "skitter" / "__main__.py")],
    pathex=[str(src)],
    datas=[(str(src / "skitter" / "resources"), "skitter/resources")],
    # pkg_resources isn't used, and older setuptools' copy fails at startup when frozen.
    excludes=["pkg_resources"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="Skitter",
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Skitter")

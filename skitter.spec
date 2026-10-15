# PyInstaller build of the app: a folder with Skitter.exe and everything it needs.
#
#     pyinstaller skitter.spec
#
# The result is dist/Skitter/ (run dist/Skitter/Skitter.exe). Build on the
# platform it is for: PyInstaller doesn't cross-compile.

from pathlib import Path

root = Path(SPECPATH)
src = root / "src"

a = Analysis(
    [str(src / "skitter" / "__main__.py")],
    pathex=[str(src)],
    datas=[
        (str(src / "skitter" / "resources"), "skitter/resources"),
        # The GPL text and notices travel with every build.
        (str(root / "LICENSE"), "."),
        (str(root / "THIRD_PARTY_NOTICES.md"), "."),
    ],
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

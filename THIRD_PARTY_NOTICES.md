# Third-party notices

Skitter is licensed under the GNU General Public License, version 3 or later (see `LICENSE`).

It uses the packages below. Each keeps its own license; the full texts ship inside the packages (in a standalone build, in their folders next to `Skitter.exe`). Versions are those the project was last checked against.

## Why the whole program is GPL

The Windows wheels of **PyAV** bundle an FFmpeg built with GPL components (x264, x265), and the wheels of **pillow-heif** bundle x265 (GPLv2), libheif and libde265 (LGPLv3); **pillow-jxl-plugin** is itself GPLv3 or later. A program distributed with them must be released under a GPL-compatible license, and version 3 is the one that satisfies every license in this list. PySide6 is used under the GPL (it is offered under LGPLv3, GPLv2 or GPLv3).

## Python packages

| Package | Version | License | Notes |
|---|---|---|---|
| PySide6, PySide6-Essentials, PySide6-Addons, shiboken6 (Qt for Python / Qt 6) | 6.11 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only (or commercial) | Used under the GPL. Source: https://code.qt.io/ |
| av (PyAV) | 19.0 | BSD-3-Clause (source); wheels bundle FFmpeg libraries | The wheel includes libx264 and libx265 (GPL), libvpx, libwebp, libopus, libdav1d, SVT-AV1 and libvmaf. Source: https://github.com/PyAV-Org/PyAV, https://ffmpeg.org/ |
| pillow-heif | 1.8 | BSD-3-Clause (source); wheels GPLv2 | Bundles libheif and libde265 (LGPLv3), x265 (GPLv2) and the MinGW-w64 runtime. Details: `LICENSES_bundled.txt` in the package |
| pillow-jxl-plugin | 1.3 | GPL-3.0-or-later | Bundles libjxl and libhwy (BSD-3-Clause), Brotli (MIT), Little CMS (MIT) and the MinGW-w64 / GCC runtime. Source: https://github.com/Isotr0py/pillow-jpegxl-plugin |
| Pillow | 12.3 | MIT-CMU | |
| numpy | 2.5 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | |
| scipy | 1.18 | BSD-3-Clause | Wheels also bundle OpenBLAS and the GCC runtime; see the package |
| numba | 0.68 | BSD-2-Clause | Third-party licenses in the package's `LICENSES.third-party` |
| llvmlite | 0.50 | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | |
| faiss-cpu | 1.15 | MIT | |
| moderngl | 5.12 | MIT | |
| glcontext | 3.0 | MIT | |

## Build tool

PyInstaller (GPL-2.0-or-later with a bootloader exception) is used only to build the standalone folder. Its license does not apply to the built program.

## Source code for GPL and LGPL components

Skitter's own source is at the project repository. The source of the bundled GPL and LGPL libraries is available from the links above and from the PyAV, pillow-heif, pillow-jxl-plugin, FFmpeg, x264, x265, libheif and libde265 projects. On request, the Skitter maintainers will also provide the exact source of any component shipped in a build.

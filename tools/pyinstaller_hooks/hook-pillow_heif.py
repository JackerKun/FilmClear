"""Collect the wheel's HEVC encoder and decoder libraries."""

from importlib.metadata import distribution

from PyInstaller import compat
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_delvewheel_libs_directory

binaries = collect_dynamic_libs("pillow_heif", search_patterns=["*.dll", "*.dylib", "*.so*"])
datas, binaries = collect_delvewheel_libs_directory("pillow_heif", binaries=binaries)
if compat.is_win:
    # Newer Windows wheels put the DLLs beside _pillow_heif.pyd in platlib.
    wheel = distribution("pillow-heif")
    binaries += [(str(wheel.locate_file(path)), ".") for path in wheel.files or ()
                 if len(path.parts) == 1 and path.suffix.lower() == ".dll"]

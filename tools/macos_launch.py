"""Give source launches a macOS app identity while using the project venv."""

import ctypes
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


def running_interpreter():
    get_path = ctypes.CDLL(None)._NSGetExecutablePath
    size = ctypes.c_uint32(1024)
    buffer = ctypes.create_string_buffer(size.value)
    if get_path(buffer, ctypes.byref(size)) != 0:
        buffer = ctypes.create_string_buffer(size.value)
        if get_path(buffer, ctypes.byref(size)) != 0:
            raise OSError("Cannot locate the macOS Python executable")
    return Path(os.fsdecode(buffer.value)).resolve()


def launch():
    root = Path(__file__).resolve().parents[1]
    venv = Path(sys.prefix)
    interpreter = running_interpreter()
    icon = root / "assets" / "filmclear.icns"
    bundle = venv / "FilmClear.app"
    launcher = bundle / "Contents" / "MacOS" / "FilmClear"
    env = dict(os.environ, PYTHONEXECUTABLE=sys.executable)
    env.pop("__PYVENV_LAUNCHER__", None)
    try:
        sources = (interpreter, icon, Path(__file__), venv / "pyvenv.cfg")
        if not launcher.exists() or any(path.stat().st_mtime > launcher.stat().st_mtime for path in sources):
            with TemporaryDirectory(dir=venv) as directory:
                staged = Path(directory) / bundle.name
                contents = staged / "Contents"
                executable = contents / "MacOS" / "FilmClear"
                executable.parent.mkdir(parents=True)
                shutil.copy(interpreter, executable)
                resources = contents / "Resources"
                resources.mkdir()
                shutil.copy(icon, resources / icon.name)
                with (contents / "Info.plist").open("wb") as stream:
                    plistlib.dump({
                        "CFBundleIdentifier": "com.filmclear.source",
                        "CFBundleName": "FilmClear",
                        "CFBundleDisplayName": "FilmClear",
                        "CFBundleExecutable": "FilmClear",
                        "CFBundlePackageType": "APPL",
                        "CFBundleIconFile": icon.name,
                        "NSHighResolutionCapable": True,
                    }, stream)
                subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", str(staged)], check=True)
                # Some custom Python builds rely on libraries relative to the executable.
                subprocess.run([str(executable), "-c",
                                f"import sys; assert sys.prefix == {str(venv)!r}"],
                               env=env, check=True)
                if bundle.exists():
                    shutil.rmtree(bundle)
                staged.rename(bundle)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"FilmClear launcher setup failed: {error}", file=sys.stderr)
        os.execv(sys.executable, [sys.executable, str(root / "main.py"), *sys.argv[1:]])
    os.execve(launcher, [str(launcher), str(root / "main.py"), *sys.argv[1:]], env)


if __name__ == "__main__":
    launch()

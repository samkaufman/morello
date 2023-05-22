import pathlib
import subprocess
import sys


def test_main_script_can_be_run(request):
    rootdir: pathlib.Path = request.config.rootdir
    main_path = rootdir / "scripts" / "main.py"
    script_args = ["--target", "x86", "--serial", "matmul", "2", "2", "2"]
    subprocess.run([sys.executable, str(main_path)] + script_args, check=True)

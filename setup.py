import setuptools
from Cython.Build import cythonize

extensions = [
    setuptools.Extension("dtypes", ["morello/dtypes.py"]),
    setuptools.Extension("specs", ["morello/specs/*.py"]),
]

setuptools.setup(ext_modules=cythonize(extensions, language_level="3"))

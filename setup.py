import setuptools
from Cython.Build import cythonize

extensions = [
    setuptools.Extension("morello.cost", ["morello/cost.py"]),
    setuptools.Extension("morello.dtypes", ["morello/dtypes.py"]),
    setuptools.Extension("morello.layouts", ["morello/layouts.py"]),
    setuptools.Extension("morello.search", ["morello/search/__init__.py"]),
    setuptools.Extension("morello.search.beam", ["morello/search/beam.py"]),
    setuptools.Extension("morello.search.common", ["morello/search/common.py"]),
    setuptools.Extension("morello.search.dp", ["morello/search/dp.py"]),
    setuptools.Extension("morello.search.naive", ["morello/search/naive.py"]),
    setuptools.Extension("morello.search.random", ["morello/search/random.py"]),
    setuptools.Extension("morello.specs", ["morello/specs/__init__.py"]),
    setuptools.Extension("morello.specs.base", ["morello/specs/base.py"]),
    setuptools.Extension("morello.specs.compose", ["morello/specs/compose.py"]),
    setuptools.Extension("morello.specs.conv", ["morello/specs/conv.py"]),
    setuptools.Extension("morello.specs.matmul", ["morello/specs/matmul.py"]),
    setuptools.Extension("morello.specs.reducesum", ["morello/specs/reducesum.py"]),
    setuptools.Extension("morello.specs.tensorspec", ["morello/specs/tensorspec.py"]),
    setuptools.Extension("morello.replace", ["morello/replace.py"]),
]

setuptools.setup(ext_modules=cythonize(extensions, language_level="3"))

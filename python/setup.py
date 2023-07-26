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
    setuptools.Extension("morello.specs", ["morello/specs/*.py"]),
]

setuptools.setup(ext_modules=cythonize(extensions, language_level="3"))

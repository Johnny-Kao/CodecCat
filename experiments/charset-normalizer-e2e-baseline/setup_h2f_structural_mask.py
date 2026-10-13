from setuptools import Extension, setup
from Cython.Build import cythonize

setup(
    name="h2f_structural_mask",
    ext_modules=cythonize(
        [Extension("h2f_structural_mask", ["h2f_structural_mask.pyx"])],
        compiler_directives={"language_level": 3, "boundscheck": False, "wraparound": False},
    ),
)

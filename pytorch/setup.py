import numpy
from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize

extensions = [
    Extension('torchsparseattn._isotonic',
              ["torchsparseattn/_isotonic.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension('torchsparseattn._fused_jv',
              ["torchsparseattn/_fused_jv.pyx"]),
    Extension('torchsparseattn._fused',
              ["torchsparseattn/_fused.pyx"])
]

extensions = cythonize(extensions)


setup(name="torchsparseattn",
      version="0.2.dev0",
      description="Sparse structured attention mechanisms for pytorch",
      author="Vlad Niculae",
      author_email="vlad@vene.ro",
      license="BSD 3-clause",
      packages=find_packages(),
      ext_modules=extensions,
      install_requires=['numpy'],
      zip_safe=False,
      classifiers=[
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers', 'License :: OSI Approved',
          'Programming Language :: C', 'Programming Language :: Python',
          'Topic :: Software Development',
          'Topic :: Scientific/Engineering',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX', 'Operating System :: Unix',
          'Operating System :: MacOS']
)

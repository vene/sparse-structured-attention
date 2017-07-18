import numpy
from setuptools import setup, find_packages, Extension

extensions = [
    Extension('torchsparseattn._isotonic',
              ["torchsparseattn/_isotonic.c"],
              include_dirs=[numpy.get_include()])
]


setup(name="torchsparseattn",
      version="0.1",
      description="Sparse attention mechanisms for pytorch",
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

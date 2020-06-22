# Always prefer setuptools over distutils
# To use a consistent encoding
from codecs import open
from os import path

from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pdom',
    python_requires='>=3.6',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.1',

    description='Object Detection Metrics',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/yfpeng/object_detection_metrics',

    # Author details
    author='Yifan Peng',
    author_email='yip4002@med.cornell.edu',

    license='BSD 3-Clause License',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',

        # Specify the Python versions you support here.
        'Programming Language :: Python',
        'Topic :: Text Processing :: Markup :: XML',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
    ],

    keywords='pdom',

    packages=find_packages(exclude=["tests.*", "tests"]),
)

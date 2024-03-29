import io
import os
import sys

from setuptools import setup
from setuptools.command.test import test as TestCommand

import rpylib

here = os.path.abspath(os.path.dirname(__file__))


def read(*filenames, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


long_description = read("README.rst")


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


setup(
    name="rpylib",
    version=rpylib.__version__,
    url="https://github.com/rpalfray/rpylib",
    license="GNU General Public licence",
    author="Romain Palfray",
    tests_require=["pytest"],
    cmdclass={"test": PyTest},
    author_email="romain.palfray@gmail.com",
    description="Pricing library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["rpylib"],
    include_package_data=True,
    platforms="any",
    test_suite="rpylib.tests",
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 2 - Pre-Alpha",
        "Natural Language :: English",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    extras_require={
        "testing": ["pytest"],
    },
)

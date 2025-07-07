import re
import setuptools
from os import path

requirements_file = path.join(path.dirname(__file__), "requirements.in")
requirements = [r for r in open(requirements_file).read().split("\n") if not re.match(r"^\-", r)]

setuptools.setup(
  name="beir_extensions",
  version="0.1",
  url="https://github.com/atypon/beir.git",
  packages=setuptools.find_packages(),
  install_requires=requirements,
  description='Extensions to BEIR for evaluation of dense retrieval models',
)

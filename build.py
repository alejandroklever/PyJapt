import os
import re


with open("pyjapt/__init__.py", "r") as f:
    version = (
        re.search(r"__version__ = \"\d\.\d\.\d\"", f.read())
        .group()
        .replace("__version__ = \"", "")[:-1]
    )

with open("pyproject.toml", "r") as f:
    s = f.read()

pattern = re.compile(r'version = "\d\.\d\.\d"')
s = s.replace(pattern.search(s).group(), f'version = "{version}"')

with open("pyproject.toml", "w") as f:
    f.write(s)

os.system("poetry build")


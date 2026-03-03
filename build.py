import os
import re
import pyjapt


version = pyjapt.__version__

with open("pyproject.toml", "r") as f:
    s = f.read()

pattern = re.compile(r'version = "\d\.\d\.\d"')
s = s.replace(pattern.search(s).group(), f'version = "{version}"')

with open("pyproject.toml", "w") as f:
    f.write(s)

os.system("poetry build")

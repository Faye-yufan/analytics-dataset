## Generate documentation with Sphinx
cd ./docs
sphinx-quickstart

cd ./src/analyticsdf
sphinx-apidoc -o ../../docs .

make sure include `modules` in `index.rst` file

edit conf.py file, extention, html_theme and
make sure include the following in `conf.py` file:
```Python
import os
import sys
sys.path.insert(0, os.path.abspath("../src/analyticsdf"))
```

make html

## Github Action to build documentation
[sphinx official tutorial](https://www.sphinx-doc.org/en/master/tutorial/deploying.html#publishing-your-html-documentation)
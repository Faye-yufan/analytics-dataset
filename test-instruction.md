## Run test on your local machine
### Create a `venv` for testing
```
python3 -m venv test-venv    
```
_test-venv_ can be changed to any name you want.
Activate the venv:
```
source test-venv/bin/activate
```

### Install package to the current virtual environment
```
pip3 install -e .  
```
If there is any error occurs, check your `pip` version first, it's likely beause of the version of pip.

### Install requirements for test tools
```
pip3 install -r ./requirements_dev.txt
```

### Run pytest/flake8/mypy
pytest:
```
pytest
```
flake8:
```
flake8 src
```
mypy:
```
mypy src
```

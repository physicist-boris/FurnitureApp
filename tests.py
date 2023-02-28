import os 
import pytest


os.chdir(os.path.join(os.path.dirname(__file__), "tests"))

pytest.main()
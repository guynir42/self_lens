import sys
from importlib import reload
from types import MethodType


def update_object(obj, function_name):
    mod = sys.modules[obj.__module__]
    reload(mod)
    cls = getattr(mod, type(obj).__name__)
    setattr(obj, function_name, MethodType(getattr(cls, function_name), obj))

# HySE/_optional.py
import importlib

def _lazy_import(name: str):
	try:
		return importlib.import_module(name)
	except ModuleNotFoundError:
		return None

sitk = _lazy_import("SimpleITK")

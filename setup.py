"""Setup shim — real metadata and dependencies live in pyproject.toml.

Proto files are pre-generated and committed under ravnest/protos/, so
no grpc_tools.protoc compilation is needed at install time. To
regenerate them during development, run: python scripts/build_protos.py
(or manually invoke grpc_tools.protoc).
"""
from setuptools import setup

setup()
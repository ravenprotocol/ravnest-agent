import subprocess
import sys
import re
from pathlib import Path
import glob
from packaging.requirements import Requirement
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.build_py import build_py

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def proto_compile(output_path=this_directory):
    install_package("grpcio-tools==1.51.3")
    import grpc_tools.protoc
    cli_args = [
        "grpc_tools.protoc",
        "--proto_path=ravnest/protos",
        "--python_out={}/ravnest/protos".format(output_path),
        "--pyi_out={}/ravnest/protos".format(output_path),
        "--grpc_python_out={}/ravnest/protos".format(output_path),
    ] + glob.glob("ravnest/protos/*.proto")
    code = grpc_tools.protoc.main(cli_args)
    
    for script in glob.iglob("{}/ravnest/protos/*.py".format(output_path)):
        with open(script, "r+") as file:
            code = file.read()
            file.seek(0)
            file.write(re.sub(r"\n(import .+_pb2.*)", "\nfrom . \\1", code))
            file.truncate()
    

class CustomInstallCommand(install):
    def run(self):
        super().run()
        proto_compile(this_directory)

class CustomDevelopCommand(develop):
    def run(self):
        super().run()
        proto_compile(this_directory) 

class CustomBuildpyCommand(build_py):
    def run(self):
        super().run()
        proto_compile(this_directory)       

class CustomEggInfoCommand(egg_info):
    def run(self):
        super().run()
        proto_compile(this_directory)

with open("requirements.txt") as requirements_file:
    install_requires = [
        str(Requirement(line.strip()))
        for line in requirements_file
        if line.strip() and not line.strip().startswith("#")
    ]

setup(
    name="ravnest",
    version="0.2.0",
    cmdclass={"install":CustomInstallCommand, 
              "build_py":CustomBuildpyCommand,
              "develop":CustomDevelopCommand,
              "egg_info":CustomEggInfoCommand},
    license='MIT',
    author="Raven Protocol",
    author_email='kailash@ravenprotocol.com',
    packages=find_packages(),
    package_data={"ravnest":["protos/*"]},
    entry_points={
        "console_scripts": [
            "ravnest=ravnest.cli:main",
        ],
    },
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ravenprotocol/ravnest',
    keywords='deep learning, distributed computing, decentralized training',
    install_requires=install_requires
)
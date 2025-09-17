'''
The setup.py file is an essential part of packaging and 
distributing Python projects. It is used by setuptools 
(or distutils in older Python versions) to define the configuration 
of your project, such as its metadata, dependencies, and more
'''

from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    """
        This function will return list of requirements
    """
    requirements_lst: List[str]=[]
    try:
        with open('requirements.txt', 'r') as file:
            # Read each line of the file
            lines=file.readlines()
            # Process each line
            for line in lines:
                requirement=line.strip()
                # Ignore empty lines and -e.
                if requirement and requirement!="-e .":
                    requirements_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirements_lst

setup(
    name="Network_Security",
    version="0.0.1",
    author="Vivek Patel",
    author_email="vivekpatel1999@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)
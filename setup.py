import os
from setuptools import setup, find_packages

#Grab the README.md for the long description
with open('README.md', 'r') as f:
    long_description = f.read()

__version__ = '0.6.1'

def setup_package():
    setup(
        name = "autocnet",
        version = __version__,
        author = "Jay Laura",
        author_email = "jlaura@usgs.gov",
        description = ("Automated control network generation."),
        long_description = long_description,
        license = "Public Domain",
        keywords = "Multi-image correspondence detection",
        url = "http://packages.python.org/autocnet",
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False,
        install_requires=[],
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Topic :: Utilities",
            "License :: Public Domain",
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
        ],
        entry_points={"console_scripts": [
        "acn_submit = autocnet.graph.cluster_submit:main"], 
        }
    )

if __name__ == '__main__':
    setup_package()

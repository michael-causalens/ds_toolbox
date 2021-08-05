import setuptools
        
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ds_toolbox",
    version="1.0.0",
    author="Michael Russell",
    author_email="michael@causalens.com",
    description="Utility functions for data science work",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=None,
    packages=setuptools.find_packages(),
    package_data={'': ['*.pkl', '*.py', '*.npy']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

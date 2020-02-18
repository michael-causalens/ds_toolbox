import setuptools
from setuptools.command.install import install 

class PostInstallCommand(install):

    def run(self):
        """
        Overwrite setup.py install to remove build products after installation
        """
        import glob, shutil
        install.run(self)
        shutil.rmtree("build")
        shutil.rmtree(glob.glob('*.egg-info')[0])

        
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ds_toolbox",
    version="0.0.1",
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
    cmdclass = {'install':PostInstallCommand}
)

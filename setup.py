import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    'numpy>=1.14',
    'pandas>=0.23' ]
	
setuptools.setup(
    name="robout", 
    version="0.1.1",
    author="pedro-r-dias",
    author_email="pedroruivodias@gmail.com",
    description="Robust scaling for numeric data with outliers",
    long_description="Scaler preserving outliers found in unscaled data. It does not discard outliers, it transforms them to a controllable proximity in relation to the higher density region of the scaled distribution.",
    long_description_content_type="text/markdown",
    url="https://github.com/pedro-r-dias/robout",
    packages=setuptools.find_packages(),
	install_requires=requirements,
	keywords=['robout','scaling','standardization','normalization','outlier'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
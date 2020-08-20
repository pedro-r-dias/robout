import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="robout", 
    version="0.0.1",
    author="pedro-r-dias",
    author_email="pedroruivodias@gmail.com",
    description="Robust scaling for numeric data with outliers",
    long_description="This scaler preserves outliers found in the unscaled data as outliers also in the scaled data, but transforms them to an acceptable proximity in relation to the higher density region of the scaled distribution.",
    long_description_content_type="text/markdown",
    url="https://github.com/pedro-r-dias/robout",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
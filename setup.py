import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="prismx-lachmann12",
    version="0.7.",
    author="Alexander Lachmann",
    author_email="alexander.lachmann@mssm.edu",
    description="Package to for gene function predictions by unsupervised gene expression partitioning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maayanlab/prismx",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_data={
        "": ["data/*.h5"]
    },
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'feather-format',
        'h5py',
        'os',
        'pickle',
        'typing',
        'progress',
        'qnorm'
    ],
    python_requires='>=3.6',
)
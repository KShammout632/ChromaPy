import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chromapy-KShammout632",
    version="0.0.1",
    author="Kareem Shammout",
    author_email="kareemshammout632@gmail.com",
    description="A python library to colour black and white images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KShammout632/ChromaPy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
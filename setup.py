from setuptools import find_packages, setup


setup(
    name="ip_adapter_art",
    version="0.1",
    description="Using reference images to control style in diffusion models",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="Using reference images to control style in diffusion models",
    license="Apache",
    author="aihao",
    author_email="aihao2000@outlook.com",
    url="https://github.com/aihao2000/IP-Adapter-Art",
    packages=find_packages(),
    python_requires=">=3.8.0",
    install_requires=[
        "diffusers",
        "transformers",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

from setuptools import setup, find_packages

with open("Readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="expliRL",
    version="0.1.0",
    author="expliRL Contributors",
    author_email="contact@explirl.dev",
    description="Unified XAI framework with SHAP, LIME, and RL-driven Counterfactuals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/soham164/explirl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    keywords="explainable-ai, xai, shap, lime, counterfactuals, reinforcement-learning, machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/soham164/explirl/issues",
        "Source": "https://github.com/soham164/explirl",
        "Documentation": "https://github.com/soham164/explirl#readme",
    },
    install_requires=[
        "numpy>=1.19.0,<2.0.0",
        "pandas>=1.2.0,<2.0.0",
        "scikit-learn>=0.24.0",
        "shap>=0.39.0",
        "lime>=0.2.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "gymnasium>=0.26.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "click>=8.0.0",
        "plotly>=5.0.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "scikit-image>=0.18.0",
    ],
    entry_points={
        "console_scripts": [
            "explirl=expliRL.cli.cli_tool:main",
        ],
    },
)

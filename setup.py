from setuptools import setup, find_packages

setup(
    name="chess_ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "chess",
        "torch",
        "numpy",
        "tqdm"
    ],
    author="Codeium User",
    description="A chess AI using neural networks and MCTS",
    python_requires=">=3.9",
)

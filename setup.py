#!/usr/bin/env python

from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="hierspeechpp",
    version="0.0.1",
    description="Hierspeech++ pip installable",
    author="Nabarun Goswami",
    author_email="nabarungoswami@mi.t.u-tokyo.ac.jp",
    packages=["hierspeechpp"],
    install_requires=required
    # [
    #     "AMFM_decompy",
    #     "Cython",
    #     "einops",
    #     "joblib",
    #     "matplotlib",
    #     "numpy",
    #     "pesq",
    #     "phonemizer",
    #     "scipy",
    #     "timm",
    #     "torch",
    #     "torchaudio",
    #     "tqdm",
    #     "transformers",
    #     "Unicode",
    # ]
)

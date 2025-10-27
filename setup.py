from setuptools import setup, find_packages

setup(
    name='LLM_annotate',
    version='0.1.0',
    author='Hannes Rosenbusch',
    author_email='h.rosenbusch@uva.nl',
    description='A package for annotating text using LLMs',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/hannesrosenbusch/LLM_annotate',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'openai',
        'jsonschema',
        'mistralai',
        'tiktoken'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
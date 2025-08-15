from setuptools import setup, find_packages

setup(
    name='luvia',
    version='0.1',
    packages=find_packages(where='src'),
    install_requires=[],
    include_package_data=True,
    package_dir={'': 'src'},
    entry_points={
        'console_scripts': [
            'luvia=luvia.main:main',
        ],
    },
)


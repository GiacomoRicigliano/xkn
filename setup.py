from setuptools import setup, find_packages

setup(
    name='xkn',
    version='0.3',
    author='Giacomo Ricigliano',
    author_email='giacomo.ricigliano@gmail.com',
    description='Semi-analytic KN framework',
    url='https://github.com/GiacomoRicigliano/xkn',
    include_package_data=True,
    packages = find_packages(),
    package_data = {
        'xkn.filter_dictionary':['*.json'],
        'xkn.flux_factor_data':['*.dat'],
        'xkn.interp_tables':['hires_sym0_results', '*.dat', '*.npz'],
                   },
    classifiers=[
        'Programming Language :: Python :: 3',
                ],
    python_requires='>=3.7',
      )


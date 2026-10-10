
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
a = generate_distutils_setup(
    packages=['predict_eval'],
    package_dir={'': 'src'},
    )
setup(**a)

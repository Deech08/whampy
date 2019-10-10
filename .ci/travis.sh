# Using file formatting based on dfm/emcee 
#!/bin/bash -x

# http://conda.pydata.org/docs/travis.html#the-travis-yml-file
if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
else
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
fi
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

# Conda Python
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create --yes -n test python=$PYTHON_VERSION
source activate test
conda install -q numpy=$NUMPY_VERSION setuptools pytest pytest-cov pip sphinx matplotlib cartopy seaborn
pip install coveralls
pip install astropy
pip install pytest-mpl
pip install spectral-cube
pip install --no-deps pyregion
pip install regions
pip install extinction
pip install dustmaps
pip install pandas
pip install spectral_cube
# Build the extension
python setup.py develop

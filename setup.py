import os
from typing import Dict, List

from setuptools import find_packages, setup  # NOQA

setup_requires: List[str] = []
install_requires: List[str] = [
    "ase>=3.18, <4.0.0",  # Note that we require ase==3.21.1 for pytest.
    "pymatgen>=2020.1.28",
]
extras_require: Dict[str, List[str]] = {
    "develop": ["pysen[lint]==0.10.5", "ase==3.21.1"],
}


__version__: str
here = os.path.abspath(os.path.dirname(__file__))
# Get __version__ variable
exec(open(os.path.join(here, "torch_dftd", "_version.py")).read())

package_data = {"torch_dftd": ["nn/params/dftd3_params.npz"]}

setup(
    name="torch-dftd",
    version=__version__,  # NOQA
    description="pytorch implementation of dftd2 & dftd3",
    packages=find_packages(),
    setup_requires=setup_requires,
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    package_data=package_data,
)

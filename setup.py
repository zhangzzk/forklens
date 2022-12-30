from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Forklens'
LONG_DESCRIPTION = 'The first public version for a shear measurement code with CNN'

packages = ['forklens']
package_dir = {'forklens':'./src'}

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="forklens", 
        version=VERSION,
        author="Zekang Zhang",
        author_email="zhangzekang@shao.ac.cn",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        #packages=find_packages('forklens'),
        packages=['forklens'],
        package_dir = package_dir,
        # install_requires=['galsim==2.3.4', 'torch==1.11.0'], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'forklens'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            # "Intended Audience :: Education",
            # "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: Linux :: Ubuntu",
            # "Operating System :: Microsoft :: Windows",
        ]
)
from setuptools import setup
from glob import glob

package_name = 'gui'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/templates', glob('templates/*')),
        ('share/' + package_name + '/static', glob('static/*')),
    ],
    install_requires=['setuptools', 'flask', 'flask-socketio'],
    zip_safe=True,
    maintainer='liza',
    maintainer_email='liza@example.com',
    description='Flask GUI package for hand gesture robot control',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'app = gui.app:main',
        ],
    },
)

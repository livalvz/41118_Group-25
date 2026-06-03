from setuptools import setup

package_name = 'hand_gesture'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='liza',
    maintainer_email='liza@example.com',
    description='Hand gesture recognition package for robot driving commands',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gesture_recognition_node = hand_gesture.gesture_recognition_node:main',
            'collect_landmark = hand_gesture.collect_landmark:main',
            'predict_live = hand_gesture.predict_live:main',
            'train_classifier = hand_gesture.train_classifier:main',
            'hand_landmark_test = hand_gesture.hand_landmark_test:main',
        ],
    },
)

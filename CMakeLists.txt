cmake_minimum_required(VERSION 3.0.2)
project(recorder_plugin)

find_package(catkin REQUIRED COMPONENTS
  rospy
  rqt_gui
  rqt_gui_py
  std_msgs
)

catkin_python_setup()

catkin_package()

install(FILES
  plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY resource
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(PROGRAMS
  src/recorder_plugin/main.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


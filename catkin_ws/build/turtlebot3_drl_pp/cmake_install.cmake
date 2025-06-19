# Install script for directory: /home/zsw/catkin_ws/src/turtlebot3_drl_pp

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/zsw/catkin_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/turtlebot3_drl_pp.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/turtlebot3_drl_pp/cmake" TYPE FILE FILES
    "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/turtlebot3_drl_ppConfig.cmake"
    "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/turtlebot3_drl_ppConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/turtlebot3_drl_pp" TYPE FILE FILES "/home/zsw/catkin_ws/src/turtlebot3_drl_pp/package.xml")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/obstacle_controller_1")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/obstacle_controller_2")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/obstacle_controller_3")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/obstacle_controller_4")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/stage_0_ddpg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/stage_1_ddpg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/stage_2_ddpg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/stage_1_ddpg_test")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/stage_2_ddpg_test")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/stage_2_ddpg_fg_test")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/stage_3_ddpg_fg_test")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/stage_0_td3")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/stage_1_td3")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/stage_2_td3")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/stage_1_td3_test")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/stage_2_td3_test")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/core.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/core.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/icm.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/core.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/core.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot3_drl_pp" TYPE PROGRAM FILES "/home/zsw/catkin_ws/build/turtlebot3_drl_pp/catkin_generated/installspace/rb.py")
endif()


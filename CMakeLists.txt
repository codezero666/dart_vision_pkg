cmake_minimum_required(VERSION 3.8)
project(dart_vision_pkg)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(geometry_msgs REQUIRED) 
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
  
# Add source files
set(SOURCES
    src/DartVision.cpp
)

# Create an executable
add_executable(DartVision ${SOURCES})

ament_target_dependencies(DartVision
  rclcpp
  std_msgs
  geometry_msgs
  cv_bridge
  OpenCV
)

install(TARGETS
DartVision
  DESTINATION lib/${PROJECT_NAME})

ament_package()

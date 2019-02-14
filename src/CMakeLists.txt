add_library( slam_share SHARED
System.cpp
Viewer.cpp
Map.cpp
Tracking.cpp
FrameDrawer.cpp
ORBextractor.cpp
Frame.cpp
KeyFrame.cpp
MapPoint.cpp
KeypointMatcher.cpp
Optimization.cpp
MapDrawer.cpp
LocalMap.cpp
PointCloudMap.cpp
)


add_executable(Viewer main.cpp)
target_link_libraries(Viewer slam_share
${catkin_LIBRARIES}
${OpenCV_LIBRARIES}
${PCL_LIBRARIES}
${kinect2_bridge_LIBRARIES}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/include/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/include/Thirdparty/g2o/lib/libg2o.so

)


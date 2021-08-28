#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <g2o/core/optimizable_graph.h>
#include <boost/smart_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <slam_datatypes/framepoint.hpp>
#include <slam_datatypes/camera.hpp>

struct Frame{
    cv::Mat image_l, image_r; //< raw image data
    Eigen::Transform<float,3,2> T_cam2world, T_world2cam; // Eigen 3D affine transform to represent pose
    std::vector<Framepoint> points;  // framepoints owned by the Frame
    Camera camera_l;
    Camera camera_r;
};
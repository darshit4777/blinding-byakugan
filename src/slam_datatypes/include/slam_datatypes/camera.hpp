#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <g2o/core/optimizable_graph.h>
#include <boost/smart_ptr.hpp>
#include <boost/shared_ptr.hpp>

struct Camera{
       // Calibrated camera object
       Eigen::Matrix3f intrinsics;
       std::vector<float> distortion_coeffs;
};
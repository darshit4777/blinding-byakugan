#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <g2o/core/optimizable_graph.h>
#include <boost/smart_ptr.hpp>
#include <boost/shared_ptr.hpp>

struct KeypointWD{
       // Keypoint with a descriptor
       cv::KeyPoint keypoint;
       cv::Mat descriptor; 
};
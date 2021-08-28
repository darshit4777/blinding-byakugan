#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <g2o/core/optimizable_graph.h>
#include <boost/smart_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <slam_datatypes/framepoint.hpp>

// Forward Declaration 
struct Framepoint;


class Landmark{
    // Landmark hold the information of multiple framepoints and their world location
    public:
    Eigen::Vector3f world_coordinates;
    boost::shared_ptr<Framepoint> origin;
    Eigen::Matrix3f omega;
    Eigen::Vector3f nu;
};
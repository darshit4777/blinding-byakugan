#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <g2o/core/optimizable_graph.h>
#include <boost/smart_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <slam_datatypes/keypoint.hpp>
#include <slam_datatypes/landmark.hpp>

// Forward Declaration
struct Landmark;

struct Framepoint{
    // A framepoint holds the information of a keypoint and its track
    KeypointWD keypoint_l, keypoint_r;
    Eigen::Vector3f camera_coordinates;
    Eigen::Vector3f world_coordinates;
    //Framepoint *next, *previous;
    //boost::shared_ptr<Framepoint> next,previous;
    Framepoint *next,*previous;
 
    bool inlier;
    //boost::shared_ptr<Landmark> associated_landmark;
    boost::shared_ptr<Landmark> associated_landmark;
    boost::shared_ptr<Frame> parent_frame; //<< Whose your daddy? haha
    bool landmark_set;
};
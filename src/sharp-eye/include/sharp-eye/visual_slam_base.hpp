#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <g2o/core/optimizable_graph.h>
#include <boost/smart_ptr.hpp>
#include <boost/shared_ptr.hpp>
class VisualSlamBase{
    /**
     * Base class to define custom data types for Visual SLAM. 
     * All of the datatypes are implementations influenced by the ProSLAM paper
    */

   public:
   struct KeypointWD{
       // Keypoint with a descriptor
       cv::KeyPoint keypoint;
       cv::Mat descriptor; 
   };
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
       bool landmark_set;
   };

   class Landmark{
       // Landmark hold the information of multiple framepoints and their world location
       public:
       Eigen::Vector3f world_coordinates;
       boost::shared_ptr<Framepoint> origin;
       Eigen::Matrix3f omega;
       Eigen::Vector3f nu;
   };
   struct Camera{
       // Calibrated camera object
       Eigen::Matrix3f intrinsics;
       std::vector<float> distortion_coeffs;
   } camera_l, camera_r;

   struct Frame{
       cv::Mat image_l, image_r; //< raw image data
       Eigen::Transform<float,3,2> T_cam2world, T_world2cam; // Eigen 3D affine transform to represent pose
       std::vector<Framepoint> points;  // framepoints owned by the Frame
       Camera camera_l;
       Camera camera_r;
   };
};
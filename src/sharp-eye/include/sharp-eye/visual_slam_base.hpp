#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <g2o/core/optimizable_graph.h>
class VisualSlamBase{
    /**
     * Base class to define custom data types for Visual SLAM. 
     * All of the datatypes are implementations influenced by the ProSLAM paper
    */

   public:
   struct KeypointWD{
       // Keypoint with a descriptor
       int pixel_x;  
       int pixel_y; 
       cv::Mat descriptor; 
       double response; 
   };

   typedef struct framepoint Framepoint;
   typedef struct landmark Landmark;


   struct Framepoint{
       // A framepoint holds the information of a keypoint and its track
       KeypointWD keypoint_l, keypoint_r;
       Eigen::Vector3d camera_coordinates;
       Eigen::Vector3d world_coordinates;
       Framepoint *next, *previous;
       bool inlier;
       Landmark *associated_landmark;
   };

   struct Landmark{
       // Landmark hold the information of multiple framepoints and their world location
       Eigen::Vector3d world_coordinates;
       Framepoint *origin;
       Eigen::Matrix3d omega;
       Eigen::Vector3d nu;
   };
   struct Camera{
       // Calibrated camera object
       cv::Mat intrinsics;
       std::vector<double> distortion_coeffs;
   } camera_l, camera_r;

   struct Frame{
       cv::Mat image_l, image_r; //< raw image data
       Eigen::Transform<double,3,2> T_cam2world, T_world2cam; // Eigen 3D affine transform to represent pose
       std::vector<Framepoint> points;  // framepoints owned by the Frame
   };

   struct LocalMap{
       Eigen::Transform<double,3,2> T_map2world, T_world2map;
       std::vector<Frame> frames; // enclosed frames
       std::vector<Landmark> associated_landmarks; // enclosed Landmarks
   };

   struct WorldMap{
       Eigen::Transform<double,3,2> T_map2world, T_world2map;
       std::vector<LocalMap> local_maps;
       std::vector<Landmark> enclosed_landmarks;
       g2o::OptimizableGraph pose_graph;
   };
};
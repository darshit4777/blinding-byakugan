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
       Landmark* associated_landmark;
       bool landmark_set;
   };

   struct Landmark{
       // Landmark hold the information of multiple framepoints and their world location
       Eigen::Vector3f world_coordinates;
       //boost::shared_ptr<Framepoint> origin;
       Framepoint* origin;
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

   class LocalMap{
       typedef boost::shared_ptr<Frame> FrameSharedPtr;
       typedef boost::shared_ptr<Landmark> LandmarkSharedPtr;
       typedef std::vector<VisualSlamBase::Framepoint> FramepointVector;
       public:
       Eigen::Transform<float,3,2> T_map2world, T_world2map;
       std::vector<FrameSharedPtr> frames; // enclosed frames
       std::vector<LandmarkSharedPtr> associated_landmarks; // enclosed Landmarks

       public:
       /**
        * @brief Construct a new Local Map object
        * 
        */
       LocalMap();

       /**
        * @brief Create a New Frame object
        * 
        * @param T_world2cam 
        */
       void CreateNewFrame(cv::Mat image_left,cv::Mat image_right,FramepointVector fp_vector,Eigen::Transform<float,3,2> T_world2cam = Eigen::Transform<float,3,2>::Identity());

       /**
        * @brief Add the landmark to the std vector of landmarks
        * 
        * @param landmark_ptr 
        */
       void AddLandmark(LandmarkSharedPtr landmark_ptr);
       
       /**
        * @brief Get the Last Frame object
        * 
        * @return Frame* 
        */
       Frame* GetLastFrame();

       /**
        * @brief Get the Previous Frame object
        * 
        * @return Frame* 
        */
       Frame* GetPreviousFrame();

       /**
        * @brief Destroy the Local Map object
        * 
        */
       ~LocalMap();
   };

   class WorldMap{

       typedef boost::shared_ptr<LocalMap> LocalMapSharedPtr;  
       typedef boost::shared_ptr<Landmark> LandmarkSharedPtr;
       public:
       Eigen::Transform<float,3,2> T_map2world, T_world2map;
       std::vector<LocalMapSharedPtr> local_maps;
       std::vector<LandmarkSharedPtr> enclosed_landmarks;
       g2o::OptimizableGraph* pose_graph_ptr;

       /**
        * @brief Construct a new World Map object
        * 
        */
       WorldMap();
       /**
        * @brief Destroy the World Map object
        * 
        */
       ~WorldMap();

       /**
        * @brief Creates a New Local Map object
        * and stored the pointer to access it.
        * 
        */
       LocalMapSharedPtr CreateNewLocalMap(Eigen::Transform<float,3,2> T_map2world = Eigen::Transform<float,3,2>::Identity());

       /**
        * @brief Adds a landmark to the list of landmarks
        * 
        */
       void AddLandmark(LandmarkSharedPtr landmark_ptr);

       VisualSlamBase::LocalMap* GetLastLocalMap();
   };
};
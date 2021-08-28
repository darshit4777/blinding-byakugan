#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <g2o/core/optimizable_graph.h>
#include <boost/smart_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <slam_datatypes/frame.hpp>
#include <slam_datatypes/landmark.hpp>
#include <slam_datatypes/framepoint.hpp>

class LocalMap{
    typedef boost::shared_ptr<Frame> FrameSharedPtr;
    typedef boost::shared_ptr<Landmark> LandmarkSharedPtr;
    typedef std::vector<Framepoint> FramepointVector;
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
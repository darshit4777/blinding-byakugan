#pragma once
#include <sharp-eye/visual_slam_base.hpp>

typedef std::vector<VisualSlamBase::KeypointWD> FeatureVector;
typedef std::vector<VisualSlamBase::Framepoint> FramepointVector;
typedef std::vector<VisualSlamBase::Frame> FrameVector;
typedef std::vector<std::pair<VisualSlamBase::KeypointWD,VisualSlamBase::KeypointWD>> MatchVector;
typedef VisualSlamBase::Camera Camera;


class VisualTracking{
    /**
     * @brief Visual Tracking class provides methods to track framepoints through 
     * multiple frames, to calculate and optimize pose estimates from such captured
     * framepoints.
     * 
     */

    public:
    
    // Vector to store frames
    FrameVector frames;
    Camera camera_left;
    Camera camera_right;

    // Image Size
    int img_height;
    int img_width;

    // Descriptor Matcher
    cv::FlannBasedMatcher matcher;
    
    /**
     * @brief Construct a new Visual Tracking object
     * Takes the camera specifics as arguments
     * @param camera_left 
     * @param camera_right 
     */
    VisualTracking(Camera &cam_left,Camera &cam_right);

    /**
     * @brief Finds corresponding framepoints between the previous and current frames
     * 
     * @param previous_frame 
     * @param current_frame 
     * @return FramepointVector 
     */
    FramepointVector FindCorrespondences(FramepointVector &previous_frame,FramepointVector &current_frame);

};
#pragma once
#include <sharp-eye/visual_slam_base.hpp>
#include <chrono>
typedef std::vector<VisualSlamBase::KeypointWD> FeatureVector;
typedef std::vector<VisualSlamBase::Framepoint> FramepointVector;
typedef std::vector<VisualSlamBase::Frame> FrameVector;
typedef std::vector<std::pair<VisualSlamBase::KeypointWD,VisualSlamBase::KeypointWD>> MatchVector;
typedef VisualSlamBase::Camera Camera;
typedef std::chrono::high_resolution_clock _Clock; 
typedef std::chrono::_V2::high_resolution_clock::time_point _Time;



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

    struct TimeDerivative{
        Eigen::Transform<double,3,2> deltaT;
        _Time prediction_call;
        _Time differential_call;
        double differential_interval;
        double prediction_interval;
        bool clock_set;
    } pose_derivative;
    
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

    /**
     * @brief Calculates and returns the Jacobian matrix of the projection equation. 
     * The jacobian is of the form D h(exp(x) * T_w2c * cam_coordinates) / dx 
     * The jacobian thus is that of a projection equation and will be calculated using the chain rule
     * The first derivative in the chain will be D( h(A) )/ DA - where A = T_w2c * cam_coordinates
     * The second derivative is of the form D(exp(x) * A)/Dx - which is a standard form.
     * 
     * @param frame_ptr 
     * @param cam_coordinates << Coordinates of the specific framepoint in the camera frame of reference
     * @param T_w2c << Estimate of the pose of the camera in the world
     * @return Eigen::Matrix<double,4,6> 
     */
    Eigen::Matrix<double,4,6> FindJacobian(Eigen::Vector3d& left_cam_coordinates,Eigen::Vector3d& right_cam_coordinates,Camera& camera_l,Camera& camera_r);

    /**
     * @brief Estimates the new pose after a change in motion by solving an optimization problem
     * between two sets of corresponding 3D points.
     * 
     * @param frame_ptr 
     * @return Eigen::Transform<double,3,2> 
     */
    Eigen::Transform<double,3,2> EstimateIncrementalMotion(VisualSlamBase::Frame &frame_ptr);

    /**
     * @brief Calculates the jacobian of the motion wrt time.
     * Is used in the motion model to predict the next pose
     * 
     * @param current_frame_ptr 
     * @param previous_frame_ptr
     * @return TimeDerivative 
     */
    TimeDerivative CalculateMotionJacobian(VisualSlamBase::Frame* current_frame_ptr,VisualSlamBase::Frame* previous_frame_ptr);

    /**
     * @brief Calculates the predicted pose using a constant 
     * velocity motion model
     * 
     * @param frame_ptr 
     * @param time_derivative 
     * @return Eigen::Transform<double,3,2> 
     */
    Eigen::Transform<double,3,2> CalculatePosePrediction(VisualSlamBase::Frame* frame_ptr, TimeDerivative time_derivative);

    /**
     * @brief Sets the Prediction Call Time 
     * 
     * @param time_derivative_ptr 
     */
    void SetPredictionCallTime(TimeDerivative* time_derivative_ptr);

    /**
     * @brief Sets the Differential Call Time 
     * 
     * @param time_derivative_ptr 
     */
    void SetDifferentialCallTime(TimeDerivative* time_derivative_ptr);

};
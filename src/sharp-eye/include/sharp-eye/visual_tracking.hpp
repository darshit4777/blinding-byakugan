#pragma once
#include<slam_datatypes/slam_datatypes.hpp>
#include<sharp-eye/pose_optimizer.hpp>
#include <chrono>
typedef std::vector<KeypointWD> FeatureVector;
typedef std::vector<Framepoint> FramepointVector;
typedef std::vector<Framepoint> FramepointVector;
typedef std::vector<boost::shared_ptr<Framepoint>> FramepointPointerVector;
typedef std::vector<std::pair<KeypointWD,KeypointWD>> MatchVector;
typedef Camera Camera;
typedef std::chrono::high_resolution_clock _Clock; 
typedef std::chrono::_V2::high_resolution_clock::time_point _Time;
typedef std::vector<boost::shared_ptr<Landmark>> LandmarkPointerVector;


class VisualTracking{
    /**
     * @brief Visual Tracking class provides methods to track framepoints through 
     * multiple frames, to calculate and optimize pose estimates from such captured
     * framepoints.
     * 
     */

    public:
    
    // Vector to store frames
    WorldMap map;
    Camera camera_left;
    Camera camera_right;

    // Imaages
    cv::Mat img_l;
    cv::Mat img_r;

    // Image Size
    int img_height;
    int img_width;

    // Descriptor Matcher
    cv::FlannBasedMatcher matcher;

    // Framepoint Vector
    FramepointPointerVector framepoint_vec;
    FramepointPointerVector lost_points;

    // Correspondences
    int frame_correspondences;

    // Pose optimizer
    PoseOptimizer* optimizer;

    // Landmarks
    LandmarkPointerVector actively_tracked_landmarks;
    struct ManifoldDerivative{
        Eigen::Transform<float,3,2> deltaT;
        _Time prediction_call;
        _Time previous_prediction_call;
        _Time previous_differential_call;
        float differential_interval;
        float prediction_interval;
        bool clock_set;
        bool deltaT_set;
    } state_jacobian;
    
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
     * @return int << returns the number of correspondences 
     */
    int FindCorrespondences(FramepointPointerVector& previous_frame,FramepointPointerVector& current_frame);

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
     * @return Eigen::Matrix<float,4,6> 
     */
    Eigen::Matrix<float,4,6> FindJacobian(Eigen::Vector3f& left_cam_coordinates,Eigen::Vector3f& right_cam_coordinates,Camera& camera_l,Camera& camera_r,float omega);

    /**
     * @brief Estimates the new pose after a change in motion by solving an optimization problem
     * between two sets of corresponding 3D points.
     * 
     * @param frame_ptr 
     * @return Eigen::Transform<float,3,2> 
     */
    Eigen::Transform<float,3,2> EstimateIncrementalMotion();

    /**
     * @brief Calculates the jacobian of the motion wrt time.
     * Is used in the motion model to predict the next pose
     * 
     * @param current_frame_ptr 
     * @param previous_frame_ptr
     * @return TimeDerivative 
     */
    ManifoldDerivative CalculateMotionJacobian(Frame* current_frame_ptr,Frame* previous_frame_ptr);

    /**
     * @brief Calculates the predicted pose using a constant 
     * velocity motion model
     * 
     * @param frame_ptr 
     * @param time_derivative 
     * @return Eigen::Transform<float,3,2> 
     */
    Eigen::Transform<float,3,2> CalculatePosePrediction(Frame* frame_ptr);

    /**
     * @brief Sets the Prediction Call Time 
     * 
     * @param time_derivative_ptr 
     */
    void SetPredictionCallTime();

    /**
     * @brief Initializes a new Frame or a new Local Map based on the availability of 
     * previous correspondences or pose derivatives
     * 
     */
    void InitializeNode();

    void SetFramepointVector(FramepointVector& framepoints);
    
    bool HasInf(Eigen::Vector3f vec);

    void CreateAndUpdateLandmarks(Frame* current_frame_ptr,LocalMap* lmap_ptr);
    
    void RecoverLostPoints(Frame* current_frame_ptr);

    void RANSACOutlierRejection();
    
    private:
    void InitializeStateJacobian();

    

};
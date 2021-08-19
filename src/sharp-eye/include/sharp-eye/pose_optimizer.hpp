#pragma once
#include <sharp-eye/visual_slam_base.hpp>
typedef std::vector<VisualSlamBase::KeypointWD> FeatureVector;
typedef std::vector<VisualSlamBase::Framepoint> FramepointVector;
typedef std::vector<VisualSlamBase::Frame> FrameVector;
typedef std::vector<std::pair<VisualSlamBase::KeypointWD,VisualSlamBase::KeypointWD>> MatchVector;
typedef VisualSlamBase::Camera Camera;

class PoseOptimizer{
    public:
    
    struct params{
        bool ignore_outliers;
        double kernel_maximum_error;
        double minimum_depth;
        double maximum_depth;
        double maximum_reliable_depth;
        int max_iterations;

        // Movement Thresholds
        double angular_delta;
        double translation_delta;

        // Correspondences
        int min_correspondences;

        // Inliers
        int min_inliers;

        // Inter camera transform
        Eigen::Transform<double,3,2> T_caml2camr;

    } parameters;

    // Error
    double iteration_error;
    double total_error;
    
    // Optimization variables
    Eigen::Transform<double,3,2> T_prev2curr;
    Eigen::Matrix<double,6,6> H;
    Eigen::VectorXd b;
    Eigen::Matrix4d omega;
    Eigen::Vector4d reproj_error;
    double translation_factor;

    // Inliers
    int measurements;
    int inliers;

    // Frame Pointers
    VisualSlamBase::Frame* current_frame_ptr;
    VisualSlamBase::Frame* previous_frame_ptr;

    // Local Map
    VisualSlamBase::LocalMap* lmap_ptr;

    // Camera Coordinates
    Eigen::Vector3d p_caml,p_camr;

    /**
     * @brief Creates a pose optimizer object and initializes all parameters
     * 
     */
    PoseOptimizer();

    ~PoseOptimizer();

    /**
     * @brief Initializes all the pose optimizer variables
     * 
     */
    void Initialize(VisualSlamBase::Frame* current_frame_ptr,VisualSlamBase::Frame* previous_frame_ptr);

    /**
     * @brief Runs the optimization loop once
     * 
     */
    void OptimizeOnce();

    /**
     * @brief Uses the optimize once along with convergence conditions
     * to generate a solution
     * 
     */
    void Converge();

    private:
    /**
     * @brief Extracts the camera coordinates from previous frame
     * checks for invalid points and then computes the reprojection error
     * 
     */
    void ComputeError(VisualSlamBase::Framepoint& fp);
    
    /**
     * @brief Assembles the H, b and omega matrices
     * 
     */
    void Linearize(VisualSlamBase::Framepoint& fp);

    /**
     * @brief Solves for Dx and transforms it into the SE3 form
     * 
     */
    void Solve();

    /**
     * @brief Updates the T_prev2current matrix and ensures the rotations 
     * are correct
     * 
     */
    void Update();

    bool HasInf(Eigen::Vector3d vec);

    Eigen::Matrix<double,4,6> FindJacobian(Eigen::Vector3d& left_cam_coordinates,Eigen::Vector3d& right_cam_coordinates,Camera& camera_l,Camera& camera_r,double omega);

};
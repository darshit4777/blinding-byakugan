#pragma once
#include <slam_datatypes/slam_datatypes.hpp>
typedef std::vector<KeypointWD> FeatureVector;
typedef std::vector<Framepoint> FramepointVector;
typedef std::vector<Frame> FrameVector;
typedef std::vector<std::pair<KeypointWD,KeypointWD>> MatchVector;
typedef Camera Camera;

class PoseOptimizer{
    public:
    
    struct params{
        bool ignore_outliers;
        float kernel_maximum_error;
        float minimum_depth;
        float maximum_depth;
        float maximum_reliable_depth;
        int max_iterations;

        // Movement Thresholds
        float angular_delta;
        float translation_delta;

        // Correspondences
        int min_correspondences;

        // Inliers
        int min_inliers;

        // Inter camera transform
        Eigen::Transform<float,3,2> T_caml2camr;

    } parameters;

    // Error
    float iteration_error;
    float total_error;
    
    // Optimization variables
    Eigen::Transform<float,3,2> T_prev2curr;
    Eigen::Matrix<float,6,6> H;
    Eigen::VectorXf b;
    Eigen::Matrix4f omega;
    Eigen::Vector4f reproj_error;
    float translation_factor;

    // Inliers
    int measurements;
    int inliers;

    // Frame Pointers
    Frame* current_frame_ptr;
    Frame* previous_frame_ptr;

    // Local Map
    LocalMap* lmap_ptr;

    // Camera Coordinates
    Eigen::Vector3f p_caml,p_camr;

    // Opencv Named Windows
    std::string left_cam; 
    std::string right_cam;


    private:
    bool compute_success;
    cv::Mat _img_left;
    cv::Mat _img_right;

    public:
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
    void Initialize(Frame* curr_frame_ptr,Frame* prev_frame_ptr,LocalMap* local_map_ptr);

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

    /**
     * @brief Visualization function to draw a framepoint on an image
     * 
     * @param image 
     */
    void VisualizeFramepoints(FramepointVector fp_vec,cv::Mat& image,int cam,cv::Scalar color);

    /**
     * @brief Compare Framepoints drawn on two images. Useful for checking correspondences
     * 
     * @param fp_1 
     * @param image_1 
     * @param fp_2 
     * @param image_2 
     */
    void VisualizeFramepointComparision(FramepointVector fp_vec1,cv::Mat& image_1,FramepointVector fp_vec2,cv::Mat& image_2);

    /**
     * @brief Visualize two groups of framepoints in a single image, useful for seeing
     * optimization convergence
     * 
     * @param fp_1 
     * @param fp_2 
     * @param image 
     */
    void VisualizeMultiFramepointComparision(FramepointVector fp_vec1,FramepointVector fp_vec2,cv::Mat image);

    private:
    /**
     * @brief Extracts the camera coordinates from previous frame
     * checks for invalid points and then computes the reprojection error
     * 
     */
    void ComputeError(Framepoint fp);
    
    /**
     * @brief Assembles the H, b and omega matrices
     * 
     */
    void Linearize(Framepoint fp);

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

    bool HasInf(Eigen::Vector3f vec);

    Eigen::Matrix<float,4,6> FindJacobian(Eigen::Vector3f& left_cam_coordinates,Eigen::Vector3f& right_cam_coordinates,Camera& camera_l,Camera& camera_r,float omega);

};
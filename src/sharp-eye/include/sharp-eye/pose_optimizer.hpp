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
        float solver_maximum_error;
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

    struct ransac_parameters{
        int n; //< No of datapoints
        float p; //< probability of success 
        float e; //< Ratio of outliers to datapoints
        int s; //< Min no of points
        int t; //< No of trials

        int max_inliers;
        std::vector<int> valid_point_indices;
        std::vector<int> inlier_vector_indices;
    } ransac_params;

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
     * @brief Runs the optimization loop once.
     * 
     * @param frame_ptr //< The frame on which optimization is to be run
     */
    void OptimizeOnce(Frame* frame_ptr);

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
    float ComputeError(Framepoint* fp,bool check_landmark = true);
    
    /**
     * @brief Assembles the H, b and omega matrices
     * 
     */
    void Linearize(Framepoint* fp);

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

    /**
     * @brief Checks if the vector has an "inf"
     * 
     * @param vec 
     * @return true 
     * @return false 
     */
    bool HasInf(Eigen::Vector3f vec);

    /**
     * @brief Calculates the jacobian matrix
     * 
     * @param left_cam_coordinates 
     * @param right_cam_coordinates 
     * @param camera_l 
     * @param camera_r 
     * @param omega 
     * @return Eigen::Matrix<float,4,6> 
     */
    Eigen::Matrix<float,4,6> FindJacobian(Eigen::Vector3f& left_cam_coordinates,Eigen::Vector3f& right_cam_coordinates,Camera& camera_l,Camera& camera_r,float omega);

    /**
     * @brief Initializes RANSAC 
     * 
     * @param current_frame_ptr 
     * @param p //< the probability of success of finding a single set with no outliers. Default 0.99.
     * @param e //< the ratio of outliers to datapoints. Default 0.1.
     * @param s //< the minimum number of points required to define a model. Default 3. 
     */

    public:
    void InitializeRANSAC(Frame* current_frame_ptr, float p = 0.99, float e = 0.1, int s = 3);

    /**
     * @brief Chooses a random subset of points, computes a model and inliers
     * for the model.  
     * 
     * @return returns the number of inliers obtained
     */

    int RANSACIterateOnce();

    /**
     * @brief Perform a single RANSAC iteration T times. Where T is the number 
     * of trials calculated during initialization. 
     * 
     */
    void RANSACConverge();

    

};
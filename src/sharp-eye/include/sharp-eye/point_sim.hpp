#include <slam_datatypes/slam_datatypes.hpp>

class PointSim{
    /**
     * @brief Tiny simulator to create random points that can be projected onto images
     * Will be used for testing the pose optimization algorithm for incremental 
     * pose estimation
     * 
     */

    public:
    int n; //< Number of points
    std::vector<Eigen::Vector3f> point_list;
    float d; //< Size of the cube in which the points will be simulated.

    // Transforms
    Eigen::Transform<float,3,2> T_world2cam1;
    Eigen::Transform<float,3,2> T_world2cam2;
    Eigen::Transform<float,3,2> T_cam12cam2;
    Eigen::Transform<float,3,2> T_caml2camr;

    // Frames
    Frame current_frame;
    Frame previous_frame;

    // Cameras
    Camera camera_l;
    Camera camera_r;

    /**
     * @brief Construct a new Point Sim object
     * 
     */
    PointSim();

    /**
     * @brief Create Cameras
     * 
     */
    void CreateCameras(Camera& camera_left,Camera& camera_right);

    /**
     * @brief Set the positions of the cameras ie, their T_world2cam transforms.
     * 
     * @param camera_initial 
     * @param camera_final 
     */
    void SetCameraPositions(Eigen::Transform<float,3,2> camera_initial,Eigen::Transform<float,3,2> camera_final);

    /**
     * @brief Sets the T_caml2camr transform
     * 
     */
    void SetInterCameraTransform(Eigen::Transform<float,3,2> transform);

    /**
     * @brief Set the initial position of the camera and the change in its position ie T_world2cam1 and T_cam22cam1
     * 
     * @param camera_initial 
     * @param transform 
     */
    void SetCameraPositionAndTransform(Eigen::Transform<float,3,2> camera_initial,Eigen::Transform<float,3,2> transform);
    /**
     * @brief Creates a list of random 3D points enclosed by a cube of the specified size. 
     * The cube will be centered at the origin by default
     * 
     * @param no_of_points 
     * @param cube_dimension 
     */
    void CreateRandomPoints(int no_of_points,float cube_dimension);

    /**
     * @brief Creates the curent and previous frames by projecting points, creating 
     * correspondences and arranging the points into the frame datastructures
     * 
     */
    void CreateFrames(); 

    cv::KeyPoint ProjectPoints(Eigen::Vector3f point_3d,Camera camera);

    bool InFieldOfView(cv::KeyPoint keypoint_l,cv::KeyPoint keypoint_r);
};
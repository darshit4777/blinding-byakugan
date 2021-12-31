#pragma once
#include <slam_datatypes/slam_datatypes.hpp>


class VisualTriangulation{
    /**
     * Visual triangulation class that includes methods for feature detection,
     * matchnig and triangulation required to create a point cloud from stereo
     * images.
     */


    public:    
    // Cameras
    Camera camera_l; //< Left camera intrinsics
    Camera camera_r; //< Right camera intrinsics
    float camera_baseline;
    float focal_length_x;
    float focal_length_y;

    // Feature Handling
    cv::Ptr<cv::FeatureDetector> orb_detector;
    cv::Ptr<cv::DescriptorExtractor> feature_descriptor;
    cv::Ptr<cv::FastFeatureDetector> fast_detector;

    std::vector<KeypointWD> keypoints_l;
    std::vector<KeypointWD> keypoints_r;
    int detection_threshold;

    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> keypoint_matches;

    int vertical_bins;
    int horizontal_bins;

    // Image Handling
    cv::Mat image_l;
    cv::Mat image_r;

    typedef std::vector<KeypointWD> FeatureVector;
    typedef std::vector<Framepoint> FramepointVector;
    typedef std::vector<std::pair<KeypointWD,KeypointWD>> MatchVector;

    // Methods
    /**
     * @brief Construct a new Visual Triangulation object.
     * Initializes the camera objects,feature detector and descriptor extractor.
     * 
     * @param left camera object 
     * @param right camera object
     */
    VisualTriangulation();

    /**
     * @brief Detects the ORB features in an image and retursn them in a vector
     * of keypoints
     * 
     * @param img_ptr 
     * @return std::vector<KeypointWD> 
     */
    FeatureVector DetectFeatures(cv::Mat* img_ptr,bool draw);

    /**
     * @brief Extracts the descriptors for a keypoint vector and adds them to the 
     * descriptor fields for the KeypointWD
     * 
     * @param keypoint_vec
     * @return FeatureVector with descriptors
     */
    FeatureVector ExtractKeypointDescriptors(cv::Mat* img_ptr,FeatureVector &keypoint_vec);

    /**
     * @brief Detects keypoints and computes features in a single function
     * 
     * @param img_ptr 
     * @param keypoint_vec 
     * @param draw 
     * @return FeatureVector 
     */
    FeatureVector DetectAndComputeFeatures(cv::Mat* img_ptr,FeatureVector &keypoint_vec,bool draw);
    /**
     * @brief Get the Keypoint Matches between two Keypoint/Feature Vectors
     * 
     * @param left_vec left camera features
     * @param right_vec right camera features
     * @return MatchVector 
     */
    MatchVector GetKeypointMatches(FeatureVector &left_vec, FeatureVector &right_vec);

    /**
     * @brief Generates 3D coordinates for all matched keypoints using triangulation.
     * 
     * @param matched_features 
     * @return FramepointVector 
     */
    FramepointVector Generate3DCoordinates(MatchVector &matched_features, FramepointVector &framepoints_in, float baseline, float focal_length,Eigen::Matrix3f camera_intrinsics);

    /**
     * @brief Get matches with epipolar constraints
     * 
     * @param left_vec 
     * @param right_vec 
     * @return MatchVector 
     */
    MatchVector GetEpipolarMatches(FeatureVector &left_vec, FeatureVector &right_vec);
    /**
     * @brief Destroy the Visual Triangulation object
     * 
     */
    ~VisualTriangulation();
    
};
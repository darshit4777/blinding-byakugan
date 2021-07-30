#include <sharp-eye/visual_slam_base.hpp>


class VisualTriangulation{
    /**
     * Visual triangulation class that includes methods for feature detection,
     * matchnig and triangulation required to create a point cloud from stereo
     * images.
     */


    public:    
    // Cameras
    VisualSlamBase::Camera camera_l; //< Left camera intrinsics
    VisualSlamBase::Camera camera_r; //< Right camera intrinsics
    
    // Feature Handling
    cv::Ptr<cv::FeatureDetector> orb_detector;
    cv::Ptr<cv::DescriptorExtractor> orb_descriptor;
    std::vector<VisualSlamBase::KeypointWD> keypoints_l;
    std::vector<VisualSlamBase::KeypointWD> keypoints_r;
    int detection_threshold;

    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> keypoint_matches;

    // Image Handling
    cv::Mat image_l;
    cv::Mat image_r;

    typedef std::vector<VisualSlamBase::KeypointWD> FeatureVector;
    typedef std::vector<std::pair<VisualSlamBase::KeypointWD,VisualSlamBase::KeypointWD>> MatchVector;

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
     * @return std::vector<VisualSlamBase::KeypointWD> 
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
     * @brief Get the Keypoint Matches between two Keypoint/Feature Vectors
     * 
     * @param left_vec left camera features
     * @param right_vec right camera features
     * @return MatchVector 
     */
    MatchVector GetKeypointMatches(FeatureVector &left_vec, FeatureVector &right_vec);

    /**
     * @brief Destroy the Visual Triangulation object
     * 
     */
    ~VisualTriangulation();
    
};
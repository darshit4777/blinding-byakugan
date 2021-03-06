#include <sharp-eye/utils.hpp>
#include <sharp-eye/visual_tracking.hpp>
#include <sharp-eye/visual_triangulation.hpp>
#include <sharp-eye/visual_triangulation_fixtures.hpp>
#include <gtest/gtest.h>

class VisualTrackingTest : public ::testing::Test{

    protected: 
    // Opencv image matrices
    cv::Mat image_l;
    cv::Mat image_r;
    cv::Mat undistorted_l;
    cv::Mat undistorted_r;
    
    // Image lists
    std::vector<std::vector<std::string>> cam_left_image_list;
    std::vector<std::vector<std::string>> cam_right_image_list;
    
    // Opencv Windows
    std::string OPENCV_WINDOW_LEFT; 
    std::string OPENCV_WINDOW_RIGHT;

    // Image filepaths
    std::string image_path_left;
    std::string image_path_right;

    // Triangulation
    VisualTriangulation triangulator;
    FeatureVector features_l, features_r;
    VisualTriangulation::MatchVector matches;

    float focal_length;
    float baseline;

    // Cameras
    Camera cam_left;
    Camera cam_right;
    
    // Visual Tracking Object
    VisualTracking* tracker;

    /**
     * @brief Construct a new Visual Tracking Test object
     * 
     */
    VisualTrackingTest();

    /**
     * @brief Destroy the Visual Tracking Test object
     * 
     */
    ~VisualTrackingTest();

    /**
     * @brief Setup function for the test fixture
     * 
     */
    void SetUp() override;

    /**
     * @brief Tear down function for the test fixture
     * 
     */
    void TearDown() override;

    /**
     * @brief Get a Frame object. Creates and returns a frame with framepoints.
     * The frame can be used for correspondence matching. The inputs are the 
     * left and right camera images.
     * 
     * @param image_l 
     * @param image_r 
     * @return Frame 
     */
    boost::shared_ptr<Frame> GetFrame(cv::Mat image_l, cv::Mat image_r);

    /**
     * @brief Get image specified by string and image idx 
     * 
     * @param image_idx 
     * @return cv::Mat 
     */
    cv::Mat GetImage(int image_idx,std::string cam);

};
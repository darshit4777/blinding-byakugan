#include <sharp-eye/utils.hpp>
#include <sharp-eye/visual_tracking.hpp>
#include <sharp-eye/visual_triangulation.hpp>
#include <sharp-eye/visual_triangulation_fixtures.hpp>
#include <gtest/gtest.h>

class VisualTrackingTest : public VisualTriangulationTest{

    protected:
    // Since image handling is taken care of by Visual Triangulation Test,
    // We focus only on members and methods that are strictly required by,
    // tracking. 

    // Triangulation
    VisualTriangulation triangulator;
    FeatureVector features_l, features_r;
    VisualTriangulation::MatchVector matches;

    float focal_length;
    float baseline;

    // Cameras
    Camera cam_left;
    Camera cam_right;

    // Local Map
    LocalMap* lmap_ptr;


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

};
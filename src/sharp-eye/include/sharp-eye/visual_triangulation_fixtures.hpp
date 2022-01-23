#pragma once
#include <opencv2/imgproc.hpp>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <sharp-eye/utils.hpp>
#include <fstream>

class VisualTriangulationTest : public ::testing::Test {
    protected:
    // You can remove any or all of the following functions if their bodies would
    // be empty.
    
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

    /**
     * @brief Construct a new Visual Triangulation Test fixture
     * 
     */
    VisualTriangulationTest();
    
    /**
     * @brief Destroy the Visual Triangulation Test fixture
     * 
     */
    ~VisualTriangulationTest();

    /**
     * @brief Set the Up fixture
     * 
     */
    void SetUp() override;

    /**
     * @brief Tear down the fixture
     * 
     */
    void TearDown() override;

    /**
     * @brief Get the Left Image 
     * 
     * @param image_idx 
     * @return cv::Mat 
     */
    cv::Mat GetImage(int image_idx,std::string cam);

};

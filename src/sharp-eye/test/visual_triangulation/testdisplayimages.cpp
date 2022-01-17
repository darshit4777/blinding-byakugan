#include <opencv2/imgproc.hpp>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <sharp-eye/utils.hpp>
#include <fstream>

// The fixture for testing class Foo.
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

    // Image indexes
    unsigned int image_idx;
    unsigned int image_idx_max;

    // Image filepaths
    std::string image_path_left;
    std::string image_path_right;

    VisualTriangulationTest() {
        
        // Create display windows
        OPENCV_WINDOW_LEFT = "Left Image window";
        OPENCV_WINDOW_RIGHT = "Right Image window";
        cv::namedWindow(OPENCV_WINDOW_LEFT);
        cv::namedWindow(OPENCV_WINDOW_RIGHT);

        // Set image indices
        image_idx = 0;
        image_idx_max = 0;

        // Set image file paths
        image_path_left ="/home/darshit/Code/blinding-byakugan/MH_01_easy/MH_01_easy.txt.d/left_camera/";
        image_path_right = "/home/darshit/Code/blinding-byakugan/MH_01_easy/MH_01_easy.txt.d/right_camera/";

        return;
    };

    ~VisualTriangulationTest() override {
     // You can do clean-up work that doesn't throw exceptions here.
    }

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    void SetUp() override {

        std::string image_list_left_file = "/home/darshit/Code/blinding-byakugan/MH_01_easy/MH_01_easy.txt.d/left_cam_fnames.txt"; 
        std::string image_list_right_file = "/home/darshit/Code/blinding-byakugan/MH_01_easy/MH_01_easy.txt.d/right_cam_fnames.txt";
        
        std::filebuf image_list_left_buf;
        std::filebuf image_list_right_buf;
    
        image_list_left_buf.open(image_list_left_file,std::ios::in);
        image_list_right_buf.open(image_list_right_file,std::ios::in);
    
        cam_left_image_list = GetImageFilenamesFromBuffer(image_list_left_buf);
        cam_right_image_list = GetImageFilenamesFromBuffer(image_list_right_buf);

        image_idx = 0;
        image_idx_max = 10;

    }

    void TearDown() override {
        cv::destroyAllWindows();
      
    }

  // Class members declared here can be used by all tests in the test suite
  // for Foo.
};

void DisplayImage(cv::Mat image, std::string window){
    cv::imshow(window,image);
    cv::waitKey(1);
    return;
}

// Tests that the Foo::Bar() method does Abc.
TEST_F(VisualTriangulationTest, DisplayImage) {
  
  while(image_idx < image_idx_max){
      std::string left_image_folder_path = "/home/darshit/Code/blinding-byakugan/MH_01_easy/MH_01_easy.txt.d/left_camera/";
      std::string left_filename = left_image_folder_path + cam_left_image_list[image_idx][0];
      image_l = GetImageFromFilename(left_filename);
      std::string right_image_folder_path = "/home/darshit/Code/blinding-byakugan/MH_01_easy/MH_01_easy.txt.d/right_camera/";
      std::string right_filename = right_image_folder_path + cam_right_image_list[image_idx][0];
      image_r = GetImageFromFilename(right_filename);
      
      DisplayImage(image_l,OPENCV_WINDOW_LEFT);
      DisplayImage(image_r,OPENCV_WINDOW_RIGHT);
      image_idx++;
  }
  return;
};

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


#include <opencv2/imgproc.hpp>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <sharp-eye/utils.hpp>
#include <fstream>
#include <sharp-eye/visual_triangulation_fixtures.hpp>

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


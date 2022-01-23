#include <opencv2/imgproc.hpp>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <sharp-eye/visual_triangulation_fixtures.hpp>

void DisplayImage(cv::Mat image, std::string window){
    cv::imshow(window,image);
    cv::waitKey(1);
    return;
}

// Tests that the Foo::Bar() method does Abc.
TEST_F(VisualTriangulationTest, DisplayImage) {
  int image_idx = 0;
  int image_idx_max = 10;

  while(image_idx < image_idx_max){
      
      image_l = GetImage(image_idx,"left");
      image_r = GetImage(image_idx,"right");

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


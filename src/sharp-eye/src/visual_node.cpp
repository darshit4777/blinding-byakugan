#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <boost/bind.hpp>

/**
 * This file will create an executable which will serve as a test node and 
 * later on will be repurposed to form the ROS layer
*/

cv::Mat image_l;
cv::Mat image_r;
 static const std::string OPENCV_WINDOW = "Image window";

void CameraCallback(const sensor_msgs::ImageConstPtr& msg,cv::Mat &image){
    // Simply store the ros image into an opencv format
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_8UC1);
      image = cv_ptr->image;
      cv::imshow(OPENCV_WINDOW,image);
      return;
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber imageSub_l = it.subscribe("camera/image", 1, boost::bind(CameraCallback,_1,image_l));
  image_transport::Subscriber imageSub_r = it.subscribe("camera/image", 1, boost::bind(CameraCallback,_1,image_r));
  ros::spin();
}

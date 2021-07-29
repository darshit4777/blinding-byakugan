#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <boost/bind.hpp>
#include <sharp-eye/visual_triangulation.hpp>

/**
 * This file will create an executable which will serve as a test node and 
 * later on will be repurposed to form the ROS layer
*/

cv::Mat image_l;
cv::Mat image_r;
bool received_l,received_r;
static const std::string OPENCV_WINDOW = "Image window";


void CameraCallback(const sensor_msgs::ImageConstPtr& msg,int cam){
    // Simply store the ros image into an opencv format
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_8UC1);
        if(cam == 0){
            image_l = cv_ptr->image;
            received_l = true;
        }
        else if(cam == 1){
            image_r = cv_ptr->image;
            received_r = true;
        };
        
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
  image_transport::Subscriber imageSub_l = it.subscribe("cam0/image_raw", 1, boost::bind(CameraCallback,_1,0));
  image_transport::Subscriber imageSub_r = it.subscribe("cam1/image_raw", 1, boost::bind(CameraCallback,_1,1));
  cv::namedWindow(OPENCV_WINDOW);

  VisualTriangulation triangulator;
  std::vector<VisualSlamBase::KeypointWD> features;
  while(ros::ok()){
      ros::spinOnce();
      if(received_l){
          triangulator.DetectFeatures(&image_l,true);
      }
      cv::imshow(OPENCV_WINDOW,image_l);
      cv::waitKey(0.1);
  }
  cv::destroyAllWindows();
  
}

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
//static const std::string OPENCV_WINDOW = "Image window";
static const std::string OPENCV_WINDOW_LEFT = "Left Image window";
static const std::string OPENCV_WINDOW_RIGHT = "Right Image window";
typedef std::vector<std::pair<VisualSlamBase::KeypointWD,VisualSlamBase::KeypointWD>> MatchVector;

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


class TestDetectFeatures{
    public:
    void TestMain(){
        

        VisualTriangulation triangulator;
        std::vector<VisualSlamBase::KeypointWD> features;
        while(ros::ok()){
            ros::spinOnce();
            if(received_l){
                triangulator.DetectAndComputeFeatures(&image_l,features,true);
                cv::imshow(OPENCV_WINDOW_LEFT,image_l);
                cv::waitKey(5);
            }
            if(received_r){
                triangulator.DetectAndComputeFeatures(&image_r,features,true);
                cv::imshow(OPENCV_WINDOW_RIGHT,image_r);
                cv::waitKey(5);
            }
        };
        cv::destroyAllWindows();
    };

    TestDetectFeatures(int argc, char** argv){
        TestMain();
    };

    ~TestDetectFeatures(){
        return;
    }
};

class TestGetMatchedKeypoints{
    public:
    TestGetMatchedKeypoints(){
        TestMain();
        return;
    }
    void TestMain(){
        VisualTriangulation triangulator;
        std::vector<VisualSlamBase::KeypointWD> features_l;
        std::vector<VisualSlamBase::KeypointWD> features_r;
        int count = 0;
        while(ros::ok()){
            ros::spinOnce();
            if(received_l){
                features_l.clear();
                features_l = triangulator.DetectAndComputeFeatures(&image_l,features_l,false);
                //count = count + 1;
            }
            if(received_r){
                features_r.clear();
                features_r = triangulator.DetectAndComputeFeatures(&image_r,features_r,false);
            }
            if(received_l && received_r){
                // Get Matches
                MatchVector matches = triangulator.GetKeypointMatches(features_l,features_r);
                DrawMatches(matches,&image_l,&image_r);
                cv::imshow(OPENCV_WINDOW_LEFT,image_l);
                cv::waitKey(5);
                cv::imshow(OPENCV_WINDOW_RIGHT,image_r);
                cv::waitKey(5);   
            }
        };
        cv::destroyAllWindows();
    }

    void DrawMatches(MatchVector matches,cv::Mat* left_img, cv::Mat* right_img){
        std::vector<cv::KeyPoint> keypoints_l;
        std::vector<cv::KeyPoint> keypoints_r;
        //std::cout<<matches.size()<<std::endl;
        if(matches.empty()){
            return;
        }
        
        for(int i = 0; i < matches.size(); i++){
            cv::KeyPoint keypoint_l;
            cv::KeyPoint keypoint_r;

            keypoint_l = matches[i].first.keypoint;
            keypoint_r = matches[i].second.keypoint;

            keypoints_l.push_back(keypoint_l);
            keypoints_r.push_back(keypoint_r);
        };
        
        // Draw the left matches
        cv::drawKeypoints(*left_img, keypoints_l, *left_img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

        // Draw the left matches
        cv::drawKeypoints(*right_img, keypoints_r, *right_img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

        return;
    };
};

int main(int argc, char **argv){
    ros::init(argc,argv,"image_listener");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber imageSub_l = it.subscribe("cam0/image_raw", 1, boost::bind(CameraCallback,_1,0));
    image_transport::Subscriber imageSub_r = it.subscribe("cam1/image_raw", 1, boost::bind(CameraCallback,_1,1));
    //TestDetectFeatures test(argc, argv);
    TestGetMatchedKeypoints();
    return 0;
}

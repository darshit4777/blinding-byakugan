#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <boost/bind.hpp>
#include <sharp-eye/visual_triangulation.hpp>
#include <sharp-eye/visual_tracking.hpp>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
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
typedef std::vector<VisualSlamBase::Framepoint> FramepointVector;

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

class TestGenerate3DCoordinates{
    public:
    double focal_length = 457.975;
    double baseline = 0.11;
    Eigen::Matrix3d cam_intrinsics;
    typedef pcl::PointXYZI PointGray;
    ros::Publisher pub;
    ros::NodeHandle nh;

    public:
    TestGenerate3DCoordinates(const ros::NodeHandle &nodehandle){

        cam_intrinsics <<  458.654,     0.0,    367.215,
                               0.0, 457.296,    248.375,
                               0.0,     0.0,        1.0;

        nh = nodehandle;
        TestMain();
        return;
    };

    void TestMain(){
        VisualTriangulation triangulator;
        std::vector<VisualSlamBase::KeypointWD> features_l;
        std::vector<VisualSlamBase::KeypointWD> features_r;
        pub = nh.advertise< sensor_msgs::PointCloud2 > ("point_cloud", 5);
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
                //std::cout<<"No of matches "<<matches.size()<<std::endl;
                FramepointVector framepoints;
                triangulator.Generate3DCoordinates(matches,framepoints,baseline,focal_length,cam_intrinsics);
                //std::cout<<"No of framepoints "<<framepoints.size()<<std::endl;
                DrawPointCloud(framepoints,&image_l);
            };
        };
        cv::destroyAllWindows();
    };
    
    void DrawPointCloud(FramepointVector &framepoint_vec,cv::Mat* image_l){
        
        // Point Cloud ROS Msg
        sensor_msgs::PointCloud2 point_cloud2;

        // Declaring the point cloud
        pcl::PointCloud<PointGray> pcl_cloud;

        // Creating the point cloud
        for(auto framepoint : framepoint_vec){
            PointGray point;
            point.x = framepoint.camera_coordinates[0];
            point.y = framepoint.camera_coordinates[1];
            point.z = framepoint.camera_coordinates[2];

            double x,y;
            x = framepoint.keypoint_l.keypoint.pt.x;
            y = framepoint.keypoint_l.keypoint.pt.y;
            cv::Scalar intensity;
            intensity = image_l->at<uchar>(x,y);
            point.intensity = intensity.val[0];

            pcl_cloud.points.push_back(point);
        };

        // Choosing a few fancy options
        pcl_cloud.is_dense = false;
        pcl_cloud.height = 1;
        pcl_cloud.width = pcl_cloud.points.size();
        //std::cout<<"Point Cloud Size "<<pcl_cloud.points.size()<<std::endl;
        pcl::toROSMsg<PointGray>(pcl_cloud,point_cloud2);
        point_cloud2.header.frame_id = "world";
        pub.publish(point_cloud2);
        return;
    };
};

class TestFindCorrespondences{
    
    public:
    Camera cam_left;
    Camera cam_right;
    TestFindCorrespondences(){
        // Initializing Camera Matrices
        cam_left.intrinsics << 458.654,     0.0,    367.215,
                               0.0, 457.296,    248.375,
                               0.0,     0.0,        1.0;

        cam_right.intrinsics << 457.587,        0.0, 379.999,
                                    0.0,    456.134, 255.238,
                                    0.05,        0.0,    1.0;                               

        TestMain();
    };

    void TestMain(){
        VisualTracking tracking(cam_left,cam_right);
        VisualTriangulation triangulator;
        std::vector<VisualSlamBase::KeypointWD> features_l;
        std::vector<VisualSlamBase::KeypointWD> features_r;

        while(ros::ok()){
            ros::spinOnce();
            if(received_l){
                features_l.clear();
                features_l = triangulator.DetectAndComputeFeatures(&image_l,features_l,false);
            }
            if(received_r){
                features_r.clear();
                features_r = triangulator.DetectAndComputeFeatures(&image_r,features_r,false);
            }
            if(received_l && received_r){
                // Get Matches
                MatchVector matches = triangulator.GetKeypointMatches(features_l,features_r);

                FramepointVector framepoints;
                triangulator.Generate3DCoordinates(matches,framepoints,0.11,457.95,cam_left.intrinsics);
                VisualSlamBase::Frame current_frame;
                current_frame.points = framepoints;
                tracking.frames.push_back(current_frame);
                current_frame.points.clear();
                if(tracking.frames.size() > 1){
                    // Skip the first frame, from the second frame onwards..
                    int previous_index = tracking.frames.size() - 2;
                    int current_index = tracking.frames.size() - 1;
                    tracking.FindCorrespondences(tracking.frames[previous_index].points,tracking.frames[current_index].points);

                }
                
            };
        };
        //cv::destroyAllWindows();
    };  
};


int main(int argc, char **argv){
    ros::init(argc,argv,"image_listener");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber imageSub_l = it.subscribe("cam0/image_raw", 1, boost::bind(CameraCallback,_1,0));
    image_transport::Subscriber imageSub_r = it.subscribe("cam1/image_raw", 1, boost::bind(CameraCallback,_1,1));
    //TestDetectFeatures test;
    //TestGetMatchedKeypoints test;
    //TestGenerate3DCoordinates test(nh);
    TestFindCorrespondences test;
    return 0;
}

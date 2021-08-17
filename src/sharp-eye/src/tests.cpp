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
cv::Mat undistorted_l;
cv::Mat undistorted_r;
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
    std::vector<VisualSlamBase::Frame> frames;
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
                frames.push_back(current_frame);
                current_frame.points.clear();
                if(frames.size() > 1){
                    // Skip the first frame, from the second frame onwards..
                    int previous_index = frames.size() - 2;
                    int current_index = frames.size() - 1;
                    int correspondences;
                    correspondences = tracking.FindCorrespondences(frames[previous_index].points,frames[current_index].points);
                    std::cout<<correspondences<<std::endl;

                }
                
            };
        };
    };  
};

class TestIncrementalMotion{
    public:

    Eigen::Transform<double,3,2> T_body2caml;
    Eigen::Transform<double,3,2> T_body2camr;
    Eigen::Transform<double,3,2> T_caml2camr;

    Camera cam_left;
    Camera cam_right;

    cv::Mat cam_l_intrinsics;
    cv::Mat cam_r_intrinsics;

    // Triangulation
    VisualTriangulation triangulator;
    std::vector<VisualSlamBase::KeypointWD> features_l;
    std::vector<VisualSlamBase::KeypointWD> features_r;

    VisualTracking* tracking;
    
    FramepointVector framepoints;
    MatchVector matches;
    
    TestIncrementalMotion(){
        // Initializing Camera Matrices
        cam_left.intrinsics << 458.654,     0.0,    367.215,
                               0.0, 457.296,    248.375,
                               0.0,     0.0,        1.0;

        cam_left.distortion_coeffs.push_back(-0.28340811);
        cam_left.distortion_coeffs.push_back(0.07395907);
        cam_left.distortion_coeffs.push_back(0.00019359);
        cam_left.distortion_coeffs.push_back(1.76187114e-05);

        cam_right.intrinsics << 457.587,        0.0, 379.999,
                                    0.0,    456.134, 255.238,
                                    0.05,        0.0,    1.0;

        cam_right.distortion_coeffs.push_back(-0.28368365);
        cam_right.distortion_coeffs.push_back(0.07451284);
        cam_right.distortion_coeffs.push_back(-0.00010473);
        cam_right.distortion_coeffs.push_back(-3.55590700e-05);

        cam_l_intrinsics = cv::Mat(3,3,cv::DataType<double>::type);    
        cam_l_intrinsics.at<double>(0,0) = cam_left.intrinsics(0,0);
        cam_l_intrinsics.at<double>(0,1) = cam_left.intrinsics(0,1);
        cam_l_intrinsics.at<double>(0,2) = cam_left.intrinsics(0,2);
        cam_l_intrinsics.at<double>(1,0) = cam_left.intrinsics(1,0);
        cam_l_intrinsics.at<double>(1,1) = cam_left.intrinsics(1,1);
        cam_l_intrinsics.at<double>(1,2) = cam_left.intrinsics(1,2);
        cam_l_intrinsics.at<double>(2,0) = cam_left.intrinsics(2,0);
        cam_l_intrinsics.at<double>(2,1) = cam_left.intrinsics(2,1);
        cam_l_intrinsics.at<double>(2,2) = cam_left.intrinsics(2,2);

        cam_r_intrinsics = cv::Mat(3,3,cv::DataType<double>::type);
        cam_r_intrinsics.at<double>(0,0) = cam_right.intrinsics(0,0);
        cam_r_intrinsics.at<double>(0,1) = cam_right.intrinsics(0,1);
        cam_r_intrinsics.at<double>(0,2) = cam_right.intrinsics(0,2);
        cam_r_intrinsics.at<double>(1,0) = cam_right.intrinsics(1,0);
        cam_r_intrinsics.at<double>(1,1) = cam_right.intrinsics(1,1);
        cam_r_intrinsics.at<double>(1,2) = cam_right.intrinsics(1,2);
        cam_r_intrinsics.at<double>(2,0) = cam_right.intrinsics(2,0);
        cam_r_intrinsics.at<double>(2,1) = cam_right.intrinsics(2,1);
        cam_r_intrinsics.at<double>(2,2) = cam_right.intrinsics(2,2);                                    

        T_body2caml.matrix()<<   0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
                                    0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
                                -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
                                0.0, 0.0, 0.0, 1.0;

        T_body2camr.matrix()<<  0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
                                0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
                                -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
                                0.0, 0.0, 0.0, 1.0;

        tracking = new VisualTracking(cam_left,cam_right);
        tracking->T_caml2camr = T_body2caml.inverse() * T_body2camr;

        TestMain();
        return;
    };

    void TestMain(){

        while(ros::ok()){
            ros::spinOnce();
            if(received_l && received_r){
                // Set the time for recieving a new frame

                // Calibrate
                UndistortImages(cam_l_intrinsics,cam_r_intrinsics,cam_left.distortion_coeffs,cam_right.distortion_coeffs,image_l,image_r);
                std::cout<<"New Frame"<<std::endl;
                tracking->SetPredictionCallTime();

                // Visual Triangulation 
                DetectFeatures();
                Calculate3DCoordinates();
                
                // Store the framepoints for tracking
                tracking->SetFramepointVector(framepoints);

                // Initialize a node
                /// This step will initialize a new frame or local map.
                tracking->InitializeNode();

                // Perform tracking by estimating the new pose
                VisualSlamBase::Frame* current_frame;
                
                Eigen::Transform<double,3,2> new_pose;
                current_frame = tracking->GetCurrentFrame();
                
                new_pose = tracking->EstimateIncrementalMotion(*current_frame);
                std::cout<<"Debug : New Pose"<<std::endl;
                std::cout<<new_pose.matrix()<<std::endl;
                
                // Calculate Motion Derivative
                if(tracking->map.local_maps[0].frames.size() > 1){
                    VisualSlamBase::Frame* previous_frame;
                    previous_frame = tracking->GetPreviousFrame();
                    tracking->CalculateMotionJacobian(current_frame,previous_frame);
                }
            };
        };
    };

    void DetectFeatures(){
        if(received_l){
                features_l.clear();
                features_l = triangulator.DetectAndComputeFeatures(&undistorted_l,features_l,false);
            }
        if(received_r){
            features_r.clear();
            features_r = triangulator.DetectAndComputeFeatures(&undistorted_r,features_r,false);
        }

        return;
    };

    void Calculate3DCoordinates(){
        // Get Matches
            matches = triangulator.GetKeypointMatches(features_l,features_r);
            // TODO : Put parametized arguments for baseline and fx
            triangulator.Generate3DCoordinates(matches,framepoints,0.110074,457.95,cam_left.intrinsics);
            return;
    };

    void UndistortImages(cv::Mat cam_l_intrinsics,cv::Mat cam_r_intrinsics,std::vector<double> l_distortion,std::vector<double> r_distortion,cv::Mat& image_l,cv::Mat& image_r){

        cv::undistort(image_l,undistorted_l,cam_l_intrinsics,l_distortion);
        cv::undistort(image_r,undistorted_r,cam_r_intrinsics,r_distortion);
        return;

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
    //TestIncrementalMotion test;
    return 0;
}

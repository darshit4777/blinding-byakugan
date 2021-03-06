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
#include <nav_msgs/Odometry.h>
#include <sharp-eye/point_sim.hpp>
#include <slam_datatypes/slam_datatypes.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
/**
 * This file will create an executable which will serve as a test node and 
 * later on will be repurposed to form the ROS layer
*/

cv::Mat image_l;
cv::Mat image_r;
cv::Mat undistorted_l;
cv::Mat undistorted_r;
bool received_l,received_r;
std::vector<std::vector<std::string>> cam_left_image_list;
std::vector<std::vector<std::string>> cam_right_image_list;
//static const std::string OPENCV_WINDOW = "Image window";
static const std::string OPENCV_WINDOW_LEFT = "Left Image window";
static const std::string OPENCV_WINDOW_RIGHT = "Right Image window";
typedef std::vector<std::pair<KeypointWD,KeypointWD>> MatchVector;
typedef std::vector<Framepoint> FramepointVector;



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

std::vector<std::vector<std::string>> GetImageFilenames(std::filebuf &fb){
    // Reads the image filenames from a csv - uses opencv to load the images and 
    // assigns the images to image_l and image_r

    std::istream file(&fb);
    
    std::vector<std::vector<std::string>> row;
    std::vector<std::string>   result;
    std::string                line;
    while(!file.eof()){
        
        // Get a new line
        std::getline(file,line);

        // Add the line to a string stream
        std::stringstream          lineStream(line);
        std::string                cell;

        while(std::getline(lineStream,cell, ','))
        {
            result.push_back(cell);
        }

        // This checks for a trailing comma with no data after it.
        if (!lineStream && cell.empty())
        {
            // If there was a trailing comma then add an empty element.
            result.push_back("");
        }
        row.push_back(result);
        result.clear();
    }
    return row;
};

void GetCameraImages(int image_idx){

    std::string image_path_left = "/home/darshit/Code/blinding-byakugan/MH_01_easy/MH_01_easy.txt.d/left_camera/";
    std::string image_path_right = "/home/darshit/Code/blinding-byakugan/MH_01_easy/MH_01_easy.txt.d/right_camera/";

    std::vector<std::string> fname_row_l = cam_left_image_list[image_idx];
    std::vector<std::string> fname_row_r = cam_right_image_list[image_idx];
    
    std::string image_fname_left = fname_row_l[0];
    std::string image_fname_right = fname_row_r[0];

    image_path_left = image_path_left+image_fname_left;
    image_path_right = image_path_right+image_fname_right;

    image_l = cv::imread(image_path_left);
    image_r = cv::imread(image_path_right);

    //cv::imshow(OPENCV_WINDOW_LEFT,image_l);
    //cv::waitKey(0);

    return;
};

class TestDetectFeatures{
    public:
    int image_idx;
    int image_idx_max;
    void TestMain(){
        

        VisualTriangulation triangulator;
        std::vector<KeypointWD> features;
        while(image_idx < image_idx_max){
            
            // Get new camera images
            GetCameraImages(image_idx);

            triangulator.DetectAndComputeFeatures(&image_l,features,true);
            cv::imshow(OPENCV_WINDOW_LEFT,image_l);
            cv::waitKey(0);
            
            triangulator.DetectAndComputeFeatures(&image_r,features,true);
            cv::imshow(OPENCV_WINDOW_RIGHT,image_r);
            cv::waitKey(1);
            
            image_idx++;
        };
        cv::destroyAllWindows();
    };

    TestDetectFeatures(){
        image_idx = 0;
        image_idx_max = 200;
        TestMain();
    };

    ~TestDetectFeatures(){
        return;
    }
};

class TestGetMatchedKeypoints{
    public:
    int image_idx;
    int image_idx_max;

    TestGetMatchedKeypoints(){
        image_idx = 1;
        image_idx_max = 10;
        TestMain();
        return;
    }
    void TestMain(){
        VisualTriangulation triangulator;
        std::vector<KeypointWD> features_l;
        std::vector<KeypointWD> features_r;
        int count = 0;
        while(image_idx < image_idx_max){
            
            // Get new images
            //GetCameraImages(image_idx);
            GetCameraImages(image_idx);
            features_l.clear();
            features_l = triangulator.DetectAndComputeFeatures(&image_l,features_l,false);
            
            features_r.clear();
            features_r = triangulator.DetectAndComputeFeatures(&image_r,features_r,false);
            
            // Get Matches
            MatchVector matches = triangulator.GetEpipolarMatches(features_l,features_r);
            DrawMatches(matches,&image_l,&image_r);        
            image_idx++;
            
        };
        cv::destroyAllWindows();
    }

    void DrawMatches(MatchVector matches,cv::Mat* left_img, cv::Mat* right_img){
        cv::Mat combined_image;
        cv::hconcat(*left_img,*right_img,combined_image);

        //std::cout<<matches.size()<<std::endl;
        if(matches.empty()){
            return;
        }
        
        for(int i = 0; i < matches.size(); i++){
            cv::Point2f left_point;
            cv::Point2f right_point;

            left_point = matches[i].first.keypoint.pt;
            right_point = matches[i].second.keypoint.pt;
            right_point.x = right_point.x + left_img->cols;
            cv::line(combined_image,left_point,right_point,(0,0,255),1);
        };
        cv::imshow(OPENCV_WINDOW_LEFT,combined_image);
        cv::waitKey(0);

        return;
    };
};

class TestGenerate3DCoordinates{
    public:
    float focal_length = 457.975;
    float baseline = 0.11;
    Eigen::Matrix3f cam_intrinsics;
    typedef pcl::PointXYZI PointGray;
    ros::Publisher pub;
    ros::NodeHandle nh;
    int image_idx;
    int image_idx_max;

    public:
    TestGenerate3DCoordinates(const ros::NodeHandle &nodehandle){

        cam_intrinsics <<  458.654,     0.0,    367.215,
                               0.0, 457.296,    248.375,
                               0.0,     0.0,        1.0;

        nh = nodehandle;
        image_idx = 1;
        image_idx_max = 50;
        TestMain();
        return;
    };

    void TestMain(){
        VisualTriangulation triangulator;
        std::vector<KeypointWD> features_l;
        std::vector<KeypointWD> features_r;
        pub = nh.advertise< sensor_msgs::PointCloud2 > ("point_cloud", 5);
        int count = 0;
        while(image_idx < image_idx_max){
            
            GetCameraImages(image_idx);
            features_l.clear();
            features_l = triangulator.DetectAndComputeFeatures(&image_l,features_l,false);
            
            features_r.clear();
            features_r = triangulator.DetectAndComputeFeatures(&image_r,features_r,false);
            
            // Get Matches
            MatchVector matches = triangulator.GetEpipolarMatches(features_l,features_r);
            //std::cout<<"No of matches "<<matches.size()<<std::endl;
            FramepointVector framepoints;
            triangulator.Generate3DCoordinates(matches,framepoints,baseline,focal_length,cam_intrinsics);
            //std::cout<<"No of framepoints "<<framepoints.size()<<std::endl;
            cv::imshow(OPENCV_WINDOW_LEFT,image_l);
            cv::waitKey(0);
            DrawPointCloud(framepoints,&image_l);
            
            image_idx++;
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

            float x,y;
            x = framepoint.keypoint_l.keypoint.pt.x;
            y = framepoint.keypoint_l.keypoint.pt.y;
            
            cv::Scalar intensity;
            intensity = image_l->at<uchar>(y,x);
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
    std::vector<Frame> frames;
    int image_idx;
    int image_idx_max;
    TestFindCorrespondences(){
        // Initializing Camera Matrices
        cam_left.intrinsics << 458.654,     0.0,    367.215,
                               0.0, 457.296,    248.375,
                               0.0,     0.0,        1.0;

        cam_right.intrinsics << 457.587,        0.0, 379.999,
                                    0.0,    456.134, 255.238,
                                    0.05,        0.0,    1.0;                               

        image_idx = 0;
        image_idx_max = 3;
        TestMain();
    };

    void TestMain(){
        VisualTracking tracking(cam_left,cam_right);
        VisualTriangulation triangulator;
        std::vector<KeypointWD> features_l;
        std::vector<KeypointWD> features_r;

        while(image_idx < image_idx_max){
            GetCameraImages(image_idx);
            features_l.clear();
            features_l = triangulator.DetectAndComputeFeatures(&image_l,features_l,false);
            
            features_r.clear();
            features_r = triangulator.DetectAndComputeFeatures(&image_r,features_r,false);

            
            // Get Matches
            MatchVector matches = triangulator.GetEpipolarMatches(features_l,features_r);

            FramepointVector framepoints;
            triangulator.Generate3DCoordinates(matches,framepoints,0.110074,435.20,cam_left.intrinsics);
            Frame current_frame;
            std::vector<boost::shared_ptr<Framepoint>> framepoint_ptr_vector;
            for(int i =0; i < framepoints.size(); i++){
                boost::shared_ptr<Framepoint> framepoint_ptr; 
                framepoint_ptr = boost::make_shared<Framepoint>(framepoints[i]);
                framepoint_ptr_vector.push_back(framepoint_ptr);
            }
            current_frame.points = framepoint_ptr_vector;
            current_frame.image_l = image_l;
            frames.push_back(current_frame);
            current_frame.points.clear();
            if(frames.size() > 1){
                // Skip the first frame, from the second frame onwards..
                int previous_index = frames.size() - 2;
                int current_index = frames.size() - 1;
                int correspondences;
                correspondences = tracking.FindCorrespondences(frames[previous_index].points,frames[current_index].points);
                // Drawing the correspondences on a compound image
                std::cout<<"No of correspondences "<<correspondences<<std::endl;
                for(auto point : frames[current_index].points){
                    if(point->previous != NULL){
                        // A correspondence exists
                        cv::Point2f previous_frame_point;
                        cv::Point2f current_frame_point;
                        cv::Mat joined_image;
                        cv::hconcat(frames[previous_index].image_l,frames[current_index].image_l,joined_image);
                        previous_frame_point = point->previous->keypoint_l.keypoint.pt;
                        current_frame_point = point->keypoint_l.keypoint.pt;
                        current_frame_point.x += frames[previous_index].image_l.cols;

                        cv::line(joined_image,previous_frame_point,current_frame_point,(0,0,255),1);
                        cv::imshow(OPENCV_WINDOW_LEFT,joined_image);
                        cv::waitKey(1);
                    } 
                }
                
                
            }
                
            image_idx++;
        };
    };  
};

class TestIncrementalMotion{
    public:

    Eigen::Transform<float,3,2> T_body2caml;
    Eigen::Transform<float,3,2> T_body2camr;
    Eigen::Transform<float,3,2> T_caml2camr;

    Camera cam_left;
    Camera cam_right;

    // Triangulation
    VisualTriangulation triangulator;
    std::vector<KeypointWD> features_l;
    std::vector<KeypointWD> features_r;

    VisualTracking* tracking;
    
    FramepointVector framepoints;
    MatchVector matches;
    
    // ROS
    ros::NodeHandle nodehandle;
    ros::Publisher pose_publisher;

    // Camera Images
    int image_idx;
    int image_idx_max;
    
    TestIncrementalMotion(ros::NodeHandle& nh){
        // Initializing Camera Matrices
        cam_left.intrinsics << 435.20,     0.0,    367.215,
                               0.0, 435.20,    252.375,
                               0.0,     0.0,        1.0;

        cam_right.intrinsics << 435.20,        0.0, 367.215,
                                    0.0,    435.134, 252.375,
                                    0.0,        0.0,    1.0;
        
        tracking = new VisualTracking(cam_left,cam_right);

        nodehandle = nh;
        pose_publisher = nh.advertise<nav_msgs::Odometry>("/visual_odometry",10);

        image_idx = 0;
        image_idx_max = 3;
        TestMain();
        delete tracking;
        
        return;
    };

    void TestMain(){

        while(image_idx < image_idx_max){
            GetCameraImages(image_idx);
            // Undistort
            //UndistortImages(cam_l_intrinsics,cam_r_intrinsics,cam_left.distortion_coeffs,cam_right.distortion_coeffs,image_l,image_r);
            std::cout<<"New Frame"<<std::endl;
            tracking->SetPredictionCallTime();
            // Visual Triangulation 
            DetectFeatures();
            Calculate3DCoordinates();
            
            // Store the framepoints for tracking
            std::cout<<"Framepoint Vector Size"<<std::endl;
            std::cout<<framepoints.size()<<std::endl;
            tracking->SetFramepointVector(framepoints);
            framepoints.clear();
            // Initialize a node
            /// This step will initialize a new frame or local map.
            tracking->img_l = image_l;
            tracking->img_r = image_r;
            tracking->InitializeNode();
            // Perform tracking by estimating the new pose
            Frame* current_frame;
            
            Eigen::Transform<float,3,2> new_pose;
            LocalMap* lmap_ptr = tracking->map.GetLastLocalMap();
            current_frame = lmap_ptr->GetLastFrame();
            
            
            // TODO : This is a band-aid patch - not quite elegant. A better solution would be to use class variables 
            // that keep track of current and previous frames
            if(tracking->frame_correspondences > 0){
                new_pose = tracking->EstimateIncrementalMotion();
            }
            else{
                new_pose = current_frame->T_world2cam;
            };
            //tracking->CreateAndUpdateLandmarks(current_frame,lmap_ptr);
            //Eigen::AngleAxisf z_rotation = Eigen::AngleAxisf(-M_PI/2,Eigen::Vector3f::UnitZ());
            //Eigen::AngleAxisf x_rotation = Eigen::AngleAxisf(-M_PI/2,Eigen::Vector3f::UnitX());
            //Eigen::AngleAxisf y_rotation = Eigen::AngleAxisf(M_PI/2,Eigen::Vector3f::UnitY());
            //new_pose = x_rotation * new_pose;
            //new_pose = z_rotation * new_pose;
            
            //new_pose = new_pose * y_rotation;
            //PublishPose(new_pose);
            std::cout<<tracking->optimizer->T_prev2curr.matrix()<<std::endl;
            
            // Calculate Motion Derivative
            if(tracking->map.local_maps[0]->frames.size() > 1){
                Frame* previous_frame;
                previous_frame = lmap_ptr->GetPreviousFrame();
                tracking->CalculateMotionJacobian(current_frame,previous_frame);
            }
            image_idx++;
        };
    };

    void DetectFeatures(){
        features_l.clear();
        features_l = triangulator.DetectAndComputeFeatures(&image_l,features_l,false);
        features_r.clear();
        features_r = triangulator.DetectAndComputeFeatures(&image_r,features_r,false);
        
        cv::Mat image_l_copy, image_r_copy;
        image_l_copy = image_l;
        image_r_copy = image_r;
        
        std::vector<cv::KeyPoint> left_keypoint_vector;
        std::vector<cv::KeyPoint> right_keypoint_vector;

        for(auto feature : features_l){
            cv::KeyPoint keypoint = feature.keypoint;
            left_keypoint_vector.push_back(keypoint);
        }
        for(auto feature : features_r){
            cv::KeyPoint keypoint = feature.keypoint;
            right_keypoint_vector.push_back(keypoint);
        }

        cv::drawKeypoints(image_l_copy, left_keypoint_vector, image_l_copy, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
        cv::drawKeypoints(image_r_copy, right_keypoint_vector, image_r_copy, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

        cv::imshow(OPENCV_WINDOW_LEFT,image_l_copy);
        cv::waitKey(1);
        cv::imshow(OPENCV_WINDOW_RIGHT,image_r_copy);
        cv::waitKey(0);
        return;
    };

    void Calculate3DCoordinates(){
        // Get Matches
        matches = triangulator.GetEpipolarMatches(features_l,features_r);
        // Drawing the correspondences on a compound image
        
        for(auto match : matches){
            cv::Point2f left_point = match.first.keypoint.pt;
            cv::Point2f right_point = match.second.keypoint.pt;
            cv::Mat joined_image;
            cv::hconcat(image_l,image_r,joined_image);
            right_point.x = right_point.x + image_l.cols;        
            cv::line(joined_image,left_point,right_point,(0,0,255),1);
            cv::imshow(OPENCV_WINDOW_LEFT,joined_image);
            cv::waitKey(1);
             
        };
        framepoints.clear();
        triangulator.Generate3DCoordinates(matches,framepoints,0.110074,435.20,cam_left.intrinsics);
        matches.clear();
        return;
    };

    void UndistortImages(cv::Mat cam_l_intrinsics,cv::Mat cam_r_intrinsics,std::vector<float> l_distortion,std::vector<float> r_distortion,cv::Mat& image_l,cv::Mat& image_r){

        cv::Mat mapl1, mapl2;
        cv::Mat mapr1, mapr2;

        std::cout<<cam_l_intrinsics<<std::endl;
        cv::Mat new_intrinsics_l, new_intrinsics_r;
        new_intrinsics_l = cv::Mat(3,3,cv::DataType<float>::type);    
        new_intrinsics_l.at<float>(0,0) = 435.20;
        new_intrinsics_l.at<float>(0,1) = 0.0;
        new_intrinsics_l.at<float>(0,2) = 367.20;
        new_intrinsics_l.at<float>(1,0) = 0.0;
        new_intrinsics_l.at<float>(1,1) = 435.45;
        new_intrinsics_l.at<float>(1,2) = 252.0;
        new_intrinsics_l.at<float>(2,0) = 0.0;
        new_intrinsics_l.at<float>(2,1) = 0.0;
        new_intrinsics_l.at<float>(2,2) = 1.0;

        new_intrinsics_r = cv::Mat(3,3,cv::DataType<float>::type);    
        new_intrinsics_r.at<float>(0,0) = 435.654;
        new_intrinsics_r.at<float>(0,1) = 0.0;
        new_intrinsics_r.at<float>(0,2) = 367.20;
        new_intrinsics_r.at<float>(1,0) = 0.0;
        new_intrinsics_r.at<float>(1,1) = 435.20;
        new_intrinsics_r.at<float>(1,2) = 252.0;
        new_intrinsics_r.at<float>(2,0) = 0.0;
        new_intrinsics_r.at<float>(2,1) = 0.0;
        new_intrinsics_r.at<float>(2,2) = 1.0;

        cv::Mat identity;
        identity = cv::Mat(3,3,cv::DataType<float>::type);
        cv::setIdentity(identity);                               

        cv::fisheye::initUndistortRectifyMap(cam_l_intrinsics, l_distortion, cv::Matx33d::eye(), new_intrinsics_l, image_l.size(), CV_32FC1, mapl1, mapl2);
        cv::fisheye::initUndistortRectifyMap(cam_r_intrinsics, l_distortion, cv::Matx33d::eye(), new_intrinsics_r, image_r.size(), CV_32FC1, mapr1, mapr2);

        cv::remap(image_l, undistorted_l, mapl1, mapl2, cv::INTER_LINEAR);
        cv::remap(image_r, undistorted_r, mapr1, mapr2, cv::INTER_LINEAR);
        cv::imshow(OPENCV_WINDOW_LEFT,undistorted_l);
        cv::imshow(OPENCV_WINDOW_RIGHT,undistorted_r);
        cam_left.intrinsics << 435.20, 0.0, 367.20,
                                0.0, 435.20, 252.0,
                                0.0, 0.0, 1.0;

        cam_right.intrinsics = cam_left.intrinsics;

        cv::waitKey(0);
        return;

    };

    void PublishPose(Eigen::Transform<float,3,2> pose){

        nav_msgs::Odometry odom_msg;
        odom_msg.child_frame_id = "base";
        odom_msg.header.frame_id = "world";
        odom_msg.pose.pose.position.x = pose.translation().x();
        odom_msg.pose.pose.position.y = pose.translation().y();
        odom_msg.pose.pose.position.z = pose.translation().z();

        Eigen::Matrix3f rot_matrix = pose.rotation();
        Eigen::Quaternionf q(rot_matrix);
        odom_msg.pose.pose.orientation.x = q.x();
        odom_msg.pose.pose.orientation.y = q.y();
        odom_msg.pose.pose.orientation.z = q.z();
        odom_msg.pose.pose.orientation.w = q.w();

        pose_publisher.publish(odom_msg);
        return;
    }

};

class TestPoseOptimizer{
    public:
    PoseOptimizer* optimizer;
    PointSim* point_sim;

    TestPoseOptimizer(){
        optimizer = new PoseOptimizer;
        point_sim = new PointSim;

        // Initializations
        Camera cam_left, cam_right;
        cam_left.intrinsics << 458.654,     0.0,    367.215,
                               0.0, 457.296,    248.375,
                               0.0,     0.0,        1.0;

        cam_right.intrinsics << 457.587,        0.0, 379.999,
                                    0.0,    456.134, 255.238,
                                    0.05,        0.0,    1.0;

        Eigen::Transform<float,3,2> T_caml2camr;

        Eigen::Transform<float,3,2> T_body2caml,T_body2camr;
        T_body2caml.matrix()<<   0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
                                    0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
                                -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
                                0.0, 0.0, 0.0, 1.0;

        T_body2camr.matrix()<<  0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
                                0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
                                -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
                                0.0, 0.0, 0.0, 1.0;
        T_caml2camr = T_body2caml.inverse() * T_body2camr;                                
        point_sim->CreateCameras(cam_left,cam_right);
        point_sim->SetInterCameraTransform(T_caml2camr);
        point_sim->CreateRandomPoints(1000,10);

        // Creating camera poses
        Eigen::Transform<float,3,2> T_world2cam1,T_world2cam2;
        T_world2cam1.setIdentity();
        T_world2cam1.translation().x() =0;
        T_world2cam1.translation().y() =0;
        T_world2cam1.translation().z() =-5.0; 
        T_world2cam2.setIdentity();
        T_world2cam2.translation().x() = 0.1;
        T_world2cam2.translation().y() = 0.1;
        T_world2cam2.translation().z() = -5.0;
        //Eigen::AngleAxis<float> z(Degrees2Radians(15),Eigen::Vector3f(0,0,1));
        Eigen::AngleAxis<float> y(Degrees2Radians(5),Eigen::Vector3f(0,1,0));
        //Eigen::AngleAxis<float> x(Degrees2Radians(3),Eigen::Vector3f(1,0,0));
        

        //T_world2cam2 = T_world2cam2 * z;
        T_world2cam2 = T_world2cam2 * y;
        //T_world2cam2 = T_world2cam2 * x; 

        std::cout<<"Correct Solution"<<std::endl;
        std::cout<<T_world2cam2.matrix()<<std::endl;
        point_sim->SetCameraPositions(T_world2cam1,T_world2cam2);
        //point_sim->SetCameraPositionAndTransform(T_world2cam1,T_world2cam2);
        point_sim->CreateFrames();
        TestMain();
        return;
    };

    void TestMain(){

        optimizer->parameters.T_caml2camr = point_sim->T_caml2camr;
        LocalMap* lmap_ptr;
        optimizer->Initialize(&point_sim->current_frame,&point_sim->previous_frame,lmap_ptr);

        //optimizer->OptimizeOnce();
        optimizer->Converge();
        std::cout<<"Debug : Final Pose"<<std::endl;
        std::cout<<point_sim->current_frame.T_world2cam.matrix()<<std::endl;
        return;
    };

    ~TestPoseOptimizer(){

        delete optimizer;
        delete point_sim;
    }

    float Degrees2Radians(float degrees){
        return (degrees * CV_PI / 180);
    }
};

class TestLandmarkOptimization{
    public:
    Eigen::Vector3f landmark_position;
    int n; //< No of measurements
    Camera cam_l;
    std::vector<boost::shared_ptr<Framepoint>> framepoint_vector;

    TestLandmarkOptimization(Eigen::Vector3f position, int measurements){
        landmark_position = position;
        n = measurements;

        cam_l.intrinsics << 458.654,     0.0,    367.215,
                               0.0, 457.296,    248.375,
                               0.0,     0.0,        1.0;

        CreateRandomMeasurements();

        TestMain();
    }

    void CreateRandomMeasurements(){
        for(int i = 0; i < n; i++){
            // Create random vectors upto a range of 15.625m
            Eigen::Vector3f random_vector;
            random_vector.setRandom();
            random_vector = random_vector * 2.5;

            // Measurement translation
            Eigen::Vector3f measurement_translation;
            measurement_translation = landmark_position + random_vector;

            // Measurement rotation
            bool valid_measurement = false;
            while(!valid_measurement){
                // Create a random rotation
                Eigen::Quaternionf random_quat;
                random_quat = Eigen::Quaternionf::UnitRandom();

                Eigen::Transform<float,3,2> T_world2cam;
                T_world2cam.setIdentity();
                T_world2cam.translation() = measurement_translation;
                T_world2cam.rotate(random_quat.toRotationMatrix());  

                // Test the projection
                Eigen::Vector3f p_caml;
                p_caml = cam_l.intrinsics * T_world2cam.inverse() * landmark_position;

                if(p_caml.z() == 0 ){
                    continue;
                }


                p_caml.x() = p_caml.x() / p_caml.z();
                p_caml.y() = p_caml.y() / p_caml.z();

                if((p_caml.x() > 720) || (p_caml.x() < 0)){
                    continue;
                }
                if((p_caml.y() > 480) || (p_caml.y() < 0)){
                    continue;
                }
                if(p_caml.hasNaN()){
                    continue;
                }

                // If it passes all the tests - its a valid measurement

                // Creating camera coordinates for the framepoint
                boost::shared_ptr<Frame> frame_ptr = boost::make_shared<Frame>();
                frame_ptr->T_world2cam = T_world2cam;
                frame_ptr->T_cam2world = T_world2cam.inverse();

                boost::shared_ptr<Framepoint> framepoint_ptr = boost::make_shared<Framepoint>();
                framepoint_ptr->camera_coordinates = T_world2cam.inverse() * landmark_position;

                // lets add some noise to it
                framepoint_ptr->camera_coordinates = framepoint_ptr->camera_coordinates + 0.15 * Eigen::Vector3f::Random();
                framepoint_ptr->world_coordinates = landmark_position;

                // Add the measurement to the vector
                framepoint_ptr->parent_frame = frame_ptr.get();
                framepoint_vector.push_back(framepoint_ptr);

                std::cout<<"Valid measurement created"<<std::endl;
                std::cout<<"The pixels for the point are "<<p_caml.x()<<" "<<p_caml.y()<<std::endl;
                valid_measurement = true; 
            };
        }
        return;
    };

    void TestMain(){
        // Create the first landmark
        Landmark landmark(framepoint_vector.front());

        // Now we try landmark updates for every landmark
        for(int i = 1; i < framepoint_vector.size(); i++){
            landmark.UpdateLandmark(framepoint_vector[i]);
            std::cout<<"The updated landmark position is "<<std::endl;
            std::cout<<landmark.world_coordinates<<std::endl;
            std::cout<<"The no of updates are "<<landmark.updates<<std::endl;
        };
        return;
    }

};

class TestSharedPointers{
    public:
    Framepoint fp;

    boost::shared_ptr<Framepoint> fp_ptr;
    boost::shared_ptr<Landmark> lmark_ptr;

    TestSharedPointers(){
        
        std::cout<<"Testing Framepoint"<<std::endl;
        Eigen::Vector3f camera_coordinates;
        camera_coordinates.setRandom();
        std::cout<<"Initial Camera coordiantes"<<std::endl;
        std::cout<<camera_coordinates<<std::endl;

        fp.camera_coordinates = camera_coordinates;
        fp_ptr = boost::make_shared<Framepoint>(fp);
        std::cout<<"Shared pointer Camera coordinates"<<std::endl;
        std::cout<<fp_ptr.get()->camera_coordinates<<std::endl;


        std::cout<<"Testing Landmark"<<std::endl;
        Eigen::Vector3f world_coordinates;
        world_coordinates.setRandom();
        std::cout<<"Initial World Coordiantes"<<std::endl;
        std::cout<<world_coordinates<<std::endl;
        fp_ptr.get()->world_coordinates = world_coordinates;
        lmark_ptr = boost::make_shared<Landmark>(fp_ptr);
        std::cout<<"Shared pointer World coordinates"<<std::endl;
        std::cout<<lmark_ptr.get()->world_coordinates<<std::endl;
        
        return;
    };
};


int main(int argc, char **argv){
    ros::init(argc,argv,"image_listener");
    ros::NodeHandle nh;

    std::string image_list_left_file = argv[1];
    std::string image_list_right_file = argv[2];
    std::filebuf image_list_left_buf;
    std::filebuf image_list_right_buf;
    
    image_list_left_buf.open(image_list_left_file,std::ios::in);
    image_list_right_buf.open(image_list_right_file,std::ios::in);
    
    cam_left_image_list = GetImageFilenames(image_list_left_buf);
    cam_right_image_list = GetImageFilenames(image_list_right_buf);

    cv::namedWindow(OPENCV_WINDOW_LEFT);
    cv::namedWindow(OPENCV_WINDOW_RIGHT);
    
    TestIncrementalMotion test(nh);
    //TestFindCorrespondences test;
    //TestDetectFeatures test;
    ///TestGetMatchedKeypoints test;
    return 0;
}

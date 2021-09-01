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


class TestDetectFeatures{
    public:
    void TestMain(){
        

        VisualTriangulation triangulator;
        std::vector<KeypointWD> features;
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
        std::vector<KeypointWD> features_l;
        std::vector<KeypointWD> features_r;
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
    float focal_length = 457.975;
    float baseline = 0.11;
    Eigen::Matrix3f cam_intrinsics;
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
        std::vector<KeypointWD> features_l;
        std::vector<KeypointWD> features_r;
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

            float x,y;
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
    std::vector<Frame> frames;
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
        std::vector<KeypointWD> features_l;
        std::vector<KeypointWD> features_r;

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
                Frame current_frame;
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

    Eigen::Transform<float,3,2> T_body2caml;
    Eigen::Transform<float,3,2> T_body2camr;
    Eigen::Transform<float,3,2> T_caml2camr;

    Camera cam_left;
    Camera cam_right;

    cv::Mat cam_l_intrinsics;
    cv::Mat cam_r_intrinsics;

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
    
    TestIncrementalMotion(ros::NodeHandle& nh){
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

        cam_l_intrinsics = cv::Mat(3,3,cv::DataType<float>::type);    
        cam_l_intrinsics.at<float>(0,0) = cam_left.intrinsics(0,0);
        cam_l_intrinsics.at<float>(0,1) = cam_left.intrinsics(0,1);
        cam_l_intrinsics.at<float>(0,2) = cam_left.intrinsics(0,2);
        cam_l_intrinsics.at<float>(1,0) = cam_left.intrinsics(1,0);
        cam_l_intrinsics.at<float>(1,1) = cam_left.intrinsics(1,1);
        cam_l_intrinsics.at<float>(1,2) = cam_left.intrinsics(1,2);
        cam_l_intrinsics.at<float>(2,0) = cam_left.intrinsics(2,0);
        cam_l_intrinsics.at<float>(2,1) = cam_left.intrinsics(2,1);
        cam_l_intrinsics.at<float>(2,2) = cam_left.intrinsics(2,2);

        cam_r_intrinsics = cv::Mat(3,3,cv::DataType<float>::type);
        cam_r_intrinsics.at<float>(0,0) = cam_right.intrinsics(0,0);
        cam_r_intrinsics.at<float>(0,1) = cam_right.intrinsics(0,1);
        cam_r_intrinsics.at<float>(0,2) = cam_right.intrinsics(0,2);
        cam_r_intrinsics.at<float>(1,0) = cam_right.intrinsics(1,0);
        cam_r_intrinsics.at<float>(1,1) = cam_right.intrinsics(1,1);
        cam_r_intrinsics.at<float>(1,2) = cam_right.intrinsics(1,2);
        cam_r_intrinsics.at<float>(2,0) = cam_right.intrinsics(2,0);
        cam_r_intrinsics.at<float>(2,1) = cam_right.intrinsics(2,1);
        cam_r_intrinsics.at<float>(2,2) = cam_right.intrinsics(2,2);                                    

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

        nodehandle = nh;
        pose_publisher = nh.advertise<nav_msgs::Odometry>("/visual_odometry",10);

        TestMain();
        delete tracking;
        
        return;
    };

    void TestMain(){

        while(ros::ok()){
            ros::spinOnce();
            if(received_l && received_r){
                // Set the time for recieving a new frame

                // Calibrate
                //UndistortImages(cam_l_intrinsics,cam_r_intrinsics,cam_left.distortion_coeffs,cam_right.distortion_coeffs,image_l,image_r);
                std::cout<<"New Frame"<<std::endl;
                tracking->SetPredictionCallTime();

                // Visual Triangulation 
                DetectFeatures();
                Calculate3DCoordinates();
                
                // Store the framepoints for tracking
                tracking->SetFramepointVector(framepoints);

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
                    new_pose = tracking->EstimateIncrementalMotion(*current_frame);
                }
                else{
                    new_pose = current_frame->T_world2cam;
                };

                PublishPose(new_pose);
                
                // Calculate Motion Derivative
                if(tracking->map.local_maps[0]->frames.size() > 1){
                    Frame* previous_frame;
                    previous_frame = lmap_ptr->GetPreviousFrame();
                    tracking->CalculateMotionJacobian(current_frame,previous_frame);
                }
            };
        };
    };

    void DetectFeatures(){
        if(received_l){
                features_l.clear();
                features_l = triangulator.DetectAndComputeFeatures(&image_l,features_l,false);
            }
        if(received_r){
            features_r.clear();
            features_r = triangulator.DetectAndComputeFeatures(&image_r,features_r,false);
        }
        //std::cout<<"Debug : Features size"<<std::endl;
        //std::cout<<features_l.size()<<std::endl;
        return;
    };

    void Calculate3DCoordinates(){
        // Get Matches
            matches = triangulator.GetKeypointMatches(features_l,features_r);
            //std::cout<<"Debug : Matches size"<<std::endl;
            //std::cout<<matches.size()<<std::endl;
            // TODO : Put parametized arguments for baseline and fx
            framepoints.clear();
            triangulator.Generate3DCoordinates(matches,framepoints,0.110074,457.95,cam_left.intrinsics);
            //std::cout<<"Debug : Framepoints size"<<std::endl;
            //std::cout<<framepoints.size()<<std::endl;
            return;
    };

    void UndistortImages(cv::Mat cam_l_intrinsics,cv::Mat cam_r_intrinsics,std::vector<float> l_distortion,std::vector<float> r_distortion,cv::Mat& image_l,cv::Mat& image_r){

        cv::undistort(image_l,undistorted_l,cam_l_intrinsics,l_distortion);
        cv::undistort(image_r,undistorted_r,cam_r_intrinsics,r_distortion);
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
        point_sim->CreateRandomPoints(1000,5);

        // Creating camera poses
        Eigen::Transform<float,3,2> T_world2cam1,T_world2cam2;
        T_world2cam1.setIdentity();
        T_world2cam1.translation().x() =0;
        T_world2cam1.translation().y() =0;
        T_world2cam1.translation().z() =-5.0; 
        T_world2cam2.setIdentity();
        T_world2cam2.translation().x() = 0.0;
        T_world2cam2.translation().y() = 0.0;
        T_world2cam2.translation().z() = -5.0;
        Eigen::AngleAxis<float> z(Degrees2Radians(3),Eigen::Vector3f(0,0,1));
        Eigen::AngleAxis<float> y(Degrees2Radians(3),Eigen::Vector3f(0,1,0));
        Eigen::AngleAxis<float> x(Degrees2Radians(3),Eigen::Vector3f(1,0,0));
        

        T_world2cam2 = T_world2cam2 * z;
        T_world2cam2 = T_world2cam2 * y;
        T_world2cam2 = T_world2cam2 * x; 

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

        optimizer->OptimizeOnce();
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
        return degrees * CV_PI / 180;
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

        //TestMain();
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

                // p_caml however is a highly accurate measurement - lets add some noise to it
                framepoint_ptr->camera_coordinates = framepoint_ptr->camera_coordinates + 0.25 * Eigen::Vector3f::Random();

                // Add the measurement to the vector
                framepoint_ptr->parent_frame = frame_ptr;
                framepoint_vector.push_back(framepoint_ptr);

                std::cout<<"Valid measurement created"<<std::endl;
                std::cout<<"The pixels for the point are "<<p_caml.x()<<" "<<p_caml.y()<<std::endl;
                valid_measurement = true; 
            };
        }
        return;
    };

    void TestMain();

};


int main(int argc, char **argv){
    ros::init(argc,argv,"image_listener");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber imageSub_l = it.subscribe("cam0/image_raw", 1, boost::bind(CameraCallback,_1,0));
    image_transport::Subscriber imageSub_r = it.subscribe("cam1/image_raw", 1, boost::bind(CameraCallback,_1,1));
    //TestIncrementalMotion test(nh);
    Eigen::Vector3f landmark_position;
    landmark_position.setZero();

    TestLandmarkOptimization test(landmark_position,5);
    return 0;
}

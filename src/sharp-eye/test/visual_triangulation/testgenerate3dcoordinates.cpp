#include <opencv2/imgproc.hpp>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sharp-eye/visual_triangulation.hpp>
#include <sharp-eye/visual_triangulation_fixtures.hpp>

typedef std::vector<KeypointWD> FeatureVector;
typedef std::vector<Framepoint> FramepointVector;
typedef std::vector<std::pair<KeypointWD,KeypointWD>> MatchVector;

class TestGenerate3DCoordinates{
    public:
    float focal_length = 457.975;
    float baseline = 0.11;
    Eigen::Matrix3f cam_intrinsics;
    typedef pcl::PointXYZI PointGray;

    VisualTriangulation triangulator;
    std::vector<KeypointWD> features_l;
    std::vector<KeypointWD> features_r;

    MatchVector matches;

    public:
    TestGenerate3DCoordinates(){

        cam_intrinsics <<  458.654,     0.0,    367.215,
                               0.0, 457.296,    248.375,
                               0.0,     0.0,        1.0;
        return;
    };

    void Generate3DPoints(cv::Mat image_l, cv::Mat image_r){

        features_l.clear();
        features_l = triangulator.DetectAndComputeFeatures(&image_l,features_l,false);
        
        features_r.clear();
        features_r = triangulator.DetectAndComputeFeatures(&image_r,features_r,false);
        
        // Get Matches
        matches = triangulator.GetEpipolarMatches(features_l,features_r);
        FramepointVector framepoints;
        
        triangulator.Generate3DCoordinates(matches,framepoints,baseline,focal_length,cam_intrinsics);
        DrawPointCloud(framepoints,&image_l);
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
        return;
    };
};

TEST_F(VisualTriangulationTest,TestGenerate3DPoints){
    TestGenerate3DCoordinates test;

    // Set image indices
    int image_idx = 0;
    int image_idx_max = 10;

    cv::Mat image_l,image_r;

    while(image_idx < image_idx_max){
        image_l = GetImage(image_idx,"left");
        image_r = GetImage(image_idx,"right");

        test.Generate3DPoints(image_l,image_r);

        image_idx++;
    }
    return;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
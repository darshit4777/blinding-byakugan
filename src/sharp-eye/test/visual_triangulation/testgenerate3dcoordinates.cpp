#include <opencv2/imgproc.hpp>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sharp-eye/visual_triangulation.hpp>
#include <sharp-eye/visual_triangulation_fixtures.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>


typedef std::vector<KeypointWD> FeatureVector;
typedef std::vector<Framepoint> FramepointVector;
typedef std::vector<std::pair<KeypointWD,KeypointWD>> MatchVector;

class TestGenerate3DCoordinates{
    public:
    float focal_length = 457.975;
    float baseline = 0.11;
    Eigen::Matrix3f cam_intrinsics;
    typedef pcl::PointXYZ PointGray;
    pcl::PointCloud<PointGray> pcl_cloud;

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
        // TODO : Add visualization using PCL
        // https://pcl.readthedocs.io/projects/tutorials/en/latest/pcl_visualizer.html#
        // Point Cloud ROS Msg

        // Declaring the point cloud
        

        // Creating the point cloud
        for(auto framepoint : framepoint_vec){
            PointGray point;
            point.x = framepoint.camera_coordinates[0];
            point.y = framepoint.camera_coordinates[1];
            point.z = framepoint.camera_coordinates[2];

            // float x,y;
            // x = framepoint.keypoint_l.keypoint.pt.x;
            // y = framepoint.keypoint_l.keypoint.pt.y;
            
            // cv::Scalar intensity;
            // intensity = image_l->at<uchar>(y,x);
            // point.intensity = intensity.val[0];

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
    cv::destroyAllWindows();
    // Set image indices
    int image_idx = 100;
    int image_idx_max = 500;

    cv::Mat image_l,image_r;
    // Creating a PCL Viewer
    
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud_ptr(&test.pcl_cloud);
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ> (cloud_ptr,"3d points");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "3d points");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    
    while(image_idx < image_idx_max){
        image_l = GetImage(image_idx,"left");
        image_r = GetImage(image_idx,"right");
        test.Generate3DPoints(image_l,image_r);
        viewer->spinOnce(100);
        viewer->updatePointCloud(cloud_ptr,"3d points");
        test.pcl_cloud.clear();
        image_idx++;
    }   
    return;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
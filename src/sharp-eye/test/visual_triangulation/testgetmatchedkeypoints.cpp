#include <opencv2/imgproc.hpp>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <sharp-eye/visual_triangulation.hpp>
#include <sharp-eye/visual_triangulation_fixtures.hpp>
class TestGetMatchedKeypoints{
    public:
    VisualTriangulation triangulator;
    std::vector<KeypointWD> features_l;
    std::vector<KeypointWD> features_r;
    VisualTriangulation::MatchVector matches;

    void GetMatchedKeypoints(cv::Mat image_l, cv::Mat image_r){

        features_l.clear();
        features_l = triangulator.DetectAndComputeFeatures(&image_l,features_l,false);
            
        features_r.clear();
        features_r = triangulator.DetectAndComputeFeatures(&image_r,features_r,false);

        matches = triangulator.GetEpipolarMatches(features_l,features_r);
    };

    void DrawMatches(cv::Mat* left_img, cv::Mat* right_img,std::string opencv_window){
        

        //std::cout<<matches.size()<<std::endl;
        if(matches.empty()){
            return;
        };
        for(int i = 0; i < matches.size(); i++){
            cv::Point2f left_point;
            cv::Point2f right_point;

            cv::Mat combined_image;
            cv::hconcat(*left_img,*right_img,combined_image);

            left_point = matches[i].first.keypoint.pt;
            right_point = matches[i].second.keypoint.pt;
            right_point.x = right_point.x + left_img->cols;
            cv::line(combined_image,left_point,right_point,(0,0,255),1);
            cv::imshow(opencv_window,combined_image);
            cv::waitKey(2);
        };
        
        return;
    };
};

TEST_F(VisualTriangulationTest,GetEpipolarMatches){
    TestGetMatchedKeypoints test;
    cv::destroyWindow(OPENCV_WINDOW_RIGHT);

    int image_idx = 0;
    int image_idx_max = 10;
    while(image_idx < image_idx_max){
        image_l = GetImage(image_idx,"left");
        image_r = GetImage(image_idx,"right");
        
        test.GetMatchedKeypoints(image_l,image_r);
        test.DrawMatches(&image_l,&image_r,OPENCV_WINDOW_LEFT);
        image_idx++;
    }
    return;
};
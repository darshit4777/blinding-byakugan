#include <sharp-eye/visual_tracking_fixture.hpp>
#include <sharp-eye/visual_tracking.hpp>
#include <sharp-eye/visual_triangulation.hpp>
#include <slam_datatypes/slam_datatypes.hpp>

TEST_F(VisualTrackingTest,TestFindCorrespondences){
    int image_idx = 0;
    int image_idx_max = 2;

    LocalMap* lmap_ptr = new LocalMap;
    while(image_idx < image_idx_max){
        cv::Mat image_l, image_r;
        image_l = GetImage(image_idx,"left");
        image_r = GetImage(image_idx,"right");

        // Create a Frame and return
        boost::shared_ptr<Frame> curr_frame_ptr = GetFrame(image_l,image_r);

        // Add Frame to the Local Map frame vector
        lmap_ptr->frames.push_back(curr_frame_ptr);

        // Check if you have more than one frame
        if(lmap_ptr->frames.size() > 1){
            Frame* prev_frame_ptr = lmap_ptr->GetPreviousFrame();
            int correspondences = 0;

            correspondences = tracker->FindCorrespondences(prev_frame_ptr->points,curr_frame_ptr->points);
            
            // Simple check to see if more than once correspondence is found
            EXPECT_GT(correspondences,0);

            // Ideally we want to draw the correspondences.
            // Drawing the correspondences on a compound image
            for(auto point : curr_frame_ptr->points){
                if(point->previous != NULL){
                    // A correspondence exists
                    cv::Point2f previous_frame_point;
                    cv::Point2f current_frame_point;
                    cv::Mat joined_image;
                    cv::hconcat(prev_frame_ptr->image_l,curr_frame_ptr->image_l,joined_image);
                    previous_frame_point = point->previous->keypoint_l.keypoint.pt;
                    current_frame_point = point->keypoint_l.keypoint.pt;
                    current_frame_point.x += image_l.cols;
                    cv::line(joined_image,previous_frame_point,current_frame_point,(0,0,255),1);
                    cv::imshow(OPENCV_WINDOW_LEFT,joined_image);
                    cv::waitKey(200);
                } 
            }

        }
        image_idx++;
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

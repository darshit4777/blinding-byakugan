#include <sharp-eye/visual_tracking_fixture.hpp>
#include <sharp-eye/visual_tracking.hpp>
#include <sharp-eye/visual_triangulation.hpp>
#include <slam_datatypes/slam_datatypes.hpp>

TEST_F(VisualTrackingTest,TestLandmarkCreation){
    int image_idx = 0;
    int image_idx_max = 20;

    LocalMap lmap;
    boost::shared_ptr<LocalMap> lmap_shared_ptr = boost::make_shared<LocalMap>(lmap);
    tracker->map.local_maps.push_back(lmap_shared_ptr);
    cv::destroyWindow(OPENCV_WINDOW_RIGHT);
    while(image_idx < image_idx_max){
        cv::Mat image_l, image_r;
        image_l = GetImage(image_idx,"left");
        image_r = GetImage(image_idx,"right");

        // Create a Frame and return
        boost::shared_ptr<Frame> curr_frame_ptr = GetFrame(image_l,image_r);

        // Add Frame to the Local Map frame vector
        lmap_shared_ptr->frames.push_back(curr_frame_ptr);

        // Check if you have more than one frame
        if(lmap_shared_ptr->frames.size() > 1){
            Frame* prev_frame_ptr = lmap_shared_ptr->GetPreviousFrame();
            int correspondences = 0;

            // New Landmarks are created as a part of the find correspondences function
            correspondences = tracker->FindCorrespondences(prev_frame_ptr->points,curr_frame_ptr->points);
            
            // Plot the actively tracked landmarks
            for(int i = 0; i < tracker->actively_tracked_landmarks.size(); i++){
                Landmark* lm_ptr = tracker->actively_tracked_landmarks[i].get();

                // Get the track length of the landmark from the latest added framepoint
                int track_length = lm_ptr->optimizer.measurement_vector[0]->track_length;
                bool track_broken = false;
                // Set the origin of the landmark to a shifting pointer
                Framepoint* shifting_ptr = lm_ptr->origin.get();
                while(!track_broken){
                    cv::Point2f first_point, second_point;
                    if(shifting_ptr->next == NULL){
                        break;
                    }
                    first_point = shifting_ptr->keypoint_l.keypoint.pt;
                    second_point = shifting_ptr->next->keypoint_l.keypoint.pt;

                    // Draw a line
                    cv::line(image_l,first_point,second_point,(0,0,255),1);
                    

                    // Now switching points
                    shifting_ptr = shifting_ptr->next;

                    if(shifting_ptr->next == NULL){
                        track_broken = true;
                    }
                }
            }       
        }
        cv::imshow(OPENCV_WINDOW_LEFT,image_l);
        cv::waitKey(20);
        image_idx++;
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


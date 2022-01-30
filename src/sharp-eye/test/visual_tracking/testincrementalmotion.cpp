#include <sharp-eye/visual_tracking_fixture.hpp>
#include <sharp-eye/visual_tracking.hpp>
#include <sharp-eye/visual_triangulation.hpp>
#include <slam_datatypes/slam_datatypes.hpp>

TEST_F(VisualTrackingTest,TestIncrementalMotion){
    int image_idx = 0;
    int image_idx_max = 2;

    LocalMap lmap;
    boost::shared_ptr<LocalMap> lmap_shared_ptr = boost::make_shared<LocalMap>(lmap);
    tracker->map.local_maps.push_back(lmap_shared_ptr);
    Eigen::Transform<float,3,2> new_pose;
    Eigen::Transform<float,3,2> expected_pose;

    expected_pose.translation() << -0.000283922, -0.0153873, -0.00686548;
    Eigen::Matrix<float,3,3,0,3,3> rotation; 
    rotation << 0.999986,  -0.00112345,  -0.00521478,
                0.00113881,     0.999995,   0.00294359,
                0.00521145,  -0.00294949,     0.999982;
    
    expected_pose.matrix().block<3,3>(0,0) = rotation;


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

            correspondences = tracker->FindCorrespondences(prev_frame_ptr->points,curr_frame_ptr->points);

            
            tracker->EstimateIncrementalMotion();
            new_pose = tracker->optimizer->T_prev2curr;
        }
        image_idx++;
    }
    std::cout<<"The calculated pose is "<<std::endl;
    std::cout<<new_pose.matrix()<<std::endl;

    std::cout<<"The expected pose is "<<std::endl;
    std::cout<<expected_pose.matrix()<<std::endl;

    // Now checking if the new pose obtained is almost equal to the expected pose.
    Eigen::Transform<float,3,2> pose_difference = expected_pose.inverse() * new_pose;

    float x_diff, y_diff, z_diff, translation_tolerance;
    translation_tolerance = 0.001; // 1 mm worth of translation tolerance
    
    x_diff = pose_difference.translation().x();
    y_diff = pose_difference.translation().y();
    z_diff = pose_difference.translation().z();
    std::cout<<"Pose Difference"<<std::endl;
    std::cout<<pose_difference.matrix()<<std::endl;


    // Asserts for translation
    EXPECT_LE(fabs(x_diff),translation_tolerance);
    EXPECT_LE(fabs(y_diff),translation_tolerance);
    EXPECT_LE(fabs(z_diff),translation_tolerance);

    // For rotations we need to convert to Euler Angles / RPY to make intuitive sense.
    Eigen::Vector3f euler_angles = pose_difference.rotation().eulerAngles(0,1,2);

    // Asserts for Rotation
    float rotational_tolerance = 0.01; // Radians
    EXPECT_LE(fabs(euler_angles.x()),rotational_tolerance);
    EXPECT_LE(fabs(euler_angles.y()),rotational_tolerance);
    EXPECT_LE(fabs(euler_angles.z()),rotational_tolerance);

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}





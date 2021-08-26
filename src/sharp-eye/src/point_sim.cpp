#include<sharp-eye/point_sim.hpp>

PointSim::PointSim(){
    std::cout<<"Point Sim Created"<<std::endl;
    return;
};

void PointSim::CreateCameras(VisualSlamBase::Camera& camera_left,VisualSlamBase::Camera& camera_right){
    camera_l = camera_left;
    camera_r = camera_right;
    return;
};

void PointSim::SetCameraPositions(Eigen::Transform<float,3,2> camera_initial,Eigen::Transform<float,3,2> camera_final){
    T_world2cam1 = camera_initial;
    T_world2cam2 = camera_final;

    return;
};

void PointSim::SetCameraPositionAndTransform(Eigen::Transform<float,3,2> camera_initial,Eigen::Transform<float,3,2> transform){
    T_world2cam1 = camera_initial;
    //T_cam12cam2 = transform;
    T_world2cam2 = T_world2cam1*T_cam12cam2;
    return;
};

void PointSim::CreateRandomPoints(int no_of_points,float cube_dimension){
    n = no_of_points;
    d = cube_dimension;

    point_list.clear();

    for(int i = 0; i<n; i++){
        Eigen::Vector3f random_point;
        random_point.setRandom();
        random_point = random_point * d;
        point_list.push_back(random_point);
    };

    return;
};

cv::KeyPoint PointSim::ProjectPoints(Eigen::Vector3f point_3d,VisualSlamBase::Camera camera){
    Eigen::Vector3f pixel_coordinates;

    pixel_coordinates = camera.intrinsics * point_3d;

    pixel_coordinates[0] = pixel_coordinates[0]/pixel_coordinates[2];
    pixel_coordinates[1] = pixel_coordinates[1]/pixel_coordinates[2];

    cv::KeyPoint keypoint;
    keypoint.pt.x = pixel_coordinates[0];
    keypoint.pt.y = pixel_coordinates[1];

    return keypoint;
}
void PointSim::CreateFrames(){
    // Assign cameras
    current_frame.camera_l = camera_l;
    current_frame.camera_r = camera_r;
    previous_frame.camera_l = camera_l;
    previous_frame.camera_r = camera_r;

    // The transforms for the previous frame are supposed to be known
    previous_frame.T_cam2world = T_world2cam1.inverse();
    previous_frame.T_world2cam = T_world2cam1;

    // Assuming zero motion as the initial guess 
    current_frame.T_cam2world = previous_frame.T_cam2world;
    current_frame.T_world2cam = previous_frame.T_world2cam;

    


    for(auto point : point_list){
        VisualSlamBase::Framepoint* fp1 = new VisualSlamBase::Framepoint;
        VisualSlamBase::Framepoint* fp2 = new VisualSlamBase::Framepoint;

        // Initialize
        fp1->associated_landmark = NULL;
        fp1->inlier = false;
        fp1->landmark_set = false;

        fp2->associated_landmark = NULL;
        fp2->inlier = false;
        fp2->landmark_set = false;


        // Assign world coordinates
        fp1->world_coordinates = point;
        
        // Assign Camera coordinates
        fp1->camera_coordinates = T_world2cam1.inverse() * point;
        fp2->camera_coordinates = T_world2cam2.inverse() * point;

        // Create left projections
        fp1->keypoint_l.keypoint = ProjectPoints(fp1->camera_coordinates,camera_l);
        fp2->keypoint_l.keypoint = ProjectPoints(fp2->camera_coordinates,camera_l);

        // Create right projection
        Eigen::Vector3f right_cam_coordinates_1,right_cam_coordinates_2;
        right_cam_coordinates_1 = T_caml2camr.inverse() * fp1->camera_coordinates;
        right_cam_coordinates_2 = T_caml2camr.inverse() * fp2->camera_coordinates;

        fp1->keypoint_r.keypoint = ProjectPoints(right_cam_coordinates_1,camera_r);
        fp2->keypoint_r.keypoint = ProjectPoints(right_cam_coordinates_2,camera_r);
        
        // Check if the projections or pixels are out of bounds - if they are remove
        /// Cam 1
        if(InFieldOfView(fp1->keypoint_l.keypoint,fp1->keypoint_r.keypoint) && InFieldOfView(fp2->keypoint_l.keypoint,fp2->keypoint_r.keypoint)){
            // If the points are in field of view then we assign them to their
            // respective frames

            fp1->next = fp2;
            fp2->previous = fp1;
            previous_frame.points.push_back(*fp1);
            current_frame.points.push_back(*fp2);

            //// Now we assign the previous and next pointers - Simple right?
            //current_frame.points.back().previous = &previous_frame.points.back();
            //previous_frame.points.back().next = &current_frame.points.back();
        }
        else{
            continue;
        }
    };
    return;
};

bool PointSim::InFieldOfView(cv::KeyPoint keypoint_l,cv::KeyPoint keypoint_r){

    if(keypoint_l.pt.x > 720 || keypoint_l.pt.x < 0){
        return false;
    };
    if(keypoint_r.pt.x > 720 || keypoint_r.pt.x < 0){
        return false;
    };
    if(keypoint_l.pt.y > 480 || keypoint_l.pt.y < 0){
        return false;
    };
    if(keypoint_r.pt.y > 480 || keypoint_r.pt.y < 0){
        return false;
    };
    return true;
}; 

void PointSim::SetInterCameraTransform(Eigen::Transform<float,3,2> transform){
    T_caml2camr = transform;
    return;
};
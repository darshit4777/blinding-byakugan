#include <slam_datatypes/local_map.hpp>
typedef boost::shared_ptr<LocalMap> LocalMapSharedPtr;  
typedef boost::shared_ptr<Landmark> LandmarkSharedPtr;
typedef boost::shared_ptr<Frame> FrameSharedPtr;
typedef std::vector<boost::shared_ptr<Framepoint>> FramepointPointerVector;

LocalMap::LocalMap(){
    T_map2world.setIdentity();
    T_world2map.setIdentity();

    frames.clear();
    associated_landmarks.clear();
    return;
};

void LocalMap::CreateNewFrame(cv::Mat image_left,cv::Mat image_right,FramepointPointerVector fp_vector,Eigen::Transform<float,3,2> T_world2cam){
    FrameSharedPtr frame_ptr = boost::make_shared<Frame>();
    frame_ptr->T_world2cam = T_world2cam;
    frame_ptr->T_cam2world = T_world2cam.inverse();
    frame_ptr->image_l = image_left;
    frame_ptr->image_r = image_right;
    frame_ptr->points = fp_vector;

    frames.push_back(frame_ptr);
    return;
};

void LocalMap::AddLandmark(LandmarkSharedPtr landmark_ptr){
    // Creating a copy of the shared ptr
    LandmarkSharedPtr copy_ptr = landmark_ptr;
    associated_landmarks.push_back(copy_ptr);
    return;
};

Frame* LocalMap::GetLastFrame(){
    // Return a raw pointer to the last frame
    Frame* frame_ptr = frames.back().get();
    return frame_ptr;  
};

Frame* LocalMap::GetPreviousFrame(){
    // Return a raw pointer to the frame previous to last
    int previous_frame_idx = frames.size() - 2;
    Frame* frame_ptr = frames[previous_frame_idx].get();

    return frame_ptr;
};

LocalMap::~LocalMap(){
    return;
}
#include <sharp-eye/visual_slam_base.hpp>
typedef boost::shared_ptr<VisualSlamBase::LocalMap> LocalMapSharedPtr;  
typedef boost::shared_ptr<VisualSlamBase::Landmark> LandmarkSharedPtr;
typedef boost::shared_ptr<VisualSlamBase::Frame> FrameSharedPtr;

VisualSlamBase::LocalMap::LocalMap(){
    T_map2world.setIdentity();
    T_world2map.setIdentity();

    frames.clear();
    associated_landmarks.clear();
    return;
};

void VisualSlamBase::LocalMap::CreateNewFrame(cv::Mat image_left,cv::Mat image_right,FramepointVector fp_vector,Eigen::Transform<double,3,2> T_world2cam){
    FrameSharedPtr frame_ptr = boost::make_shared<VisualSlamBase::Frame>();
    frame_ptr->T_world2cam = T_world2cam;
    frame_ptr->T_cam2world = T_world2cam.inverse();
    frame_ptr->image_l = image_left;
    frame_ptr->image_r = image_right;
    frame_ptr->points = fp_vector;

    frames.push_back(frame_ptr);
    return;
};

void VisualSlamBase::LocalMap::AddLandmark(LandmarkSharedPtr landmark_ptr){
    // Creating a copy of the shared ptr
    LandmarkSharedPtr copy_ptr = landmark_ptr;
    associated_landmarks.push_back(copy_ptr);
    return;
};

VisualSlamBase::Frame* VisualSlamBase::LocalMap::GetLastFrame(){
    // Return a raw pointer to the last frame
    VisualSlamBase::Frame* frame_ptr = frames.back().get();
    return frame_ptr;  
};

VisualSlamBase::Frame* VisualSlamBase::LocalMap::GetPreviousFrame(){
    // Return a raw pointer to the frame previous to last
    int previous_frame_idx = frames.size() - 2;
    VisualSlamBase::Frame* frame_ptr = frames[previous_frame_idx].get();

    return frame_ptr;
};

VisualSlamBase::WorldMap::WorldMap(){
    // Create a new world map
    T_map2world.setIdentity();
    T_world2map.setIdentity();

    local_maps.clear();
    enclosed_landmarks.clear();

    pose_graph_ptr = new g2o::OptimizableGraph;
};

LocalMapSharedPtr VisualSlamBase::WorldMap::CreateNewLocalMap(Eigen::Transform<double,3,2> T_map2world){
    LocalMapSharedPtr lmap_ptr = boost::make_shared<VisualSlamBase::LocalMap>();
    
    // Location where we want to create a new local map
    lmap_ptr->T_map2world = T_map2world;
    lmap_ptr->T_world2map = lmap_ptr->T_map2world.inverse();

    local_maps.push_back(lmap_ptr);

    return lmap_ptr;
};

void VisualSlamBase::WorldMap::AddLandmark(LandmarkSharedPtr landmark_ptr){
    /**
     * @brief Add a landmark to the list of landmarks
     * 
     */
    enclosed_landmarks.push_back(landmark_ptr);
    return;
};

VisualSlamBase::WorldMap::~WorldMap(){
    delete pose_graph_ptr;
    return;
};

VisualSlamBase::LocalMap* VisualSlamBase::WorldMap::GetLastLocalMap(){
    // Return a raw pointer for local usage
    return local_maps.back().get();
};
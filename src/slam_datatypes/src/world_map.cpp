#include <slam_datatypes/world_map.hpp>

typedef boost::shared_ptr<LocalMap> LocalMapSharedPtr;  
typedef boost::shared_ptr<Landmark> LandmarkSharedPtr;
typedef boost::shared_ptr<Frame> FrameSharedPtr;

WorldMap::WorldMap(){
    // Create a new world map
    T_map2world.setIdentity();
    T_world2map.setIdentity();

    local_maps.clear();
    enclosed_landmarks.clear();

    //pose_graph_ptr = new g2o::OptimizableGraph;
};

LocalMapSharedPtr WorldMap::CreateNewLocalMap(Eigen::Transform<float,3,2> T_map2world){
    LocalMapSharedPtr lmap_ptr = boost::make_shared<LocalMap>();
    
    // Location where we want to create a new local map
    lmap_ptr->T_map2world = T_map2world;
    lmap_ptr->T_world2map = lmap_ptr->T_map2world.inverse();

    local_maps.push_back(lmap_ptr);

    return lmap_ptr;
};

void WorldMap::AddLandmark(LandmarkSharedPtr landmark_ptr){
    /**
     * @brief Add a landmark to the list of landmarks
     * 
     */
    enclosed_landmarks.push_back(landmark_ptr);
    return;
};

WorldMap::~WorldMap(){
    //delete pose_graph_ptr;
    return;
};

LocalMap* WorldMap::GetLastLocalMap(){
    // Return a raw pointer for local usage
    return local_maps.back().get();
};
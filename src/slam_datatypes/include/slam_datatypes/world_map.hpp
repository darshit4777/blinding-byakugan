#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <g2o/core/optimizable_graph.h>
#include <boost/smart_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <slam_datatypes/local_map.hpp>

class WorldMap{
    typedef boost::shared_ptr<LocalMap> LocalMapSharedPtr;  
    typedef boost::shared_ptr<Landmark> LandmarkSharedPtr;
    public:
    Eigen::Transform<float,3,2> T_map2world, T_world2map;
    std::vector<LocalMapSharedPtr> local_maps;
    std::vector<LandmarkSharedPtr> enclosed_landmarks;
    g2o::OptimizableGraph* pose_graph_ptr;
    /**
     * @brief Construct a new World Map object
     * 
     */
    WorldMap();
    /**
     * @brief Destroy the World Map object
     * 
     */
    ~WorldMap();
    /**
     * @brief Creates a New Local Map object
     * and stored the pointer to access it.
     * 
     */
    LocalMapSharedPtr CreateNewLocalMap(Eigen::Transform<float,3,2> T_map2world = Eigen::Transform<float,3,2>::Identity());
    /**
     * @brief Adds a landmark to the list of landmarks
     * 
     */
    void AddLandmark(LandmarkSharedPtr landmark_ptr);
    LocalMap* GetLastLocalMap();
};
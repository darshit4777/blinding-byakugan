# blinding-byakugan

## WIP Software Package that Implements 3D Visual SLam
This package is based on a Visual SLAM implementation inspired from Programmer's SLAM or ProSLAM https://arxiv.org/abs/1709.04377

Currently the repository is still a work in progress. 

## Directory Structure and Code Arrangement. 
The code is arranged into two packages called sharp-eye and slam_datatypes sharp-eye is the package with visual slam algorithm implementations which include multiple libraries namely
1. Visual Triangulation
2. Visual Tracking
3. Pose Optimizer
4. Point Sim

Slam Datatypes is a package to store all the datatypes used in the code. The definitions of the datatypes is heavily inspired from ProSLAM and as such the paper becomes a good reference for the code.

## Library Descriptions 
### Visual Triangulation
Visual triangulation consists of methods used for performing triangulation - computing 3D points given two images. 

### Visual Tracking 
Visual tracking implements methods and sub-methods for performing the task of incremental motion estimation and visual odometry. 

### Pose Optimizer 
Pose optimizer can be treated as a separate package of its own. It provides methods to implement an optimization problem for computing relative camera pose given two images and a an initial pose. This is a classic bundle adjustment for computing the camera pose. 

## Testing and Major issues 

### Testing 
Currently the code gets tested on the ETH ASL indoor quadcopter dataset. 

### Major issues

1. A major issue hindering progress right now is unstable incremental pose estimation. 
Incremental Pose estimation has been tested out to be working in the simulator (Point Sim). However it fails when subjected to real images. This points to possible issues in :</br>
    1. Input correspondence matching
    2. Camera Calibration





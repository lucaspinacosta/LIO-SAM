#include "utility.h"
#include "lio_sam/cloud_info.h"
#include "lio_sam/save_map.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

#include "Scancontext.h"

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;



class mapOptimization : public ParamServer
{

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;
    Eigen::MatrixXf poseCovariance_f;

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubLaserOdometryGlobal;
    ros::Publisher pubLaserOdometryIncremental;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubLoopConstraintEdge;

    ros::Publisher pubSLAMInfo;

    ros::Subscriber subCloud;
    ros::Subscriber subGPS;
    ros::Subscriber subLoop;

    ros::ServiceServer srvSaveMap;
    //ros::ServiceClient srvSaveImage;
    // std_srvs::Empty srvImage;

    std::deque<nav_msgs::Odometry> gpsQueue;
    lio_sam::cloud_info cloudInfo;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses2D; // Scan Context
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf feature set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::VoxelGrid<PointType> downSizeFilterSC; // Scan Context
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    
    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6];

    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    // cv::Mat matP;
    Eigen::Matrix<float, 6, 6> matP; // Scan Context

    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    // map<int, int> loopIndexContainer; // from new to old
    multimap<int, int> loopIndexContainer; // from new to old // Scan Context

    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    // vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue; // Diagonal <- Gausssian <- Base
    vector<gtsam::SharedNoiseModel> loopNoiseQueue; // Scan Context for polymorhpisam (Diagonal <- Gausssian <- Base)

    deque<std_msgs::Float64MultiArray> loopInfoVec;

    nav_msgs::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;

    // loop detector
    SCManager scManager; // Scan Context

    // ************** Localization ***********
    std::string saveNodePCDDirectory;
    std::string saveCornerKeyFramePCDDirectory;
    std::string saveSurfKeyFramePCDDirectory;
    std::string saveSCDDirectory; // Scan Context
    std::fstream pgTimeSaveStream;
    std::fstream similarityScoreSaveStream;
    std::fstream cmScoreSaveStream;
    std::fstream fitnessScoreSaveStream;

    NonlinearFactorGraph gtSAMgraphFromFile;
    Values initialEstimateFromFile;
    Values isamCurrentEstimateFromFile;

    pcl::PointCloud<PointType>::Ptr cloudGlobalMap;
    pcl::PointCloud<PointType>::Ptr cloudGlobalMapDS;
    pcl::PointCloud<PointType>::Ptr cloudScanForInitialize;

    pcl::PointCloud<PointType>::Ptr laserCloudRaw;
    pcl::PointCloud<PointType>::Ptr laserCloudRawDS;

    pcl::PointCloud<PointType>::Ptr cloudCornerFromFile;
    pcl::PointCloud<PointType>::Ptr cloudSurfFromFile;
    pcl::PointCloud<PointType>::Ptr cloudGlobalFromFile;

    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3DFromFile;
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D_Q_FromFile;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6DFromFile;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFramesFromFile;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFramesFromFile;

    std::mutex mtx_general;

    enum InitializedFlag
    {
        NonInitialized,
        Initializing,
        Initialized
    };
    InitializedFlag initializedFlag;

    float initialPose[6];
    double laserCloudRawTime;
    float transformInTheWorld[6];

    // Scan Context
    std::vector<Eigen::MatrixXd> polarcontexts_FromFile;
    std::vector<Eigen::MatrixXd> polarcontexts_Q_FromFile;

    mapOptimization()
    {
        // ISAM2Params parameters;
        // parameters.relinearizeThreshold = 0.1;
        // parameters.relinearizeSkip = 1;
        // if (factorization == 1)
        //     parameters.factorization = gtsam::ISAM2Params::QR;
        // isam = new ISAM2(parameters);

        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/trajectory", 1);
        pubLaserCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_global", 1);
        pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry", 1);
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry_incremental", 1);
        pubPath                     = nh.advertise<nav_msgs::Path>("lio_sam/mapping/path", 1);

        subCloud = nh.subscribe<lio_sam::cloud_info>("lio_sam/feature/cloud_info", 1, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subGPS   = nh.subscribe<nav_msgs::Odometry> (gpsTopic, 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        subLoop  = nh.subscribe<std_msgs::Float64MultiArray>("lio_loop/loop_closure_detection", 1, &mapOptimization::loopInfoHandler, this, ros::TransportHints().tcpNoDelay());

        srvSaveMap  = nh.advertiseService("lio_sam/save_map", &mapOptimization::saveMapService, this);
        // srvSaveImage = nh.serviceClient<std_srvs::Empty>("image_saver/save");

        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("lio_sam/mapping/loop_closure_constraints", 1);

        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_local", 1);
        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1);

        const float kSCFilterSize = 0.5; // Scan Context
        downSizeFilterSC.setLeafSize(kSCFilterSize, kSCFilterSize, kSCFilterSize); // Scan Context

        pubSLAMInfo           = nh.advertise<lio_sam::cloud_info>("lio_sam/mapping/slam_info", 1);

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        allocateMemory();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR); // Scan Context

        switch(liosamMode) {
            case 0:
                initNameFolder(std::getenv("ROS_MAP_PATH_DEFAULT"));
                slamModeInit();
                break;
            case 1:
                initNameFolder(std::getenv("ROS_MAP_PATH_DEFAULT"));
                locModeInit();
                break;
            case 2:
                appModeInit();
                break;
            default:
                cout << "LIO-SAM must be initialized." << endl;
        }
    }

    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses2D.reset(new pcl::PointCloud<PointType>()); // Scan Context
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        // matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0)); // Scan Context
        matP.setZero();

        // ************** Localization ***********
        cloudGlobalMap.reset(new pcl::PointCloud<PointType>());
	    cloudGlobalMapDS.reset(new pcl::PointCloud<PointType>());
        cloudScanForInitialize.reset(new pcl::PointCloud<PointType>());

        laserCloudRaw.reset(new pcl::PointCloud<PointType>());
        laserCloudRawDS.reset(new pcl::PointCloud<PointType>());

        cloudCornerFromFile.reset(new pcl::PointCloud<PointType>());
	    cloudSurfFromFile.reset(new pcl::PointCloud<PointType>());
        cloudGlobalFromFile.reset(new pcl::PointCloud<PointType>());

        cloudKeyPoses3DFromFile.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses3D_Q_FromFile.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6DFromFile.reset(new pcl::PointCloud<PointTypePose>());

        for (int i = 0; i < 6; ++i){
            initialPose[i] = 0;
        }

        for (int i = 0; i < 6; ++i){
            transformInTheWorld[i] = 0;
        }
    }

    void slamModeInit()
    {
        cout << "****************************************************" << endl;
        cout << "LIO-SAM: SLAM Mode" << endl;
        cout << "****************************************************" << endl;
        initializedFlag = Initialized;
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        parameters.factorization = gtsam::ISAM2Params::CHOLESKY;
        isam = new ISAM2(parameters);
        if (savePCD == true)
            initFolder();
        feedback_mode = 0;
    }

    void locModeInit()
    {
        cout << "****************************************************" << endl;
        cout << "LIO-SAM: Localization Mode" << endl;
        cout << "****************************************************" << endl;
        initializedFlag = NonInitialized;
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        parameters.factorization = gtsam::ISAM2Params::QR;
        isam = new ISAM2(parameters);
        savePCD = false;
        saveGTSAM = false;

        if (!loadGraph())
        {
        ROS_ERROR("It was not possible to load graph.");
        ros::shutdown();
        }

        if (!cloudGlobalLoad())
        {
        ROS_ERROR("It was not possible to load global map.");
        }

        if (!(loadFiles() == 0))
        {
        ROS_ERROR("It was not possible to load files.");
        ros::shutdown();
        }

        *cloudKeyPoses3D = *cloudKeyPoses3DFromFile;
        *cloudKeyPoses6D = *cloudKeyPoses6DFromFile;

        std::cout << " **** Points (cloudKeyPoses3D): " << cloudKeyPoses3D->points.size() << std::endl;
        std::cout << " **** Points (cloudKeyPoses6D): " << cloudKeyPoses6D->points.size() << std::endl;

        for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++)
        {
        cornerCloudKeyFrames.push_back(cornerCloudKeyFramesFromFile[i]);
        surfCloudKeyFrames.push_back(surfCloudKeyFramesFromFile[i]);
        }
        feedback_mode = 1;
    }

    void appModeInit()
    {
        cout << "****************************************************" << endl;
        cout << "LIO-SAM: App Mode" << endl;
        cout << "****************************************************" << endl;
        cout << "" << endl;
        cout << "Waiting...." << endl;
        cout << "" << endl;

        app_waiting_user = true;
        feedback_mode = 3;
    }

    void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr& msgIn)
    {
        // extract time stamp
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = msgIn->header.stamp.toSec();

        // extract info and feature cloud
        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_corner,  *laserCloudCornerLast);
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

        pcl::fromROSMsg(msgIn->cloud_deskewed,  *laserCloudRaw);
        laserCloudRawTime = cloudInfo.header.stamp.toSec();

        // Waiting Application
        if (app_waiting_user == true)
            return;

        if ((liosamMode == 2) && (app_initialized != true))
        {
            initNameFolder(app_map_name);
            if (app_liosam_mode == 0) {
                slamModeInit();
                app_initialized = true;

            } else if (app_liosam_mode == 1) {
                locModeInit();
                app_initialized = true;
            } else {
                ROS_ERROR("Reapet the message: wrong mode.");
                app_waiting_user = true;
            }
        }

        if (initializedFlag == NonInitialized || initializedFlag == Initializing)
        {
            if (cloudScanForInitialize->points.size() == 0)
            {
                initLocalization();
            }
            return;
        }

        std::lock_guard<std::mutex> lock(mtx);

        static double timeLastProcessing = -1;
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        {
            timeLastProcessing = timeLaserInfoCur;

            updateInitialGuess();

            extractSurroundingKeyFrames();

            downsampleCurrentScan();

            scan2MapOptimization();

            saveKeyFramesAndFactor();

            correctPoses();

            publishOdometry();

            publishFrames();
        }
    }

    void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {
        gpsQueue.push_back(*gpsMsg);
    }

    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    













    bool saveMapService(lio_sam::save_mapRequest& req, lio_sam::save_mapResponse& res)
    {
      if (cloudKeyPoses3D->points.empty()){
        cout << "Input point cloud has no data" << endl;
        return false;
      }

      string saveMapDirectory;

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files ..." << endl;
    //   if(req.destination.empty()) saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
    //   else saveMapDirectory = std::getenv("HOME") + req.destination;
      if(req.destination.empty()) saveMapDirectory = saveMainDirectory;
      else saveMapDirectory = std::getenv("HOME") + req.destination;
      cout << "Save destination: " << saveMapDirectory << endl;
      // create directory and remove old files;
      // int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
      // unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
      // save key frame transformations
      pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
      pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
      // extract global point cloud map
      pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
      for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
          *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
          *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
          cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
      }

      if(req.resolution != 0)
      {
        cout << "\n\nSave resolution: " << req.resolution << endl;

        // down-sample and save corner cloud
        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        downSizeFilterCorner.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterCorner.filter(*globalCornerCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);
        // down-sample and save surf cloud
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterSurf.filter(*globalSurfCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
      }
      else
      {
        // save corner cloud
        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
        // save surf cloud
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
      }

      // save global point cloud map
      *globalMapCloud += *globalCornerCloud;
      *globalMapCloud += *globalSurfCloud;

      int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
      res.success = ret == 0;

      downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
      downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files completed\n" << endl;

      return true;
    }

    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            if (globalMapVisualizationFile == true)
                publishMap();
            else
                publishGlobalMap();
        }

        if (savePCD == false)
            return;

        lio_sam::save_mapRequest  req;
        lio_sam::save_mapResponse res;

        if(!saveMapService(req, res)){
            cout << "Fail to save map" << endl;
        }
    }

    void publishGlobalMap()
    {
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        for(auto& pt : globalMapKeyPosesDS->points)
        {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
        }

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }












    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(loopClosureFrequency);
        while (ros::ok())
        {
            rate.sleep();
            if (performRSLoopClosureFlag)
                performLoopClosure();
            if (performSCLoopClosureFlag)
                performSCLoopClosure(); // Scan Context
            visualizeLoopClosure();
        }
    }

    void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopMsg->data.size() != 2)
            return;

        loopInfoVec.push_back(*loopMsg);

        while (loopInfoVec.size() > 5)
            loopInfoVec.pop_front();
    }

    void performLoopClosure()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        copy_cloudKeyPoses2D->clear(); // Scan Context
        *copy_cloudKeyPoses2D = *cloudKeyPoses3D; // Scan Context
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // find keys
        int loopKeyCur;
        int loopKeyPre;
        if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)
            if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
                return;

        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        }

        // ICP Settings
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // publish corrected cloud
        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
        }

        // Get pose transformation
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();
        // transform from world origin to wrong pose
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();

        // add loop constriant
        // loopIndexContainer[loopKeyCur] = loopKeyPre;
        loopIndexContainer.insert(std::pair<int, int>(loopKeyCur, loopKeyPre)); // Scan Context for multimap
    }


    void performSCLoopClosure()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        copy_cloudKeyPoses2D->clear(); // Scan Context
        *copy_cloudKeyPoses2D = *cloudKeyPoses3D; // Scan Context
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        if (scManager.polarcontexts_.size() == 0)
            return;

        // find keys
        auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;;
        int loopKeyPre = detectResult.first;
        float yawDiffRad = detectResult.second; // not use for v1 (because pcl icp withi initial somthing wrong...)
        if( loopKeyPre == -1 /* No loop found */)
            return;

        // std::cout << "SC loop found! between " << loopKeyCur << " and " << loopKeyPre << "." << std::endl; // giseop

        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {

            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
            
            // loopFindNearKeyframesWithRespectTo(cureKeyframeCloud, loopKeyCur, 0, loopKeyPre); // giseop
            // loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);

            // int base_key = 0;
            // loopFindNearKeyframesWithRespectTo(cureKeyframeCloud, loopKeyCur, 0, base_key); // giseop
            // loopFindNearKeyframesWithRespectTo(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum, base_key); // giseop

            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        }

        // ICP Settings
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);
        // giseop
        // TODO icp align with initial

        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore) {
            // std::cout << "ICP fitness test failed (" << icp.getFitnessScore() << " > " << historyKeyframeFitnessScore << "). Reject this SC loop." << std::endl;
            return;
        // } else {
        //     std::cout << "ICP fitness test passed (" << icp.getFitnessScore() << " < " << historyKeyframeFitnessScore << "). Add this SC loop." << std::endl;
        }

        // publish corrected cloud
        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
        }

        // Get pose transformation
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();

        // transform from world origin to wrong pose
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);

        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();

        // add loop constriant
        // loopIndexContainer[loopKeyCur] = loopKeyPre;
        loopIndexContainer.insert(std::pair<int, int>(loopKeyCur, loopKeyPre)); // Scan Context for multimap

        // // giseop
        // pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
        // gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        // gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

        // // giseop, robust kernel for a SC loop
        // float robustNoiseScore = 0.5; // constant is ok...
        // gtsam::Vector robustNoiseVector6(6);
        // robustNoiseVector6 << robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore;
        // noiseModel::Base::shared_ptr robustConstraintNoise;
        // robustConstraintNoise = gtsam::noiseModel::Robust::Create(
        //     gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure, but with a good front-end loop detector, Cauchy is empirically enough.
        //     gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6)
        // ); // - checked it works. but with robust kernel, map modification may be delayed (i.e,. requires more true-positive loop factors)

        // // Add pose constraint
        // mtx.lock();
        // loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        // loopPoseQueue.push_back(poseFrom.between(poseTo));
        // loopNoiseQueue.push_back(robustConstraintNoise);
        // mtx.unlock();

        // // add loop constriant
        // // loopIndexContainer[loopKeyCur] = loopKeyPre;
        // loopIndexContainer.insert(std::pair<int, int>(loopKeyCur, loopKeyPre)); // Scan Context for multimap
    } // performSCLoopClosure

    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        // kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        // kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

        for (int i = 0; i < (int)copy_cloudKeyPoses2D->size(); i++) // Scan Context
            copy_cloudKeyPoses2D->points[i].z = 1.1; // to relieve the z-axis drift, 1.1 is just foo val

        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses2D); // Scan Context
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses2D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0); // Scan Context

        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
                break;
            }
        }

        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    bool detectLoopClosureExternal(int *latestID, int *closestID)
    {
        // this function is not used yet, please ignore it
        int loopKeyCur = -1;
        int loopKeyPre = -1;

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopInfoVec.empty())
            return false;

        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();

        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
            return false;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return false;

        // latest key
        loopKeyCur = cloudSize - 1;
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        loopKeyPre = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        if (loopKeyCur == loopKeyPre)
            return false;

        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    void loopFindNearKeyframesWithRespectTo(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum, const int _wrt_key)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[_wrt_key]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[_wrt_key]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    void visualizeLoopClosure()
    {
        // if (loopIndexContainer.empty())
        //     return;

        visualization_msgs::MarkerArray markerArray;
        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);
    }







    



    void updateInitialGuess()
    {
        // save current transformation before any processing
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

        static Eigen::Affine3f lastImuTransformation;
        // initialization
        if (cloudKeyPoses3D->points.empty() || initializedFlag == NonInitialized)
        {
            transformTobeMapped[0] = cloudInfo.imuRollInit;
            transformTobeMapped[1] = cloudInfo.imuPitchInit;
            transformTobeMapped[2] = cloudInfo.imuYawInit;

            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;

            // lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            lastImuTransformation = pcl::getTransformation(initialPose[3], initialPose[4], initialPose[5], //
                    cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }

        // use imu pre-integration estimation for pose guess
        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation;
        if (cloudInfo.odomAvailable == true)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX,    cloudInfo.initialGuessY,     cloudInfo.initialGuessZ, 
                                                               cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
            if (lastImuPreTransAvailable == false)
            {
                lastImuPreTransformation = transBack;
                lastImuPreTransAvailable = true;
            } else {
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                Eigen::Affine3f transFinal = transTobe * transIncre;
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

                lastImuPreTransformation = transBack;

                // lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
                lastImuTransformation = pcl::getTransformation(initialPose[3], initialPose[4], initialPose[5], //
                    cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
                return;
            }
        }

        // use imu incremental estimation for pose guess (only rotation)
        if (cloudInfo.imuAvailable == true)
        {
            // Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
            Eigen::Affine3f transBack = pcl::getTransformation(initialPose[3], initialPose[4], initialPose[5], //
                    cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            lastImuTransformation = pcl::getTransformation(initialPose[3], initialPose[4], initialPose[5], //
                    cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            //lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }
    }

    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(cloudToExtract);
    }

    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
        for(auto& pt : surroundingKeyPosesDS->points)
        {
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }

        // also extract some latest key frames in case the robot rotates in one position
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(surroundingKeyPosesDS);
    }

    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // fuse the map
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;

            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) 
            {
                // transformed cloud available
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            } else {
                // transformed cloud not available
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap   += laserCloudSurfTemp;
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
            
        }

        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // clear map cache if too large
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }

    void extractSurroundingKeyFrames()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return; 
        
        // if (loopClosureEnableFlag == true)
        // {
        //     extractForLoopClosure();    
        // } else {
        //     extractNearby();
        // }

        extractNearby();
    }

    void downsampleCurrentScan()
    {
        // Scan Context
        laserCloudRawDS->clear();
        downSizeFilterSC.setInputCloud(laserCloudRaw);
        downSizeFilterSC.filter(*laserCloudRawDS);

        // Downsample cloud from current scan
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    void cornerOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
                    
            if (pointSearchSqDis[4] < 1.0) {
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                cv::eigen(matA1, matD1, matV1);

                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;

                    float s = 1 - 0.9 * fabs(ld2);

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    if (s > 0.1) {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    void surfOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel); 
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointOri.x * pointOri.x
                            + pointOri.y * pointOri.y + pointOri.z * pointOri.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i){
            if (laserCloudOriCornerFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // reset flag for next iteration
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    bool LMOptimization(int iterCount)
    {
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0)); // Scan Context

        PointType pointOri, coeff;

        for (int i = 0; i < laserCloudSelNum; i++) {
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
            // camera -> lidar
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            matA.at<float>(i, 3) = coeff.z;
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {

            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        return false; // keep optimizing
    }

    void scan2MapOptimization()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization();
                surfOptimization();

                combineOptimizationCoeffs();

                if (LMOptimization(iterCount) == true)
                    break;              
            }

            transformUpdate();
        } else {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

    void transformUpdate()
    {
        if (cloudInfo.imuAvailable == true)
        {
            if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            {
                double imuWeight = imuRPYWeight;
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
        {
            lastSaveSecs = ros::Time::now().toSec();
            return true;
        }

        if (sensor == SensorType::LIVOX)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->back().time > 1.0)
                return true;
        }

        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        if (saveLastPCD == true)
        {
            if (((float)(ros::Time::now().toSec() - lastSaveSecs) > 5.0) && waitSaveLastPCD == true)
            {
                // std::cout << "Time" << std::endl;
                lastSaveSecs = ros::Time::now().toSec();
                waitSaveLastPCD = false;
                return true;
            }
        }

        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        waitSaveLastPCD = true;
        lastSaveSecs = ros::Time::now().toSec();

        return true;
    }

    void addOdomFactor()
    {
        if (cloudKeyPoses3D->points.empty())
        {
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << priorNoiseVector, priorNoiseVector, priorNoiseVector, priorNoiseVector, priorNoiseVector, priorNoiseVector).finished()); // rad*rad, meter*meter
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }else{
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    void addGPSFactor()
    {
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down
        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {
            if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;
        }

        // pose covariance small, no need to correct
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;

                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                if (!useGpsElevation)
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);

                aLoopIsClosed = true;
                break;
            }
        }
    }

    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            // gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i]; // original
            auto noiseBetween = loopNoiseQueue[i]; // Scan Context for polymorhpism // shared_ptr<gtsam::noiseModel::Base>, typedef noiseModel::Base::shared_ptr gtsam::SharedNoiseModel
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }

    void saveKeyFramesAndFactor()
    {
        if (saveFrame() == false)
            return;

        // odom factor
        addOdomFactor();

        // gps factor
        addGPSFactor();

        // loop factor
        addLoopFactor();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // update iSAM
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        if (aLoopIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }

        if (saveGTSAM == true)
                saveGraph();

        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);
        if ((poseCovariance(3,3) < covarianceMedium) && (poseCovariance(4,4) < covarianceMedium))
            feedback_quality = 1;
        else
            if ((poseCovariance(3,3) > covarianceHigh) || (poseCovariance(4,4) > covarianceHigh))
                feedback_quality = 3;
            else
                feedback_quality = 2;
        // cout << "****************************************************" << endl;
        // cout << "Feedback Quality:" << endl;
        // cout << poseCovariance(3,3) << endl;
        // cout << poseCovariance(4,4) << endl;
        // cout << feedback_quality << endl << endl;

        // save updated transform
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // save key frame cloud
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // Scan Context loop detector - giseop
        // - SINGLE_SCAN_FULL: using downsampled original point cloud (/full_cloud_projected + downsampling)
        // - SINGLE_SCAN_FEAT: using surface feature as an input point cloud for scan context (2020.04.01: checked it works.)
        // - MULTI_SCAN_FEAT: using NearKeyframes (because a MulRan scan does not have beyond region, so to solve this issue ... )
        if( scdInput == SCInputType::SINGLE_SCAN_FULL ) {
            pcl::PointCloud<PointType>::Ptr thisRawCloudKeyFrame(new pcl::PointCloud<PointType>());
            pcl::copyPointCloud(*laserCloudRawDS,  *thisRawCloudKeyFrame);
            scManager.makeAndSaveScancontextAndKeys(*thisRawCloudKeyFrame);
        }  
        else if (scdInput == SCInputType::SINGLE_SCAN_FEAT) {
            pcl::PointCloud<PointType>::Ptr scanFeat(new pcl::PointCloud<PointType>());
            *scanFeat += *thisCornerKeyFrame;
            *scanFeat += *thisSurfKeyFrame;
            scManager.makeAndSaveScancontextAndKeys(*scanFeat);
        }
        else if (scdInput == SCInputType::MULTI_SCAN_FEAT) {
            pcl::PointCloud<PointType>::Ptr multiKeyFrameFeatureCloud(new pcl::PointCloud<PointType>());
            loopFindNearKeyframes(multiKeyFrameFeatureCloud, cloudKeyPoses6D->size() - 1, historyKeyframeSearchNum);
            scManager.makeAndSaveScancontextAndKeys(*multiKeyFrameFeatureCloud);
        }

        // save path for visualization
        updatePath(thisPose6D);

        // save keyframe cloud as file
        pcl::PointCloud<PointType>::Ptr thisKeyFrameCloud(new pcl::PointCloud<PointType>());
        if (savePCD)
        {
            cout << "****************************************************" << endl;
            cout << "Saving Key Frame: " << cornerCloudKeyFrames.size() << endl;
            if(saveRawCloud)
            {
                *thisKeyFrameCloud += *laserCloudRaw;
            }
            else
            {
                *thisKeyFrameCloud += *thisCornerKeyFrame;
                *thisKeyFrameCloud += *thisSurfKeyFrame;
            }
            std::string curr_scd_node_idx = padZeros(cornerCloudKeyFrames.size() - 1);
            pcl::io::savePCDFileBinary(saveNodePCDDirectory + curr_scd_node_idx + ".pcd", *thisKeyFrameCloud);
            pcl::io::savePCDFileBinary(saveCornerKeyFramePCDDirectory + "C" + curr_scd_node_idx + ".pcd", *thisCornerKeyFrame);
            pcl::io::savePCDFileBinary(saveSurfKeyFramePCDDirectory + "S" + curr_scd_node_idx + ".pcd", *thisSurfKeyFrame);

            // save sc data
            const auto& curr_scd = scManager.getConstRefRecentSCD();
            curr_scd_node_idx = padZeros(scManager.polarcontexts_.size() - 1);
            saveSCD(saveSCDDirectory + curr_scd_node_idx + ".scd", curr_scd);
        }

        // Save time stream
        pgTimeSaveStream << laserCloudRawTime << std::endl;

        // toSave = false;

        // Add image service
        // srvSaveImage.call(srvImage);
    }

    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (aLoopIsClosed == true)
        {
            // clear map cache
            laserCloudMapContainer.clear();
            // clear path
            globalPath.poses.clear();
            // update key poses
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                updatePath(cloudKeyPoses6D->points[i]);
            }

            aLoopIsClosed = false;
        }
    }

    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    void publishOdometry()
    {
        // Publish odometry for ROS (global)
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        poseCovariance_f = poseCovariance.cast <float> ();  // Pose covariance
        for (int i = 0; i < 36; i++) {
            laserOdometryROS.pose.covariance[i] =  poseCovariance_f(i);
        }
        pubLaserOdometryGlobal.publish(laserOdometryROS);
        
        // Publish TF
        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                      tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");
        br.sendTransform(trans_odom_to_lidar);

        // Publish odometry for ROS (incremental)
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
        static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometryROS;
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        } else {
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
            if (cloudInfo.imuAvailable == true)
            {
                if (std::abs(cloudInfo.imuPitchInit) < 1.4)
                {
                    double imuWeight = 0.1;
                    tf::Quaternion imuQuaternion;
                    tf::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;

                    // slerp roll
                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;

                    // slerp pitch
                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odometryFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
            if (isDegenerate)
                laserOdomIncremental.pose.covariance[0] = 1;
            else
                laserOdomIncremental.pose.covariance[0] = 0;
        }
        pubLaserOdometryIncremental.publish(laserOdomIncremental);
    }

    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // publish key poses
        publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
        // Publish surrounding key frames
        publishCloud(pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
        // publish registered key frame
        if (pubRecentKeyFrame.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish registered high-res raw cloud
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            publishCloud(pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish path
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
        // publish SLAM infomation for 3rd-party usage
        static int lastSLAMInfoPubSize = -1;
        if (pubSLAMInfo.getNumSubscribers() != 0)
        {
            if (lastSLAMInfoPubSize != cloudKeyPoses6D->size())
            {
                lio_sam::cloud_info slamInfo;
                slamInfo.header.stamp = timeLaserInfoStamp;
                pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
                *cloudOut += *laserCloudCornerLastDS;
                *cloudOut += *laserCloudSurfLastDS;
                slamInfo.key_frame_cloud = publishCloud(ros::Publisher(), cloudOut, timeLaserInfoStamp, lidarFrame);
                slamInfo.key_frame_poses = publishCloud(ros::Publisher(), cloudKeyPoses6D, timeLaserInfoStamp, odometryFrame);
                pcl::PointCloud<PointType>::Ptr localMapOut(new pcl::PointCloud<PointType>());
                *localMapOut += *laserCloudCornerFromMapDS;
                *localMapOut += *laserCloudSurfFromMapDS;
                slamInfo.key_frame_map = publishCloud(ros::Publisher(), localMapOut, timeLaserInfoStamp, odometryFrame);
                pubSLAMInfo.publish(slamInfo);
                lastSLAMInfoPubSize = cloudKeyPoses6D->size();
            }
        }
    }

    // Localization Functions

    bool saveGraph()
    {
        string saveDirectory;
        saveDirectory = saveMainDirectory + saveGTSAMDirectory;

        // cout << "Save destination: " << saveDirectory << endl;
        // create directory and remove old files;
        int unused_g = system((std::string("exec rm -r ") + saveDirectory).c_str());
        unused_g = system((std::string("mkdir -p ") + saveDirectory).c_str());
        gtsam::writeG2o(isam->getFactorsUnsafe(), isam->calculateEstimate(), saveDirectory + "map.g2o");
        ofstream os(saveDirectory + "map_graph.dot");
        gtSAMgraph.saveGraph(os, isam->calculateEstimate());
        cout << "****************************************************" << endl;
        if (unused_g == 0)
            cout << "Saving Factor Graph completed" << endl;

        return true;
    }

    bool loadGraph()
    {
        NonlinearFactorGraph::shared_ptr graph;
        Values::shared_ptr initial;
        Key firstKey = 0;

        string saveDirectory;

        saveDirectory = saveMainDirectory + saveGTSAMDirectory;
        // cout << "Save destination: " << saveDirectory << endl;
        const char *saveDirectoryPath = saveDirectory.c_str();

        if (!dirExists(saveDirectoryPath))
        {
            cout << "GTSAM directory Path failed. "<< endl;
            return false;
        }

        boost::tie(graph, initial) = gtsam::readG2o(saveDirectory + "map.g2o", true);
        gtSAMgraphFromFile = *graph;
        initialEstimateFromFile = *initial;
        // gtSAMgraphFromFile.print("GTSAM Graph:\n");
        // initialEstimateFromFile.print("Initial estimate: ");

        noiseModel::Diagonal::shared_ptr odometryNoise = //
            noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());

        gtSAMgraphFromFile.add(PriorFactor<Pose3>(firstKey, initialEstimateFromFile.at<Pose3>(0), odometryNoise));

        try {
            // update iSAM
            isam->update(gtSAMgraphFromFile, initialEstimateFromFile);

            // Perform iSAM2 update on the loaded iSAM2 object
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();

            if (graphTest == true)
            {
                ofstream os(saveDirectory + "map_lio.dot");
                gtSAMgraphFromFile.saveGraph(os, isam->calculateEstimate());
                gtsam::writeG2o(isam->getFactorsUnsafe(), isam->calculateEstimate(), saveDirectory + "map1.g2o");
            }

            gtSAMgraphFromFile.resize(0);
            initialEstimateFromFile.clear();

        } catch (const gtsam::IndeterminantLinearSystemException& e) {
            // Handle the exception
            cout << "IndeterminantLinearSystemException: " << e.what() << endl;
            // Additional error handling or recovery logic can be added here
            return false;
        }

        isamCurrentEstimateFromFile = isam->calculateEstimate();
        cout << "****************************************************" << endl;
        std::cout << "iSAM estimate size: " << isamCurrentEstimateFromFile.size() << std::endl;

        // //save key poses
        // Pose3 latestEstimate;
        // isamCurrentEstimateFromFile = isam->calculateEstimate();
        // latestEstimate = isamCurrentEstimateFromFile.at<Pose3>(isamCurrentEstimateFromFile.size()-1);
        // cout << "****************************************************" << endl;
        // latestEstimate.print("Current estimate: ");

        return true;
    }

    bool initNameFolder(string mapName)
    {
        if ((liosamMode == 0) || (liosamMode == 1)) {
            saveMainDirectory = mapName + savePCDDirectory;
        } else if (liosamMode == 2) {
            saveMainDirectory = std::getenv("ROS_MAP_PATH") + std::string("/") + mapName + savePCDDirectory;
        } else {
            // ROS_ERROR("Repeat the message: wrong mode.");
            cout << "Repeat the message: wrong mode." << endl;
            return false;
        }
        cout << "Save destination: " << saveMainDirectory << endl;
        return true;
    }

    bool initFolder()
    {
        int unused;
        string saveDirectory;

        saveDirectory = saveMainDirectory;
        // cout << "Save destination: " << saveDirectory << endl;
        const char *saveDirectoryPath = saveDirectory.c_str();

        if (dirExists(saveDirectoryPath))
        {
            if (eraseLOAMFolder)
            {
                cout << "eraseLOAM param set to true - Removing LOAM Directory" << saveDirectory << endl;
                unused = system((std::string("exec rm -r ") + saveDirectory).c_str());
            }
        }
        else
        {
            cout << "No LOAM directory Found - Creating LOAM Directory" << saveDirectory << endl;
		    unused = system((std::string("mkdir ") + saveDirectory).c_str());
        }

        saveSCDDirectory = saveDirectory + "SCDs/"; // SCD: scan context descriptor
        unused = system((std::string("mkdir -p ") + saveSCDDirectory).c_str());

        saveNodePCDDirectory = saveDirectory + "Scans/";
        unused = system((std::string("mkdir -p ") + saveNodePCDDirectory).c_str());

        saveCornerKeyFramePCDDirectory = saveNodePCDDirectory + "CornerFrames/";
        unused = system((std::string("mkdir -p ") + saveCornerKeyFramePCDDirectory).c_str());

        saveSurfKeyFramePCDDirectory = saveNodePCDDirectory + "SurfFrames/";
        unused = system((std::string("mkdir -p ") + saveSurfKeyFramePCDDirectory).c_str());

        pgTimeSaveStream = std::fstream(saveDirectory + "times.txt", std::fstream::out); pgTimeSaveStream.precision(dbl::max_digits10);

        cout << "****************************************************" << endl;
        if (unused == 0)
            cout << "Folders created." << endl;

        return true;
    }

    bool dirExists(const char *path)
    {
	    struct stat info;

	    if(stat( path, &info ) != 0)
		    return false;
	    else if(info.st_mode & S_IFDIR)
		    return true;
	    else
		    return false;
    }

    bool cloudGlobalLoad()
    {
        string saveDirectory;

        saveDirectory = saveMainDirectory;
        // cout << "Save destination: " << saveDirectory << endl;
        const char *saveDirectoryPath = saveDirectory.c_str();

        if (!dirExists(saveDirectoryPath))
        {
            cout << "PCD directory Path failed. "<< endl;
            return false;
        }

        pcl::io::loadPCDFile(saveDirectory + "GlobalMap.pcd", *cloudGlobalMap);

        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(cloudGlobalMap);
        downSizeFilterICP.filter(*cloud_temp);
        *cloudGlobalMapDS = *cloud_temp;

        cout << "****************************************************" << endl;
        cout << "Loading Global Map" << endl;
        cout << "The size of global cloud: " << cloudGlobalMap->points.size() << endl;
        cout << "The size of global map after filter: " << cloudGlobalMapDS->points.size() << endl;

        return true;
    }

    void publishMap()
    {
        sensor_msgs::PointCloud2 point_cloud_msg;
        if (globalMapVisualizationFileFilter == true)
        {
            pcl::toROSMsg(*cloudGlobalMapDS, point_cloud_msg);
        }
        else
        {
            pcl::toROSMsg(*cloudGlobalMap, point_cloud_msg);
        }
        point_cloud_msg.header.stamp = ros::Time::now();
        point_cloud_msg.header.frame_id = "map";
        pubLaserCloudSurround.publish(point_cloud_msg);
    }

    int loadFiles()
    {
        string saveDirectory;

        saveDirectory = saveMainDirectory;
        // saveDirectory = std::getenv("HOME") + savePCDDirectory;
        // cout << "Save destination: " << saveDirectory << endl;

        if (pcl::io::loadPCDFile<PointType>(saveDirectory + "trajectory.pcd", *cloudKeyPoses3DFromFile) == -1)
        {
            PCL_ERROR("Couldn't read file trajectory.pcd \n No cloudKeyPoses3D");
            return (-1);
        }

        if (pcl::io::loadPCDFile<PointTypePose>(saveDirectory + "transformations.pcd", *cloudKeyPoses6DFromFile) == -1)
        {
            PCL_ERROR("Couldn't read file transformations.pcd \n No cloudKeyPoses6D");
            return (-1);
        }

        cout << "****************************************************" << endl;
        std::cout << " **** Points (cloudKeyPoses3DFromFile): " << cloudKeyPoses3DFromFile->points.size() << std::endl;
        std::cout << " **** Points (cloudKeyPoses6DFromFile): " << cloudKeyPoses6DFromFile->points.size() << std::endl;
        // std::cout << "Points (cloudKeyPoses3DFromFile[0].x): " << (*cloudKeyPoses3DFromFile)[0].x << std::endl;
        // std::cout << "Points (cloudKeyPoses3DFromFile[0].y): " << (*cloudKeyPoses3DFromFile)[0].y << std::endl;
        // std::cout << "Points (cloudKeyPoses3DFromFile[0].z): " << (*cloudKeyPoses3DFromFile)[0].z << std::endl;

        saveNodePCDDirectory = saveDirectory + "Scans/";
        saveCornerKeyFramePCDDirectory = saveNodePCDDirectory + "CornerFrames/";

        for (int i = 0; i < (int)cloudKeyPoses3DFromFile->points.size(); i++)
        {
            std::string curr_scd_node_idx = padZeros(i);
            std::string pathString = saveCornerKeyFramePCDDirectory + "C" + curr_scd_node_idx + ".pcd";
            pcl::PointCloud<PointType>::Ptr thisCornerKeyFrameCloud(new pcl::PointCloud<PointType>());

            if (pcl::io::loadPCDFile<PointType> (pathString, *thisCornerKeyFrameCloud) == -1)
            {
                PCL_ERROR ("Couldn't read file cloudCorner.pcd \n");
                return (-1);
            }
            else
            {
                cornerCloudKeyFramesFromFile.push_back(thisCornerKeyFrameCloud);
            }
        }

        cout << " **** Loading Corner Key Frames: "<< cornerCloudKeyFramesFromFile.size() << endl;

        saveNodePCDDirectory = saveDirectory + "Scans/";
        saveSurfKeyFramePCDDirectory = saveNodePCDDirectory + "SurfFrames/";

        for (int i = 0; i < (int)cloudKeyPoses3DFromFile->points.size(); i++)
        {
            std::string curr_scd_node_idx = padZeros(i);
            std::string pathString = saveSurfKeyFramePCDDirectory + "S" + curr_scd_node_idx + ".pcd";
            pcl::PointCloud<PointType>::Ptr thisCornerKeyFrameCloud(new pcl::PointCloud<PointType>());

            if (pcl::io::loadPCDFile<PointType> (pathString, *thisCornerKeyFrameCloud) == -1)
            {
                PCL_ERROR ("Couldn't read file cloudSurf.pcd \n");
                return (-1);
            }
            else
            {
                surfCloudKeyFramesFromFile.push_back(thisCornerKeyFrameCloud);
            }
        }

        cout << " **** Loading Surf Key Frames: "<< surfCloudKeyFramesFromFile.size() << endl;

        saveSCDDirectory = saveDirectory + "SCDs/";

        for (int i = 0; i < (int)cloudKeyPoses3DFromFile->points.size(); i++)
        {
            std::string curr_scd_node_idx = padZeros(i);
            std::string pathString = saveSCDDirectory + curr_scd_node_idx + ".scd";

            polarcontexts_FromFile.push_back(loadSCD(pathString, scManager.PC_NUM_RING, scManager.PC_NUM_SECTOR));
            scManager.loadMakeSaveScancontextAndKeys(polarcontexts_FromFile.back());
        }

        cout << " **** Loading SCD with " << scManager.PC_NUM_RING << " rings and " << scManager.PC_NUM_SECTOR << " sectors: : "<< scManager.polarcontexts_.size() << endl;

        if (buildConfusionMatrix == true)
        {
            saveDirectory = saveMainDirectory + savePCDDirectory_Q;

            if (pcl::io::loadPCDFile<PointType>(saveDirectory + "trajectory.pcd", *cloudKeyPoses3D_Q_FromFile) == -1)
            {
                PCL_ERROR("Couldn't read file trajectory.pcd \n No cloudKeyPoses3D");
                return (-1);
            }
            saveSCDDirectory = saveDirectory + "SCDs/";

            for (int i = 0; i < (int)cloudKeyPoses3D_Q_FromFile->points.size(); i++)
            {
                std::string curr_scd_node_idx = padZeros(i);
                std::string pathString = saveSCDDirectory + curr_scd_node_idx + ".scd";

                polarcontexts_Q_FromFile.push_back(loadSCD(pathString, scManager.PC_NUM_RING, scManager.PC_NUM_SECTOR));
            }

            cout << " **** Loading SCD Query with " << scManager.PC_NUM_RING << " rings and " << scManager.PC_NUM_SECTOR << " sectors: : "<< polarcontexts_Q_FromFile.size() << endl;
        }

        return 0;
    }

    bool performInitPCD()
    {
        double auxFitnessScore[2] = {0, 100}; // {node, minimum_fitness_score}
        float fitnessScoreVector[cornerCloudKeyFrames.size()];
        std::fill_n(fitnessScoreVector, cornerCloudKeyFrames.size(), 0.0);

        pcl::PointCloud<PointType>::Ptr laserCloudIn(new pcl::PointCloud<PointType>());
        mtx_general.lock();
        *laserCloudIn += *cloudScanForInitialize;
        mtx_general.unlock();

        if (laserCloudIn->points.size() == 0)
            return false;
        // cloudScanForInitialize->clear();
        cout << "****************************************************" << endl;
        std::cout << "Perform Init PCD " << std::endl;
        std::cout << "The size of incoming lasercloud: " << laserCloudIn->points.size() << std::endl;

        // fitnessScoreSaveStream = std::fstream(std::getenv("HOME") + savePCDDirectory + "fitness.txt", std::fstream::out);
        fitnessScoreSaveStream = std::fstream(saveMainDirectory + "fitness.txt", std::fstream::out);

        // ICP Settings
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        for (int indexInitial : node_vector(cornerCloudKeyFrames.size()-1))
        {
            if (indexInitial > (int)(cornerCloudKeyFrames.size()-1))
                break;

            // std::cout << "indexInitial: " << indexInitial << std::endl;

            pcl::PointCloud<PointType>::Ptr framefrom(new pcl::PointCloud<PointType>());
            *framefrom += *cornerCloudKeyFrames[indexInitial];
            *framefrom += *surfCloudKeyFrames[indexInitial];

            // Align clouds
            icp.setInputSource(laserCloudIn);
            icp.setInputTarget(framefrom);
            pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
            icp.align(*unused_result);

            // if (icp.hasConverged() == false)
            // {
            //     std::cout << "Initial ICP fitness test failed for node "<< indexInitial << " (" << icp.getFitnessScore() << ")." << std::endl;
            //     std::cout << "/**********************************************************************/" << std::endl;
            // }
            // else
            // {
            //     std::cout << "Initial ICP fitness test passed for node "<< indexInitial << " (" << icp.getFitnessScore() << ")." << std::endl;
            //     std::cout << "/**********************************************************************/" << std::endl;
            // }

            if ((icp.getFitnessScore() < auxFitnessScore[1]) && (icp.getFitnessScore() > 0.00001))
            {
                auxFitnessScore[0] = indexInitial;
                auxFitnessScore[1] = icp.getFitnessScore();
            }

            fitnessScoreVector[indexInitial] = icp.getFitnessScore();

            if (auxFitnessScore[1] <= pcdLocalizationThreshold)
                break;
        }

        cout << "The Minimum Fitness Score is " << auxFitnessScore[1] << " and the initial node is : " << auxFitnessScore[0] <<  endl;

        if (auxFitnessScore[1] > 1000.0)
            return false;

        std::stringstream ss;
        for(size_t i = 0; i < cornerCloudKeyFrames.size(); ++i)
        {
            if(i != 0)
                ss << ",";
            ss << fitnessScoreVector[i];
        }
        std::string s = ss.str();
        fitnessScoreSaveStream << s << std::endl;

        initialPose[3] = (*cloudKeyPoses6D)[auxFitnessScore[0]].x;
        initialPose[4] = (*cloudKeyPoses6D)[auxFitnessScore[0]].y;
        initialPose[5] = (*cloudKeyPoses6D)[auxFitnessScore[0]].z;
        initialPose[0] = (*cloudKeyPoses6D)[auxFitnessScore[0]].roll;
        initialPose[1] = (*cloudKeyPoses6D)[auxFitnessScore[0]].pitch;
        initialPose[2] = (*cloudKeyPoses6D)[auxFitnessScore[0]].yaw;

        cout << initialPose[3] << endl;
        cout << initialPose[4] << endl;
        cout << initialPose[5] << endl;
        cout << initialPose[0] << endl;
        cout << initialPose[1] << endl;
        cout << initialPose[2] << endl;

        return true;
    }

    bool performInitSCD()
    {
        double auxSimilarityScore[2] = {0, 1};
        float similarityScoreVector[polarcontexts_FromFile.size()];
        std::fill_n(similarityScoreVector, polarcontexts_FromFile.size(), 0.0);

        // similarityScoreSaveStream = std::fstream(std::getenv("HOME") + savePCDDirectory + "similarity.txt", std::fstream::out);
        similarityScoreSaveStream = std::fstream(saveMainDirectory + "similarity.txt", std::fstream::out);

        // pcl::PointCloud<PointType>::Ptr laserCloudIn(new pcl::PointCloud<PointType>());
        // mtx_general.lock();
        // //*laserCloudIn += *laserCloudRawDS;
        // *laserCloudIn += *laserCloudSurfLastDS;
        // mtx_general.unlock();

        // if (laserCloudIn->points.size() == 0)
        //     return false;
        // cloudScanForInitialize->clear();
        //std::cout << "performInitSCD " << std::endl;
        //std::cout << "the size of incoming lasercloud: " << laserCloudIn->points.size() << std::endl;

        cout << "****************************************************" << endl;
        std::cout << "Perform Init SCD " << std::endl;

        // save all the received edge and surf points
        // pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        // pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        // pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        // pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // pcl::PointCloud<PointType>::Ptr laserCloudIn(new pcl::PointCloud<PointType>());

        // // - SINGLE_SCAN_FULL: using downsampled original point cloud (/full_cloud_projected + downsampling)
        // // - SINGLE_SCAN_FEAT: using surface feature as an input point cloud for scan context (2020.04.01: checked it works.)
        // // - MULTI_SCAN_FEAT: using NearKeyframes (because a MulRan scan does not have beyond region, so to solve this issue ... )

        // if( scdInput == SCInputType::SINGLE_SCAN_FULL ) 
        // {
        //     std::cout << "Scan Type Mode: single scan" << std::endl;
        //     pcl::copyPointCloud(*laserCloudRawDS,  *laserCloudIn);
        // }  
        // else if (scdInput == SCInputType::SINGLE_SCAN_FEAT) {
        //     std::cout << "Scan Type Mode: single feature" << std::endl;
        //     mtx_general.lock();
        //     *laserCloudIn += *thisCornerKeyFrame;
        //     *laserCloudIn += *thisSurfKeyFrame;
        //     mtx_general.unlock();
        // }
        // else if (scdInput == SCInputType::MULTI_SCAN_FEAT) {
        //     std::cout << "Scan Type Mode: Multi feature" << std::endl;
        //     mtx_general.lock();
        //     *laserCloudIn += *thisCornerKeyFrame;
        //     *laserCloudIn += *thisSurfKeyFrame;
        //     mtx_general.unlock();
        // }

        // scManager.makeAndSaveScancontextAndKeys(*laserCloudIn);
        std::cout << "The size of incoming SCD data: " << scManager.polarcontexts_.size() << std::endl;

        //std::cout << "SC: " << scManager.polarcontexts_ << std::endl;

        auto curr_desc = scManager.polarcontexts_.back();

        for (int indexInitial : node_vector(polarcontexts_FromFile.size()-1))
        {
            if (indexInitial > (int)(polarcontexts_FromFile.size()-1))
                break;
            std::pair<double, int> sc_dist_result = //
                scManager.distanceBtnScanContext(curr_desc, polarcontexts_FromFile.at(indexInitial));
            // cout << indexInitial << ": "<< sc_dist_result.first << endl;

            if ((sc_dist_result.first < auxSimilarityScore[1]))
            {
                auxSimilarityScore[0] = indexInitial;
                auxSimilarityScore[1] = sc_dist_result.first;
            }
        }

        cout << "The Minimum is " << auxSimilarityScore[1] << " and the initial node is : " << auxSimilarityScore[0] <<  endl;

        std::stringstream ss_sim;
        for(size_t i = 0; i < cornerCloudKeyFrames.size(); ++i)
        {
            if(i != 0)
                ss_sim << ",";
            ss_sim << auxSimilarityScore[i];
        }
        std::string s_sim = ss_sim.str();
        similarityScoreSaveStream << s_sim << std::endl;

        initialPose[3] = (*cloudKeyPoses6D)[auxSimilarityScore[0]].x;
        initialPose[4] = (*cloudKeyPoses6D)[auxSimilarityScore[0]].y;
        initialPose[5] = (*cloudKeyPoses6D)[auxSimilarityScore[0]].z;
        initialPose[0] = (*cloudKeyPoses6D)[auxSimilarityScore[0]].roll;
        initialPose[1] = (*cloudKeyPoses6D)[auxSimilarityScore[0]].pitch;
        initialPose[2] = (*cloudKeyPoses6D)[auxSimilarityScore[0]].yaw;

        cout << initialPose[3] << endl;
        cout << initialPose[4] << endl;
        cout << initialPose[5] << endl;
        cout << initialPose[0] << endl;
        cout << initialPose[1] << endl;
        cout << initialPose[2] << endl;

        return true;
    }

    void confusionMatrix()
    {
        // Confusion Matrix
        cout << "****************************************************" << endl;
        std::cout << "Confusion Matrix: start" << std::endl;

        cmScoreSaveStream = std::fstream(std::getenv("HOME") + savePCDDirectory + "scores.txt", std::fstream::out);

        std::stringstream ss;
        for (int dataIndex : node_vector(polarcontexts_FromFile.size()-1) )
        {
            if (dataIndex > (int)(polarcontexts_FromFile.size()-1))
                break;

            for (int queryIndex : node_vector(polarcontexts_Q_FromFile.size()-1))
            {
                if (queryIndex > (int)(polarcontexts_Q_FromFile.size()-1))
                    break;
                std::pair<double, int> sc_dist_result = //
                    scManager.distanceBtnScanContext(polarcontexts_FromFile.at(dataIndex), polarcontexts_Q_FromFile.at(queryIndex));
                if(queryIndex != 0)
                    ss << ",";
                ss << sc_dist_result.first;
            }
            ss << std::endl;
        }
        std::string s = ss.str();
        cmScoreSaveStream << s << std::endl;

        std::cout << "Confusion Matrix: end" << std::endl;
    }

    std::vector<int> node_vector(int sizeFrames)
    {
        std::vector<int> nodes;
        int indexInitial;
        // int sizeFrames = cornerCloudKeyFrames.size()-1;

        if (icpLocalizationAscendingOrder == true)
        {
            for (indexInitial = 0 ; indexInitial <= sizeFrames; indexInitial += icpLocalizationStep)
            {
                nodes.push_back(indexInitial);
            }
        }
        else
        {
            for (indexInitial = sizeFrames ; indexInitial >= 0; indexInitial -= icpLocalizationStep)
            {
                nodes.push_back(indexInitial);
            }
        }
        return nodes;
    }

    void initLocalization()
    {
        if (cloudScanForInitialize->points.size() == 0)
        {
            downsampleCurrentScan();

            mtx_general.lock();
            *cloudScanForInitialize += *laserCloudCornerLastDS;
            *cloudScanForInitialize += *laserCloudSurfLastDS;
            mtx_general.unlock();

            // SCD 
            pcl::PointCloud<PointType>::Ptr laserCloudIn(new pcl::PointCloud<PointType>());
            if( scdInput == SCInputType::SINGLE_SCAN_FULL ) 
            {
                std::cout << "Scan Type Mode: single scan" << std::endl;
                mtx_general.lock();
                *laserCloudIn = *laserCloudRawDS;
                mtx_general.unlock();
            }  
            else if (scdInput == SCInputType::SINGLE_SCAN_FEAT) {
                std::cout << "Scan Type Mode: single feature" << std::endl;
                mtx_general.lock();
                *laserCloudIn += *laserCloudCornerLastDS;
                *laserCloudIn += *laserCloudSurfLastDS;
                mtx_general.unlock();
            }
            else if (scdInput == SCInputType::MULTI_SCAN_FEAT) {
                std::cout << "Scan Type Mode: Multi feature" << std::endl;
                mtx_general.lock();
                *laserCloudIn += *laserCloudCornerLastDS;
                *laserCloudIn += *laserCloudSurfLastDS;
                mtx_general.unlock();
            }
            scManager.makeAndSaveScancontextAndKeys(*laserCloudIn);

            // Preliminary ICP - Not implemented
            if (icpLocalizationPreliminary == true)
            {
                auto begin_00 = std::chrono::high_resolution_clock::now();
                ICPLocalizeInitialize();
                auto end_00 = std::chrono::high_resolution_clock::now();
                auto elapsed_00  = std::chrono::duration_cast<std::chrono::nanoseconds>(end_00 - begin_00);
                printf("Time measured (PRE): %.9f seconds.\n", elapsed_00.count() * 1e-9);
            }

            // Scan Context Descriptor
            if (scdInitialLocalization == true)
            {
                auto begin_01 = std::chrono::high_resolution_clock::now();
                if (!performInitSCD())
                {
                    cloudScanForInitialize->clear();
                    return;
                }
                auto end_01 = std::chrono::high_resolution_clock::now();
                auto elapsed_01  = std::chrono::duration_cast<std::chrono::nanoseconds>(end_01 - begin_01);
                printf("Time measured (SCD): %.9f seconds.\n", elapsed_01.count() * 1e-9);
            }

            // Point Cloud ICP
            if (pcdInitialLocalization == true)
            {
                auto begin = std::chrono::high_resolution_clock::now();
                if (!performInitPCD())
                {
                    cloudScanForInitialize->clear();
                    return;
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
                printf("Time measured (PCD): %.9f seconds.\n", elapsed.count() * 1e-9);
            }
            
            // Debug data
            if (buildConfusionMatrix)
            {
                confusionMatrix();
                while(1);
            }
                
            updateInitialGuess();

            transformTobeMapped[3] = initialPose[3];
            transformTobeMapped[4] = initialPose[4];
            transformTobeMapped[5] = initialPose[5];

            // odom factor
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(initialPose);
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), trans2gtsamPose(initialPose));

            cout << "****************************************************" << endl;
            gtSAMgraph.print("GTSAM Graph:\n");

            // update iSAM
            isam->update(gtSAMgraph, initialEstimate);
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();

            gtSAMgraph.resize(0);
            initialEstimate.clear();

            //save key poses
            PointType thisPose3D;
            PointTypePose thisPose6D;
            Pose3 latestEstimate;

            isamCurrentEstimate = isam->calculateEstimate();
            latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
            cout << "****************************************************" << endl;
            latestEstimate.print("Current estimate: ");

            thisPose3D.x = latestEstimate.translation().x();
            thisPose3D.y = latestEstimate.translation().y();
            thisPose3D.z = latestEstimate.translation().z();
            thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
            cloudKeyPoses3D->push_back(thisPose3D);

            thisPose6D.x = thisPose3D.x;
            thisPose6D.y = thisPose3D.y;
            thisPose6D.z = thisPose3D.z;
            thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
            thisPose6D.roll  = latestEstimate.rotation().roll();
            thisPose6D.pitch = latestEstimate.rotation().pitch();
            thisPose6D.yaw   = latestEstimate.rotation().yaw();
            thisPose6D.time = timeLaserInfoCur;
            cloudKeyPoses6D->push_back(thisPose6D);

            // cout << "****************************************************" << endl;
            // cout << "Pose covariance:" << endl;
            // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
            poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

            // save updated transform
            transformTobeMapped[0] = latestEstimate.rotation().roll();
            transformTobeMapped[1] = latestEstimate.rotation().pitch();
            transformTobeMapped[2] = latestEstimate.rotation().yaw();
            transformTobeMapped[3] = latestEstimate.translation().x();
            transformTobeMapped[4] = latestEstimate.translation().y();
            transformTobeMapped[5] = latestEstimate.translation().z();

            // save all the received edge and surf points
            pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
            pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
            pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

            // save key frame cloud
            cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
            surfCloudKeyFrames.push_back(thisSurfKeyFrame);

            cout << "****************************************************" << endl;
            cout << "cornerCloudKeyFrames size: " << cornerCloudKeyFrames.size() << endl;
            cout << "surfCloudKeyFrames size: " << surfCloudKeyFrames.size() << endl;

            cout << "Points (cloudKeyPoses3D): " << cloudKeyPoses3D->points.size() << endl;
            cout << "Points (cloudKeyPoses6D): " << cloudKeyPoses6D->points.size() << endl;

            // save path for visualization
            updatePath(thisPose6D);

            correctPoses();

            publishOdometry();

            publishFrames();

            if (savePCD)
            {
                // save keyframe cloud as file
                pcl::PointCloud<PointType>::Ptr thisKeyFrameCloud(new pcl::PointCloud<PointType>());

                cout << "****************************************************" << endl;
                cout << "Saving Key Frame: " << cornerCloudKeyFrames.size() << endl;
                if(saveRawCloud)
                {
                    *thisKeyFrameCloud += *laserCloudRaw;
                }
                else
                {
                    *thisKeyFrameCloud += *thisCornerKeyFrame;
                    *thisKeyFrameCloud += *thisSurfKeyFrame;
                }
                std::string curr_pcd_node_idx = padZeros(cornerCloudKeyFrames.size() - 1);
                pcl::io::savePCDFileBinary(saveNodePCDDirectory + curr_pcd_node_idx + ".pcd", *thisKeyFrameCloud);
                pcl::io::savePCDFileBinary(saveCornerKeyFramePCDDirectory + "C" + curr_pcd_node_idx + ".pcd", *thisCornerKeyFrame);
                pcl::io::savePCDFileBinary(saveSurfKeyFramePCDDirectory + "S" + curr_pcd_node_idx + ".pcd", *thisSurfKeyFrame);

                // save sc data
                const auto& curr_scd = scManager.getConstRefRecentSCD();
                std::string curr_scd_node_idx = padZeros(scManager.polarcontexts_.size() - 1);
                saveSCD(saveSCDDirectory + curr_scd_node_idx + ".scd", curr_scd);

            }

            initializedFlag = Initialized;
        }
    }

    void ICPLocalizeInitialize()
    {
        pcl::PointCloud<PointType>::Ptr laserCloudIn(new pcl::PointCloud<PointType>());

        mtx_general.lock();
        *laserCloudIn += *cloudScanForInitialize;
        mtx_general.unlock();

        if (laserCloudIn->points.size() == 0)
            return;

        pcl::NormalDistributionsTransform<PointType, PointType> ndt;
        ndt.setTransformationEpsilon(0.01);
        ndt.setResolution(1.0);

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        // Set the maximum distance threshold between two correspondent points in source <-> target.
        // If the distance is larger than this threshold, the points will be ignored in the alignment process.
        icp.setMaxCorrespondenceDistance(setMaxCorrespondenceDistance);
        // Set the maximum number of iterations the internal optimization should run for.
        icp.setMaximumIterations(setMaximumIterations);
        // The maximum difference between two consecutive transformations in order to consider convergence (user defined)
        icp.setTransformationEpsilon(setTransformationEpsilon);
        // Set the maximum allowed Euclidean error between two consecutive steps in the ICP loop, before the algorithm is considered to have converged.
        // The error is estimated as the sum of the differences between correspondences in an Euclidean sense, divided by the number of correspondences.
        icp.setEuclideanFitnessEpsilon(setEuclideanFitnessEpsilon);
        // Set the number of iterations RANSAC should run for.
        icp.setRANSACIterations(setRANSACIterations);

        ndt.setInputSource(laserCloudIn);
        ndt.setInputTarget(cloudGlobalMapDS);
        pcl::PointCloud<PointType>::Ptr unused_result_0(new pcl::PointCloud<PointType>());

        PointTypePose thisPose6DInWorld = trans2PointTypePose(transformInTheWorld);
        Eigen::Affine3f T_thisPose6DInWorld = pclPointToAffine3f(thisPose6DInWorld);
        ndt.align(*unused_result_0, T_thisPose6DInWorld.matrix());

        // use the outcome of ndt as the initial guess for ICP
        icp.setInputSource(laserCloudIn);
        icp.setInputTarget(cloudGlobalMapDS);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result, ndt.getFinalTransformation());

        std::cout << "the icp score in initializing process is: " << icp.getFitnessScore() << std::endl;
        std::cout << "the pose after initializing process is: " << icp.getFinalTransformation() << std::endl;
    }
};

// Constructor of the LIO-SAM Server
ExecuteLioSamServer::ExecuteLioSamServer(ros::NodeHandle* node_handler, std::string action_server_name) : action_server_(*node_handler, action_server_name, false)
{
    action_server_.registerGoalCallback(boost::bind(&ExecuteLioSamServer::goal_callback, this));
    action_server_.registerPreemptCallback(boost::bind(&ExecuteLioSamServer::preempt_callback, this));
    action_server_.start();
}

void ExecuteLioSamServer::goal_callback()
{
    // If the server is already processing a goal
    if (busy_)
    {
        ROS_ERROR("The action will not be processed because the server is already busy with another action. "
                "Please preempt the latter or wait before sending a new goal");
        return;
    }
    ROS_INFO("Action callback");
    // The first step is to accept a new goal. If you want to access to the input field, you should write
    // new_goal_->input;
    new_goal_ = action_server_.acceptNewGoal();
    // Set busy to true
    busy_ = true;
    // Set wait
    //ros::Rate r(2);

    app_liosam_mode = new_goal_->liosam_mode;
    app_map_name = new_goal_->map_name;
    app_waiting_user = false;

    // std::cout << "App liosam_mode: " << app_liosam_mode << std::endl;
    // std::cout << "App map_name: " << app_map_name << std::endl;
    // std::cout << "Waiting: " << app_waiting_user << std::endl;

    if ((app_liosam_mode == 0) && (feedback_mode == 3))
    {
        ROS_INFO("Set SLAM");
        action_feedback_.liosam_feedback_mode = 0;
        action_feedback_.liosam_feedback_info = 0;
    }

    if ((app_liosam_mode == 1) && (feedback_mode == 3))
    {
        ROS_INFO("Set Localization");
        action_feedback_.liosam_feedback_mode = 1;
        action_feedback_.liosam_feedback_info = 0;
    }

    if (app_liosam_mode == 8)
    {
        ROS_INFO("Status");
        action_feedback_.liosam_feedback_mode = feedback_mode;
        action_feedback_.liosam_feedback_info = feedback_quality;
    }

    if ((app_liosam_mode == 10) && (feedback_mode == 3))
    {
        ROS_INFO("Kill command sent by app.");
        app_initialized = false;
        app_waiting_user = true;
        app_liosam_mode = 9;
    }

    if (app_liosam_mode == 9)
    {
        ROS_INFO("Finish");
        action_feedback_.liosam_feedback_mode = 2;
        action_feedback_.liosam_feedback_info = 0;
        toKill = true;
    }

    action_server_.publishFeedback(action_feedback_);
    //r.sleep();

    // Return the response of the action.
    action_result_.liosam_result = feedback_quality;
    action_server_.setSucceeded(action_result_);
    // Set busy to false
    busy_ = false;
    
    if (toKill == true)
    {
        std_msgs::String msg;
        msg.data = "all";
        pubKillNodes.publish(msg);
        ROS_INFO("Sent kill signal to node: %s", msg.data.c_str());
    }
}

void ExecuteLioSamServer::preempt_callback()
{
    ROS_INFO("Action preempted");
    action_server_.setPreempted();
}

void shutdownNode(const std_msgs::String::ConstPtr& msg) {
    ROS_INFO("Shutting down: Map Optimization.");
    ros::shutdown();
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    mapOptimization MO;

    ExecuteLioSamServer execute_liosam_server_(&MO.nh, "lio_sam/execute_mode");
    pubKillNodes = MO.nh.advertise<std_msgs::String>("lio_sam/kill_nodes", 1);
    ros::Subscriber killSub = MO.nh.subscribe("lio_sam/kill_nodes", 10, shutdownNode);

    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");
    
    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::spin();

    loopthread.join();
    visualizeMapThread.join();

    return 0;
}

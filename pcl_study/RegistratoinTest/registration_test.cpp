#include <unistd.h>
#include <stdlib.h>
#include <ctime>

#include <boost/thread/thread.hpp>
#include <Eigen/Core>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>

typedef pcl::PointXYZ PointXYZ;
typedef pcl::PointNormal PointN;
typedef pcl::PointCloud<PointXYZ> PointCloudXYZ;
typedef pcl::PointCloud<PointN> PointCloudN;


typedef enum {
  SVD,
  LM,
  LM_Plane,
  LLS_Plane
} TransformationEstimationAlgorithm;


int main (int argc, char **argv)
{
  // check argument
  std::string input_pcd_file = "../pcd/chef.pcd";
  TransformationEstimationAlgorithm alg = SVD;
  double delta_x = 0.5, delta_yaw = 1.0;
  int opt;
  opterr = 0;
  while ((opt = getopt(argc, argv, "i:a:t:r:")) != -1) {
    std::string optarg_string = optarg;
    switch (opt) {
    case 'i':
      input_pcd_file = optarg;
      break;
    case 'a':
      if (optarg_string == "svd" || optarg_string == "SVD") {
        alg = SVD;
      } else if (optarg_string == "lm" || optarg_string == "LM") {
        alg = LM;
      } else if (optarg_string == "lm_plane" || optarg_string == "LM_Plane" || optarg_string == "LM_PLANE") {
        alg = LM_Plane;
      } else if (optarg_string == "lls_plane" || optarg_string == "LLS_Plane" || optarg_string == "LLS_PLANE") {
        alg = LLS_Plane;
      }
      break;
    case 't':
      delta_x = atof(optarg);
      break;
    case 'r':
      delta_yaw = atof(optarg);
      break;
    }
  }

  // load point cloud
  PointCloudXYZ::Ptr object_raw (new PointCloudXYZ);
  PointCloudXYZ::Ptr object (new PointCloudXYZ);
  PointCloudXYZ::Ptr object_act_transformed (new PointCloudXYZ);
  PointCloudXYZ::Ptr object_est_transformed (new PointCloudXYZ);
  PointCloudN::Ptr object_act_transformed_normal (new PointCloudN);
  pcl::console::print_highlight ("Loading point clouds...\n");
  if (pcl::io::loadPCDFile<PointXYZ> (input_pcd_file, *object_raw) < 0) {
    pcl::console::print_error ("Error loading object/scene file!\n");
    return 1;
  } else {
    pcl::console::print_info ("PCD file: %s\n", input_pcd_file.c_str());
  }
  std::vector<int> remove_nan_indices;
  pcl::removeNaNFromPointCloud(*object_raw, *object, remove_nan_indices);

  // transform pointcloud with actual transformation
  Eigen::Affine3f act_trans = Eigen::Affine3f::Identity();
  Eigen::Affine3f est_trans;
  act_trans.translation () << delta_x, 0.0, 0.0;
  act_trans.rotate (Eigen::AngleAxisf (delta_yaw, Eigen::Vector3f::UnitZ()));
  pcl::transformPointCloud (*object, *object_act_transformed, act_trans);

  // calculate norm for plane
  if (alg == LM_Plane || alg == LLS_Plane) {
    pcl::NormalEstimation< PointXYZ, PointN > ne;
    ne.setInputCloud (object_act_transformed);
    pcl::search::KdTree< PointXYZ >::Ptr tree (new pcl::search::KdTree< PointXYZ > ());
    ne.setSearchMethod (tree);
    ne.setRadiusSearch (0.03);
    ne.compute (*object_act_transformed_normal);
  }

  // estimate transformation
  clock_t begin_clock = clock();
  boost::shared_ptr< pcl::registration::TransformationEstimation< PointXYZ, PointXYZ > > est;
  boost::shared_ptr< pcl::registration::TransformationEstimation< PointXYZ, PointN > > est_normal;
  switch (alg) {
  case SVD:
    pcl::console::print_info ("Algorithm: SVD\n");
    est.reset ( new pcl::registration::TransformationEstimationSVD < PointXYZ, PointXYZ > () );
    break;
  case LM:
    pcl::console::print_info ("Algorithm: LM\n");
    est.reset ( new pcl::registration::TransformationEstimationLM < PointXYZ, PointXYZ > () );
    break;
  case LM_Plane:
    pcl::console::print_info ("Algorithm: LM_Plane\n");
    est_normal.reset ( new pcl::registration::TransformationEstimationPointToPlane < PointXYZ, PointN > () );
    break;
  case LLS_Plane:
    pcl::console::print_info ("Algorithm: LLS_Plane\n");
    est_normal.reset ( new pcl::registration::TransformationEstimationPointToPlaneLLS < PointXYZ, PointN > () );
    break;
  default:
    pcl::console::print_error ("Invalid algorithm for estimating transformation!\n");
    return 1;
  }
  if (alg == LM_Plane || alg == LLS_Plane) {
    est_normal->estimateRigidTransformation(*object, *object_act_transformed_normal, est_trans.matrix());
  } else {
    est->estimateRigidTransformation(*object, *object_act_transformed, est_trans.matrix());
  }
  clock_t end_clock = clock();
  double elapsed_time = double(end_clock - begin_clock) / CLOCKS_PER_SEC;

  // transform pointcloud with estimated transformation
  pcl::transformPointCloud (*object, *object_est_transformed, est_trans);

  // calc error
  Eigen::Matrix3f act_rot = act_trans.matrix().topLeftCorner(3, 3);
  Eigen::Vector3f act_rpy = act_rot.eulerAngles(0, 1, 2);
  Eigen::Vector3f act_xyz = act_trans.matrix().block (0, 3, 3, 1);
  Eigen::Matrix3f est_rot = est_trans.matrix().topLeftCorner(3, 3);
  Eigen::Vector3f est_rpy = est_rot.eulerAngles(0, 1, 2);
  Eigen::Vector3f est_xyz = est_trans.matrix().block (0, 3, 3, 1);
  Eigen::Vector3f xyz_error = act_xyz - est_xyz;
  double xyz_error_norm = xyz_error.norm();
  Eigen::Matrix3f rot_error = act_rot.transpose() * est_rot;
  Eigen::Vector3f rpy_error = rot_error.eulerAngles(0, 1, 2) ;
  Eigen::AngleAxisf angle_axis_error;
  double rpy_error_norm = angle_axis_error.fromRotationMatrix(rot_error).angle();

  // print result
  // pcl::console::print_info ("[xyz]  actual: %f %f %f  estimated: %f %f %f\n",
  //                           act_xyz[0], act_xyz[1], act_xyz[2], est_xyz[0], est_xyz[1], est_xyz[2]);
  // pcl::console::print_info ("[rpy]  actual: %f %f %f  estimated: %f %f %f\n",
  //                           act_rpy[0], act_rpy[1], act_rpy[2], est_rpy[0], est_rpy[1], est_rpy[2]);
  std::cout << "actual transformation " << std::endl << act_trans.matrix() << std::endl;
  std::cout << "estimated transformation " << std::endl << est_trans.matrix() << std::endl;
  std::cout << "[result]  alg: " << alg << "  xyz_error: " << xyz_error_norm << " [m]  rpy_error: " << rpy_error_norm
            << " [rad]  time: " << elapsed_time << " [sec]" << std::endl;

  // view point cloud
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_object (object, 255, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (object, rgb_object, "object");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "object");
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_object_act_transformed (object_act_transformed, 0, 255, 0);
  viewer->addPointCloud<pcl::PointXYZ> (object_act_transformed, rgb_object_act_transformed, "object_act_transformed");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "object_act_transformed");
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_object_est_transformed (object_est_transformed, 0, 0, 255);
  viewer->addPointCloud<pcl::PointXYZ> (object_est_transformed, rgb_object_est_transformed, "object_est_transformed");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "object_est_transformed");
  viewer->spin();

  return 0;
}

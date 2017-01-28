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
#include "transformation_estimation_point_to_line.h"

typedef pcl::PointXYZ PointXYZ;
typedef pcl::PointNormal PointNormal;
typedef pcl::PointCloud<PointXYZ> PointCloudXYZ;
typedef pcl::PointCloud<PointNormal> PointCloudNormal;


typedef enum {
  SVD,
  LM,
  LM_Plane,
  LLS_Plane,
  LM_Line,
} TransformationEstimationAlgorithm;


void addEdgePoints (PointCloudNormal &cloud, const Eigen::Vector3f p_start, const Eigen::Vector3f p_end, const int np)
{
  for (float i = 0.0; i <= 1.0; i += (1.0 / np)) {
    Eigen::Vector3f p_vec(p_start + i * (p_end - p_start));
    Eigen::Vector3f d_vec((p_end - p_start).normalized());
    PointNormal p;
    p.x = p_vec(0);
    p.y = p_vec(1);
    p.z = p_vec(2);
    p.normal_x = d_vec(0);
    p.normal_y = d_vec(1);
    p.normal_z = d_vec(2);
    cloud.points.push_back(p);
  }
}

void addEdgeLine (PointCloudNormal &cloud, const Eigen::Vector3f p_start, const Eigen::Vector3f p_end, const int np)
{
  Eigen::Vector3f p_mid_vec(p_start);
  // Eigen::Vector3f p_mid_vec((p_start + p_end) * 0.5);
  Eigen::Vector3f d_vec((p_end - p_start));
  PointNormal p;
  p.x = p_mid_vec(0);
  p.y = p_mid_vec(1);
  p.z = p_mid_vec(2);
  p.x = 0;
  p.y = 0;
  p.z = 0;
  p.normal_x = d_vec(0);
  p.normal_y = d_vec(1);
  p.normal_z = d_vec(2);
  for (float i = 0.0; i <= 1.0; i += (1.0 / np)) {
    cloud.points.push_back(p);
  }
}

void generateCubeEdgePointCloud (PointCloudNormal &cloud)
{
  // generate object pointcloud
  addEdgePoints (cloud, Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0.1, 0, 0), 10);
  addEdgePoints (cloud, Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0.0, 0.1, 0), 20);
  addEdgePoints (cloud, Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0.0, 0.0, 0.2), 10);
  cloud.width = (int) cloud.points.size ();
  cloud.height = 1;
}

void generateCubeEdgeLines (PointCloudNormal &cloud)
{
  // generate object pointcloud
  addEdgeLine (cloud, Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0.1, 0, 0), 10);
  addEdgeLine (cloud, Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0.0, 0.1, 0), 20);
  addEdgeLine (cloud, Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0.0, 0.0, 0.2), 10);
  cloud.width = (int) cloud.points.size ();
  cloud.height = 1;
}


int main (int argc, char **argv)
{
  // check argument
  std::string input_pcd_file = "";
  TransformationEstimationAlgorithm alg = SVD;
  double delta_x = 0, delta_y = 0, delta_z = 0, delta_roll = 0, delta_pitch = 0, delta_yaw = 0;
  int opt;
  opterr = 0;
  while ((opt = getopt(argc, argv, "i:a:x:y:z:R:P:Y:")) != -1) {
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
      } else if (optarg_string == "lm_line" || optarg_string == "LM_Line" || optarg_string == "LM_LINE") {
        alg = LM_Line;
      }
      break;
    case 'x':
      delta_x = atof(optarg);
      break;
    case 'y':
      delta_y = atof(optarg);
      break;
    case 'z':
      delta_z = atof(optarg);
      break;
    case 'R':
      delta_roll = atof(optarg);
      break;
    case 'P':
      delta_pitch = atof(optarg);
      break;
    case 'Y':
      delta_yaw = atof(optarg);
      break;
    }
  }

  // load point cloud
  PointCloudXYZ::Ptr object_raw (new PointCloudXYZ);
  PointCloudXYZ::Ptr object (new PointCloudXYZ);
  PointCloudNormal::Ptr object_line (new PointCloudNormal);
  PointCloudXYZ::Ptr object_model (new PointCloudXYZ);
  PointCloudNormal::Ptr object_model_line (new PointCloudNormal);
  PointCloudXYZ::Ptr object_act_transformed (new PointCloudXYZ);
  PointCloudXYZ::Ptr object_est_transformed (new PointCloudXYZ);
  PointCloudNormal::Ptr object_act_transformed_normal (new PointCloudNormal);
  pcl::console::print_highlight ("Loading point clouds...\n");
  if (input_pcd_file == "") {
    generateCubeEdgePointCloud (*object_line);
    generateCubeEdgeLines (*object_model_line);
    // pcl::copyPointCloud (*object_line, *object_model_line);
    pcl::copyPointCloud (*object_line, *object);
    pcl::copyPointCloud (*object_model_line, *object_model);
  } else {
    if (pcl::io::loadPCDFile<PointXYZ> (input_pcd_file, *object_raw) < 0) {
      pcl::console::print_error ("Error loading object/scene file!\n");
      return 1;
    } else {
      pcl::console::print_info ("PCD file: %s\n", input_pcd_file.c_str());
    }
    std::vector<int> remove_nan_indices;
    pcl::removeNaNFromPointCloud(*object_raw, *object, remove_nan_indices);
    pcl::copyPointCloud (*object, *object_model);
  }

  // transform pointcloud with actual transformation
  Eigen::Affine3f act_trans = Eigen::Affine3f::Identity();
  Eigen::Affine3f est_trans;
  act_trans.translation () << delta_x, delta_y, delta_z;
  act_trans.rotate (Eigen::AngleAxisf (delta_roll, Eigen::Vector3f::UnitX()));
  act_trans.rotate (Eigen::AngleAxisf (delta_pitch, Eigen::Vector3f::UnitY()));
  act_trans.rotate (Eigen::AngleAxisf (delta_yaw, Eigen::Vector3f::UnitZ()));
  pcl::transformPointCloud (*object, *object_act_transformed, act_trans);

  // calculate norm for plane
  if (alg == LM_Plane || alg == LLS_Plane) {
    pcl::copyPointCloud(*object_act_transformed, *object_act_transformed_normal);
    pcl::NormalEstimation< PointXYZ, PointNormal > ne;
    ne.setInputCloud (object_act_transformed);
    pcl::search::KdTree< PointXYZ >::Ptr tree (new pcl::search::KdTree< PointXYZ > ());
    ne.setSearchMethod (tree);
    ne.setRadiusSearch (0.02);
    ne.compute (*object_act_transformed_normal);
  }

  // estimate transformation
  clock_t begin_clock = clock();
  boost::shared_ptr< pcl::registration::TransformationEstimation< PointXYZ, PointXYZ > > est;
  boost::shared_ptr< pcl::registration::TransformationEstimation< PointXYZ, PointNormal > > est_normal;
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
    est_normal.reset ( new pcl::registration::TransformationEstimationPointToPlane < PointXYZ, PointNormal > () );
    break;
  case LLS_Plane:
    pcl::console::print_info ("Algorithm: LLS_Plane\n");
    est_normal.reset ( new pcl::registration::TransformationEstimationPointToPlaneLLS < PointXYZ, PointNormal > () );
    break;
  case LM_Line:
    pcl::console::print_info ("Algorithm: LM_Line\n");
    est_normal.reset ( new pcl::registration::TransformationEstimationPointToLine < PointXYZ, PointNormal > () );
    break;
  default:
    pcl::console::print_error ("Invalid algorithm for estimating transformation!\n");
    return 1;
  }
  if (alg == LM_Plane || alg == LLS_Plane) {
    est_normal->estimateRigidTransformation(*object_model, *object_act_transformed_normal, est_trans.matrix());
  } else if (alg == LM_Line) {
    est_normal->estimateRigidTransformation(*object_act_transformed, *object_model_line, est_trans.matrix());
    est_trans = est_trans.inverse(); // inverse transformation because src and dest is flipped in estimation
  } else {
    est->estimateRigidTransformation(*object_model, *object_act_transformed, est_trans.matrix());
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
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerCustom<PointXYZ> rgb_object (object, 255, 0, 0);
  viewer->addPointCloud<PointXYZ> (object_model, rgb_object, "object");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "object");
  pcl::visualization::PointCloudColorHandlerCustom<PointXYZ> rgb_object_act_transformed (object_act_transformed, 0, 255, 0);
  viewer->addPointCloud<PointXYZ> (object_act_transformed, rgb_object_act_transformed, "object_act_transformed");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "object_act_transformed");
  pcl::visualization::PointCloudColorHandlerCustom<PointXYZ> rgb_object_est_transformed (object_est_transformed, 0, 0, 255);
  viewer->addPointCloud<PointXYZ> (object_est_transformed, rgb_object_est_transformed, "object_est_transformed");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "object_est_transformed");
  if (alg == LM_Plane || alg == LLS_Plane) {
    viewer->addPointCloudNormals<PointXYZ, PointNormal> (object_act_transformed, object_act_transformed_normal, 10, 0.02, "object_act_transformed_normal");
  } else if (alg == LM_Line) {
    viewer->addPointCloudNormals<PointXYZ, PointNormal> (object_model, object_model_line, 6, 1.0, "object_line");
  }
  viewer->spin();

  return 0;
}

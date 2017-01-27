#include <unistd.h>
#include <stdlib.h>

#include <boost/thread/thread.hpp>
#include <Eigen/Core>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>

typedef pcl::PointXYZ PointXYZ;
typedef pcl::PointCloud<PointXYZ> PointCloudT;


typedef enum {
  SVD, LM
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
  PointCloudT::Ptr object_raw (new PointCloudT);
  PointCloudT::Ptr object (new PointCloudT);
  PointCloudT::Ptr object_act_transformed (new PointCloudT);
  PointCloudT::Ptr object_est_transformed (new PointCloudT);
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

  // estimate transformation
  boost::shared_ptr< pcl::registration::TransformationEstimation< PointXYZ, PointXYZ > > est;
  switch (alg) {
  case SVD:
    pcl::console::print_info ("Algorithm: SVD\n");
    est.reset ( new pcl::registration::TransformationEstimationSVD < PointXYZ, PointXYZ > () );
    break;
  case LM:
    pcl::console::print_info ("Algorithm: LM\n");
    est.reset ( new pcl::registration::TransformationEstimationLM < PointXYZ, PointXYZ > () );
    break;
  default:
    pcl::console::print_error ("Invalid algorithm for estimating transformation!\n");
    return 1;
  }
  est->estimateRigidTransformation(*object, *object_act_transformed, est_trans.matrix());

  // transform pointcloud with estimated transformation
  pcl::transformPointCloud (*object, *object_est_transformed, est_trans);

  // print result
  std::cout << "actual transformation " << std::endl << act_trans.matrix() << std::endl;
  std::cout << "estimated transformation " << std::endl << est_trans.matrix() << std::endl;

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

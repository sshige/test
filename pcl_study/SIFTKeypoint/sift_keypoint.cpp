#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>

void Surface_normals(pcl::PointCloud<pcl::PointNormal>::Ptr cloud)
{
  pcl::NormalEstimation<pcl::PointNormal, pcl::PointNormal> ne;
  ne.setInputCloud (cloud); // 法線の計算を行いたい点群を指定する

  pcl::search::KdTree<pcl::PointNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointNormal> ()); // KDTREEを作る
  ne.setSearchMethod (tree); // 検索方法にKDTREEを指定する

  ne.setRadiusSearch (0.5); // 検索する半径を指定する

  ne.compute (*cloud); // 法線情報の出力先を指定する

  return;
}


pcl::PointCloud<pcl::PointWithScale> Extract_SIFT(pcl::PointCloud<pcl::PointNormal>::Ptr cloud)
{
  // SIFT特徴量計算のためのパラメータ
  const float min_scale = 0.01f;
  const int n_octaves = 3;
  const int n_scales_per_octave = 4;
  const float min_contrast = 0.001f;
  pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
  pcl::PointCloud<pcl::PointWithScale> result;

  pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal> ());
  sift.setSearchMethod(tree);

  sift.setScales(min_scale, n_octaves, n_scales_per_octave);

  sift.setMinimumContrast(0.00);

  sift.setInputCloud(cloud);

  sift.compute(result);

  std::cout << "Number of SIFT points in the result are " << result.points.size () << std::endl;

  return result;
}


int main(int argc, char** argv)

{
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud (new pcl::PointCloud<pcl::PointNormal>);
  // pcl::io::loadPCDFile<pcl::PointNormal> (argv[1], *cloud); // どちらでもいい
  pcl::PCDReader reader;
  reader.read<pcl::PointNormal> (argv[1], *cloud);

  // 法線を計算
  Surface_normals(cloud);

  // 視覚化のためSIFT計算の結果をcloud_siftにコピー
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_sift (new pcl::PointCloud<pcl::PointNormal>);
  copyPointCloud(Extract_SIFT(cloud), *cloud_sift);
  std::cout << "SIFT points in the cloud_sift are " << cloud_sift->points.size () << std::endl;

  // 入力点群と計算された特徴点を表示
  pcl::visualization::PCLVisualizer viewer("PCL Viewer");
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> keypoints_color_handler (cloud_sift, 0, 255, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> cloud_color_handler (cloud, 255, 0, 0);
  viewer.setBackgroundColor( 0.0, 0.0, 0.0 );
  viewer.addPointCloud(cloud, cloud_color_handler, "cloud");
  viewer.addPointCloud(cloud_sift, keypoints_color_handler, "keypoints");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints");
  viewer.spin ();
  return (0);
}

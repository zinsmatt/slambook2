#pragma once

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <sophus/se3.hpp>


const int boarder = 20;
const int width = 640;
const int height = 480;
const double fx = 481.2f;
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 3;
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1);
const double min_cov = 0.1;
const double max_cov = 10;
const double epsilon = 1e-10;



bool epipolar_search(cv::Mat ref, cv::Mat cur, const Sophus::SE3d& Tcr,
                     const Eigen::Vector2d& pt, double depth_mu, double depth_sigma2,
                     Eigen::Vector2d& best_pc, Eigen::Vector2d& epipolar_dir);

void update_depth_filter(const Eigen::Vector2d& pr, const Eigen::Vector2d& pc,
                         const Sophus::SE3d& Tcr, const Eigen::Vector2d& epipolar_dir, 
                         cv::Mat depth, cv::Mat cov2);

bool epipolar_search_cuda(const unsigned char* ref, const unsigned char* cur, 
                          const double Tcr[3][4], const double pt[2], 
                          double depth_mu, double depth_sigma2, 
                          double best_pc[2], double epipolar_dir[2]);


void update_depth_filter_cuda(const double pr[2], const double pc[2],
                              const double Trc[3][4], const double epipolar_dir[2], 
                              double *depth, double *cov2);


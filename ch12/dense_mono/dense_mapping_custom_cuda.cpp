#include <iostream>
#include <vector>
#include <fstream>
#include <thread>
#include <cmath>
#include <chrono>

using namespace std;

#include <boost/timer.hpp>

// for sophus
#include <sophus/se3.hpp>

using Sophus::SE3d;

// for eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>


#include "cuda_wrapper.h"
#include "plot.h"

using namespace cv;



std::ofstream debug("debug_custom.txt");


void plotDepth(const Mat &depth_truth, const Mat &depth_estimate);
void plotCur(const Mat &cur);

bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    vector<SE3d> &poses,
    cv::Mat &ref_depth
);

void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr);

void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr);

void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate);
// ------------------------------------------------------------------


void update(cv::Mat ref, cv::Mat cur, const Sophus::SE3d& Tcr, cv::Mat depth, cv::Mat cov2)
{
    Eigen::Vector2d pc;
    Eigen::Vector2d epipolar_dir;
    double pc_out[3];
    double epipolar_dir_out[2];

    Sophus::SE3d Trc = Tcr.inverse();
    double Tcr_data[3][4];
    double Trc_data[3][4];


    Eigen::Matrix<double, 3, 4> temp_cr = Tcr.matrix3x4();
    Eigen::Matrix<double, 3, 4> temp_rc = Trc.matrix3x4();
    for (int i = 0; i < 3; ++i)
    {
        
        for (int j = 0; j < 4; ++j)
        {
            Tcr_data[i][j] = temp_cr(i, j);
            Trc_data[i][j] = temp_rc(i, j);
        }
    }



    for (int j = boarder; j < width-boarder; ++j)
    {
        for (int i = boarder; i < height-boarder; ++i)
        {
            double depth_mu = depth.at<double>(i, j);
            double depth_sigma2 = cov2.at<double>(i, j);
            if (depth_sigma2 < min_cov || depth_sigma2 > max_cov) 
                continue;
            Eigen::Vector2d pr(j, i);
            // bool found = epipolar_search(ref, cur, Tcr, pr, depth_mu, depth_sigma2, pc, epipolar_dir);

            bool found = epipolar_search_cuda(ref.ptr<unsigned char>(0), cur.ptr<unsigned char>(0), 
                                               Tcr_data, pr.data(),
                                               depth_mu, depth_sigma2, 
                                               pc.data(), epipolar_dir.data());
          

            // std::cout << found << " " << found2 << "\n";
            if (!found)
                continue;
            // debug << epipolar_dir.transpose() << "\n";

            // showEpipolarMatch(ref, cur, pr, pc);

            //update_depth_filter(pr, pc, Tcr, epipolar_dir, depth, cov2);
            update_depth_filter_cuda(pr.data(), pc.data(), Trc_data, epipolar_dir.data(), depth.ptr<double>(0), cov2.ptr<double>(0));
        }
    }
    // std::cout << depth << "\n";

}



int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: dense_mapping path_to_test_dataset" << endl;
        return -1;
    }

    // Read dataset
    vector<string> color_image_files;
    vector<SE3d> poses_TWC;
    Mat ref_depth;
    bool ret = readDatasetFiles(argv[1], color_image_files, poses_TWC, ref_depth);
    if (ret == false) {
        cout << "Reading image files failed!" << endl;
        return -1;
    }
    cout << "read total " << color_image_files.size() << " files." << endl;

    // Initial depth image
    Mat ref = imread(color_image_files[0], 0); // gray-scale image
    SE3d pose_ref_TWC = poses_TWC[0];
    double init_depth = 3.0;
    double init_cov2 = 3.0;
    Mat depth(height, width, CV_64F, init_depth);
    Mat depth_cov2(height, width, CV_64F, init_cov2);

    for (int index = 1; index < color_image_files.size(); index++) {
        cout << "*** loop " << index << " ***" << endl;
        Mat curr = imread(color_image_files[index], 0);
        if (curr.data == nullptr) continue;
        SE3d pose_curr_TWC = poses_TWC[index];
        SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC;   // T_C_W * T_W_R = T_C_R
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        update(ref, curr, pose_T_C_R, depth, depth_cov2);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        std::cout << "Time used: " << time_used.count() << "s\n";
        evaludateDepth(ref_depth, depth);
        plotDepth(ref_depth, depth);
        plotCur(curr);
        // imshow("image", curr);
        // waitKey(1);
    }

    cout << "estimation returns, saving depth map ..." << endl;
    imwrite("depth.png", depth);
    cout << "done." << endl;

    debug.close();
    return 0;
}

bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    std::vector<SE3d> &poses,
    cv::Mat &ref_depth) {
    ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin) return false;

    while (!fin.eof()) {
        // 数据格式：图像文件名 tx, ty, tz, qx, qy, qz, qw ，注意是 TWC 而非 TCW
        string image;
        fin >> image;
        double data[7];
        for (double &d:data) fin >> d;

        color_image_files.push_back(path + string("/images/") + image);
        poses.push_back(
            SE3d(Quaterniond(data[6], data[3], data[4], data[5]),
                 Vector3d(data[0], data[1], data[2]))
        );
        if (!fin.good()) break;
    }
    fin.close();

    // load reference depth
    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = cv::Mat(height, width, CV_64F);
    if (!fin) return false;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            double depth = 0;
            fin >> depth;
            ref_depth.ptr<double>(y)[x] = depth / 100.0;
        }

    return true;
}



void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate) {
    double ave_depth_error = 0;
    double ave_depth_error_sq = 0;
    int cnt_depth_data = 0;
    for (int y = boarder; y < depth_truth.rows - boarder; y++)
        for (int x = boarder; x < depth_truth.cols - boarder; x++) {
            double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
            ave_depth_error += error;
            ave_depth_error_sq += error * error;
            cnt_depth_data++;
        }
    ave_depth_error /= cnt_depth_data;
    ave_depth_error_sq /= cnt_depth_data;

    cout << "Average squared error = " << ave_depth_error_sq << ", average error: " << ave_depth_error << endl;
}

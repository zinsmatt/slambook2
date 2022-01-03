#include <iostream>
#include <vector>
#include <fstream>
#include <thread>

#include "kernels.cuh"
#include "cuda_wrapper.h"
#include "constants.h"


void update_cuda(cv::Mat ref, cv::Mat cur, const Sophus::SE3d& Tcr, cv::Mat depth, cv::Mat cov2)
{
    Sophus::SE3d Trc = Tcr.inverse();
    double Tcr_data[3][4];
    double Trc_data[3][4];

    Eigen::Matrix<double, 3, 4> temp_cr = Tcr.matrix3x4();
    Eigen::Matrix<double, 3, 4> temp_rc = Trc.matrix3x4();
    // std::cout << temp_cr << "\n";
    // std::cout << temp_rc << "\n";
    for (int i = 0; i < 3; ++i)
    {

        for (int j = 0; j < 4; ++j)
        {
            Tcr_data[i][j] = temp_cr(i, j);
            Trc_data[i][j] = temp_rc(i, j);
        }
    }
    wrapper_update_cuda(ref.ptr<unsigned char>(0), cur.ptr<unsigned char>(0), Tcr_data, Trc_data, depth.ptr<double>(0), cov2.ptr<double>(0));


    // cv::setNumThreads(std::thread::hardware_concurrency());
    // cv::parallel_for_(cv::Range(boarder, depth.rows-boarder),
    //     [&](const cv::Range& range){

    //         // Eigen::Vector2d pc;
    //         // Eigen::Vector2d epipolar_dir;

    //         double pc[2];
    //         double epipolar_dir[2];
    //         for (int i = range.start; i < range.end; ++i)
    //         {
    //             for (int j = boarder; j < width-boarder; ++j)
    //             {
    //                 double depth_mu = depth.at<double>(i, j);
    //                 double depth_sigma2 = cov2.at<double>(i, j);
    //                 if (depth_sigma2 < min_cov || depth_sigma2 > max_cov)
    //                     continue;
    //                 Eigen::Vector2d pr(j, i);
    //                 // bool found = epipolar_search(ref, cur, Tcr, pr, depth_mu, depth_sigma2, pc, epipolar_dir);
    //                 bool found = epipolar_search_cuda(ref.ptr<unsigned char>(0), cur.ptr<unsigned char>(0),
    //                                                 Tcr_data, pr.data(),
    //                                                 depth_mu, depth_sigma2,
    //                                                 pc, epipolar_dir);

    //                 if (!found)
    //                     continue;
    //                 // showEpipolarMatch(ref, cur, pr, pc);
    //                 //update_depth_filter(pr, pc, Tcr, epipolar_dir, depth, cov2);
    //                 update_depth_filter_cuda(pr.data(), pc, Trc_data, epipolar_dir, depth.ptr<double>(0), cov2.ptr<double>(0));
    //             }
    //         }
    //     }
    //     );
}







inline double getBilinearInterpolatedValue(const cv::Mat &img, const Eigen::Vector2d &pt) {
    uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
    double xx = pt(0, 0) - floor(pt(0, 0));
    double yy = pt(1, 0) - floor(pt(1, 0));
    return ((1 - xx) * (1 - yy) * double(d[0]) +
            xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[img.step]) +
            xx * yy * double(d[img.step + 1])) / 255.0;
}



inline Eigen::Vector3d pix2cam(const Eigen::Vector2d px) {
    return Eigen::Vector3d(
        (px(0, 0) - cx) / fx,
        (px(1, 0) - cy) / fy,
        1
    );
}

inline Eigen::Vector2d cam2pix(const Eigen::Vector3d p_cam) {
    return Eigen::Vector2d(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx,
        p_cam(1, 0) * fy / p_cam(2, 0) + cy
    );
}





double ZNCC(cv::Mat im1, const Eigen::Vector2d& pt1, cv::Mat im2, Eigen::Vector2d& pt2)
{
    // no need to consider block partly outside because of boarder
    std::vector<double> v1(ncc_area, 0.0), v2(ncc_area, 0.0);
    double s1 = 0.0, s2 = 0.0;
    int idx = 0;
    for (int i = -ncc_window_size; i <= ncc_window_size; ++i)
    {
        for (int j = -ncc_window_size; j <= ncc_window_size; ++j)
        {
            double val_1 = static_cast<double>(im1.at<uchar>(pt1.y()+i, pt1.x()+j)) / 255;
            double val_2 = getBilinearInterpolatedValue(im2, pt2 + Eigen::Vector2d(j, i));
            s1 += val_1;
            s2 += val_2;
            v1[idx] = val_1;
            v2[idx] = val_2;
            ++idx;
        }
    }

    double mean_1 = s1 / ncc_area;
    double mean_2 = s2 / ncc_area;

    double numerator = 0.0;
    double den1 = 0.0, den2 = 0.0;
    for (int i = 0; i < v1.size(); ++i)
    {
        double zv1 = v1[i] - mean_1;
        double zv2 = v2[i] - mean_2;
        numerator += zv1*zv2;
        den1 += zv1 * zv1;
        den2 += zv2 * zv2;
    }
    auto zncc =  numerator / (std::sqrt(den1 * den2 + epsilon));
    // std::cout << "zncc = " << zncc << "\n";
    return zncc;
}

bool epipolar_search(cv::Mat ref, cv::Mat cur, const Sophus::SE3d& Tcr, const Eigen::Vector2d& pt, double depth_mu, double depth_sigma2, Eigen::Vector2d& best_pc, Eigen::Vector2d& epipolar_dir)
{
    double depth_sigma = std::sqrt(depth_sigma2);
    double dmax = depth_mu + 3 * depth_sigma;
    double dmin = depth_mu - 3 * depth_sigma;
    dmin = std::max(0.1, dmin);

    Eigen::Vector3d pn((pt.x()-cx) / fx, (pt.y() - cy) / fy, 1.0);
    pn.normalize();
    Eigen::Vector3d P_max = pn * dmax;
    Eigen::Vector3d P_min = pn * dmin;
    Eigen::Vector3d P_mu = pn * depth_mu;


    Eigen::Vector2d pc_max = cam2pix(Tcr * P_max);
    Eigen::Vector2d pc_min = cam2pix(Tcr * P_min);
    Eigen::Vector2d pc_mu = cam2pix(Tcr * P_mu);

    Eigen::Vector2d epipolar_line = pc_max - pc_min;
    epipolar_dir = epipolar_line.normalized();

    double step = 0.7;
    int nb_samples = std::ceil(epipolar_line.norm() / step);


    double half_range = 0.5 * epipolar_line.norm();
    if (half_range > 100) half_range = 100;

    Eigen::Vector2d p = pc_min;
    double best_zncc = -1.0;
    best_pc = pc_mu;
    // for (int i = 0; i < nb_samples; ++i)
    for (double l = -half_range; l<= half_range; l+= 0.7)
    {
        Eigen::Vector2d p = pc_mu + l * epipolar_dir;

        if (p.x() < boarder || p.x() >= width-boarder || p.y() < boarder || p.y() >= height-boarder)
            continue; // p is outside the cur image

        double zncc = ZNCC(ref, pt, cur, p);
        if (zncc > best_zncc)
        {
            best_zncc = zncc;
            best_pc = p;
        }

        // p += epipolar_dir * step;
    }

    // std::cout << best_zncc << "\n";
    if (best_zncc < 0.85)
        return false;
    else
        return true;
}


void update_depth_filter(const Eigen::Vector2d& pr, const Eigen::Vector2d& pc, const Sophus::SE3d& Tcr, const Eigen::Vector2d& epipolar_dir, cv::Mat depth, cv::Mat cov2)
{
    Sophus::SE3d Trc = Tcr.inverse();

    Eigen::Vector3d fr = pix2cam(pr);
    fr.normalize();
    Eigen::Vector3d fc = pix2cam(pc);
    fc.normalize();
    Eigen::Vector3d f2 = Trc.so3() * fc;
    Eigen::Vector3d trc = Trc.translation();


    // Solve the system of equation for triangulating depth
    Eigen::Matrix2d A;
    Eigen::Vector2d b;
    A(0, 0) = fr.dot(fr);
    A(0, 1) = -fr.dot(f2);
    A(1, 0) = f2.dot(fr);
    A(1, 1) = -f2.dot(f2);
    b[0] = fr.dot(trc);
    b[1] = f2.dot(trc);
    Eigen::Vector2d res = A.inverse() * b;
    Eigen::Vector3d P1 = fr * res[0];
    Eigen::Vector3d P2 = trc + fc * res[1];
    Eigen::Vector3d P_est = (P1 + P2) * 0.5;
    double depth_obs = P_est.norm(); //depth obs
    // debug << depth_obs << "\n";

    // Estimate depth uncertainty 
    Eigen::Vector3d P = fr * depth_obs;
    Eigen::Vector3d a = P - trc;
    Eigen::Vector3d t = trc.normalized();
    double alpha = std::acos(fr.dot(t));
    double beta = std::acos(a.normalized().dot(-t));
    // debug << alpha << " " << beta << "\n";
    Eigen::Vector2d pc2 = pc + epipolar_dir;
    Eigen::Vector3d fc2 = pix2cam(pc2);
    fc2.normalize();
    double beta_2 = std::acos(fc2.dot(-t));
    // debug << beta_2 << "\n";
    double gamma = M_PI - alpha - beta_2;
    double d_noise = trc.norm() * std::sin(beta_2) / std::sin(gamma); // sinus law
    double sigma_obs = depth_obs - d_noise;
    double sigma2_obs = sigma_obs * sigma_obs; // sigma2 obs
    // debug << sigma2_obs  <<"\n";

    // Depth fusion
    double d = depth.at<double>(static_cast<int>(pr.y()), static_cast<int>(pr.x()));
    double sigma2 = cov2.at<double>(static_cast<int>(pr.y()), static_cast<int>(pr.x()));

    double d_fused = (sigma2_obs * d + sigma2 * depth_obs) / (sigma2 + sigma2_obs);
    double sigma2_fused = (sigma2 * sigma2_obs) / (sigma2 + sigma2_obs);

    // debug << d_fused << "\n";
    depth.at<double>(static_cast<int>(pr.y()), static_cast<int>(pr.x())) = d_fused;
    cov2.at<double>(static_cast<int>(pr.y()), static_cast<int>(pr.x())) = sigma2_fused;
}

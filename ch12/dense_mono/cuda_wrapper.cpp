#include <iostream>
#include <vector>
#include <fstream>


#include "cuda_wrapper.h"

inline double getBilinearInterpolatedValue_cuda(const unsigned char *img, double pt[2]) {
    const unsigned char* d = &img[(int)pt[1] * width + (int)pt[0]];
    double xx = pt[0] - std::floor(pt[0]);
    double yy = pt[1] - std::floor(pt[1]);
    return ((1 - xx) * (1 - yy) * double(d[0]) +
            xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[width]) +
            xx * yy * double(d[width + 1])) / 255.0;
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

inline void pix2cam_cuda(const double in[2], double out[3]) {
    out[0] = (in[0] - cx) / fx;
    out[1] =  (in[1] - cy) / fy;
    out[2] = 1.0;
}

inline Eigen::Vector2d cam2pix(const Eigen::Vector3d p_cam) {
    return Eigen::Vector2d(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx,
        p_cam(1, 0) * fy / p_cam(2, 0) + cy
    );
}
inline void cam2pix_cuda(const double in[3], double out[2]) {
    out[0] = in[0] * fx / in[2] + cx;
    out[1] = in[1] * fy / in[2] + cy;
}

double norm3_cuda(double in[3])
{
    return std::sqrt(in[0]*in[0] + in[1]*in[1] + in[2]*in[2]);
}
// inplace normalization
inline void normalize3_cuda(double in_out[3]) {
    double d = std::sqrt(in_out[0]*in_out[0] 
                       + in_out[1]*in_out[1]
                       + in_out[2]*in_out[2]);
    in_out[0] /= d;
    in_out[1] /= d;
    in_out[2] /= d;
}

inline void normalize2_cuda(double in_out[2]) {
    double d = std::sqrt(in_out[0]*in_out[0] + in_out[1]*in_out[1]);
    in_out[0] /= d;
    in_out[1] /= d;
}


void transform_cuda(double x[3], const double T[3][4], double out[3])
{
    for (int i = 0; i < 3; ++i)
    {
        out[i] = x[0] * T[i][0] + x[1] * T[i][1] + x[2] * T[i][2] +  T[i][3];
    }
}

// inline bool inside(const Eigen::Vector2d &pt) {
//     return pt(0, 0) >= boarder && pt(1, 0) >= boarder
//            && pt(0, 0) + boarder < width && pt(1, 0) + boarder <= height;
// }


double ZNCC_cuda(const unsigned char *im1, const double pt1[2], const unsigned char *im2, const double pt2[2])
{
    // no need to consider block partly outside because of boarder
    double v1[ncc_area], v2[ncc_area];
    double s1 = 0.0, s2 = 0.0;
    int idx = 0;
    for (int i = -ncc_window_size; i <= ncc_window_size; ++i)
    {
        for (int j = -ncc_window_size; j <= ncc_window_size; ++j)
        {
            double val_1 = ((double) im1[((int)pt1[1] + i) * width + (int)pt1[0] + j]) / 255;
            double temp_p2[2] = {pt2[0] + j, pt2[1] + i};
            double val_2 = getBilinearInterpolatedValue_cuda(im2, temp_p2);
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
    for (int i = 0; i < ncc_area; ++i)
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

bool epipolar_search_cuda(const unsigned char* ref, const unsigned char* cur, 
const double Tcr[3][4], const double pt[2],
double depth_mu, double depth_sigma2, 
double best_pc[2], double epipolar_dir[2])
{
    double depth_sigma = std::sqrt(depth_sigma2);
    double dmax = depth_mu + 3 * depth_sigma;
    double dmin = depth_mu - 3 * depth_sigma;
    dmin = std::max(0.1, dmin);

    double pn[3];
    pix2cam_cuda(pt, pn);
    normalize3_cuda(pn);
    double P_max[3] = {pn[0] * dmax, pn[1] * dmax, pn[2] * dmax};
    double P_min[3] = {pn[0] * dmin, pn[1] * dmin, pn[2] * dmin};
    double P_mu[3] = {pn[0] * depth_mu, pn[1] * depth_mu, pn[2] * depth_mu};

    double P_max_cur[3], P_min_cur[3], P_mu_cur[3];
    transform_cuda(P_max, Tcr, P_max_cur);
    transform_cuda(P_min, Tcr, P_min_cur);
    transform_cuda(P_mu, Tcr, P_mu_cur);


    double pc_max[2], pc_min[2], pc_mu[2];
    cam2pix_cuda(P_max_cur, pc_max);
    cam2pix_cuda(P_min_cur, pc_min);
    cam2pix_cuda(P_mu_cur, pc_mu);


    double epipolar_line[2] = {pc_max[0] - pc_min[0], pc_max[1] - pc_min[1]};
    epipolar_dir[0] = epipolar_line[0];
    epipolar_dir[1] = epipolar_line[1];
    normalize2_cuda(epipolar_dir);
    double epipolar_line_norm = norm3_cuda(epipolar_line);

    // double step = 0.7;
    // int nb_samples = std::ceil(epipolar_line.norm() / step);

    double half_range = 0.5 * epipolar_line_norm;
    if (half_range > 100) half_range = 100;

    double best_zncc = -1.0;
    for (double l = -half_range; l<= half_range; l+= 0.7)
    {
        double p[2] = {pc_mu[0] + l * epipolar_dir[0], pc_mu[1] + l * epipolar_dir[1]};

        if (p[0] < boarder || p[0] >= width-boarder || p[1] < boarder || p[1] >= height-boarder)
            continue; // p is outside the cur image

        double zncc = ZNCC_cuda(ref, pt, cur, p);
        if (zncc > best_zncc)
        {
            best_zncc = zncc;
            best_pc[0] = p[0];
            best_pc[1] = p[1];
        }
    }
    // std::cout << best_zncc << "\n";
    if (best_zncc < 0.85)
        return false;
    else
        return true;
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

double dot3_cuda(const double a[3], const double b[3])
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

double det2(const double A[2][2])
{
    return A[0][0] * A[1][1] - A[1][0] * A[0][1];
}

void solve_Axb2_cuda(const double A[2][2], const double b[2], double res[2])
{
    double det_inv = 1.0 / det2(A);
    double A_inv[2][2];
    A_inv[0][0] = det_inv * A[1][1];
    A_inv[0][1] = -det_inv * A[0][1];
    A_inv[1][0] = -det_inv * A[1][0];
    A_inv[1][1] = det_inv * A[0][0];

    res[0] = A_inv[0][0] * b[0] + A_inv[0][1] * b[1];
    res[1] = A_inv[1][0] * b[0] + A_inv[1][1] * b[1];
}



void update_depth_filter_cuda(const double pr[2], const double pc[2], const double Trc[3][4], const double epipolar_dir[2], double *depth, double *cov2)
{
    double fr[3];
    pix2cam_cuda(pr, fr);
    normalize3_cuda(fr);

    double fc[3];
    pix2cam_cuda(pc, fc);
    normalize3_cuda(fc);
    
    double f2[3] = {dot3_cuda(Trc[0], fc),
                    dot3_cuda(Trc[1], fc),
                    dot3_cuda(Trc[2], fc)};

    double trc[3] = {Trc[0][3], Trc[1][3], Trc[2][3]};
    double A[2][2];
    double b[2];

    A[0][0] = dot3_cuda(fr, fr);
    A[0][1] = dot3_cuda(fr, f2);
    A[1][0] = dot3_cuda(f2, fr);
    A[1][1] = dot3_cuda(f2, f2);
    A[0][1] *= -1;
    A[1][1] *= -1;
    
    b[0] = dot3_cuda(fr, trc);
    b[1] = dot3_cuda(f2, trc);

    double res[2];
    solve_Axb2_cuda(A, b, res);
    double P1[3] = {fr[0] * res[0], fr[1] * res[0], fr[2] * res[0]};
    double P2[3] = {trc[0] + fc[0] * res[1], trc[1] + fc[1] * res[1], trc[2] + fc[2] * res[1]};
    double P_est[3] = {(P1[0] + P2[0]) * 0.5, 
                       (P1[1] + P2[1]) * 0.5, 
                       (P1[2] + P2[2]) * 0.5};
    double depth_obs = norm3_cuda(P_est);

    double P[3] = {fr[0] * depth_obs, fr[1] * depth_obs, fr[2] * depth_obs};
    double a[3] = {P[0] - trc[0], P[1] - trc[1], P[2] - trc[2]};

    double t[3] = {trc[0], trc[1], trc[2]};
    normalize3_cuda(t);

    double alpha = std::acos(dot3_cuda(fr, t));
    double beta = std::acos(-dot3_cuda(a, t) / norm3_cuda(a));

    double pc2[2] = {pc[0] + epipolar_dir[0], pc[1] + epipolar_dir[1]};
    double fc2[3];
    pix2cam_cuda(pc2, fc2);
    normalize3_cuda(fc2);
    double beta_2 = std::acos(-dot3_cuda(fc2, t));

    double gamma = M_PI - alpha - beta_2;

    double d_noise = norm3_cuda(trc) * std::sin(beta_2) / std::sin(gamma); // sinus law
    double sigma_obs = depth_obs - d_noise;
    double sigma2_obs = sigma_obs * sigma_obs;


    // Depth fusion
    double d = depth[(int)pr[1] * width + (int)pr[0]];
    double sigma2 = cov2[(int)pr[1] * width + (int)pr[0]];

    double d_fused = (sigma2_obs * d + sigma2 * depth_obs) / (sigma2 + sigma2_obs);
    double sigma2_fused = (sigma2 * sigma2_obs) / (sigma2 + sigma2_obs);

    // debug << d_fused << "\n";
    depth[(int)pr[1] * width + (int)pr[0]] = d_fused;
    cov2[(int)pr[1] * width + (int)pr[0]] = sigma2_fused;
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

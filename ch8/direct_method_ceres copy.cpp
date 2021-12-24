#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// Camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;

// baseline
double baseline = 0.573;
// paths
string left_file = "../left.png";
string disparity_file = "../disparity.png";
boost::format fmt_others("../%06d.png");    // other files

// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;


// bilinear interpolation
inline float get(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}

Eigen::Vector3d get_3D_point_from_depth(const Eigen::Vector2d& p, double depth)
{
    return Eigen::Vector3d(depth * (p.x() - cx) / fx,
                           depth * (p.y() - cy) / fy,
                           depth);
}


using namespace Sophus;
// Local parameterization needed to handle SE3 from Sophus (from Sophus/test/ceres/)
class LocalParameterizationSE3 : public ceres::LocalParameterization {
 public:
  virtual ~LocalParameterizationSE3() {}

  // SE3 plus operation for Ceres
  //
  //  T * exp(x)
  //
  virtual bool Plus(double const* T_raw, double const* delta_raw,
                    double* T_plus_delta_raw) const {
    Eigen::Map<SE3d const> const T(T_raw);
    Eigen::Map<Vector6d const> const delta(delta_raw);
    Eigen::Map<SE3d> T_plus_delta(T_plus_delta_raw);
    T_plus_delta = T * SE3d::exp(delta); ///// check if good left or righ multiply ??????????????????
    // std::cout << (SE3d::exp(delta) * T).matrix() << "\n";
    // std::cout << (T * SE3d::exp(delta)).matrix() << "\n\n";
    return true;
  }

  // Jacobian of SE3 plus operation for Ceres
  //
  // Dx T * exp(x)  with  x=0
  //
  virtual bool ComputeJacobian(double const* T_raw,
                               double* jacobian_raw) const {
    Eigen::Map<SE3d const> T(T_raw);
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> jacobian(
        jacobian_raw);
    jacobian = T.Dx_this_mul_exp_x_at_0();
    return true;
  }

  virtual int GlobalSize() const { return SE3d::num_parameters; }

  virtual int LocalSize() const { return SE3d::DoF; }
};

struct PhotometricError: public ceres::SizedCostFunction<1, 7>
{
    PhotometricError(const cv::Mat& img1, const cv::Mat& img2, const Eigen::Vector2d& p1, const Eigen::Vector3d& P1, const Eigen::Matrix3d& K)
    : _img1(img1), _img2(img2), _p1(p1), _P1(P1), _K(K) {}

    virtual bool Evaluate(double const* const *params,
                          double *residuals,
                          double **jacobians) const {
        const Eigen::Map<const Sophus::SE3d> Rt(params[0]);

        Eigen::Vector3d P2 = Rt * _P1;
        Eigen::Vector3d p2 = _K * P2;
        p2 /= p2.z();

        double v1 = get(_img1, _p1.x(), _p1.y());
        double v2 = get(_img2, p2.x(), p2.y());
        double err = v1 - v2;
        residuals[0] = err;

        if (!jacobians) return true;
        if (!jacobians[0]) return true;

        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double cx = _K(0, 2);
        double cy = _K(1, 2);
        double X2 = std::pow(P2.x(), 2);
        double Y2 = std::pow(P2.y(), 2);
        double Z2 = std::pow(P2.z(), 2);


        int xx = 0, yy = 0;
        double dx = 0.5 * (get(_img2, p2.x() + xx + 1, p2.y() + yy) - get(_img2, p2.x() + xx - 1, p2.y() + yy));
        double dy = 0.5 * (get(_img2, p2.x() + xx, p2.y() + yy + 1) - get(_img2, p2.x() + xx, p2.y() + yy - 1));
        Eigen::Vector2d dIdu(dx, dy);
        Eigen::Matrix<double, 2, 6> dudRt;
        dudRt << fx/P2.z(), 0.0, -fx*P2.x() / Z2,-fx * P2.x() * P2.y() / Z2, fx + fx * X2 / Z2, -fx * P2.y() / P2.z(),
                 0.0, fy / P2.z(), -fy*P2.y()/Z2, -fy-fy * Y2 / Z2, fy * P2.x() * P2.y() / Z2, fy * P2.x() / P2.z();
        Eigen::Matrix<double, 6, 1> J = -(dIdu.transpose() * dudRt).transpose();
        jacobians[0][0] = J(0, 0);
        jacobians[0][1] = J(1, 0);
        jacobians[0][2] = J(2, 0);
        jacobians[0][3] = J(3, 0);
        jacobians[0][4] = J(4, 0);
        jacobians[0][5] = J(5, 0);
        jacobians[0][6] = 0.0;

        return true;
    }

    private:
        cv::Mat _img1, _img2;
        Eigen::Vector2d _p1;
        Eigen::Vector3d _P1;
        Eigen::Matrix3d _K;
};


/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    const Eigen::Matrix3d& K,
    Sophus::SE3d &Rt // points from cam1 reference frame to cam2
)
{
    int nb_iters = 10;
    int half_w_size = 2;
    double prev_cost = 0.0;

    ceres::Problem problem;

    problem.AddParameterBlock(Rt.data(), 7, new LocalParameterizationSE3());
std::cout << "AFTER add parameters bloxk" << std::endl;
    for (int i = 0; i < px_ref.size(); ++i)
    {
        const auto& p1 = px_ref[i];
        Eigen::Vector3d P1 = get_3D_point_from_depth(p1, depth_ref[i]);
        problem.AddResidualBlock(
            new PhotometricError(img1, img2, p1, P1, K),
            nullptr,
            Rt.data()
        );
    }
std::cout << "AFTER ad dresiduals block" << std::endl;

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double >>(t2 - t1);
    cout << "optimization with ceres costs time: " << time_used.count() << " seconds." << endl;
    

    std::cout << "translation: " << Rt.translation().transpose() << "\n";
    std::cout << "rotation: " << Rt.so3().unit_quaternion().toRotationMatrix() << "\n";

}


void DirectPoseEstimationPyramidal(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    const Eigen::Matrix3d& K,
    Sophus::SE3d &Rt)
{
    int nb_levels = 1;
    double factor = 2.0;
    double scale = 1.0 / std::pow(factor, nb_levels-1);


    for (int l = 0; l < nb_levels; ++l)
    {
        cv::Mat img1_r, img2_r;
        cv::resize(img1, img1_r, cv::Size(), scale, scale);
        cv::resize(img2, img2_r, cv::Size(), scale, scale);

        Eigen::Matrix3d K_r = K;
        K_r(0, 0) *= scale;
        K_r(1, 1) *= scale;
        K_r(0, 2) *= scale;
        K_r(1, 2) *= scale;
        auto p_r = px_ref;
        for (auto& p : p_r)
        {
            p *= scale;
        }

        DirectPoseEstimationSingleLayer(img1_r, img2_r, p_r, depth_ref, K_r, Rt);

        scale *= factor;
    }
}


int main(int argc, char **argv) {

    cv::Mat left_img = cv::imread(left_file, 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng(1994);
    int nPoints = 2000;
    int boarder = 20;
    VecVector2d pixels_ref;
    vector<double> depth_ref;


    // generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++) {
        int x = rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
        int y = rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
        int disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity; // you know this is disparity to depth
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }

    // estimates 01~05.png's pose using this information
    Sophus::SE3d Rt;
    Eigen::Matrix3d K;
    K << fx, 0.0, cx,
         0.0, fy, cy,
         0.0, 0.0, 1.0;

    for (int i = 1; i < 2; i++) {  // 1~10
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        // try single layer by uncomment this line
        DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, K, Rt);
        // DirectPoseEstimationPyramidal(left_img, img, pixels_ref, depth_ref, K, Rt);


        // plot the projected pixels here
        cv::Mat img2_show;
        cv::cvtColor(img, img2_show, CV_GRAY2BGR);
        std::vector<Eigen::Vector2d> projections(pixels_ref.size());
        for (int i = 0; i < pixels_ref.size(); ++i)
        {
            Eigen::Vector3d P_ref = get_3D_point_from_depth(pixels_ref[i], depth_ref[i]);
            Eigen::Vector3d uv = K * (Rt * P_ref);
            projections[i] = uv.hnormalized();
        }

        for (size_t i = 0; i < pixels_ref.size(); ++i) {
            auto p_ref = pixels_ref[i];
            auto p_cur = projections[i];
            if (p_cur[0] > 0 && p_cur[1] > 0) {
                cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
                cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
                        cv::Scalar(0, 250, 0));
            }
        }
        cv::imshow("current", img2_show);
        cv::waitKey();
        // cv::imwrite("img_"+std::to_string(i) + ".png", img2_show);

    }
    return 0;
}

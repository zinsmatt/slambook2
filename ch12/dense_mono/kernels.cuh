#pragma once

struct Mat3x4
{
    double a00, a01, a02, a03, 
           a10, a11, a12, a13, 
           a20, a21, a22, a23;
};

bool epipolar_search_cuda(const unsigned char* ref, const unsigned char* cur, 
                          const double Tcr[3][4], const double pt[2], 
                          double depth_mu, double depth_sigma2, 
                          double best_pc[2], double epipolar_dir[2], double *debug, int &a);



void update_depth_filter_cuda(const double pr[2], const double pc[2],
                              const double Trc[3][4], const double epipolar_dir[2], 
                              double *depth, double *cov2);

void wrapper_update_cuda(const unsigned char* ref, const unsigned char* cur, double Tcr[3][4], double Trc[3][4], double *depth, double *cov2);
/*
* C++版本
* 
*/
#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "OsqpEigen/OsqpEigen.h"

using namespace Eigen;

// 五阶伯恩斯坦多项式
class QuinticBernsteinPolynomial {
public:
    QuinticBernsteinPolynomial(Eigen::VectorXd params, double time_allocation)
        : params_(params), time_allocation_(time_allocation) {
        assert(params.size() == 6);
    }

    double value(double t) {
        double u = t / time_allocation_;
        return params_[0] * pow(1 - u, 5) +
               params_[1] * 5 * pow(1 - u, 4) * u +
               params_[2] * 10 * pow(1 - u, 3) * pow(u, 2) +
               params_[3] * 10 * pow(1 - u, 2) * pow(u, 3) +
               params_[4] * 5 * (1 - u) * pow(u, 4) +
               params_[5] * pow(u, 5);
    }

    double derivative(double t) {
        double u = t / time_allocation_;
        return (1 / time_allocation_) *
               (params_[0] * (-5) * pow(1 - u, 4) +
                params_[1] * (5 * pow(1 - u, 4) - 20 * pow(1 - u, 3) * u) +
                params_[2] * (20 * pow(1 - u, 3) * u - 30 * pow(1 - u, 2) * pow(u, 2)) +
                params_[3] * (30 * pow(1 - u, 2) * pow(u, 2) - 20 * (1 - u) * pow(u, 3)) +
                params_[4] * (20 * (1 - u) * pow(u, 3) - 5 * pow(u, 4)) +
                params_[5] * 5 * pow(u, 4));
    }

    double secondOrderDerivative(double t) {
        double u = t / time_allocation_;
        return pow(1 / time_allocation_, 2) *
               (params_[0] * 20 * pow(1 - u, 3) +
                params_[1] * 5 * (-8 * pow(1 - u, 3) + 12 * pow(1 - u, 2) * u) +
                params_[2] * 10 * (2 * pow(1 - u, 3) - 12 * pow(1 - u, 2) * u + 6 * (1 - u) * pow(u, 2)) +
                params_[3] * 10 * (6 * pow(1 - u, 2) * u - 12 * (1 - u) * pow(u, 2) + 2 * pow(u, 3)) +
                params_[4] * 5 * (12 * (1 - u) * pow(u, 2) - 8 * pow(u, 3)) +
                params_[5] * 20 * pow(u, 3));
    }

    double thirdOrderDerivative(double t) {
        double u = t / time_allocation_;
        return pow(1 / time_allocation_, 3) *
               (params_[0] * (-60) * pow(1 - u, 2) +
                params_[1] * 5 * (36 * pow(1 - u, 2) - 24 * (1 - u) * u) +
                params_[2] * 10 * (-18 * pow(1 - u, 2) + 36 * (1 - u) * u - 6 * pow(u, 2)) +
                params_[3] * 10 * (6 * pow(1 - u, 2) - 36 * (1 - u) * u + 18 * pow(u, 2)) +
                params_[4] * 5 * (24 * (1 - u) * u - 36 * pow(u, 2)) +
                params_[5] * 60 * pow(u, 2));
    }

private:
    Eigen::VectorXd params_;
    double time_allocation_;
};

class PieceWiseTrajectory {
public:
    int segment_num_;
    std::vector<double> time_segments_;
    std::vector<std::vector<QuinticBernsteinPolynomial>> trajectory_segments_;

public:
    PieceWiseTrajectory(std::vector<Eigen::VectorXd> x_params, std::vector<Eigen::VectorXd>  y_params, std::vector<Eigen::VectorXd> z_params, std::vector<double> time_allocations) {
        segment_num_ = time_allocations.size();
        time_segments_.resize(segment_num_);
        double cumulative_time = 0.0;
        for (int i = 0; i < segment_num_; ++i) {
            cumulative_time += time_allocations[i];
            time_segments_[i] = cumulative_time;
            std::vector<QuinticBernsteinPolynomial> ploys;
            ploys.push_back(QuinticBernsteinPolynomial(x_params[i], time_allocations[i]));
            ploys.push_back(QuinticBernsteinPolynomial(y_params[i], time_allocations[i]));
            ploys.push_back(QuinticBernsteinPolynomial(z_params[i], time_allocations[i]));
            trajectory_segments_.emplace_back(ploys);
        }
    }

    int index(double t) {
        for (int i = 0; i < segment_num_; ++i) {
            if (t <= time_segments_[i]) {
                return i;
            }
        }
        return -1;
    }

    std::vector<double> getPos(double t) {
        int idx = index(t);
        if (idx > 0) {
            t -= time_segments_[idx - 1];
        }
        return {trajectory_segments_[idx][0].value(t), trajectory_segments_[idx][1].value(t), trajectory_segments_[idx][2].value(t)};
    }

    std::vector<double> getVel(double t) {
        int idx = index(t);
        if (idx > 0) {
            t -= time_segments_[idx - 1];
        }
        return {trajectory_segments_[idx][0].derivative(t), trajectory_segments_[idx][1].derivative(t), trajectory_segments_[idx][2].derivative(t)};
    }

    // double getYew(double t) {
    //     auto [vx, vy] = getVel(t);
    //     vx += 1e-12;
    //     double degree = std::atan2(vy, vx) * 180 / M_PI;
    //     if (vx < 0) {
    //         degree += 180;
    //     }
    //     return degree;
    // }

    // double getDYew(double t) {
    //     auto [ax, ay] = getAcc(t);
    //     ax += 1e-12;
    //     double beta = std::atan2(ay, ax) * 180 / M_PI;
    //     if (ax < 0) {
    //         beta += 180;
    //     }
    //     return beta;
    // }

    std::vector<double> getAcc(double t) {
        int idx = index(t);
        if (idx > 0) {
            t -= time_segments_[idx - 1];
        }
        return {trajectory_segments_[idx][0].secondOrderDerivative(t), trajectory_segments_[idx][1].secondOrderDerivative(t), trajectory_segments_[idx][2].secondOrderDerivative(t)};
    }

    std::vector<double> getJerk(double t) {
        int idx = index(t);
        if (idx > 0) {
            t -= time_segments_[idx - 1];
        }
        return {trajectory_segments_[idx][0].thirdOrderDerivative(t), trajectory_segments_[idx][1].thirdOrderDerivative(t), trajectory_segments_[idx][2].thirdOrderDerivative(t)};
    }
};

class TrajectoryOptimizer {
private:
    double vel_max_, acc_max_, jerk_max_;
    int dim_, degree_, freedom_;
    Eigen::MatrixXd M_k;
    Eigen::MatrixXd M_, Q_;
    PieceWiseTrajectory* piece_wise_trajectory;
    bool get_traj_;

public:
    TrajectoryOptimizer(double vel_max, double acc_max, double jerk_max)
            : vel_max_(vel_max), acc_max_(acc_max), jerk_max_(jerk_max),
              dim_(3), degree_(5), freedom_(degree_ + 1) {
        get_traj_ = false;
        M_k.resize(6,6);
        M_k << 1, 0, 0, 0, 0, 0,
                -5, 5, 0, 0, 0, 0,
                10, -20, 10, 0, 0, 0,
                -10, 30, -30, 10, 0, 0,
                5, -20, 30, -20, 5, 0,
                -1, 5, -10, 10, -5, 1;
    }

    // Other methods
    void optimize(Eigen::VectorXd start_state, Eigen::VectorXd end_state, 
                std::vector<Eigen::Vector3d>& line_points, vec_E<Polyhedron3D> polyArray) {
        // Implementation
        int segment_num = line_points.size()-1;
        std::vector<double> time_allocations;
        for(int i=0; i<segment_num; i++){
            time_allocations.push_back((line_points[i + 1] - line_points[i]).norm() / vel_max_ * 2);
        }
        optimizeIter(start_state, end_state, segment_num, time_allocations, polyArray);
    }

    double factorial(int n) {  
        int result = 1;  
        for (int i = 2; i <= n; ++i) {  
            result *= i;  
        }  
        return (double)result;  
    }  
    
    void getQM(int n_seg, int n_order, const std::vector<double>& ts) {
        int d_order = (n_order + 1) / 2;
        int len = n_seg * (n_order + 1);
        M_.resize(len, len);
        Q_.resize(len, len);
        Eigen::MatrixXd Q_k(n_order + 1, n_order + 1);
        for (int k=1; k<n_seg+1; k++) {
            Q_k.setZero();
            for(int i=d_order; i<n_order+1; i++) {
                for(int j=d_order; j<n_order+1; j++) {
                    Q_k(i, j) = (
                            factorial(i) / factorial(i - d_order) *
                            factorial(j) / factorial(j - d_order) /
                            (double)(i + j - n_order) * pow(ts[k - 1], (i + j - n_order))
                    );
                }
            }
            Q_.block((n_order + 1)*(k-1), (n_order + 1)*(k-1), (n_order + 1), (n_order + 1)) = Q_k;
            M_.block((n_order + 1)*(k-1), (n_order + 1)*(k-1), (n_order + 1), (n_order + 1)) = M_k;
        }
    }

    // Other methods
    void optimizeIter(Eigen::VectorXd start_state, Eigen::VectorXd end_state, int segment_num,
                        std::vector<double> time_allocations, vec_E<Polyhedron3D> polyArray) {

        getQM(segment_num, 5, time_allocations);
        Eigen::MatrixXd P_k;
        P_k = M_.transpose()*Q_*M_;
        int row_p = P_k.rows();
        int col_p = P_k.cols();
        Eigen::MatrixXd P(3*row_p, 3*col_p);
        for(int i=0; i<3; i++) {
            P.block(row_p*i,col_p*i,row_p, col_p) = P_k;
        }
        Eigen::SparseMatrix<double> Sparse_P = P.sparseView();
        Eigen::VectorXd q = Eigen::VectorXd::Zero(dim_ * segment_num * freedom_);
        int equality_constraints_num = 5 * dim_ + 3 * (segment_num - 1) * dim_;
        int param_num = dim_ * segment_num * freedom_;
        int inequality_constraints_num = 0;
        for (const auto& polygon : polyArray) { 
            inequality_constraints_num += freedom_ * polygon.hyperplanes().size();
        }
        // 速度的超平面不等式约束
        inequality_constraints_num += (freedom_ - 1) * segment_num * dim_;
        // 加速度的超平面不等式约束
        inequality_constraints_num += (freedom_ - 2) * segment_num * dim_;
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(equality_constraints_num + inequality_constraints_num, param_num);
        Eigen::VectorXd lb = -Eigen::VectorXd::Ones(equality_constraints_num + inequality_constraints_num) * 999999;//std::numeric_limits<double>::infinity();
        Eigen::VectorXd ub = Eigen::VectorXd::Ones(equality_constraints_num + inequality_constraints_num) * 999999;//std::numeric_limits<double>::infinity();
        // ROS_INFO("P:%d, %d, Q:%d", P.rows(), row_p, q.size());
        int constraints_index = 0;
        // 起点x,y,z位置
        for(int i=0; i<3; i++) {
            A(constraints_index, i*segment_num*freedom_) = 1;
            lb(constraints_index) = start_state(i);
            ub(constraints_index) = start_state(i);
            constraints_index++;
        }
        // 起点x,y,z速度
        for(int i=0; i<3; i++) {
            A(constraints_index, i*segment_num*freedom_) = -5/time_allocations[0];
            A(constraints_index, i*segment_num*freedom_+1) = 5/time_allocations[0];
            lb(constraints_index) = start_state(5+i);
            ub(constraints_index) = start_state(5+i);
            constraints_index++;
        }
        // 起点x,y,z加速度
        for(int i=0; i<3; i++) {
            A(constraints_index, i*segment_num*freedom_) = 20 / pow(time_allocations[0], 2);
            A(constraints_index, i*segment_num*freedom_+1) = -40 / pow(time_allocations[0], 2);
            A(constraints_index, i*segment_num*freedom_+2) = 20 / pow(time_allocations[0], 2);
            lb(constraints_index) = start_state(8+i);
            ub(constraints_index) = start_state(8+i);
            constraints_index++;
        }
        // 终点x,y,z位置
        for(int i=0; i<3; i++) {
            A(constraints_index, (i+1)*segment_num*freedom_-1) = 1;
            lb(constraints_index) = end_state(i);
            constraints_index++;
        }
        // 终点x,y,z速度
        for(int i=0; i<3; i++) {
            A(constraints_index, (i+1)*segment_num*freedom_-1) = 5/time_allocations.back();
            A(constraints_index, (i+1)*segment_num*freedom_-2) = -5/time_allocations.back();
            lb(constraints_index) = end_state(5+i);
            ub(constraints_index) = end_state(5+i);
            constraints_index++;
        }

        // 连接处的零阶导数相等
        constraints_index = 15;
        for(int sigma=0; sigma<dim_; sigma++) {
            for(int n=0; n<segment_num-1; n++) {
                A(constraints_index,sigma * segment_num * freedom_ + n * freedom_ + freedom_ - 1) = 1;
                A(constraints_index,sigma * segment_num * freedom_ + (n + 1) * freedom_) = -1;
                lb(constraints_index) = 0;
                ub(constraints_index) = 0;
                constraints_index += 1;
            }
        }
        //连接处的一阶导数相等
        for(int sigma=0; sigma<dim_; sigma++) {
            for(int n=0; n<segment_num-1; n++) {
                A(constraints_index, sigma * segment_num * freedom_ + n * freedom_ + freedom_ - 1) = 5 / time_allocations[n];
                A(constraints_index, sigma * segment_num * freedom_ + n * freedom_ + freedom_ - 2) = -5 / time_allocations[n];
                A(constraints_index, sigma * segment_num * freedom_ + (n + 1) * freedom_) = 5 / time_allocations[n+1];
                A(constraints_index, sigma * segment_num * freedom_ + (n + 1) * freedom_ + 1) = -5 / time_allocations[n+1];
                lb(constraints_index) = 0;
                ub(constraints_index) = 0;
                constraints_index += 1;
            }
        }
        //连接处的二阶导数相等
        for(int sigma=0; sigma<dim_; sigma++) {
            for(int n=0; n<segment_num-1; n++) {
                A(constraints_index, sigma * segment_num * freedom_ + n * freedom_ + freedom_ - 1) = 20 / time_allocations[n] / time_allocations[n];
                A(constraints_index, sigma * segment_num * freedom_ + n * freedom_ + freedom_ - 2) = -40 / time_allocations[n] / time_allocations[n];
                A(constraints_index, sigma * segment_num * freedom_ + n * freedom_ + freedom_ - 3) = 20 / time_allocations[n] / time_allocations[n];
                A(constraints_index, sigma * segment_num * freedom_ + (n + 1) * freedom_) = -20 / time_allocations[n+1] / time_allocations[n+1];
                A(constraints_index, sigma * segment_num * freedom_ + (n + 1) * freedom_ + 1) = 40 / time_allocations[n+1] / time_allocations[n+1];
                A(constraints_index, sigma * segment_num * freedom_ + (n + 1) * freedom_ + 2) = -20 / time_allocations[n+1] / time_allocations[n+1];
                lb(constraints_index) = 0;
                ub(constraints_index) = 0;
                constraints_index += 1;
            }
        }

        // 速度约束
        for(int d=0; d<dim_; d++) {
            for (int n = 0; n < segment_num; ++n) {
                for (int k = 0; k < degree_; ++k) {
                    A(constraints_index, d * segment_num * freedom_ + n * freedom_ + k) = -degree_ / time_allocations[n];
                    A(constraints_index, d * segment_num * freedom_ + n * freedom_ + k + 1) = degree_ / time_allocations[n];
                    ub[constraints_index] = vel_max_;
                    lb[constraints_index] = -vel_max_;
                    ++constraints_index;
                }
            }
        }
        // 加速度约束
        int pAcc = degree_ * (degree_ - 1);
        for(int d=0; d<dim_; d++) {
            for (int n = 0; n < segment_num; ++n) {
                for (int k = 0; k < degree_ - 1; ++k) {
                    A(constraints_index, d * segment_num * freedom_ + n * freedom_ + k) = pAcc / time_allocations[n] / time_allocations[n];
                    A(constraints_index, d * segment_num * freedom_ + n * freedom_ + k + 1) = -2 * pAcc / time_allocations[n] / time_allocations[n];
                    A(constraints_index, d * segment_num * freedom_ + n * freedom_ + k + 2) = pAcc / time_allocations[n] / time_allocations[n];
                    ub[constraints_index] = acc_max_;
                    lb[constraints_index] = -acc_max_;
                    ++constraints_index;
                }
            }
        }

        //不等式约束
        for(int n=0; n<segment_num;n++) {
            for(int k=0; k<freedom_; k++) {
                for(auto hyper_plane:polyArray[n].hyperplanes()) {
                    A(constraints_index, n * freedom_ + k) = hyper_plane.n_[0];
                    A(constraints_index, segment_num * freedom_ + n * freedom_ + k) = hyper_plane.n_[1];
                    A(constraints_index, 2 * segment_num * freedom_ + n * freedom_ + k) = hyper_plane.n_[2];
                    ub(constraints_index) = hyper_plane.n_.dot(hyper_plane.p_);
                    constraints_index += 1;
                }
            }
        }
        std::cout<<"!!!equ: "<<(constraints_index==equality_constraints_num + inequality_constraints_num)<<std::endl;
        //solve
        OsqpEigen::Solver solver;
        // settings
        solver.settings()->setVerbosity(false);
        solver.settings()->setWarmStart(true);
        // set the initial data of the QP solver
        solver.data()->setNumberOfVariables(P.rows());   //变量数n
        Eigen::SparseMatrix<double> Sparse_A = A.sparseView();
        solver.data()->setNumberOfConstraints(A.rows()); //约束数m
        if (!solver.data()->setHessianMatrix(Sparse_P))
            return;
        if (!solver.data()->setGradient(q))
            return;
        if (!solver.data()->setLinearConstraintsMatrix(Sparse_A))
            return;
        if (!solver.data()->setLowerBound(lb))
            return;
        if (!solver.data()->setUpperBound(ub))
            return;
    
        // instantiate the solver
        if (!solver.initSolver())
            return;
    

        // solve the QP problem
        if (!solver.solve())
            return;
            
        get_traj_ = true;
        Eigen::VectorXd QPSolution;
        QPSolution = solver.getSolution();
        std::vector<Eigen::VectorXd> trajectory_x_params, trajectory_y_params, trajectory_z_params;
        for(int n=0; n<segment_num; n++) {
            trajectory_x_params.push_back(QPSolution.segment(freedom_*n, freedom_));
            trajectory_y_params.push_back(QPSolution.segment(segment_num*freedom_+freedom_*n, freedom_));
            trajectory_z_params.push_back(QPSolution.segment(2*segment_num*freedom_+freedom_*n, freedom_));
        }
        piece_wise_trajectory = new PieceWiseTrajectory(trajectory_x_params, trajectory_y_params, trajectory_z_params, time_allocations);
    }

    PieceWiseTrajectory* getTraj() {
        return piece_wise_trajectory;
    }

    /**
    * 是否成功优化得到轨迹
    */
    bool isGetTraj() { return get_traj_; }

    /**
    * 判断是否正定
    */
    void is(Eigen::MatrixXd Matrix) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(Matrix);
        if (eigensolver.info() == Eigen::Success) {
            Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
            bool isPositiveDefinite = true;
            bool isPositiveSemidefinite = true;
            for (int i = 0; i < eigenvalues.size(); ++i) {
                if (eigenvalues(i) <= 0) {
                    isPositiveDefinite = false;
                    if (eigenvalues(i) < 0) {
                        isPositiveSemidefinite = false;
                    }
                }
            }
            if (isPositiveDefinite) {
                std::cout << "The matrix is positive definite." << std::endl;
            } else if (isPositiveSemidefinite) {
                std::cout << "The matrix is positive semidefinite." << std::endl;
            } else {
                std::cout << "The matrix is neither positive definite nor positive semidefinite." << std::endl;
            }
        } else {
            std::cerr << "Eigenvalue computation failed." << std::endl;
        }
    }
};

inline double getAngle(const Eigen::Vector2d& data) {
    double theta = std::atan2(data[1], (data[0] + 1e-12)) * 180 / M_PI;
    if (data[0] < 0) {
        theta += 180;
    }
    return theta;
}
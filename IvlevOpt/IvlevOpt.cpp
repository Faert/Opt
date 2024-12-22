#include <iostream>
#include <filesystem>
#include "nlohmann/json.hpp"
#include "Matrix.h"

// /openmp:experimental
// /Qvec-report:2

int main()
{
    std::srand(time(NULL));

    //input param and create base log
    std::ifstream in;
    in.open("input.json");
    nlohmann::json data = nlohmann::json::parse(in);
    in.close();

    std::cout << data << '\n';

    //a - main reservoir; b - qubit(transmon); c - small reservoir
    const double alpha = double(data.at("alpha"));// * 2 * PI
    double omega = double(data.at("omega"));// * 2 * PI
    const double st_omega = omega;
    const double delta = double(data.at("delta"));// * 2 * PI
    const double g_r = double(data.at("g_r"));// * 2 * PI
    const double delta_r = double(data.at("delta_r"));// * 2 * PI
    const double g_a = double(data.at("g_a"));// * 2 * PI
    const double delta_a = double(data.at("delta_a"));// * 2 * PI
    const double gamma_a = double(data.at("gamma_a"));
    const double gamma_q = double(data.at("gamma_q"));
    const double gamma_f = double(data.at("gamma_f"));
    const double gamma_r = double(data.at("gamma_r"));
    const double start = double(data.at("start"));//middle point
    const bool log_flag = bool(data.at("log"));//save log
    const bool test_flag = bool(data.at("log"));//for test
    std::vector<double> param = { gamma_a, gamma_q, gamma_f, gamma_r };

    size_t size_a = data.at("size_r");//r
    size_t size_b = data.at("size_q");//q
    size_t size_c = data.at("size_a");//a

    size_t N_size = size_a * size_b * size_c;//Size sistem

    std::vector<size_t> size_s = { size_a, size_b, size_c , N_size };

    //Preparing matrices for calculations
    Matrix a(size_a, size_a);
    a.Set_d();
    Matrix a_d = a.dagger();
    Matrix I_r(size_a, size_a);
    I_r.Set_identity();

    Matrix b(size_b, size_b);
    b.Set_d();
    Matrix b_d = b.dagger();
    Matrix I_q(size_b, size_b);
    I_q.Set_identity();

    Matrix c(size_c, size_c);
    c.Set_d();
    Matrix c_d = c.dagger();
    Matrix I_a(size_c, size_c);
    I_a.Set_identity();

    Matrix I__(N_size, N_size);
    I__.Set_identity();

    Matrix A_ = (a.kroneckerProduct(I_q)).kroneckerProduct(I_a);
    Matrix A_D = (a_d.kroneckerProduct(I_q)).kroneckerProduct(I_a);
    Matrix A_D_A = A_D.multiply(A_);

    Matrix B_ = (I_r.kroneckerProduct(b)).kroneckerProduct(I_a);
    Matrix B_D = (I_r.kroneckerProduct(b_d)).kroneckerProduct(I_a);
    Matrix B_D_B = B_D.multiply(B_);

    Matrix C_ = (I_r.kroneckerProduct(I_q)).kroneckerProduct(c);
    Matrix C_D = (I_r.kroneckerProduct(I_q)).kroneckerProduct(c_d);
    Matrix C_D_C = C_D.multiply(C_);

    //Preparation of the Hamiltonian
    Matrix q_temp_1 = (B_D_B.multiply(B_D_B.sum(I__.mult_c(-1)))).mult_c(alpha / 2);
    Matrix q_temp_2 = (B_D.sum(B_)).mult_c(omega / 2);
    Matrix H_q = ((B_D_B.mult_c(delta)).sum(q_temp_1)).sum(q_temp_2);
    Matrix H_r = A_D_A.mult_c(delta_r);
    Matrix H_a = C_D_C.mult_c(delta_a);
    Matrix H_q_r = ((B_.multiply(A_D)).sum(B_D.multiply(A_))).mult_c(g_r);
    Matrix H_q_a = ((B_.multiply(C_D)).sum(B_D.multiply(C_))).mult_c(g_a);
    Matrix H = H_q.sum(H_r.sum(H_a.sum(H_q_r.sum(H_q_a))));
    //H.Output_to_file("Sparse_test");
    Ellpack H_S(H);
    //H_S.Print_Ellpack();
    //H_S.Output_to_file("Sparse_test2");
    /*
    Ellpack H_S_M = H_S.mult_c(-1);
    std::cout << "\n\n";
    H_S_M.Print_Ellpack();
    std::cout << "\n\n";
    (H_S_M.sum(H_S)).Print_Ellpack();*/

    Matrix Ro(N_size, N_size);
    Ro.Set_random();

    std::cout << "START_S_F" << std::endl;
    double itime = omp_get_wtime();
    /*
    Matrix T = H.multiply(Ro);
    T.Output_to_file("T");
    Matrix T_S = H_S.multiply(Ro);
    T_S.Output_to_file("T_S");*/
    H_S.multiply(Ro);

    std::cout << "Time: " << omp_get_wtime() - itime << std::endl;

    std::cout << "START_F_S" << std::endl;
    itime = omp_get_wtime();
    /*
    Matrix T = Ro.multiply(H);
    T.Output_to_file("T");
    Matrix T_S = Ro.multiply(H_S);
    T_S.Output_to_file("T_S");*/
    Ro.multiply(H_S);

    std::cout << "Time: " << omp_get_wtime() - itime << std::endl;

    return 0;
}


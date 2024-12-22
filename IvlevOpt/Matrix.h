#pragma once
#include <iostream>
#include <stdlib.h>
#include <complex>
#include <fstream>
#include <vector>
#include <omp.h>

#define eps 0.0000000001

class Ellpack {
    friend class Matrix;
private:
    size_t rows;
    size_t cols;
    std::vector<std::complex<double>> value;
    std::vector<size_t> col;
public:
    Ellpack(size_t rows_, size_t cols_, bool flag);

    Ellpack(const Ellpack& other);

    Ellpack(const Matrix& other);

    //void resize();

    bool operator==(const Ellpack& other) noexcept;

    std::pair<size_t, size_t> Size(bool flag) const;

    void Print_Ellpack() const;

    void Output_to_file(std::string name) const;

    Ellpack mult_c(std::complex<double> c) const;

    Matrix multiply(const Matrix& other) const;

    //Ellpack sum(const Ellpack& other) const;

    void clear();
};

class Matrix {
    friend class Ellpack;
private:
    size_t rows;
    size_t cols;
    std::vector<std::complex<double>> matrix;
public:
    Matrix(size_t rows_ = 1, size_t cols_ = 1, bool flag = true) : rows(rows_), cols(cols_) {
        if (flag) {
            matrix.resize(rows * cols);
        }
        else {
            matrix.reserve(rows * cols);
        }
    }

    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
        matrix = other.matrix;
    }

    bool operator==(const Matrix& other) noexcept {
        return (matrix == other.matrix);
    }

    void setValue(size_t row, size_t col, std::complex<double> value) {
        matrix[row * cols + col] = value;
    }

    void setValue_pair(std::pair<size_t, size_t> pair_, std::complex<double> value) {
        matrix[pair_.first * cols + pair_.second] = value;
    }

    std::complex<double> getValue(size_t row, size_t col) const {
        return matrix[row * cols + col];
    }

    std::complex<double> getValue_pair(std::pair<size_t, size_t> pair_) const {
        return matrix[pair_.first * cols + pair_.second];
    }

    std::pair<size_t, size_t> Size(bool flag = false) const {
        if (flag) {
            std::cout << "Matrix size: " << rows << " x " << cols << '\n';
        }
        return std::make_pair(rows, cols);
    }

    size_t Non_zero_elements(bool flag = false) const {
        size_t count = 0;

        for (size_t i = 0; i < rows * cols; i++) {
            if (std::abs(matrix[i]) != 0) {
                count += 1;
            }
        }

        if (flag) {
            std::cout << "Non-zero elements: " << count << '\n';
        }
        return count;
    }

    void Output_to_file(std::string name = "out") const {
        std::ofstream out(name + ".csv");
        //out.imbue(std::locale(""));

        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                out << matrix[i*cols + j].real();
                if (matrix[i * cols + j].imag() >= 0)
                    out << '+' << matrix[i * cols + j].imag() << 'j';
                else
                    out << matrix[i * cols + j].imag() << 'j';
                if (j != cols - 1)
                    out << ';';
            }
            out << '\n';
        }

        out.close();
    };

    void printFullMatrix() const {
        auto iter = matrix.begin();

        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                std::cout << *iter << ';';
                iter++;
            }
            std::cout << '\n';
        }
    }

    Matrix kroneckerProduct(const Matrix& other) const {
        Matrix result(rows * other.rows, cols * other.cols, false);

        #pragma omp parallel for
        for (int i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                const std::complex<double> tmp = matrix[i*cols+j];
                for (size_t p = 0; p < other.rows; ++p) {
                    for (size_t q = 0; q < other.cols; ++q) {
                        result.setValue(i * other.rows + p, j*other.cols+q, tmp*other.matrix[p*other.cols + q]);
                    }
                }
            }
        }

        return result;
    }

    Matrix transpose() const {
        Matrix result(cols, rows);

        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.setValue(j, i, matrix[i * cols + j]);
            }
        }

        return result;
    }

    Matrix dagger() const {
        Matrix result(cols, rows);

        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.setValue(j, i, std::conj(matrix[i*cols + j]));
            }
        }

        return result;
    }

    Matrix mult_c(std::complex<double> c) const {
        Matrix result(cols, rows);

        #pragma omp parallel for
        for (int i = 0; i < cols * rows; ++i) {
            result.matrix[i] = matrix[i] * c;
        }

        return result;
    }
    
    Matrix multiply(const Matrix& other) const {
        Matrix result(rows, other.cols, false);

        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                //#pragma omp simd
                for (int k = 0; k < other.cols; k++)
                    result.matrix[i * other.cols + k] += matrix[i * cols + j] * other.matrix[j * other.cols + k];
            }
        }
        return result;
    }

    Matrix multiply(const Ellpack& other) const {
        Matrix result(rows, other.rows);

#pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                for (size_t k = 0; k < other.cols; k++)
                    result.matrix[i * other.rows + other.col[j * other.cols + k]] += other.value[j * other.cols + k] * matrix[i * cols + j];
            }
        }
        return result;
    }

    Matrix sum(const Matrix& other) const {
        Matrix result(rows, cols);

        #pragma omp parallel for
        for (int i = 0; i < cols * rows; i++) {
            result.matrix[i] = matrix[i] + other.matrix[i];
        }

        return result;
    }

    void Set_identity(std::complex<double> val = 1) {
        //matrix.clear();
        if (rows == cols) {
            for (size_t i = 0; i < rows; i++) {
                setValue(i, i, val);
            }
        }
    }

    void Set_d() {
        //matrix.clear();
        if (rows == cols) {
            for (size_t i = 1; i < rows; i++) {
                setValue(i - 1, i, std::sqrt(i));
            }
        }
    }

    void Set_random(int min = -10, int max = 10) {
        #pragma omp parallel for
        for (int i = 0; i < rows * cols; i++) {
            matrix[i] = (min + rand() % (max - min + 1), min + rand() % (max - min + 1));
        }
    }

    void clear() {
        matrix.clear();
        matrix.resize(rows * cols, 0);
    }
};

Ellpack::Ellpack(size_t rows_ = 1, size_t cols_ = 1, bool flag = true) : rows(rows_), cols(cols_) {
        if (flag) {
            value.resize(rows * cols);
            col.resize(rows * cols);
        }
        else {
            value.reserve(rows * cols);
            col.reserve(rows * cols);
        }
    }

Ellpack::Ellpack(const Ellpack& other) : rows(other.rows), cols(other.cols) {
        value = other.value;
        col = other.col;
    }

Ellpack::Ellpack(const Matrix& other) : rows(other.rows) {
        size_t max_elem_r = 0;
        for (size_t i = 0; i < other.rows; ++i) {
            size_t tmp = 0;
            for (size_t j = 0; j < other.cols; ++j) {
                if (std::norm(other.matrix[i * other.cols + j]) >= eps) {
                    tmp++;
                }
            }
            max_elem_r = std::max(max_elem_r, tmp);
        }
        cols = max_elem_r;
        value.resize(rows * cols);
        col.resize(rows * cols);
        size_t k = 0;
        for (size_t i = 0; i < other.rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                if (std::norm(other.matrix[i * other.cols + j]) >= eps) {
                    if (k == 0 || col[k - 1] < j) {
                        value[k] = other.matrix[i * other.cols + j];
                        col[k] = j;
                        k++;
                    } else {
                        while (k % cols != 0) {
                            value[k] = 0;
                            col[k] = 0;
                            k++;
                        }
                        value[k] = other.matrix[i * other.cols + j];
                        col[k] = j;
                        k++;
                    }
                }
                else {

                }
                
            }
        }
    }
    /*
    void Ellpack::resize() {
        for (size_t j = 0; j < cols; ++j) {
            bool flag = true;
            for (size_t i = 0; i < rows; ++i) {
                if (std::norm(value[i * cols + j]) > eps) {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                Ellpack tmp(rows, j);
                for (size_t p = 0; p < rows; ++i) {
                    for (size_t j = 0; j < cols; ++j) {
                    bool flag = true;
                    
                        if (std::norm(value[i * cols + j]) > eps) {
                            flag = false;
                            break;
                        }
                    }
                break;
            }
        }
    }*/

    bool Ellpack::operator==(const Ellpack& other) noexcept {
        return (cols == other.cols) && (rows == other.rows) && (value == other.value) && (col == other.col);
    }

    std::pair<size_t, size_t> Ellpack::Size(bool flag = false) const {
        if (flag) {
            std::cout << "Matrix size: " << rows << ", Max elem in row: " << cols << '\n';
        }
        return std::make_pair(rows, cols);
    }

    void Ellpack::Print_Ellpack() const {
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                std::cout << value[i * cols + j] << ' ' << col[i * cols + j] << ';';
            }
            std::cout << '\n';
        }
    }

    void Ellpack::Output_to_file(std::string name = "out") const {
        std::ofstream out(name + ".csv");
        //out.imbue(std::locale(""));

        for (size_t i = 0; i < rows; i++) {
            size_t tmp = 0;
            for (size_t j = 0; j < cols; j++) {
                for (size_t k = tmp; k < col[i * cols + j]; k++) {
                    out << "0+0j;";
                }
                if (std::norm(value[i * cols + j]) != 0) {
                    tmp = col[i * cols + j] + 1;
                    out << value[i * cols + j].real();
                    if (value[i * cols + j].imag() >= 0)
                        out << '+' << value[i * cols + j].imag() << 'j';
                    else
                        out << value[i * cols + j].imag() << 'j';
                    if (col[i * cols + j] != rows - 1) {
                        out << ';';
                    }
                }

            }
            if (tmp != rows) {
                for (size_t k = tmp; k < rows - 1; k++) {
                    out << "0+0j;";
                }
                out << "0+0j";
            }
            out << '\n';
        }

        out.close();
    };
    
    Ellpack Ellpack::mult_c(std::complex<double> c) const {
        Ellpack result(rows, cols);
        result.col = col;

        #pragma omp parallel for
        for (int i = 0; i < cols * rows; ++i) {
            result.value[i] = value[i] * c;
        }

        return result;
    }

    Matrix Ellpack::multiply(const Matrix& other) const {
        Matrix result(rows, other.cols);

        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                for (size_t k = 0; k < other.cols; k++)
                    result.matrix[i * other.cols + k] += value[i * cols + j] * other.matrix[col[i * cols + j] * other.cols + k];
            }
        }
        return result;
    }

    /*
    Ellpack Ellpack::sum(const Ellpack& other) const {
        Ellpack result(rows, cols+other.cols);

        for (size_t i = 0; i < rows; i++) {
            size_t j = 0, k = 0, p = 0;
            while (j < cols && k < other.cols) {
                if (col[i * cols + j] < other.col[i * other.cols + k]) {
                    result.col[i * (cols + other.cols) + p] = col[i * cols + j];
                    result.value[i * (cols + other.cols) + p] = value[i * cols + j];
                    j++;
                    p++;
                }
                else if (col[i * cols + j] == other.col[i * other.cols + k]) {
                    std::complex<double> tmp = value[i * cols + j] + other.value[i * other.cols + k];
                    if (std::norm(tmp) > eps) {
                        result.col[i * (cols + other.cols) + p] = col[i * cols + j];
                        result.value[i * (cols + other.cols) + p] = tmp;
                    }
                    j++;
                    k++;
                    p++;
                }
                else {
                    result.col[i * (cols + other.cols) + p] = col[i * cols + j];
                    result.value[i * (cols + other.cols) + p] = other.value[i * other.cols + k];
                    k++;
                    p++;
                }
            }
            while (j < cols) {
                result.col[i * (cols + other.cols) + p] = col[i * cols + j];
                result.value[i * (cols + other.cols) + p] = value[i * cols + j];
                j++;
                p++;
            }
            while (k < other.cols) {
                result.col[i * (cols + other.cols) + p] = col[i * cols + j];
                result.value[i * (cols + other.cols) + p] = other.value[i * other.cols + k];
                k++;
                p++;
            }
        }

        //resize();

        return result;
    }*/

    void Ellpack::clear() {
        value.clear();
        value.resize(rows * cols, 0);
        col.clear();
        col.resize(rows * cols, 0);
    }

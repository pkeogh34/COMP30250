//
// Created by Patrick Ross Keogh on 26/10/2023.
//
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cblas.h>

#define MAX_MATRIX_SIZE 4096
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
void blocked_ijk(double *A, double *B, double *C, int n, int block_size) {
#pragma omp parallel for collapse(3) default(none) shared(A, B, C, n, block_size)
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int k = 0; k < n; k += block_size) {
                int M = MIN(block_size, n - i);
                int N = MIN(block_size, n - j);
                int K = MIN(block_size, n - k);

                double *subA = A + i * n + k;
                double *subB = B + k * n + j;
                double *subC = C + i * n + j;

                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            M, N, K, 1.0, subA, n, subB, n, 1.0, subC, n);
            }
        }
    }
}



void non_blocked(double *A, double *B, double *C, int n) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, A, n, B, n, 0.0, C, n);
}

void initialize_matrix(double *matrix, int n) {
    for (int i = 0; i < n * n; ++i) {
        matrix[i] = rand() % 10;
    }
}

void run_experiment(int n, int block_size) {
    double *A = malloc(n * n * sizeof(double));
    double *B = malloc(n * n * sizeof(double));
    double *C = malloc(n * n * sizeof(double));

    if (!A || !B || !C) {
        printf("Memory allocation failed for size %d\n", n);
        return;
    }

    initialize_matrix(A, n);
    initialize_matrix(B, n);

    struct timespec start, end;
    double elapsed_blocked, elapsed_non_blocked, speedup;

    clock_gettime(CLOCK_MONOTONIC, &start);
    blocked_ijk(A, B, C, n, block_size);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_blocked = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    // Non-blocked ijk algorithm timing
    clock_gettime(CLOCK_MONOTONIC, &start);
    non_blocked(A, B, C, n);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_non_blocked = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    speedup = elapsed_non_blocked / elapsed_blocked;

    printf("%d          |%d         | %lfs    | %lfs        | %lf\n",
           block_size,n, elapsed_blocked, elapsed_non_blocked, speedup);

    free(A);
    free(B);
    free(C);
}

int main() {
    srand(1);
    for (int i = 1; i < 4; ++i) {
        int matrix_sizes[] = {1024, 2048, MAX_MATRIX_SIZE};
        int block_size[] = {16, 32, 64, 128, 256};

        printf("Block size | Matrix size | Blocked Time | Non-Blocked Time | Speedup\n");
        for (int j = 0; j < sizeof(block_size) / sizeof(block_size[0]); ++j) {
            for (int k = 0; k < sizeof(matrix_sizes) / sizeof(matrix_sizes[0]); ++k)
                run_experiment(matrix_sizes[k], block_size[j]);
            printf("---------------------------------------------------------------------\n");
        }
        printf("\n");
    }

    return 0;
}





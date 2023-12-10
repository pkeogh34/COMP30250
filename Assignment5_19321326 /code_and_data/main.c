#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

float* initialise_matrix(int n, float value) {
    float* matrix = malloc(n * n * sizeof(float));
    if (matrix == NULL) {
        perror("Failed to allocate matrix");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int i = 0; i < n * n; i++)
        matrix[i] = value;

    return matrix;
}

float serial(int n) {
    double start_time = MPI_Wtime();

    float *A = initialise_matrix(n, 1.0f);
    float *B = initialise_matrix(n, 2.0f);
    float *C = initialise_matrix(n, 0.0f);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }

    return MPI_Wtime() - start_time;
}

double parallel(int n, int* p, int* process_id) {
    int submatrix_size;
    float *submatrix_a, *submatrix_b, *submatrix_c, *gathered_rows_a, *gathered_cols_b, *result_matrix = NULL;
    MPI_Comm row_comm, col_comm;

    MPI_Comm_size(MPI_COMM_WORLD, p);
    MPI_Comm_rank(MPI_COMM_WORLD, process_id);

    if (n % (int)sqrt(*p) != 0) {
        if (*process_id == 0) {
            printf("Matrix size (%d) must be divisible by the square root of p (%d)\n", n, (int)sqrt(*p));
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    submatrix_size = n / (int)sqrt(*p);
    int row_id = *process_id / (int)sqrt(*p);
    int col_id = *process_id % (int)sqrt(*p);

    submatrix_a = initialise_matrix(submatrix_size, 1.0f);
    submatrix_b = initialise_matrix(submatrix_size, 2.0f);
    submatrix_c = initialise_matrix(submatrix_size, 0.0f);

    gathered_rows_a = malloc(n * submatrix_size * sizeof(float));
    if (!gathered_rows_a) {
        perror("Failed to allocate gathered_rows_a");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    gathered_cols_b = malloc(n * submatrix_size * sizeof(float));
    if (!gathered_cols_b) {
        perror("Failed to allocate gathered_cols_b");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (*process_id == 0) {
        result_matrix = malloc(n * n * sizeof(float));
        if (!result_matrix) {
            perror("Failed to allocate result_matrix");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Comm_split(MPI_COMM_WORLD, row_id, *process_id, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col_id, *process_id, &col_comm);

    double local_start_time = MPI_Wtime();

    MPI_Allgather(submatrix_a, submatrix_size * submatrix_size, MPI_FLOAT, gathered_rows_a, submatrix_size * submatrix_size, MPI_FLOAT, row_comm);
    MPI_Allgather(submatrix_b, submatrix_size * submatrix_size, MPI_FLOAT, gathered_cols_b, submatrix_size * submatrix_size, MPI_FLOAT, col_comm);

    for (int i = 0; i < submatrix_size; i++) {
        for (int j = 0; j < submatrix_size; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += gathered_rows_a[i * n + k] * gathered_cols_b[k * n + j];
            }
            submatrix_c[i * submatrix_size + j] = sum;
        }
    }

    if (*process_id == 0) {
        MPI_Gather(submatrix_c, submatrix_size * submatrix_size, MPI_FLOAT, result_matrix, submatrix_size * submatrix_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gather(submatrix_c, submatrix_size * submatrix_size, MPI_FLOAT, NULL, 0, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    double local_end_time = MPI_Wtime();
    double local_elapsed = local_end_time - local_start_time;
    double global_elapsed;

    MPI_Reduce(&local_elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    free(submatrix_a);
    free(submatrix_b);
    free(submatrix_c);
    free(gathered_rows_a);
    free(gathered_cols_b);
    if (*process_id == 0) {
        free(result_matrix);
    }

    return *process_id == 0 ? global_elapsed : 0;
}

int experiment_1_matrix_size_impact(int n) {
    int p, process_id;
    double parallel_time = parallel(n, &p, &process_id);

    if (process_id == 0) {
        FILE *file = fopen("experiment1.csv", "a");

        if (!file) {
            perror("Error opening file");
            return -1;
        }

        fseek(file, 0, SEEK_END);
        if (ftell(file) == 0) {
            fprintf(file, "Matrix Size,Parallel Time,p\n");
        }

        fprintf(file, "%d,%f,%d\n", n, parallel_time, p);
        fclose(file);
    }
    return process_id;
}

void experiment_2_speedup_comparison(int n, int process_id) {
    if (process_id == 0) {
        FILE *file1 = fopen("experiment2.csv", "a");
        FILE *file2 = fopen("experiment1.csv", "r");

        if (!file1 || !file2) {
            perror("Error opening file");
            if (file1) fclose(file1);
            if (file2) fclose(file2);
            return;
        }

        char line[1024];
        int matrix_size;
        double parallel_time;

        while (fgets(line, sizeof(line), file2)) {
            sscanf(line, "%d,%lf", &matrix_size, &parallel_time);
            if (matrix_size == n) {
                break;
            }
        }

        fseek(file1, 0, SEEK_END);
        if (ftell(file1) == 0) {
            fprintf(file1, "Matrix Size,Serial Time,Parallel Time,Speedup\n");
        }

        double serial_time = serial(n);

        fprintf(file1, "%d,%lf,%lf,%f\n", n, serial_time, parallel_time, serial_time / parallel_time);

        fclose(file1);
        fclose(file2);
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    for(int n = 100; n <= 2800; n += 150) {
        int process_id = experiment_1_matrix_size_impact(n);
        experiment_2_speedup_comparison(n, process_id);
    }

    MPI_Finalize();

    return 0;
}
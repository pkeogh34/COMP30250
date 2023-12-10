#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

typedef struct {
    int matrix_size;
    int chunk_size;
    double speedup_static;
    double speedup_dynamic;
    double speedup_guided;
} ExperimentData;

void initialise_matrices(double **A, double **B, double **C, int n) {
    *A = (double *)malloc(n * n * sizeof(double));
    *B = (double *)malloc(n * n * sizeof(double));
    *C = (double *)malloc(n * n * sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            (*A)[i * n + j] = 1.0;
            (*B)[i * n + j] = 2.0;
            (*C)[i * n + j] = 0.0;
        }
    }
}

double parallel_infinity_norm(double *C, int n) {
    double max_norm = 0.0;
#pragma omp parallel for reduction(max: max_norm) default(none) shared(C, n)
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++) {
            row_sum += fabs(C[i * n + j]);
        }
        max_norm = fmax(max_norm, row_sum);
    }
    return max_norm;
}


double parallel(double *A, double *B, double *C, int n) {
    double start_time = omp_get_wtime();

#pragma omp parallel for default(none) shared(A, B, C, n)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }

    parallel_infinity_norm(C, n);

    double end_time = omp_get_wtime();
    return end_time - start_time;
}


double serial_infinity_norm(double *C, int n) {
    double max_norm = 0.0;
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++) {
            row_sum += fabs(C[i * n + j]);
        }
        if (row_sum > max_norm) {
            max_norm = row_sum;
        }
    }
    return max_norm;
}

double serial(double *A, double *B, double *C, int n) {
    double start_time = omp_get_wtime();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }

    serial_infinity_norm(C, n);

    double end_time = omp_get_wtime();
    return end_time - start_time;
}

double calculate_speedup(double *A, double *B, double *C, int n){
    double parallel_time = parallel(A, B, C, n);
    double serial_time = serial(A, B, C, n);
    return serial_time / parallel_time;
}

void experiment_1_matrix_size_impact(int n) {
    FILE *file = fopen("experiment1.csv", "a");
    fseek(file, 0, SEEK_END);
    if (ftell(file) == 0) {
        fprintf(file, "Matrix Size,Parallel Time\n");
    }

    double *A, *B, *C;
    initialise_matrices(&A, &B, &C, n);

    double parallel_time = parallel(A, B, C, n);

    free(A); free(B); free(C);

    fprintf(file, "%d,%lf\n", n, parallel_time);
    fclose(file);
}

void experiment_2_speedup_comparison(int n) {
    FILE *file = fopen("experiment2.csv", "a");
    fseek(file, 0, SEEK_END);
    if (ftell(file) == 0) {
        fprintf(file, "Matrix Size,Speedup\n");
    }

    double *A, *B, *C;
    initialise_matrices(&A, &B, &C, n);

    double speedup = calculate_speedup(A, B, C, n);

    free(A); free(B); free(C);

    fprintf(file, "%d,%lf\n", n, speedup);
    fclose(file);
}

void experiment_3_calculate_values(int n, const int chunk_sizes[], int num_chunks) {
    FILE *file = fopen("experiment3.csv", "a");
    fseek(file, 0, SEEK_END);
    if (ftell(file) == 0) {
        fprintf(file, "Matrix Size,Chunk Size,Speedup Static,Speedup Dynamic,Speedup Guided\n");
    }

    double *A, *B, *C;
    initialise_matrices(&A, &B, &C, n);

    for (int k = 0; k < num_chunks; k++) {
        omp_set_schedule(omp_sched_static, chunk_sizes[k]);
        double speedup_static = calculate_speedup(A, B, C, n);

        omp_set_schedule(omp_sched_dynamic, chunk_sizes[k]);
        double speedup_dynamic = calculate_speedup(A, B, C, n);

        omp_set_schedule(omp_sched_guided, chunk_sizes[k]);
        double speedup_guided = calculate_speedup(A, B, C, n);

        fprintf(file, "%d,%d,%lf,%lf,%lf\n", n, chunk_sizes[k], speedup_static, speedup_dynamic, speedup_guided);
    }

    fclose(file);
    free(A); free(B); free(C);
}

void experiment_3_schedule_comparison(int n) {
    FILE *inputFile = fopen("experiment3.csv", "r");
    FILE *outputFile = fopen("experiment4.csv", "a");

    if (inputFile == NULL || outputFile == NULL) {
        perror("Error opening file");
        if (inputFile) fclose(inputFile);
        if (outputFile) fclose(outputFile);
        return;
    }

    fseek(outputFile, 0, SEEK_END);
    if (ftell(outputFile) == 0) {
        fprintf(outputFile, "Matrix Size,Best Chunk Static,Speedup Static,Best Chunk Dynamic,Speedup Dynamic,Best Chunk Guided,Speedup Guided,Best Schedule\n");
    }

    char line[128];
    ExperimentData data;
    double best_speedup_static = -1.0, best_speedup_dynamic = -1.0, best_speedup_guided = -1.0;
    int best_chunk_static = -1, best_chunk_dynamic = -1, best_chunk_guided = -1;
    char *best_schedule = "";

    while (fgets(line, sizeof(line), inputFile)) {
        sscanf(line, "%d,%d,%lf,%lf,%lf", &data.matrix_size, &data.chunk_size,
               &data.speedup_static, &data.speedup_dynamic, &data.speedup_guided);

        if (data.matrix_size == n) {
            if (data.speedup_static > best_speedup_static) {
                best_speedup_static = data.speedup_static;
                best_chunk_static = data.chunk_size;
            }
            if (data.speedup_dynamic > best_speedup_dynamic) {
                best_speedup_dynamic = data.speedup_dynamic;
                best_chunk_dynamic = data.chunk_size;
            }
            if (data.speedup_guided > best_speedup_guided) {
                best_speedup_guided = data.speedup_guided;
                best_chunk_guided = data.chunk_size;
            }
        }
    }

    fclose(inputFile);

    // Determine the best schedule
    double max_speedup = best_speedup_static;
    best_schedule = "Static";

    if (best_speedup_dynamic > max_speedup) {
        max_speedup = best_speedup_dynamic;
        best_schedule = "Dynamic";
    }
    if (best_speedup_guided > max_speedup) {
        best_schedule = "Guided";
    }

    printf("%lf",best_speedup_guided);
    fprintf(outputFile, "%d,%d,%lf,%d,%lf,%d,%lf,%s\n", n,
            best_chunk_static, best_speedup_static,
            best_chunk_dynamic, best_speedup_dynamic,
            best_chunk_guided, best_speedup_guided, best_schedule);

    fclose(outputFile);
}


int main(int argc, char* argv[]) {
    int choice = 3;
    int p = omp_get_num_procs();
    omp_set_num_threads(p);

    int chunk_sizes[] = {10, 20, 50, 100};
    int num_chunk_sizes = sizeof(chunk_sizes) / sizeof(chunk_sizes[0]);

    for (int i = 3; i <= 11; i++) {
        int n = pow(2, i);
        if (choice == 1){
            experiment_1_matrix_size_impact(n);
        }else if(choice == 2){
            experiment_2_speedup_comparison(n);
        }else if(choice == 3){
            experiment_3_calculate_values(n, chunk_sizes, num_chunk_sizes);
            experiment_3_schedule_comparison(n);
        }
    }
    return 0;
}



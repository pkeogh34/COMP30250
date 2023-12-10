// Include necessary headers
#include <cblas.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define NUM_PROCESSORS 8

// Global variables for measuring time
struct timeval start, end;

// Structure definition for matrix data and thread synchronization
typedef struct {
    double *A, *B, *C, *norm;
    int n, start, end, range;
    pthread_mutex_t *mutex;
} MatrixData;

// Thread function for matrix multiplication
void* multiplyMatrices(void* arg) {
    MatrixData* data = (MatrixData*)arg;

    // Perform matrix multiplication using CBLAS library function
    cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            data->n, data->range, data->n,
            1.0, data->A, data->n, data->B, data->n,
            0.0, data->C, data->n
    );
    pthread_exit(0);
}

// Thread function for calculating the infinity norm of a matrix
void* calculateNorm(void* arg) {
    MatrixData* data = (MatrixData*)arg;
    double currentSum = 0.0, localMax = 0.0;
    double element;

    // Calculate the infinity norm for the assigned columns
    for (int col = data->start; col < data->end; col++) {
        currentSum = 0.0;
        for (int row = 0; row < data->n; row++) {
            element = data->C[row * data->n + col];
            currentSum += fabs(element);
        }
        if (localMax < currentSum) localMax = currentSum;
    }

    // Synchronize updating the global maximum norm with a mutex
    pthread_mutex_lock(data->mutex);
    *(data->norm) = fmax(*(data->norm), localMax);
    pthread_mutex_unlock(data->mutex);
}

// Function to perform parallel computation of matrix multiplication and norm calculation
double parallel(int n, int threadCount, double *A, double *B, double *C) {
    int block_size;
    double max_norm;
    pthread_t *multi_threads, *norm_threads;
    MatrixData *multi_data, *norm_data;
    pthread_mutex_t normMutex;

    block_size = n / threadCount;

    multi_threads = (pthread_t *)malloc(threadCount * sizeof(pthread_t));
    multi_data = (MatrixData *)malloc(threadCount * sizeof(MatrixData));
    norm_threads = (pthread_t *)malloc(threadCount * sizeof(pthread_t));
    norm_data = (MatrixData *)malloc(threadCount * sizeof(MatrixData));
    pthread_mutex_init(&normMutex, NULL);

    gettimeofday(&start, NULL);
    // Create and start threads for matrix multiplication
    for (int index = 0; index < threadCount; index++) {
        multi_data[index] = (MatrixData){
                .A = A, .B = B + index * block_size, .C = C + index * block_size,
                .n = n, .range = (index == threadCount - 1) ? (n - index * block_size) : block_size
        };
        pthread_create(&multi_threads[index], NULL, multiplyMatrices, &multi_data[index]);
    }

    // Wait for all matrix multiplication threads to complete
    for (int index = 0; index < threadCount; index++) {
        pthread_join(multi_threads[index], NULL);
    }

    max_norm = 0.0;

    // Create and start threads for norm calculation
    for (int index = 0; index < threadCount; index++) {
        norm_data[index] = (MatrixData){
                .C = C, .norm = &max_norm, .n = n,
                .start = index * block_size, .end = (index == threadCount - 1) ? n : (index + 1) * block_size,
                .mutex = &normMutex
        };
        pthread_create(&norm_threads[index], NULL, calculateNorm, &norm_data[index]);
    }

    gettimeofday(&end, NULL);

    // Wait for all norm calculation threads to complete
    for (int index = 0; index < threadCount; index++) {
        pthread_join(norm_threads[index], NULL);
    }

    gettimeofday(&end, NULL);

    // Cleanup: free allocated memory and destroy mutex
    free(multi_threads);
    free(multi_data);
    free(norm_threads);
    free(norm_data);
    pthread_mutex_destroy(&normMutex);

    // Calculate and return the elapsed time in seconds
    return (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1e6;
}

// Function to perform serial matrix multiplication and norm calculation
double serial(int n, double *A, double *B, double *C) {
    gettimeofday(&start, NULL);

    // Matrix multiplication
    cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            n, n, n, 1.0, A, n, B, n, 0.0, C, n);

    // Calculate maximum absolute row sum norm (Infinity Norm)
    double maxRowSum = 0.0;
    for (int i = 0; i < n; ++i) {
        double rowSum = 0.0;
        for (int j = 0; j < n; ++j) {
            rowSum += fabs(C[i * n + j]);
        }
        if (rowSum > maxRowSum) {
            maxRowSum = rowSum;
        }
    }

    gettimeofday(&end, NULL);

    return (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) * 1.e-6;
}


// Function to initialise matrices A, B, and C and allocate memory
void initialiseMatrices(double **A, double **B, double **C, int n) {
    int size = n * n;

    // Allocate memory for matrices
    *A = malloc(size * sizeof(double));
    *B = malloc(size * sizeof(double));
    *C = malloc(size * sizeof(double));

    // Initialise matrices with default values (A with 1s, B with 2s, and C with 0s)
    for (int i = 0; i < size; i++) {
        (*A)[i] = 1.0;
        (*B)[i] = 2.0;
        (*C)[i] = 0.0;
    }
}

// Function to manage the computation and logging the results based on the choice
void performCalculations(int choice) {
    char filename[20];
    FILE *fp;
    double *A, *B, *C;

    sprintf(filename, "q%d.csv", choice);
    fp = fopen(filename, "w");

    // Loop through different matrix sizes
    for (int n = 100; n <= 2000; n += 100) {
        initialiseMatrices(&A, &B, &C, n);
        // Compute parallel execution time
        double parallelTime = parallel(n, NUM_PROCESSORS, A, B, C);
        if (choice == 1) {
            fprintf(fp, "%d, %f\n", n, parallelTime);
        } else {
            fprintf(fp, "%d, %f\n", n, serial(n, A, B, C) / parallelTime);
        }

        // Free allocated memory for matrices
        free(A);
        free(B);
        free(C);
    }

    fclose(fp);
}

int main(int argc, char* argv[]) {
    if(argc != 2){
         printf("You must enter a question choice: 1 or 2");
         return 1;
     }
     int choice = atoi(argv[1]);

    performCalculations(choice);

    return 0;
}
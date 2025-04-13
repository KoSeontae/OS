#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

typedef struct {
    int start;      // first row (inclusive)
    int end;        // last row (exclusive)
    int common;     // common dimension (columns of Q, which equals columns of K)
    int rK;         // number of rows in K (defines M1’s column count)
    int **Q;
    int **K;
    int **M1;       // intermediate result matrix (dimensions: rQ x rK)
} ThreadData;

int rQ, cQ, rK, cK, rV, cV;
int **Q, **K, **V;
int **M1, **result;

void *compute_M1(void *arg) {
    ThreadData *data = (ThreadData*) arg;
    for (int i = data->start; i < data->end; i++){
        for (int j = 0; j < data->rK; j++){
            int sum = 0;
            for (int k = 0; k < data->common; k++){
                sum += data->Q[i][k] * data->K[j][k];
            }
            data->M1[i][j] = sum;
        }
    }
    pthread_exit(NULL);
}

int main(int argc, char *argv[]){
    if(argc < 2){
        fprintf(stderr, "Usage: %s total_thread_num\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    int total_threads = atoi(argv[1]);
    
    // Read Q dimensions and matrix.
    if (scanf("%d %d", &rQ, &cQ) != 2) {
        fprintf(stderr, "Failed to read dimensions for Q.\n");
        exit(EXIT_FAILURE);
    }
    Q = malloc(rQ * sizeof(int *));
    for (int i = 0; i < rQ; i++){
        Q[i] = malloc(cQ * sizeof(int));
        for (int j = 0; j < cQ; j++){
            if (scanf("%d", &Q[i][j]) != 1) {
                fprintf(stderr, "Failed to read Q[%d][%d].\n", i, j);
                exit(EXIT_FAILURE);
            }
        }
    }
    
    // Read K dimensions and matrix.
    if (scanf("%d %d", &rK, &cK) != 2) {
        fprintf(stderr, "Failed to read dimensions for K.\n");
        exit(EXIT_FAILURE);
    }
    if(cQ != cK){
        fprintf(stderr, "Dimension mismatch: Q columns (%d) must equal K columns (%d).\n", cQ, cK);
        exit(EXIT_FAILURE);
    }
    K = malloc(rK * sizeof(int *));
    for (int i = 0; i < rK; i++){
        K[i] = malloc(cK * sizeof(int));
        for (int j = 0; j < cK; j++){
            if (scanf("%d", &K[i][j]) != 1) {
                fprintf(stderr, "Failed to read K[%d][%d].\n", i, j);
                exit(EXIT_FAILURE);
            }
        }
    }
    
    // Read V dimensions and matrix.
    if (scanf("%d %d", &rV, &cV) != 2) {
        fprintf(stderr, "Failed to read dimensions for V.\n");
        exit(EXIT_FAILURE);
    }
    if(rV != rK){
        fprintf(stderr, "Dimension mismatch: V rows (%d) must equal K rows (%d).\n", rV, rK);
        exit(EXIT_FAILURE);
    }
    V = malloc(rV * sizeof(int *));
    for (int i = 0; i < rV; i++){
        V[i] = malloc(cV * sizeof(int));
        for (int j = 0; j < cV; j++){
            if (scanf("%d", &V[i][j]) != 1) {
                fprintf(stderr, "Failed to read V[%d][%d].\n", i, j);
                exit(EXIT_FAILURE);
            }
        }
    }
    
    // Allocate intermediate result matrix M1 and final result.
    M1 = malloc(rQ * sizeof(int *));
    for (int i = 0; i < rQ; i++){
        M1[i] = malloc(rK * sizeof(int));
    }
    result = malloc(rQ * sizeof(int *));
    for (int i = 0; i < rQ; i++){
        result[i] = malloc(cV * sizeof(int));
    }
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    // Create threads for computing M1 = Q * Kᵀ.
    pthread_t threads[total_threads];
    ThreadData thread_data[total_threads];
    int rows_per_thread = rQ / total_threads;
    int extra = rQ % total_threads;
    int current_row = 0;
    for(int i = 0; i < total_threads; i++){
        int start_row = current_row;
        int num_rows = rows_per_thread + (i < extra ? 1 : 0);
        int end_row = start_row + num_rows;
        thread_data[i].start = start_row;
        thread_data[i].end = end_row;
        thread_data[i].common = cQ;
        thread_data[i].rK = rK;
        thread_data[i].Q = Q;
        thread_data[i].K = K;
        thread_data[i].M1 = M1;
        current_row = end_row;
        if(pthread_create(&threads[i], NULL, compute_M1, (void*)&thread_data[i]) != 0){
            perror("Error creating thread");
            exit(EXIT_FAILURE);
        }
    }
    
    for(int i = 0; i < total_threads; i++){
        pthread_join(threads[i], NULL);
    }
    
    // Compute final result = M1 * V.
    for (int i = 0; i < rQ; i++){
        for (int j = 0; j < cV; j++){
            int sum = 0;
            for (int k = 0; k < rK; k++){
                sum += M1[i][k] * V[k][j];
            }
            result[i][j] = sum;
        }
    }
    
    gettimeofday(&end, NULL);
    long elapsed_ms = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000;
    
    // Output latency and result.
    printf("%ld\n", elapsed_ms);
    for (int i = 0; i < rQ; i++){
        for (int j = 0; j < cV; j++){
            printf("%d ", result[i][j]);
        }
        printf("\n");
    }
    
    // Free memory.
    for (int i = 0; i < rQ; i++){
        free(Q[i]);
        free(M1[i]);
        free(result[i]);
    }
    free(Q); free(M1); free(result);
    for (int i = 0; i < rK; i++){
        free(K[i]);
    }
    free(K);
    for (int i = 0; i < rV; i++){
        free(V[i]);
    }
    free(V);
    
    return 0;
}

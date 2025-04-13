#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/wait.h>

int rQ, cQ, rK, cK, rV, cV;
int **Q, **K, **V;
int **M1, **result;

int main(int argc, char *argv[]){
    if(argc < 2){
        fprintf(stderr, "Usage: %s total_process_num\n", argv[0]);
        exit(1);
    }
    int total_processes = atoi(argv[1]);

    // Read matrix Q.
    scanf("%d %d", &rQ, &cQ);
    Q = malloc(rQ * sizeof(int *));
    for (int i = 0; i < rQ; i++){
        Q[i] = malloc(cQ * sizeof(int));
        for (int j = 0; j < cQ; j++){
            scanf("%d", &Q[i][j]);
        }
    }

    // Read matrix K.
    scanf("%d %d", &rK, &cK);
    if(cQ != cK){
        fprintf(stderr, "Dimension mismatch: Q columns (%d) must equal K columns (%d).\n", cQ, cK);
        exit(1);
    }
    K = malloc(rK * sizeof(int *));
    for (int i = 0; i < rK; i++){
        K[i] = malloc(cK * sizeof(int));
        for (int j = 0; j < cK; j++){
            scanf("%d", &K[i][j]);
        }
    }

    // Read matrix V.
    scanf("%d %d", &rV, &cV);
    if(rV != rK){
        fprintf(stderr, "Dimension mismatch: V rows (%d) must equal K rows (%d).\n", rV, rK);
        exit(1);
    }
    V = malloc(rV * sizeof(int *));
    for (int i = 0; i < rV; i++){
        V[i] = malloc(cV * sizeof(int));
        for (int j = 0; j < cV; j++){
            scanf("%d", &V[i][j]);
        }
    }

    // Allocate the intermediate result M1 (rQ x rK) and final result (rQ x cV).
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

    // Determine the rows per process.
    int rows_per_proc = rQ / total_processes;
    int extra = rQ % total_processes;
    
    // Create an array of pipes for each process.
    int (*pipes)[2] = malloc(total_processes * sizeof(int[2]));
    pid_t *pids = malloc(total_processes * sizeof(pid_t));
    int current_row = 0;

    // Fork processes and have each compute its assigned rows of M1 = Q * Kᵀ.
    for (int i = 0; i < total_processes; i++){
        int start_row = current_row;
        int num_rows = rows_per_proc + (i < extra ? 1 : 0);
        int end_row = start_row + num_rows;
        current_row = end_row;

        if(pipe(pipes[i]) == -1){
            perror("pipe");
            exit(1);
        }

        pids[i] = fork();
        if(pids[i] < 0){
            perror("fork");
            exit(1);
        }
        if(pids[i] == 0){
            // Child process: compute M1 for rows [start_row, end_row)
            for (int i_row = start_row; i_row < end_row; i_row++){
                for (int j = 0; j < rK; j++){
                    int sum = 0;
                    // Note: Q is rQ x cQ and K is rK x cK, and we need Kᵀ so we use K[j][k].
                    for (int k = 0; k < cQ; k++){
                        sum += Q[i_row][k] * K[j][k];
                    }
                    M1[i_row][j] = sum;
                }
            }
            // Write the computed rows of M1 into the pipe.
            for (int i_row = start_row; i_row < end_row; i_row++){
                if(write(pipes[i][1], M1[i_row], rK * sizeof(int)) != rK * sizeof(int)){
                    perror("write");
                    exit(1);
                }
            }
            close(pipes[i][1]);
            exit(0);
        } else {
            // Parent process: close the write end.
            close(pipes[i][1]);
        }
    }

    // In the parent, collect computed rows from each child.
    current_row = 0;
    for (int i = 0; i < total_processes; i++){
        int rows_for_proc = rows_per_proc + (i < extra ? 1 : 0);
        // Wait for the child process to finish.
        wait(NULL);
        for (int r = 0; r < rows_for_proc; r++){
            if(read(pipes[i][0], M1[current_row], rK * sizeof(int)) != rK * sizeof(int)){
                fprintf(stderr, "Error reading from pipe for row %d\n", current_row);
                exit(1);
            }
            current_row++;
        }
        close(pipes[i][0]);
    }
    
    // Compute the final result: result = M1 * V.
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
    
    // Output the latency and the final result matrix.
    printf("%ld\n", elapsed_ms);
    for (int i = 0; i < rQ; i++){
        for (int j = 0; j < cV; j++){
            printf("%d ", result[i][j]);
        }
        printf("\n");
    }

    // Free all allocated memory.
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
    free(pipes);
    free(pids);
    
    return 0;
}

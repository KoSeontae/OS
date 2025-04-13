/*
 * multiHeadAttention.c
 *
 * This program reads the total number of attention heads.
 * For each head, it:
 *   - Reads the head-specific input: matrices Q, K, and V.
 *   - Prepares a text buffer formatted as expected by the attention_mp executable.
 *   - Sets up pipes, forks a child process, and uses execvp to run "./attention_mp".
 *   - Reads and aggregates the output matrix from each child.
 *
 * Finally, it prints the aggregated result.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <string.h>

#define BUFSIZE 8192  // Buffer size for input/output

int main(){
    int total_heads;
    if (scanf("%d", &total_heads) != 1) {
        fprintf(stderr, "Failed to read the total number of heads.\n");
        exit(EXIT_FAILURE);
    }
    
    int final_rows = 0, final_cols = 0;
    int **finalResult = NULL;  // Will be allocated after reading the first head

    // Process each head
    for (int h = 0; h < total_heads; h++){
        int pipe_in[2], pipe_out[2];
        if(pipe(pipe_in) < 0 || pipe(pipe_out) < 0){
            perror("pipe");
            exit(EXIT_FAILURE);
        }
        
        char headInput[BUFSIZE];
        memset(headInput, 0, sizeof(headInput));
        char temp[256];  // Temporary buffer
        
        // --- Read matrix Q ---
        int rQ, cQ;
        if (scanf("%d %d", &rQ, &cQ) != 2) {
            fprintf(stderr, "Error reading Q dimensions for head %d\n", h);
            exit(EXIT_FAILURE);
        }
        sprintf(temp, "%d %d\n", rQ, cQ);
        strcat(headInput, temp);
        
        if(h == 0){
            final_rows = rQ;
        }
        
        for (int i = 0; i < rQ; i++){
            for (int j = 0; j < cQ; j++){
                int num;
                if (scanf("%d", &num) != 1) {
                    fprintf(stderr, "Error reading Q matrix for head %d, element [%d][%d]\n", h, i, j);
                    exit(EXIT_FAILURE);
                }
                sprintf(temp, "%d ", num);
                strcat(headInput, temp);
            }
            strcat(headInput, "\n");
        }
        
        // --- Read matrix K ---
        int rK, cK;
        if (scanf("%d %d", &rK, &cK) != 2) {
            fprintf(stderr, "Error reading K dimensions for head %d\n", h);
            exit(EXIT_FAILURE);
        }
        sprintf(temp, "%d %d\n", rK, cK);
        strcat(headInput, temp);
        if(cQ != cK){
            fprintf(stderr, "Dimension mismatch for head %d: Q columns (%d) != K columns (%d)\n", h, cQ, cK);
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < rK; i++){
            for (int j = 0; j < cK; j++){
                int num;
                if (scanf("%d", &num) != 1) {
                    fprintf(stderr, "Error reading K matrix for head %d, element [%d][%d]\n", h, i, j);
                    exit(EXIT_FAILURE);
                }
                sprintf(temp, "%d ", num);
                strcat(headInput, temp);
            }
            strcat(headInput, "\n");
        }
        
        // --- Read matrix V ---
        int rV, cV;
        if (scanf("%d %d", &rV, &cV) != 2) {
            fprintf(stderr, "Error reading V dimensions for head %d\n", h);
            exit(EXIT_FAILURE);
        }
        sprintf(temp, "%d %d\n", rV, cV);
        strcat(headInput, temp);
        if(h == 0){
            final_cols = cV;
            finalResult = malloc(final_rows * sizeof(int *));
            for (int i = 0; i < final_rows; i++){
                finalResult[i] = calloc(final_cols, sizeof(int));
            }
        }
        if(rV != rK){
            fprintf(stderr, "Dimension mismatch for head %d: V rows (%d) != K rows (%d)\n", h, rV, rK);
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < rV; i++){
            for (int j = 0; j < cV; j++){
                int num;
                if (scanf("%d", &num) != 1) {
                    fprintf(stderr, "Error reading V matrix for head %d, element [%d][%d]\n", h, i, j);
                    exit(EXIT_FAILURE);
                }
                sprintf(temp, "%d ", num);
                strcat(headInput, temp);
            }
            strcat(headInput, "\n");
        }
        
        // Fork and execute attention_mp for this head.
        pid_t pid = fork();
        if(pid < 0){
            perror("fork");
            exit(EXIT_FAILURE);
        }
        if(pid == 0){
            // Child process: redirect stdin and stdout.
            dup2(pipe_in[0], STDIN_FILENO);
            dup2(pipe_out[1], STDOUT_FILENO);
            close(pipe_in[0]); close(pipe_in[1]);
            close(pipe_out[0]); close(pipe_out[1]);
            char *args[] = {"./attention_mp", "2", NULL};  // Adjust process count if desired.
            execvp(args[0], args);
            perror("execvp failed");
            exit(EXIT_FAILURE);
        } else {
            // Parent process.
            close(pipe_in[0]);
            if(write(pipe_in[1], headInput, strlen(headInput)) != (ssize_t)strlen(headInput)){
                perror("write to pipe_in failed");
                exit(EXIT_FAILURE);
            }
            close(pipe_in[1]);
            
            close(pipe_out[1]);
            char outputBuffer[BUFSIZE];
            memset(outputBuffer, 0, sizeof(outputBuffer));
            int n = read(pipe_out[0], outputBuffer, sizeof(outputBuffer) - 1);
            if(n < 0){
                perror("read from pipe_out failed");
                exit(EXIT_FAILURE);
            }
            close(pipe_out[0]);
            wait(NULL);
            
            // Parse the output: first line is latency (ignored), rest are matrix rows.
            char *line = strtok(outputBuffer, "\n");
            line = strtok(NULL, "\n");
            for (int i = 0; i < final_rows && line != NULL; i++){
                char *token = strtok(line, " ");
                for (int j = 0; j < final_cols && token != NULL; j++){
                    int val = atoi(token);
                    finalResult[i][j] += val; // Aggregate result via elementwise sum.
                    token = strtok(NULL, " ");
                }
                line = strtok(NULL, "\n");
            }
        }
    }
    
    // Output the aggregated final result.
    for (int i = 0; i < final_rows; i++){
        for (int j = 0; j < final_cols; j++){
            printf("%d ", finalResult[i][j]);
        }
        printf("\n");
    }
    
    // Free allocated memory.
    for (int i = 0; i < final_rows; i++){
        free(finalResult[i]);
    }
    free(finalResult);
    
    return 0;
}

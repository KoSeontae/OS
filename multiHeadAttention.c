#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <string.h>

#define BUFSIZE 8192  // Size for input and output buffers

int main(){
    int total_heads;
    if(scanf("%d", &total_heads) != 1){
        fprintf(stderr, "Failed to read total number of heads.\n");
        exit(1);
    }
    
    // Variables for final aggregated matrix dimensions.
    int final_rows = 0, final_cols = 0;
    int **finalResult = NULL;  // Will allocate after reading first head

    // Process each head one by one
    for (int h = 0; h < total_heads; h++){
        // Create two pipes for communicating with attention_mp:
        // pipe_in: parent writes head input for attention_mp.
        // pipe_out: child (attention_mp) writes its output which parent reads.
        int pipe_in[2], pipe_out[2];
        if(pipe(pipe_in) < 0 || pipe(pipe_out) < 0){
            perror("pipe");
            exit(1);
        }
        
        // Prepare a buffer to store the current head's input.
        // The expected input format for attention_mp is:
        //   Q dimensions, Q matrix, K dimensions, K matrix, V dimensions, V matrix.
        char headInput[BUFSIZE];
        memset(headInput, 0, sizeof(headInput));
        char temp[256];  // Temporary string buffer
        
        // --- Read matrix Q ---
        int rQ, cQ;
        if(scanf("%d %d", &rQ, &cQ) != 2){
            fprintf(stderr, "Error reading Q dimensions for head %d\n", h);
            exit(1);
        }
        sprintf(temp, "%d %d\n", rQ, cQ);
        strcat(headInput, temp);
        
        // For the first head, record result rows.
        if(h == 0){
            final_rows = rQ;
        }
        
        // Read Q matrix values.
        for (int i = 0; i < rQ; i++){
            for (int j = 0; j < cQ; j++){
                int num;
                if(scanf("%d", &num) != 1){
                    fprintf(stderr, "Error reading Q matrix for head %d\n", h);
                    exit(1);
                }
                sprintf(temp, "%d ", num);
                strcat(headInput, temp);
            }
            strcat(headInput, "\n");
        }
        
        // --- Read matrix K ---
        int rK, cK;
        if(scanf("%d %d", &rK, &cK) != 2){
            fprintf(stderr, "Error reading K dimensions for head %d\n", h);
            exit(1);
        }
        sprintf(temp, "%d %d\n", rK, cK);
        strcat(headInput, temp);
        
        // Check that Q and K are compatible (cQ == cK).
        if(cQ != cK){
            fprintf(stderr, "Dimension mismatch for head %d: Q columns (%d) != K columns (%d)\n", h, cQ, cK);
            exit(1);
        }
        
        // Read K matrix values.
        for (int i = 0; i < rK; i++){
            for (int j = 0; j < cK; j++){
                int num;
                if(scanf("%d", &num) != 1){
                    fprintf(stderr, "Error reading K matrix for head %d\n", h);
                    exit(1);
                }
                sprintf(temp, "%d ", num);
                strcat(headInput, temp);
            }
            strcat(headInput, "\n");
        }
        
        // --- Read matrix V ---
        int rV, cV;
        if(scanf("%d %d", &rV, &cV) != 2){
            fprintf(stderr, "Error reading V dimensions for head %d\n", h);
            exit(1);
        }
        sprintf(temp, "%d %d\n", rV, cV);
        strcat(headInput, temp);
        
        // For the first head, record final columns and allocate the finalResult matrix.
        if(h == 0){
            final_cols = cV;
            finalResult = malloc(final_rows * sizeof(int *));
            for (int i = 0; i < final_rows; i++){
                finalResult[i] = calloc(final_cols, sizeof(int));
            }
        }
        
        // Check compatibility: V rows must equal K rows.
        if(rV != rK){
            fprintf(stderr, "Dimension mismatch for head %d: V rows (%d) != K rows (%d)\n", h, rV, rK);
            exit(1);
        }
        
        // Read V matrix values.
        for (int i = 0; i < rV; i++){
            for (int j = 0; j < cV; j++){
                int num;
                if(scanf("%d", &num) != 1){
                    fprintf(stderr, "Error reading V matrix for head %d\n", h);
                    exit(1);
                }
                sprintf(temp, "%d ", num);
                strcat(headInput, temp);
            }
            strcat(headInput, "\n");
        }
        
        // At this point, 'headInput' holds the complete input for attention_mp.
        
        // Fork a child process to run attention_mp for this head.
        pid_t pid = fork();
        if(pid < 0){
            perror("fork");
            exit(1);
        }
        if(pid == 0){
            // Child process:
            // Redirect read end of pipe_in to standard input.
            dup2(pipe_in[0], STDIN_FILENO);
            // Redirect write end of pipe_out to standard output.
            dup2(pipe_out[1], STDOUT_FILENO);
            
            // Close all unused file descriptors.
            close(pipe_in[0]);
            close(pipe_in[1]);
            close(pipe_out[0]);
            close(pipe_out[1]);
            
            // Execute the attention_mp program.
            // (Here we assume a fixed process count of "2"; adjust if needed.)
            char *args[] = {"./attention_mp", "2", NULL};
            execvp(args[0], args);
            perror("execvp failed");
            exit(1);
        } else {
            // Parent process:
            // Close the read end of pipe_in; we will only write.
            close(pipe_in[0]);
            
            // Write the head input to the child's standard input.
            if(write(pipe_in[1], headInput, strlen(headInput)) != (ssize_t)strlen(headInput)){
                perror("write to pipe_in failed");
                exit(1);
            }
            close(pipe_in[1]);
            
            // Close the write end of pipe_out; we will only read.
            close(pipe_out[1]);
            
            // Read the output from the child process.
            char outputBuffer[BUFSIZE];
            memset(outputBuffer, 0, sizeof(outputBuffer));
            int n = read(pipe_out[0], outputBuffer, sizeof(outputBuffer) - 1);
            if(n < 0){
                perror("read from pipe_out failed");
                exit(1);
            }
            close(pipe_out[0]);
            
            // Wait for the child process to finish.
            wait(NULL);
            
            // --- Parse the output ---
            // The attention_mp output is expected to have:
            //   First line: latency (ignore it)
            //   Following lines: the resulting matrix (rQ rows, each with cV integers).
            char *line = strtok(outputBuffer, "\n");  // this is the latency, so ignore
            line = strtok(NULL, "\n");  // first row of result
            for (int i = 0; i < final_rows && line != NULL; i++){
                // Parse the line for cV integers.
                char *token = strtok(line, " ");
                for (int j = 0; j < final_cols && token != NULL; j++){
                    int val = atoi(token);
                    finalResult[i][j] += val; // aggregate via elementwise sum
                    token = strtok(NULL, " ");
                }
                line = strtok(NULL, "\n");
            }
        }
    } // end for each head

    // After processing all heads, output the final aggregated matrix.
    for (int i = 0; i < final_rows; i++){
        for (int j = 0; j < final_cols; j++){
            printf("%d ", finalResult[i][j]);
        }
        printf("\n");
    }
    
    // Free the allocated final result matrix.
    for (int i = 0; i < final_rows; i++){
        free(finalResult[i]);
    }
    free(finalResult);
    
    return 0;
}

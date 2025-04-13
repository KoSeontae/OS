/*
 * multiHeadAttention.c
 *
 * 이 프로그램은 명령행 인자로 total_process_num을 받고,
 * input.txt 파일에서는 첫 번째 줄에 total_heads(헤드의 개수),
 * 이후 각 헤드마다 Q, K, V 행렬의 크기와 데이터를 순서대로 읽습니다.
 *
 * 각 헤드별로, 프로그램은 자식 프로세스를 fork하여 attention_mp를 execvp로 호출합니다.
 * 자식 프로세스는 attention_mp 실행 시, 표준 입력에서 해당 헤드의 데이터를 읽어
 * 첫 번째 줄에는 지연 시간, 그 다음 줄부터 결과 행렬(예: rQ 행, 각 행에 cV 개의 정수)을 출력합니다.
 *
 * 부모 프로세스는 파이프로 자식의 출력을 전부 읽어들인 후,
 * 첫 줄(지연 시간)은 건너뛰고, 이후 rQ줄을 각각 파싱하여 최종 행렬(finalResult)에 element‑wise 합산합니다.
 *
 * 최종적으로 모든 헤드의 결과를 합산한 최종 행렬을 출력합니다.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/wait.h>

#define BUFSIZE 8192

int main(int argc, char *argv[]) {
    if(argc < 2){
        fprintf(stderr, "Usage: %s [total_process_num]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    int total_process_num = atoi(argv[1]);

    // 첫 번째 입력: 전체 헤드 개수 (total_heads)
    int total_heads;
    if (scanf("%d", &total_heads) != 1) {
        fprintf(stderr, "Failed to read total_heads from input.\n");
        exit(EXIT_FAILURE);
    }

    int final_rows = 0, final_cols = 0;
    int **finalResult = NULL;  // 첫 번째 헤드 처리 후 할당

    // 각 헤드를 순차적으로 처리
    for (int h = 0; h < total_heads; h++) {
        // 파이프 생성 (입력, 출력용)
        int pipe_in[2], pipe_out[2];
        if(pipe(pipe_in) < 0 || pipe(pipe_out) < 0) {
            perror("pipe error");
            exit(EXIT_FAILURE);
        }
        
        // 각 헤드의 데이터를 문자형 버퍼에 저장 (attention_mp가 읽을 형식)
        char headInput[BUFSIZE];
        memset(headInput, 0, sizeof(headInput));
        char temp[256];
        
        // --- Q 행렬 읽기 ---
        int rQ, cQ;
        if (scanf("%d %d", &rQ, &cQ) != 2) {
            fprintf(stderr, "Error reading Q dimensions for head %d\n", h);
            exit(EXIT_FAILURE);
        }
        sprintf(temp, "%d %d\n", rQ, cQ);
        strcat(headInput, temp);
        if (h == 0) {
            final_rows = rQ;  // 최종 행렬의 행 수
        }
        for (int i = 0; i < rQ; i++) {
            for (int j = 0; j < cQ; j++) {
                int num;
                if (scanf("%d", &num) != 1) {
                    fprintf(stderr, "Error reading Q[%d][%d] for head %d\n", i, j, h);
                    exit(EXIT_FAILURE);
                }
                sprintf(temp, "%d ", num);
                strcat(headInput, temp);
            }
            strcat(headInput, "\n");
        }
        
        // --- K 행렬 읽기 ---
        int rK, cK;
        if (scanf("%d %d", &rK, &cK) != 2) {
            fprintf(stderr, "Error reading K dimensions for head %d\n", h);
            exit(EXIT_FAILURE);
        }
        sprintf(temp, "%d %d\n", rK, cK);
        strcat(headInput, temp);
        if(cQ != cK) {
            fprintf(stderr, "Dimension mismatch for head %d: Q columns (%d) != K columns (%d)\n", h, cQ, cK);
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < rK; i++) {
            for (int j = 0; j < cK; j++) {
                int num;
                if (scanf("%d", &num) != 1) {
                    fprintf(stderr, "Error reading K[%d][%d] for head %d\n", i, j, h);
                    exit(EXIT_FAILURE);
                }
                sprintf(temp, "%d ", num);
                strcat(headInput, temp);
            }
            strcat(headInput, "\n");
        }
        
        // --- V 행렬 읽기 ---
        int rV, cV;
        if (scanf("%d %d", &rV, &cV) != 2) {
            fprintf(stderr, "Error reading V dimensions for head %d\n", h);
            exit(EXIT_FAILURE);
        }
        sprintf(temp, "%d %d\n", rV, cV);
        strcat(headInput, temp);
        if(rV != rK) {
            fprintf(stderr, "Dimension mismatch for head %d: V rows (%d) != K rows (%d)\n", h, rV, rK);
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < rV; i++) {
            for (int j = 0; j < cV; j++) {
                int num;
                if (scanf("%d", &num) != 1) {
                    fprintf(stderr, "Error reading V[%d][%d] for head %d\n", i, j, h);
                    exit(EXIT_FAILURE);
                }
                sprintf(temp, "%d ", num);
                strcat(headInput, temp);
            }
            strcat(headInput, "\n");
        }
        if (h == 0) {  // 첫 헤드에서 최종 행렬의 열 수와 메모리 할당
            final_cols = cV;
            finalResult = malloc(final_rows * sizeof(int *));
            for (int i = 0; i < final_rows; i++) {
                finalResult[i] = calloc(final_cols, sizeof(int));
            }
        }
        
        // 자식 프로세스 생성하여 attention_mp 실행
        pid_t pid = fork();
        if(pid < 0){
            perror("fork error");
            exit(EXIT_FAILURE);
        }
        if(pid == 0) {
            // 자식 프로세스: 파이프 재지정
            dup2(pipe_in[0], STDIN_FILENO);
            dup2(pipe_out[1], STDOUT_FILENO);
            close(pipe_in[0]); close(pipe_in[1]);
            close(pipe_out[0]); close(pipe_out[1]);
            // total_process_num을 인자로 전달
            char arg[16];
            sprintf(arg, "%d", total_process_num);
            char *args[] = {"./attention_mp", arg, NULL};
            execvp(args[0], args);
            perror("execvp failed");
            exit(EXIT_FAILURE);
        } else {
            // 부모 프로세스
            close(pipe_in[0]);
            if(write(pipe_in[1], headInput, strlen(headInput)) != (ssize_t)strlen(headInput)) {
                perror("write error");
                exit(EXIT_FAILURE);
            }
            close(pipe_in[1]);
            
            close(pipe_out[1]);
            // 자식의 출력을 완전히 읽어들이기 위해 버퍼에 저장
            char outputBuffer[BUFSIZE];
            memset(outputBuffer, 0, sizeof(outputBuffer));
            int totalRead = 0, n;
            while ((n = read(pipe_out[0], outputBuffer + totalRead, sizeof(outputBuffer) - totalRead - 1)) > 0) {
                totalRead += n;
                if(totalRead >= sizeof(outputBuffer) - 1)
                    break;
            }
            outputBuffer[totalRead] = '\0';
            close(pipe_out[0]);
            wait(NULL);
            
            // 자식 출력은 첫 줄에 지연 시간, 이후 rQ줄에 걸쳐 결과 행렬 출력
            // strtok로 줄 단위로 분리하여 첫 줄은 건너뛰고 나머지를 파싱
            char *line = strtok(outputBuffer, "\n");  // 첫 줄: 지연 시간 (무시)
            line = strtok(NULL, "\n");  // 첫 번째 행렬의 첫 줄
            for (int i = 0; i < rQ && line != NULL; i++) {
                char *token = strtok(line, " ");
                for (int j = 0; j < final_cols && token != NULL; j++) {
                    int val = atoi(token);
                    finalResult[i][j] += val; // elementwise 누적 합산
                    token = strtok(NULL, " ");
                }
                line = strtok(NULL, "\n");
            }
        }
    }
    
    // 최종 합산된 결과 행렬 출력
    for (int i = 0; i < final_rows; i++) {
        for (int j = 0; j < final_cols; j++) {
            printf("%d ", finalResult[i][j]);
        }
        printf("\n");
    }
    
    // 할당된 메모리 해제
    for (int i = 0; i < final_rows; i++) {
        free(finalResult[i]);
    }
    free(finalResult);
    
    return 0;
}

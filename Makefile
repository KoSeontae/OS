CC = gcc
CFLAGS = -pthread -O2

all: attention attention_mp multiHeadAttention

attention: attention.c
	$(CC) $(CFLAGS) -o attention attention.c

attention_mp: attention_mp.c
	$(CC) $(CFLAGS) -o attention_mp attention_mp.c

multiHeadAttention: multiHeadAttention.c
	$(CC) $(CFLAGS) -o multiHeadAttention multiHeadAttention.c

clean:
	rm -f attention attention_mp multiHeadAttention

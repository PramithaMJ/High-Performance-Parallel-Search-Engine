CC=gcc
CFLAGS=-Wall -O2

OBJS=main.o parser.o index.o ranking.o utils.o

all: search_engine

search_engine: $(OBJS)
	$(CC) -o $@ $^

clean:
	rm -f *.o search_engine

CC=gcc
CFLAGS=-Wall -O2
LDFLAGS=`pkg-config --libs libcurl`
CPPFLAGS=`pkg-config --cflags libcurl`

OBJS=main.o parser.o index.o ranking.o utils.o crawler.o

all: search_engine

search_engine: $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

clean:
	rm -f *.o search_engine

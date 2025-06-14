CC=gcc
CFLAGS=-Wall -O2
LDFLAGS=`pkg-config --libs libcurl`
CPPFLAGS=`pkg-config --cflags libcurl`

OBJS=main.o parser.o index.o ranking.o utils.o crawler.o

all: search_engine test_url_normalization test_medium_urls

# Production build - only builds the search engine without tests
production: search_engine

search_engine: $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)
	
test_url_normalization: test_url_normalization.c
	$(CC) $(CFLAGS) -o $@ $<
	
test_medium_urls: test_medium_urls.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f *.o search_engine test_url_normalization test_medium_urls

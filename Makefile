CC=gcc
all: search_engine test_url_normalization test_medium_urls evaluate

# Production build - only builds the search engine without tests
production: search_engine

evaluate: evaluate.o parser.o index.o ranking.o utils.o crawler.o metrics.o
	$(CC) -o $@ $^ $(LDFLAGS)GS=-Wall -O2
LDFLAGS=`pkg-config --libs libcurl`
CPPFLAGS=`pkg-config --cflags libcurl`

OBJS=main.o parser.o index.o ranking.o utils.o crawler.o metrics.o

all: search_engine test_url_normalization test_medium_urls evaluate

# Production build - only builds the search engine without tests
production: search_engine

evaluate: evaluate.o $(filter-out main.o, $(OBJS))
	$(CC) -o $@ $^ $(LDFLAGS)

search_engine: $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)
	
test_url_normalization: test_url_normalization.c
	$(CC) $(CFLAGS) -o $@ $<
	
test_medium_urls: test_medium_urls.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f *.o search_engine test_url_normalization test_medium_urls

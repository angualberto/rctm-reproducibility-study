CC = gcc
CFLAGS = -O3 -Wall -Wextra -lm
TARGET = tentmap

all: $(TARGET)

$(TARGET): tentmap.c
	$(CC) $(CFLAGS) -o $(TARGET) tentmap.c -lm

clean:
	rm -f $(TARGET) *.bin

.PHONY: all clean

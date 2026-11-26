// Controlled CPU memory load for the GB10 DRAM dispersion study: each thread
// copies between two 1 GiB buffers until SIGTERM, then the program prints the
// bytes moved per second (read plus write) as JSON.
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BUFFER_BYTES (1UL << 30)

static atomic_int stop;
static atomic_ulong moved;

static void on_term(int sig) { (void)sig; atomic_store(&stop, 1); }

static void* stream(void* arg) {
  (void)arg;
  char* a = malloc(BUFFER_BYTES);
  char* b = malloc(BUFFER_BYTES);
  if (!a || !b) { perror("malloc"); exit(1); }
  memset(a, 1, BUFFER_BYTES);
  memset(b, 2, BUFFER_BYTES);
  const size_t chunk = 64UL << 20;
  while (!atomic_load(&stop)) {
    for (size_t off = 0; off < BUFFER_BYTES && !atomic_load(&stop); off += chunk) {
      memcpy(b + off, a + off, chunk);
      atomic_fetch_add(&moved, 2 * chunk);
    }
    char* t = a; a = b; b = t;
  }
  free(a);
  free(b);
  return NULL;
}

int main(int argc, char** argv) {
  int threads = argc > 1 ? atoi(argv[1]) : 1;
  signal(SIGTERM, on_term);
  signal(SIGINT, on_term);
  struct timespec t0, t1;
  pthread_t* tid = calloc(threads, sizeof(pthread_t));
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (int i = 0; i < threads; ++i) pthread_create(&tid[i], NULL, stream, NULL);
  for (int i = 0; i < threads; ++i) pthread_join(tid[i], NULL);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
  printf("{\"threads\": %d, \"seconds\": %.3f, \"gbps\": %.3f}\n", threads, s, atomic_load(&moved) / s / 1e9);
  return 0;
}

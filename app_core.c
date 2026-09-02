/* app_core.c - Gemma 4 E2B IT QAT inference engine.
 *
 * A single translation unit that carries both the public interface and the
 * implementation.  Consumers (app_main.c, app_test.c) include this file
 * directly, which keeps the project flat and lets the compiler inline across
 * every layer.  See GUIDE.md for a full tour.
 *
 * Layer order in this file:
 *   1. public interface     - types and entry points
 *   2. platform layer       - memory, files, time, threads
 *   3. json layer           - config and tokenizer parsing
 *   4. store layer          - safetensors reader
 *   5. quant layer          - pack-quantized decode
 *   6. kernel layer         - math primitives behind a backend table
 *   7. model layer          - config, weights, graph
 *   8. token layer          - vocabulary, encode, decode
 *   9. session layer        - cache, forward, sample
 *  10. public layer         - entry point bodies
 */

#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L
#endif

#ifndef APP_CORE_INCLUDED
#define APP_CORE_INCLUDED

/* ======================================================================== */
/* 1. public interface                                                      */
/* ======================================================================== */

#include <stddef.h>
#include <stdint.h>

#define APP_NAME    "igllm"
#define APP_VERSION "0.1.0"

/* Result of any call that can fail.  Zero is success, negative is failure. */
typedef enum app_code {
  APP_OKAY = 0,
  APP_FAIL_ARGUMENT = -1,
  APP_FAIL_MEMORY = -2,
  APP_FAIL_FILE = -3,
  APP_FAIL_FORMAT = -4,
  APP_FAIL_MISSING = -5,
  APP_FAIL_SUPPORT = -6,
  APP_FAIL_STATE = -7
} app_code;

/* Tunables applied when a model is loaded. */
typedef struct app_setup {
  int   thread_count;  /* worker threads, 0 selects the host default */
  int   window_limit;  /* context length, 0 selects the model default */
  int   verbose_level; /* 0 quiet, 1 progress, 2 detail */
} app_setup;

/* Tunables applied when a token is drawn from a logit row. */
typedef struct app_taste {
  float heat_value;   /* temperature, 0 selects greedy decoding */
  int   top_count;    /* top-k cutoff, 0 disables */
  float top_portion;  /* top-p cutoff, 1 disables */
  float echo_penalty; /* repetition penalty, 1 disables */
  int   echo_window;  /* tokens considered by the repetition penalty */
  uint64_t seed_value;/* sampler seed */
} app_taste;

/* Counters collected while a session runs. */
typedef struct app_tally {
  double prime_seconds;
  double serve_seconds;
  size_t prime_tokens;
  size_t serve_tokens;
  size_t memory_bytes;
} app_tally;

typedef struct app_model   app_model;
typedef struct app_session app_session;

/* model layer -------------------------------------------------------------*/
app_code    model_load(const char *folder_path, const app_setup *setup, app_model **model_out);
void        model_free(app_model *model);
const char *model_name(const app_model *model);
int         model_layer_count(const app_model *model);
int         model_vocab_count(const app_model *model);
int         model_window_limit(const app_model *model);
size_t      model_memory_bytes(const app_model *model);

/* token layer -------------------------------------------------------------*/
int  token_encode(const app_model *model, const char *text, int lead_marker,
                  int32_t *id_list, int id_limit);
int  token_decode(const app_model *model, int32_t id_value, char *text_out, int text_limit);
int  token_frame(const app_model *model, const char *user_text,
                 int32_t *id_list, int id_limit); /* chat framing for IT models */
int  token_start_id(const app_model *model);
int  token_close_id(const app_model *model);
int  token_is_close(const app_model *model, int32_t id_value);

/* session layer -----------------------------------------------------------*/
app_code     session_open(app_model *model, app_session **session_out);
void         session_close(app_session *session);
void         session_reset(app_session *session);
app_code     session_prime(app_session *session, const int32_t *id_list, int id_count);
const float *session_step(app_session *session, int32_t id_value);
int32_t      session_pick(app_session *session, const float *logit_list, const app_taste *taste);
int          session_fill(const app_session *session); /* tokens held in cache */
app_tally    session_tally(const app_session *session);

/* helper layer ------------------------------------------------------------*/
const char *app_code_text(app_code code);
app_taste   app_taste_plain(void);
app_setup   app_setup_plain(void);

#endif /* APP_CORE_INCLUDED */

#ifndef APP_CORE_IMPLEMENTED
#define APP_CORE_IMPLEMENTED

#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  define APP_HOST_WINDOWS 1
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  define APP_HOST_POSIX 1
#  include <fcntl.h>
#  include <pthread.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <time.h>
#  include <unistd.h>
#endif

#if defined(__AVX2__)
#  include <immintrin.h>
#  define APP_SIMD_AVX2 1
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#  include <emmintrin.h>
#  define APP_SIMD_SSE2 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
#  include <arm_neon.h>
#  define APP_SIMD_NEON 1
#endif

/* ======================================================================== */
/* 2. platform layer                                                        */
/* ======================================================================== */

#define APP_ALIGN_BYTES 64

static size_t app_total_bytes = 0; /* coarse accounting for the bench report */

static void *mem_alloc(size_t byte_count) {
  void *block = NULL;
  size_t rounded = (byte_count + APP_ALIGN_BYTES - 1) & ~(size_t)(APP_ALIGN_BYTES - 1);
  if (rounded == 0) rounded = APP_ALIGN_BYTES;
#if defined(APP_HOST_WINDOWS)
  block = _aligned_malloc(rounded, APP_ALIGN_BYTES);
#else
  if (posix_memalign(&block, APP_ALIGN_BYTES, rounded) != 0) block = NULL;
#endif
  if (block) app_total_bytes += rounded;
  return block;
}

static void *mem_clear(size_t byte_count) {
  void *block = mem_alloc(byte_count);
  if (block) memset(block, 0, byte_count);
  return block;
}

static void mem_free(void *block) {
  if (!block) return;
#if defined(APP_HOST_WINDOWS)
  _aligned_free(block);
#else
  free(block);
#endif
}

static char *text_copy(const char *text, size_t length) {
  char *copy = (char *)mem_alloc(length + 1);
  if (!copy) return NULL;
  memcpy(copy, text, length);
  copy[length] = '\0';
  return copy;
}

/* A read-only mapping of a file on disk. */
typedef struct file_map {
  const uint8_t *base_data;
  size_t         byte_count;
#if defined(APP_HOST_WINDOWS)
  HANDLE file_handle;
  HANDLE view_handle;
#else
  int file_handle;
#endif
} file_map;

static app_code file_open(const char *path, file_map *map_out) {
  memset(map_out, 0, sizeof(*map_out));
#if defined(APP_HOST_WINDOWS)
  {
    LARGE_INTEGER size_value;
    HANDLE file_handle = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                                     OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file_handle == INVALID_HANDLE_VALUE) return APP_FAIL_FILE;
    if (!GetFileSizeEx(file_handle, &size_value) || size_value.QuadPart == 0) {
      CloseHandle(file_handle);
      return APP_FAIL_FILE;
    }
    {
      HANDLE view_handle = CreateFileMappingA(file_handle, NULL, PAGE_READONLY, 0, 0, NULL);
      const uint8_t *base_data;
      if (!view_handle) { CloseHandle(file_handle); return APP_FAIL_FILE; }
      base_data = (const uint8_t *)MapViewOfFile(view_handle, FILE_MAP_READ, 0, 0, 0);
      if (!base_data) { CloseHandle(view_handle); CloseHandle(file_handle); return APP_FAIL_FILE; }
      map_out->file_handle = file_handle;
      map_out->view_handle = view_handle;
      map_out->base_data = base_data;
      map_out->byte_count = (size_t)size_value.QuadPart;
    }
  }
#else
  {
    struct stat file_stat;
    int file_handle = open(path, O_RDONLY);
    void *base_data;
    if (file_handle < 0) return APP_FAIL_FILE;
    if (fstat(file_handle, &file_stat) != 0 || file_stat.st_size <= 0) {
      close(file_handle);
      return APP_FAIL_FILE;
    }
    base_data = mmap(NULL, (size_t)file_stat.st_size, PROT_READ, MAP_PRIVATE, file_handle, 0);
    if (base_data == MAP_FAILED) { close(file_handle); return APP_FAIL_FILE; }
    map_out->file_handle = file_handle;
    map_out->base_data = (const uint8_t *)base_data;
    map_out->byte_count = (size_t)file_stat.st_size;
  }
#endif
  return APP_OKAY;
}

static void file_close(file_map *map) {
  if (!map || !map->base_data) return;
#if defined(APP_HOST_WINDOWS)
  UnmapViewOfFile((LPCVOID)map->base_data);
  CloseHandle(map->view_handle);
  CloseHandle(map->file_handle);
#else
  munmap((void *)map->base_data, map->byte_count);
  close(map->file_handle);
#endif
  memset(map, 0, sizeof(*map));
}

/* Whole-file read for small documents such as config.json. */
static char *file_slurp(const char *path, size_t *byte_count_out) {
  file_map map;
  char *copy;
  if (file_open(path, &map) != APP_OKAY) return NULL;
  copy = (char *)mem_alloc(map.byte_count + 1);
  if (copy) {
    memcpy(copy, map.base_data, map.byte_count);
    copy[map.byte_count] = '\0';
    if (byte_count_out) *byte_count_out = map.byte_count;
  }
  file_close(&map);
  return copy;
}

static int file_exists(const char *path) {
#if defined(APP_HOST_WINDOWS)
  DWORD attributes = GetFileAttributesA(path);
  return attributes != INVALID_FILE_ATTRIBUTES;
#else
  struct stat file_stat;
  return stat(path, &file_stat) == 0;
#endif
}

static void path_join(char *path_out, size_t path_limit, const char *folder, const char *leaf) {
  size_t length = strlen(folder);
  if (length > 0 && (folder[length - 1] == '/' || folder[length - 1] == '\\'))
    snprintf(path_out, path_limit, "%s%s", folder, leaf);
  else
    snprintf(path_out, path_limit, "%s/%s", folder, leaf);
}

static double time_now(void) {
#if defined(APP_HOST_WINDOWS)
  LARGE_INTEGER tick_value, tick_rate;
  QueryPerformanceCounter(&tick_value);
  QueryPerformanceFrequency(&tick_rate);
  return (double)tick_value.QuadPart / (double)tick_rate.QuadPart;
#else
  struct timespec clock_value;
  clock_gettime(CLOCK_MONOTONIC, &clock_value);
  return (double)clock_value.tv_sec + (double)clock_value.tv_nsec * 1e-9;
#endif
}

static int host_thread_count(void) {
#if defined(APP_HOST_WINDOWS)
  SYSTEM_INFO host_info;
  GetSystemInfo(&host_info);
  return (int)host_info.dwNumberOfProcessors;
#else
  long count = sysconf(_SC_NPROCESSORS_ONLN);
  return count > 0 ? (int)count : 1;
#endif
}

/* A fork-join pool: every job splits one index range across the workers. */
typedef void (*pool_task)(void *state, int slice_index, int slice_count);

typedef struct pool_group {
  int        worker_count;
  int        stop_flag;
  pool_task  task_call;
  void      *task_state;
  int        task_serial;
  int        done_count;
#if defined(APP_HOST_WINDOWS)
  HANDLE            *worker_list;
  CRITICAL_SECTION   guard_lock;
  CONDITION_VARIABLE work_wake;
  CONDITION_VARIABLE done_wake;
#else
  pthread_t      *worker_list;
  pthread_mutex_t guard_lock;
  pthread_cond_t  work_wake;
  pthread_cond_t  done_wake;
#endif
} pool_group;

typedef struct pool_seat {
  pool_group *group;
  int         slice_index;
} pool_seat;

static void pool_lock(pool_group *group) {
#if defined(APP_HOST_WINDOWS)
  EnterCriticalSection(&group->guard_lock);
#else
  pthread_mutex_lock(&group->guard_lock);
#endif
}

static void pool_unlock(pool_group *group) {
#if defined(APP_HOST_WINDOWS)
  LeaveCriticalSection(&group->guard_lock);
#else
  pthread_mutex_unlock(&group->guard_lock);
#endif
}

#if defined(APP_HOST_WINDOWS)
static DWORD WINAPI pool_loop(LPVOID seat_data)
#else
static void *pool_loop(void *seat_data)
#endif
{
  pool_seat *seat = (pool_seat *)seat_data;
  pool_group *group = seat->group;
  int seen_serial = 0;
  for (;;) {
    pool_task task_call;
    void *task_state;
    int slice_count;
    pool_lock(group);
    while (group->task_serial == seen_serial && !group->stop_flag) {
#if defined(APP_HOST_WINDOWS)
      SleepConditionVariableCS(&group->work_wake, &group->guard_lock, INFINITE);
#else
      pthread_cond_wait(&group->work_wake, &group->guard_lock);
#endif
    }
    if (group->stop_flag) { pool_unlock(group); break; }
    seen_serial = group->task_serial;
    task_call = group->task_call;
    task_state = group->task_state;
    slice_count = group->worker_count + 1;
    pool_unlock(group);

    if (task_call) task_call(task_state, seat->slice_index, slice_count);

    pool_lock(group);
    group->done_count += 1;
#if defined(APP_HOST_WINDOWS)
    WakeAllConditionVariable(&group->done_wake);
#else
    pthread_cond_broadcast(&group->done_wake);
#endif
    pool_unlock(group);
  }
#if defined(APP_HOST_WINDOWS)
  return 0;
#else
  return NULL;
#endif
}

static pool_seat *pool_seat_list = NULL;

static app_code pool_open(pool_group *group, int thread_count) {
  int seat_index;
  memset(group, 0, sizeof(*group));
  if (thread_count < 1) thread_count = 1;
  group->worker_count = thread_count - 1; /* the caller is worker zero */
  if (group->worker_count <= 0) return APP_OKAY;

#if defined(APP_HOST_WINDOWS)
  InitializeCriticalSection(&group->guard_lock);
  InitializeConditionVariable(&group->work_wake);
  InitializeConditionVariable(&group->done_wake);
  group->worker_list = (HANDLE *)mem_clear(sizeof(HANDLE) * (size_t)group->worker_count);
#else
  pthread_mutex_init(&group->guard_lock, NULL);
  pthread_cond_init(&group->work_wake, NULL);
  pthread_cond_init(&group->done_wake, NULL);
  group->worker_list = (pthread_t *)mem_clear(sizeof(pthread_t) * (size_t)group->worker_count);
#endif
  pool_seat_list = (pool_seat *)mem_clear(sizeof(pool_seat) * (size_t)group->worker_count);
  if (!group->worker_list || !pool_seat_list) return APP_FAIL_MEMORY;

  for (seat_index = 0; seat_index < group->worker_count; ++seat_index) {
    pool_seat_list[seat_index].group = group;
    pool_seat_list[seat_index].slice_index = seat_index + 1;
#if defined(APP_HOST_WINDOWS)
    group->worker_list[seat_index] =
        CreateThread(NULL, 0, pool_loop, &pool_seat_list[seat_index], 0, NULL);
    if (!group->worker_list[seat_index]) return APP_FAIL_MEMORY;
#else
    if (pthread_create(&group->worker_list[seat_index], NULL, pool_loop,
                       &pool_seat_list[seat_index]) != 0)
      return APP_FAIL_MEMORY;
#endif
  }
  return APP_OKAY;
}

static void pool_run(pool_group *group, pool_task task_call, void *task_state) {
  int slice_count = group->worker_count + 1;
  if (group->worker_count <= 0) {
    task_call(task_state, 0, 1);
    return;
  }
  pool_lock(group);
  group->task_call = task_call;
  group->task_state = task_state;
  group->done_count = 0;
  group->task_serial += 1;
#if defined(APP_HOST_WINDOWS)
  WakeAllConditionVariable(&group->work_wake);
#else
  pthread_cond_broadcast(&group->work_wake);
#endif
  pool_unlock(group);

  task_call(task_state, 0, slice_count);

  pool_lock(group);
  while (group->done_count < group->worker_count) {
#if defined(APP_HOST_WINDOWS)
    SleepConditionVariableCS(&group->done_wake, &group->guard_lock, INFINITE);
#else
    pthread_cond_wait(&group->done_wake, &group->guard_lock);
#endif
  }
  pool_unlock(group);
}

static void pool_close(pool_group *group) {
  int seat_index;
  if (group->worker_count <= 0) return;
  pool_lock(group);
  group->stop_flag = 1;
#if defined(APP_HOST_WINDOWS)
  WakeAllConditionVariable(&group->work_wake);
#else
  pthread_cond_broadcast(&group->work_wake);
#endif
  pool_unlock(group);
  for (seat_index = 0; seat_index < group->worker_count; ++seat_index) {
#if defined(APP_HOST_WINDOWS)
    if (group->worker_list[seat_index]) {
      WaitForSingleObject(group->worker_list[seat_index], INFINITE);
      CloseHandle(group->worker_list[seat_index]);
    }
#else
    pthread_join(group->worker_list[seat_index], NULL);
#endif
  }
#if defined(APP_HOST_WINDOWS)
  DeleteCriticalSection(&group->guard_lock);
#else
  pthread_mutex_destroy(&group->guard_lock);
  pthread_cond_destroy(&group->work_wake);
  pthread_cond_destroy(&group->done_wake);
#endif
  mem_free(group->worker_list);
  mem_free(pool_seat_list);
  group->worker_list = NULL;
  pool_seat_list = NULL;
  group->worker_count = 0;
}

/* Even split of `total` items into `slice_count` bands. */
static void slice_span(int total, int slice_index, int slice_count, int *from_out, int *upto_out) {
  int band = total / slice_count;
  int extra = total % slice_count;
  int from = slice_index * band + (slice_index < extra ? slice_index : extra);
  int size = band + (slice_index < extra ? 1 : 0);
  *from_out = from;
  *upto_out = from + size;
}

/* ======================================================================== */
/* 3. json layer                                                            */
/* ======================================================================== */

typedef enum json_kind {
  JSON_VOID = 0,
  JSON_NULL,
  JSON_FLAG,
  JSON_NUMBER,
  JSON_TEXT,
  JSON_LIST,
  JSON_DICT
} json_kind;

typedef struct json_node {
  json_kind kind;
  double    number_value;
  int32_t   text_start;  /* offset into the tree text pool, -1 when absent */
  int32_t   name_start;  /* member name for dictionary children */
  int32_t   head_child;
  int32_t   next_peer;
  int32_t   child_count;
} json_node;

typedef struct json_tree {
  json_node *node_list;
  int32_t    node_count;
  int32_t    node_limit;
  char      *text_pool;
  size_t     text_size;
  size_t     text_limit;
} json_tree;

typedef struct json_scan {
  const char *text_data;
  size_t      text_size;
  size_t      text_head;
  json_tree  *tree;
  int         fail_flag;
} json_scan;

static int32_t json_take_node(json_tree *tree) {
  if (tree->node_count == tree->node_limit) {
    int32_t next_limit = tree->node_limit ? tree->node_limit * 2 : 256;
    json_node *next_list = (json_node *)mem_alloc(sizeof(json_node) * (size_t)next_limit);
    if (!next_list) return -1;
    if (tree->node_list) {
      memcpy(next_list, tree->node_list, sizeof(json_node) * (size_t)tree->node_count);
      mem_free(tree->node_list);
    }
    tree->node_list = next_list;
    tree->node_limit = next_limit;
  }
  {
    json_node *node = &tree->node_list[tree->node_count];
    memset(node, 0, sizeof(*node));
    node->text_start = -1;
    node->name_start = -1;
    node->head_child = -1;
    node->next_peer = -1;
  }
  return tree->node_count++;
}

static int32_t json_take_text(json_tree *tree, size_t byte_count) {
  if (tree->text_size + byte_count + 1 > tree->text_limit) {
    size_t next_limit = tree->text_limit ? tree->text_limit * 2 : 4096;
    char *next_pool;
    while (next_limit < tree->text_size + byte_count + 1) next_limit *= 2;
    next_pool = (char *)mem_alloc(next_limit);
    if (!next_pool) return -1;
    if (tree->text_pool) {
      memcpy(next_pool, tree->text_pool, tree->text_size);
      mem_free(tree->text_pool);
    }
    tree->text_pool = next_pool;
    tree->text_limit = next_limit;
  }
  return (int32_t)tree->text_size;
}

static void json_skip_space(json_scan *scan) {
  while (scan->text_head < scan->text_size) {
    char letter = scan->text_data[scan->text_head];
    if (letter == ' ' || letter == '\t' || letter == '\n' || letter == '\r')
      scan->text_head += 1;
    else
      break;
  }
}

static void json_write_rune(json_tree *tree, size_t *cursor, uint32_t rune) {
  char *pool = tree->text_pool;
  if (rune < 0x80) {
    pool[(*cursor)++] = (char)rune;
  } else if (rune < 0x800) {
    pool[(*cursor)++] = (char)(0xC0 | (rune >> 6));
    pool[(*cursor)++] = (char)(0x80 | (rune & 0x3F));
  } else if (rune < 0x10000) {
    pool[(*cursor)++] = (char)(0xE0 | (rune >> 12));
    pool[(*cursor)++] = (char)(0x80 | ((rune >> 6) & 0x3F));
    pool[(*cursor)++] = (char)(0x80 | (rune & 0x3F));
  } else {
    pool[(*cursor)++] = (char)(0xF0 | (rune >> 18));
    pool[(*cursor)++] = (char)(0x80 | ((rune >> 12) & 0x3F));
    pool[(*cursor)++] = (char)(0x80 | ((rune >> 6) & 0x3F));
    pool[(*cursor)++] = (char)(0x80 | (rune & 0x3F));
  }
}

static uint32_t json_read_hex(json_scan *scan) {
  uint32_t value = 0;
  int digit_index;
  for (digit_index = 0; digit_index < 4; ++digit_index) {
    char letter = scan->text_head < scan->text_size ? scan->text_data[scan->text_head++] : '0';
    value <<= 4;
    if (letter >= '0' && letter <= '9') value |= (uint32_t)(letter - '0');
    else if (letter >= 'a' && letter <= 'f') value |= (uint32_t)(letter - 'a' + 10);
    else if (letter >= 'A' && letter <= 'F') value |= (uint32_t)(letter - 'A' + 10);
    else scan->fail_flag = 1;
  }
  return value;
}

/* Copies one JSON string into the tree pool and returns its offset. */
static int32_t json_read_text(json_scan *scan) {
  size_t probe_head = scan->text_head;
  size_t raw_length = 0;
  int32_t start_offset;
  size_t cursor;

  if (probe_head >= scan->text_size || scan->text_data[probe_head] != '"') {
    scan->fail_flag = 1;
    return -1;
  }
  probe_head += 1;
  while (probe_head < scan->text_size && scan->text_data[probe_head] != '"') {
    if (scan->text_data[probe_head] == '\\') probe_head += 1;
    probe_head += 1;
    raw_length += 1;
  }
  /* Worst case four bytes per escape sequence. */
  start_offset = json_take_text(scan->tree, raw_length * 4 + 4);
  if (start_offset < 0) { scan->fail_flag = 1; return -1; }

  cursor = (size_t)start_offset;
  scan->text_head += 1;
  while (scan->text_head < scan->text_size) {
    char letter = scan->text_data[scan->text_head];
    if (letter == '"') { scan->text_head += 1; break; }
    if (letter == '\\') {
      char mark = scan->text_head + 1 < scan->text_size ? scan->text_data[scan->text_head + 1] : '"';
      scan->text_head += 2;
      switch (mark) {
        case 'n': scan->tree->text_pool[cursor++] = '\n'; break;
        case 't': scan->tree->text_pool[cursor++] = '\t'; break;
        case 'r': scan->tree->text_pool[cursor++] = '\r'; break;
        case 'b': scan->tree->text_pool[cursor++] = '\b'; break;
        case 'f': scan->tree->text_pool[cursor++] = '\f'; break;
        case '/': scan->tree->text_pool[cursor++] = '/'; break;
        case '\\': scan->tree->text_pool[cursor++] = '\\'; break;
        case '"': scan->tree->text_pool[cursor++] = '"'; break;
        case 'u': {
          uint32_t rune = json_read_hex(scan);
          if (rune >= 0xD800 && rune <= 0xDBFF && scan->text_head + 1 < scan->text_size &&
              scan->text_data[scan->text_head] == '\\' && scan->text_data[scan->text_head + 1] == 'u') {
            uint32_t trail;
            scan->text_head += 2;
            trail = json_read_hex(scan);
            rune = 0x10000 + ((rune - 0xD800) << 10) + (trail - 0xDC00);
          }
          json_write_rune(scan->tree, &cursor, rune);
          break;
        }
        default: scan->tree->text_pool[cursor++] = mark; break;
      }
    } else {
      scan->tree->text_pool[cursor++] = letter;
      scan->text_head += 1;
    }
  }
  scan->tree->text_pool[cursor++] = '\0';
  scan->tree->text_size = cursor;
  return start_offset;
}

static int32_t json_read_value(json_scan *scan);

static int32_t json_read_group(json_scan *scan, int is_dict) {
  int32_t group_index = json_take_node(scan->tree);
  int32_t tail_index = -1;
  char close_mark = is_dict ? '}' : ']';
  if (group_index < 0) { scan->fail_flag = 1; return -1; }
  scan->tree->node_list[group_index].kind = is_dict ? JSON_DICT : JSON_LIST;
  scan->text_head += 1;
  json_skip_space(scan);
  if (scan->text_head < scan->text_size && scan->text_data[scan->text_head] == close_mark) {
    scan->text_head += 1;
    return group_index;
  }
  for (;;) {
    int32_t name_start = -1;
    int32_t child_index;
    json_skip_space(scan);
    if (is_dict) {
      name_start = json_read_text(scan);
      json_skip_space(scan);
      if (scan->text_head < scan->text_size && scan->text_data[scan->text_head] == ':')
        scan->text_head += 1;
      else
        scan->fail_flag = 1;
      json_skip_space(scan);
    }
    child_index = json_read_value(scan);
    if (child_index < 0 || scan->fail_flag) { scan->fail_flag = 1; return -1; }
    scan->tree->node_list[child_index].name_start = name_start;
    if (tail_index < 0)
      scan->tree->node_list[group_index].head_child = child_index;
    else
      scan->tree->node_list[tail_index].next_peer = child_index;
    tail_index = child_index;
    scan->tree->node_list[group_index].child_count += 1;

    json_skip_space(scan);
    if (scan->text_head < scan->text_size && scan->text_data[scan->text_head] == ',') {
      scan->text_head += 1;
      continue;
    }
    if (scan->text_head < scan->text_size && scan->text_data[scan->text_head] == close_mark) {
      scan->text_head += 1;
      break;
    }
    scan->fail_flag = 1;
    return -1;
  }
  return group_index;
}

static int32_t json_read_value(json_scan *scan) {
  char letter;
  json_skip_space(scan);
  if (scan->text_head >= scan->text_size) { scan->fail_flag = 1; return -1; }
  letter = scan->text_data[scan->text_head];
  if (letter == '{') return json_read_group(scan, 1);
  if (letter == '[') return json_read_group(scan, 0);
  {
    int32_t node_index = json_take_node(scan->tree);
    if (node_index < 0) { scan->fail_flag = 1; return -1; }
    if (letter == '"') {
      int32_t text_start = json_read_text(scan);
      scan->tree->node_list[node_index].kind = JSON_TEXT;
      scan->tree->node_list[node_index].text_start = text_start;
    } else if (letter == 't' || letter == 'f') {
      scan->tree->node_list[node_index].kind = JSON_FLAG;
      scan->tree->node_list[node_index].number_value = letter == 't' ? 1.0 : 0.0;
      scan->text_head += letter == 't' ? 4 : 5;
    } else if (letter == 'n') {
      scan->tree->node_list[node_index].kind = JSON_NULL;
      scan->text_head += 4;
    } else {
      char number_text[64];
      size_t number_size = 0;
      while (scan->text_head < scan->text_size && number_size < sizeof(number_text) - 1) {
        char mark = scan->text_data[scan->text_head];
        if ((mark >= '0' && mark <= '9') || mark == '-' || mark == '+' || mark == '.' ||
            mark == 'e' || mark == 'E') {
          number_text[number_size++] = mark;
          scan->text_head += 1;
        } else {
          break;
        }
      }
      if (number_size == 0) { scan->fail_flag = 1; return -1; }
      number_text[number_size] = '\0';
      scan->tree->node_list[node_index].kind = JSON_NUMBER;
      scan->tree->node_list[node_index].number_value = strtod(number_text, NULL);
    }
    return node_index;
  }
}

static void json_free(json_tree *tree) {
  if (!tree) return;
  mem_free(tree->node_list);
  mem_free(tree->text_pool);
  mem_free(tree);
}

static json_tree *json_read(const char *text_data, size_t text_size) {
  json_scan scan;
  json_tree *tree = (json_tree *)mem_clear(sizeof(json_tree));
  if (!tree) return NULL;
  scan.text_data = text_data;
  scan.text_size = text_size;
  scan.text_head = 0;
  scan.tree = tree;
  scan.fail_flag = 0;
  if (json_read_value(&scan) < 0 || scan.fail_flag) {
    json_free(tree);
    return NULL;
  }
  return tree;
}

static const char *json_text_at(const json_tree *tree, int32_t offset) {
  return offset < 0 ? NULL : tree->text_pool + offset;
}

static int32_t json_field(const json_tree *tree, int32_t node_index, const char *name) {
  int32_t child_index;
  if (!tree || node_index < 0 || tree->node_list[node_index].kind != JSON_DICT) return -1;
  for (child_index = tree->node_list[node_index].head_child; child_index >= 0;
       child_index = tree->node_list[child_index].next_peer) {
    const char *child_name = json_text_at(tree, tree->node_list[child_index].name_start);
    if (child_name && strcmp(child_name, name) == 0) return child_index;
  }
  return -1;
}

static int32_t json_item(const json_tree *tree, int32_t node_index, int32_t slot) {
  int32_t child_index;
  if (!tree || node_index < 0) return -1;
  child_index = tree->node_list[node_index].head_child;
  while (child_index >= 0 && slot > 0) {
    child_index = tree->node_list[child_index].next_peer;
    slot -= 1;
  }
  return child_index;
}

static int32_t json_count(const json_tree *tree, int32_t node_index) {
  if (!tree || node_index < 0) return 0;
  return tree->node_list[node_index].child_count;
}

static double json_number(const json_tree *tree, int32_t node_index, double spare) {
  if (!tree || node_index < 0) return spare;
  if (tree->node_list[node_index].kind == JSON_NUMBER ||
      tree->node_list[node_index].kind == JSON_FLAG)
    return tree->node_list[node_index].number_value;
  return spare;
}

static const char *json_text(const json_tree *tree, int32_t node_index) {
  if (!tree || node_index < 0 || tree->node_list[node_index].kind != JSON_TEXT) return NULL;
  return json_text_at(tree, tree->node_list[node_index].text_start);
}

static double json_field_number(const json_tree *tree, int32_t node_index, const char *name,
                                double spare) {
  return json_number(tree, json_field(tree, node_index, name), spare);
}

static const char *json_field_text(const json_tree *tree, int32_t node_index, const char *name) {
  return json_text(tree, json_field(tree, node_index, name));
}

static int json_field_flag(const json_tree *tree, int32_t node_index, const char *name, int spare) {
  int32_t child_index = json_field(tree, node_index, name);
  if (child_index < 0 || tree->node_list[child_index].kind == JSON_NULL) return spare;
  return json_number(tree, child_index, spare) != 0.0;
}

/* ======================================================================== */
/* 4. store layer (safetensors)                                             */
/* ======================================================================== */

typedef enum store_type {
  STORE_VOID = 0,
  STORE_F32,
  STORE_F16,
  STORE_BF16,
  STORE_F64,
  STORE_I8,
  STORE_U8,
  STORE_I16,
  STORE_I32,
  STORE_I64,
  STORE_BOOL
} store_type;

typedef struct store_span {
  const char    *name_text;
  store_type     type_kind;
  int            rank_count;
  int64_t        size_list[4];
  const uint8_t *data_base;
  size_t         data_bytes;
} store_span;

typedef struct store_set {
  file_map   *map_list;
  int         map_count;
  int         map_limit;
  store_span *span_list;
  int         span_count;
  int         span_limit;
  int32_t    *slot_list; /* open addressing index into span_list, -1 when free */
  int         slot_mask;
  char      **name_pool;
  int         name_count;
  int         name_limit;
} store_set;

static store_type store_type_of(const char *name) {
  if (!name) return STORE_VOID;
  if (strcmp(name, "F32") == 0) return STORE_F32;
  if (strcmp(name, "F16") == 0) return STORE_F16;
  if (strcmp(name, "BF16") == 0) return STORE_BF16;
  if (strcmp(name, "F64") == 0) return STORE_F64;
  if (strcmp(name, "I8") == 0) return STORE_I8;
  if (strcmp(name, "U8") == 0) return STORE_U8;
  if (strcmp(name, "I16") == 0) return STORE_I16;
  if (strcmp(name, "I32") == 0 || strcmp(name, "U32") == 0) return STORE_I32;
  if (strcmp(name, "I64") == 0 || strcmp(name, "U64") == 0) return STORE_I64;
  if (strcmp(name, "BOOL") == 0) return STORE_BOOL;
  return STORE_VOID;
}

static size_t store_type_bytes(store_type type_kind) {
  switch (type_kind) {
    case STORE_F32: case STORE_I32: return 4;
    case STORE_F16: case STORE_BF16: case STORE_I16: return 2;
    case STORE_F64: case STORE_I64: return 8;
    case STORE_I8: case STORE_U8: case STORE_BOOL: return 1;
    default: return 0;
  }
}

static uint64_t store_hash(const char *text) {
  uint64_t value = 1469598103934665603ULL;
  while (*text) {
    value ^= (uint64_t)(unsigned char)*text++;
    value *= 1099511628211ULL;
  }
  return value;
}

static app_code store_index(store_set *set) {
  int slot_size = 64;
  int span_index;
  while (slot_size < set->span_count * 2) slot_size *= 2;
  mem_free(set->slot_list);
  set->slot_list = (int32_t *)mem_alloc(sizeof(int32_t) * (size_t)slot_size);
  if (!set->slot_list) return APP_FAIL_MEMORY;
  memset(set->slot_list, 0xFF, sizeof(int32_t) * (size_t)slot_size);
  set->slot_mask = slot_size - 1;
  for (span_index = 0; span_index < set->span_count; ++span_index) {
    int slot = (int)(store_hash(set->span_list[span_index].name_text) & (uint64_t)set->slot_mask);
    while (set->slot_list[slot] >= 0) slot = (slot + 1) & set->slot_mask;
    set->slot_list[slot] = span_index;
  }
  return APP_OKAY;
}

static const store_span *store_find(const store_set *set, const char *name) {
  int slot;
  if (!set->slot_list) return NULL;
  slot = (int)(store_hash(name) & (uint64_t)set->slot_mask);
  while (set->slot_list[slot] >= 0) {
    const store_span *span = &set->span_list[set->slot_list[slot]];
    if (strcmp(span->name_text, name) == 0) return span;
    slot = (slot + 1) & set->slot_mask;
  }
  return NULL;
}

static app_code store_grow(store_set *set, int extra_count) {
  if (set->span_count + extra_count <= set->span_limit) return APP_OKAY;
  {
    int next_limit = set->span_limit ? set->span_limit : 64;
    store_span *next_list;
    while (next_limit < set->span_count + extra_count) next_limit *= 2;
    next_list = (store_span *)mem_alloc(sizeof(store_span) * (size_t)next_limit);
    if (!next_list) return APP_FAIL_MEMORY;
    if (set->span_list) {
      memcpy(next_list, set->span_list, sizeof(store_span) * (size_t)set->span_count);
      mem_free(set->span_list);
    }
    set->span_list = next_list;
    set->span_limit = next_limit;
  }
  return APP_OKAY;
}

static app_code store_keep_name(store_set *set, char *name_text) {
  if (set->name_count == set->name_limit) {
    int next_limit = set->name_limit ? set->name_limit * 2 : 64;
    char **next_pool = (char **)mem_alloc(sizeof(char *) * (size_t)next_limit);
    if (!next_pool) return APP_FAIL_MEMORY;
    if (set->name_pool) {
      memcpy(next_pool, set->name_pool, sizeof(char *) * (size_t)set->name_count);
      mem_free(set->name_pool);
    }
    set->name_pool = next_pool;
    set->name_limit = next_limit;
  }
  set->name_pool[set->name_count++] = name_text;
  return APP_OKAY;
}

/* Reads one .safetensors file and appends its tensors to the set. */
static app_code store_add(store_set *set, const char *path) {
  file_map map;
  uint64_t header_bytes = 0;
  json_tree *tree;
  int32_t root_index, child_index;
  app_code code;

  if (set->map_count == set->map_limit) {
    int next_limit = set->map_limit ? set->map_limit * 2 : 8;
    file_map *next_list = (file_map *)mem_clear(sizeof(file_map) * (size_t)next_limit);
    if (!next_list) return APP_FAIL_MEMORY;
    if (set->map_list) {
      memcpy(next_list, set->map_list, sizeof(file_map) * (size_t)set->map_count);
      mem_free(set->map_list);
    }
    set->map_list = next_list;
    set->map_limit = next_limit;
  }

  code = file_open(path, &map);
  if (code != APP_OKAY) return code;
  if (map.byte_count < 8) { file_close(&map); return APP_FAIL_FORMAT; }
  memcpy(&header_bytes, map.base_data, 8);
  if (header_bytes == 0 || header_bytes + 8 > map.byte_count) {
    file_close(&map);
    return APP_FAIL_FORMAT;
  }
  tree = json_read((const char *)map.base_data + 8, (size_t)header_bytes);
  if (!tree) { file_close(&map); return APP_FAIL_FORMAT; }

  set->map_list[set->map_count++] = map;
  root_index = 0;
  for (child_index = tree->node_list[root_index].head_child; child_index >= 0;
       child_index = tree->node_list[child_index].next_peer) {
    const char *entry_name = json_text_at(tree, tree->node_list[child_index].name_start);
    int32_t type_index, shape_index, range_index;
    store_span span;
    double from_value, upto_value;
    int axis_index;

    if (!entry_name || strcmp(entry_name, "__metadata__") == 0) continue;
    type_index = json_field(tree, child_index, "dtype");
    shape_index = json_field(tree, child_index, "shape");
    range_index = json_field(tree, child_index, "data_offsets");
    if (type_index < 0 || shape_index < 0 || range_index < 0) continue;

    memset(&span, 0, sizeof(span));
    span.type_kind = store_type_of(json_text(tree, type_index));
    span.rank_count = json_count(tree, shape_index);
    if (span.rank_count > 4) { json_free(tree); return APP_FAIL_SUPPORT; }
    for (axis_index = 0; axis_index < span.rank_count; ++axis_index)
      span.size_list[axis_index] =
          (int64_t)json_number(tree, json_item(tree, shape_index, axis_index), 0.0);
    from_value = json_number(tree, json_item(tree, range_index, 0), 0.0);
    upto_value = json_number(tree, json_item(tree, range_index, 1), 0.0);
    if (upto_value < from_value || (uint64_t)upto_value + 8 + header_bytes > map.byte_count) {
      json_free(tree);
      return APP_FAIL_FORMAT;
    }
    span.data_base = map.base_data + 8 + (size_t)header_bytes + (size_t)from_value;
    span.data_bytes = (size_t)(upto_value - from_value);
    {
      char *name_copy = text_copy(entry_name, strlen(entry_name));
      if (!name_copy) { json_free(tree); return APP_FAIL_MEMORY; }
      if (store_keep_name(set, name_copy) != APP_OKAY) { json_free(tree); return APP_FAIL_MEMORY; }
      span.name_text = name_copy;
    }
    if (store_grow(set, 1) != APP_OKAY) { json_free(tree); return APP_FAIL_MEMORY; }
    set->span_list[set->span_count++] = span;
  }
  json_free(tree);
  return APP_OKAY;
}

static void store_close(store_set *set) {
  int index;
  if (!set) return;
  for (index = 0; index < set->map_count; ++index) file_close(&set->map_list[index]);
  for (index = 0; index < set->name_count; ++index) mem_free(set->name_pool[index]);
  mem_free(set->map_list);
  mem_free(set->name_pool);
  mem_free(set->span_list);
  mem_free(set->slot_list);
  memset(set, 0, sizeof(*set));
}

/* Opens every shard in a checkpoint folder, honouring the shard index when present. */
static app_code store_open(store_set *set, const char *folder_path) {
  char path_text[1024];
  memset(set, 0, sizeof(*set));

  path_join(path_text, sizeof(path_text), folder_path, "model.safetensors.index.json");
  if (file_exists(path_text)) {
    size_t text_size = 0;
    char *text_data = file_slurp(path_text, &text_size);
    json_tree *tree;
    int32_t map_index, child_index;
    char **shard_list = NULL;
    int shard_count = 0, shard_index;
    app_code code = APP_OKAY;

    if (!text_data) return APP_FAIL_FILE;
    tree = json_read(text_data, text_size);
    mem_free(text_data);
    if (!tree) return APP_FAIL_FORMAT;
    map_index = json_field(tree, 0, "weight_map");
    shard_list = (char **)mem_clear(sizeof(char *) * 1024);
    if (!shard_list) { json_free(tree); return APP_FAIL_MEMORY; }
    for (child_index = map_index >= 0 ? tree->node_list[map_index].head_child : -1;
         child_index >= 0; child_index = tree->node_list[child_index].next_peer) {
      const char *shard_name = json_text(tree, child_index);
      int seen_flag = 0;
      if (!shard_name) continue;
      for (shard_index = 0; shard_index < shard_count; ++shard_index)
        if (strcmp(shard_list[shard_index], shard_name) == 0) { seen_flag = 1; break; }
      if (!seen_flag && shard_count < 1024)
        shard_list[shard_count++] = text_copy(shard_name, strlen(shard_name));
    }
    json_free(tree);
    for (shard_index = 0; shard_index < shard_count && code == APP_OKAY; ++shard_index) {
      path_join(path_text, sizeof(path_text), folder_path, shard_list[shard_index]);
      code = store_add(set, path_text);
    }
    for (shard_index = 0; shard_index < shard_count; ++shard_index) mem_free(shard_list[shard_index]);
    mem_free(shard_list);
    if (code != APP_OKAY) { store_close(set); return code; }
    return store_index(set);
  }

  path_join(path_text, sizeof(path_text), folder_path, "model.safetensors");
  if (file_exists(path_text)) {
    app_code code = store_add(set, path_text);
    if (code != APP_OKAY) { store_close(set); return code; }
    return store_index(set);
  }
  return APP_FAIL_MISSING;
}

/* ======================================================================== */
/* 5. quant layer                                                           */
/* ======================================================================== */

static float real_from_bf16(uint16_t raw_value) {
  union { uint32_t bits; float real; } cast;
  cast.bits = (uint32_t)raw_value << 16;
  return cast.real;
}

static float real_from_f16(uint16_t raw_value) {
  uint32_t sign_bit = (uint32_t)(raw_value >> 15) & 1u;
  uint32_t exponent = (uint32_t)(raw_value >> 10) & 0x1Fu;
  uint32_t fraction = (uint32_t)raw_value & 0x3FFu;
  union { uint32_t bits; float real; } cast;
  if (exponent == 0) {
    if (fraction == 0) {
      cast.bits = sign_bit << 31;
    } else {
      exponent = 127 - 15 + 1;
      while ((fraction & 0x400u) == 0) { fraction <<= 1; exponent -= 1; }
      fraction &= 0x3FFu;
      cast.bits = (sign_bit << 31) | (exponent << 23) | (fraction << 13);
    }
  } else if (exponent == 31) {
    cast.bits = (sign_bit << 31) | 0x7F800000u | (fraction << 13);
  } else {
    cast.bits = (sign_bit << 31) | ((exponent + 127 - 15) << 23) | (fraction << 13);
  }
  return cast.real;
}

static float real_read(const void *base_data, store_type type_kind, size_t slot) {
  switch (type_kind) {
    case STORE_F32: return ((const float *)base_data)[slot];
    case STORE_BF16: return real_from_bf16(((const uint16_t *)base_data)[slot]);
    case STORE_F16: return real_from_f16(((const uint16_t *)base_data)[slot]);
    case STORE_F64: return (float)((const double *)base_data)[slot];
    case STORE_I8: return (float)((const int8_t *)base_data)[slot];
    case STORE_U8: return (float)((const uint8_t *)base_data)[slot];
    case STORE_I16: return (float)((const int16_t *)base_data)[slot];
    case STORE_I32: return (float)((const int32_t *)base_data)[slot];
    case STORE_I64: return (float)((const int64_t *)base_data)[slot];
    default: return 0.0f;
  }
}

static int64_t whole_read(const void *base_data, store_type type_kind, size_t slot) {
  switch (type_kind) {
    case STORE_I64: return ((const int64_t *)base_data)[slot];
    case STORE_I32: return ((const int32_t *)base_data)[slot];
    case STORE_I16: return ((const int16_t *)base_data)[slot];
    case STORE_I8: return ((const int8_t *)base_data)[slot];
    case STORE_U8: return ((const uint8_t *)base_data)[slot];
    default: return (int64_t)real_read(base_data, type_kind, slot);
  }
}

/* Reads one element from a dense little-endian bit stream of `bit_count` fields. */
static uint32_t pack_read(const uint8_t *code_data, size_t element_index, int bit_count) {
  size_t bit_start = element_index * (size_t)bit_count;
  size_t byte_start = bit_start >> 3;
  int shift_count = (int)(bit_start & 7u);
  uint32_t window = (uint32_t)code_data[byte_start];
  int taken = 8 - shift_count;
  int byte_step = 1;
  while (taken < bit_count) {
    window |= (uint32_t)code_data[byte_start + (size_t)byte_step] << (taken + shift_count);
    taken += 8;
    byte_step += 1;
  }
  return (window >> shift_count) & ((bit_count == 32) ? 0xFFFFFFFFu : ((1u << bit_count) - 1u));
}

/* Storage form of a weight matrix or vector. */
typedef enum plane_form { PLANE_VOID = 0, PLANE_REAL, PLANE_CODE } plane_form;

typedef struct plane {
  plane_form form;
  int        row_count;   /* output features */
  int        col_count;   /* input features */

  const void *real_data;  /* PLANE_REAL payload, row major */
  store_type  real_type;

  const uint8_t *code_data;  /* PLANE_CODE payload, dense bit stream per row */
  size_t         row_stride; /* bytes between rows of code_data */
  const void    *gain_data;  /* [row_count * group_count] */
  store_type     gain_type;
  const int8_t  *bias_data;  /* zero points, owned, NULL when symmetric */
  int            bit_count;
  int            code_bias;  /* offset folded into the stored codes */
  int            group_size;
  int            group_count;

  void *own_block; /* owned auxiliary allocation, freed with the model */
} plane;

static float plane_gain(const plane *sheet, size_t slot) {
  return real_read(sheet->gain_data, sheet->gain_type, slot);
}

/* Decodes a whole row of a plane into f32. */
static void plane_row(const plane *sheet, int row_index, float *row_out) {
  int col_index;
  if (sheet->form == PLANE_REAL) {
    size_t base_slot = (size_t)row_index * (size_t)sheet->col_count;
    if (sheet->real_type == STORE_F32) {
      memcpy(row_out, (const float *)sheet->real_data + base_slot,
             sizeof(float) * (size_t)sheet->col_count);
      return;
    }
    for (col_index = 0; col_index < sheet->col_count; ++col_index)
      row_out[col_index] = real_read(sheet->real_data, sheet->real_type, base_slot + (size_t)col_index);
    return;
  }
  {
    const uint8_t *code_row = sheet->code_data + (size_t)row_index * sheet->row_stride;
    int group_index;
    for (group_index = 0; group_index < sheet->group_count; ++group_index) {
      size_t gain_slot = (size_t)row_index * (size_t)sheet->group_count + (size_t)group_index;
      float gain_value = plane_gain(sheet, gain_slot);
      int bias_value = sheet->bias_data ? sheet->bias_data[gain_slot] : 0;
      int from_index = group_index * sheet->group_size;
      int upto_index = from_index + sheet->group_size;
      if (upto_index > sheet->col_count) upto_index = sheet->col_count;
      for (col_index = from_index; col_index < upto_index; ++col_index) {
        int code_value = (int)pack_read(code_row, (size_t)col_index, sheet->bit_count);
        row_out[col_index] = (float)(code_value - sheet->code_bias - bias_value) * gain_value;
      }
    }
  }
}

/* Rules recovered from `quantization_config` in config.json. */
typedef enum quant_plan { QUANT_NONE = 0, QUANT_TENSOR, QUANT_CHANNEL, QUANT_GROUP } quant_plan;

typedef struct quant_rule {
  int        live_flag;
  int        bit_count;
  int        group_size;
  int        symmetric_flag;
  int        dynamic_flag;
  quant_plan plan_kind;
} quant_rule;

typedef struct quant_book {
  quant_rule weight_rule;
  quant_rule input_rule;
  int        pack_flag; /* checkpoint stores packed codes rather than dense floats */
} quant_book;

static quant_plan quant_plan_of(const char *name) {
  if (!name) return QUANT_NONE;
  if (strcmp(name, "tensor") == 0) return QUANT_TENSOR;
  if (strcmp(name, "channel") == 0) return QUANT_CHANNEL;
  if (strcmp(name, "group") == 0 || strcmp(name, "tensor_group") == 0) return QUANT_GROUP;
  return QUANT_NONE;
}

static void quant_rule_read(const json_tree *tree, int32_t node_index, quant_rule *rule_out) {
  memset(rule_out, 0, sizeof(*rule_out));
  if (node_index < 0 || tree->node_list[node_index].kind != JSON_DICT) return;
  rule_out->live_flag = 1;
  rule_out->bit_count = (int)json_field_number(tree, node_index, "num_bits", 8);
  rule_out->group_size = (int)json_field_number(tree, node_index, "group_size", 0);
  rule_out->symmetric_flag = json_field_flag(tree, node_index, "symmetric", 1);
  rule_out->dynamic_flag = json_field_flag(tree, node_index, "dynamic", 0);
  rule_out->plan_kind = quant_plan_of(json_field_text(tree, node_index, "strategy"));
  if (rule_out->plan_kind == QUANT_NONE)
    rule_out->plan_kind = rule_out->group_size > 0 ? QUANT_GROUP : QUANT_CHANNEL;
}

/* Fake-quantizes activations the way the reference forward does. */
static void quant_act(float *value_list, int value_count, const quant_rule *rule) {
  float low_value = 0.0f, high_value = 0.0f, gain_value, bias_value = 0.0f;
  int value_index;
  int level_span;
  if (!rule->live_flag || rule->bit_count <= 0 || rule->bit_count >= 16) return;
  level_span = (1 << rule->bit_count) - 1;
  for (value_index = 0; value_index < value_count; ++value_index) {
    float value = value_list[value_index];
    if (value < low_value) low_value = value;
    if (value > high_value) high_value = value;
  }
  if (rule->symmetric_flag) {
    float reach = high_value > -low_value ? high_value : -low_value;
    gain_value = reach / (float)((level_span + 1) / 2);
    bias_value = 0.0f;
  } else {
    gain_value = (high_value - low_value) / (float)level_span;
    bias_value = gain_value != 0.0f ? -(float)floor((double)(low_value / gain_value) + 0.5) : 0.0f;
  }
  if (gain_value == 0.0f) return;
  {
    float low_level = rule->symmetric_flag ? -(float)((level_span + 1) / 2) : 0.0f;
    float high_level = rule->symmetric_flag ? (float)(level_span / 2) : (float)level_span;
    for (value_index = 0; value_index < value_count; ++value_index) {
      float level = (float)floor((double)(value_list[value_index] / gain_value) + 0.5) + bias_value;
      if (level < low_level) level = low_level;
      if (level > high_level) level = high_level;
      value_list[value_index] = (level - bias_value) * gain_value;
    }
  }
}

/* ======================================================================== */
/* 6. kernel layer                                                          */
/* ======================================================================== */

static float kern_dot_real(const void *row_data, store_type row_type, const float *act_data,
                           int span_count) {
  int slot;
  float total = 0.0f;
  if (row_type == STORE_F32) {
    const float *row_real = (const float *)row_data;
#if defined(APP_SIMD_AVX2)
    __m256 wide_total = _mm256_setzero_ps();
    for (slot = 0; slot + 8 <= span_count; slot += 8)
      wide_total = _mm256_fmadd_ps(_mm256_loadu_ps(row_real + slot),
                                   _mm256_loadu_ps(act_data + slot), wide_total);
    {
      __m128 half_total = _mm_add_ps(_mm256_castps256_ps128(wide_total),
                                     _mm256_extractf128_ps(wide_total, 1));
      half_total = _mm_hadd_ps(half_total, half_total);
      half_total = _mm_hadd_ps(half_total, half_total);
      total = _mm_cvtss_f32(half_total);
    }
#elif defined(APP_SIMD_NEON)
    float32x4_t wide_total = vdupq_n_f32(0.0f);
    for (slot = 0; slot + 4 <= span_count; slot += 4)
      wide_total = vmlaq_f32(wide_total, vld1q_f32(row_real + slot), vld1q_f32(act_data + slot));
    total = vgetq_lane_f32(wide_total, 0) + vgetq_lane_f32(wide_total, 1) +
            vgetq_lane_f32(wide_total, 2) + vgetq_lane_f32(wide_total, 3);
#else
    slot = 0;
#endif
    for (; slot < span_count; ++slot) total += row_real[slot] * act_data[slot];
    return total;
  }
  if (row_type == STORE_BF16) {
    const uint16_t *row_raw = (const uint16_t *)row_data;
    float part_a = 0.0f, part_b = 0.0f;
    for (slot = 0; slot + 2 <= span_count; slot += 2) {
      part_a += real_from_bf16(row_raw[slot]) * act_data[slot];
      part_b += real_from_bf16(row_raw[slot + 1]) * act_data[slot + 1];
    }
    total = part_a + part_b;
    for (; slot < span_count; ++slot) total += real_from_bf16(row_raw[slot]) * act_data[slot];
    return total;
  }
  for (slot = 0; slot < span_count; ++slot)
    total += real_read(row_data, row_type, (size_t)slot) * act_data[slot];
  return total;
}

/* Sum of activation times code over one quantization group. */
static float kern_dot_code(const uint8_t *code_row, int from_index, int span_count,
                           const float *act_data, int bit_count) {
  float total = 0.0f;
  int slot;
  if (bit_count == 4) {
    const uint8_t *byte_head = code_row + (size_t)from_index / 2;
    float part_a = 0.0f, part_b = 0.0f;
    for (slot = 0; slot + 2 <= span_count; slot += 2) {
      uint8_t pair = byte_head[slot / 2];
      part_a += (float)(pair & 0x0Fu) * act_data[slot];
      part_b += (float)(pair >> 4) * act_data[slot + 1];
    }
    total = part_a + part_b;
    return total;
  }
  if (bit_count == 8) {
    const uint8_t *byte_head = code_row + (size_t)from_index;
    for (slot = 0; slot < span_count; ++slot) total += (float)byte_head[slot] * act_data[slot];
    return total;
  }
  if (bit_count == 2) {
    const uint8_t *byte_head = code_row + (size_t)from_index / 4;
    for (slot = 0; slot + 4 <= span_count; slot += 4) {
      uint8_t quad = byte_head[slot / 4];
      total += (float)(quad & 3u) * act_data[slot];
      total += (float)((quad >> 2) & 3u) * act_data[slot + 1];
      total += (float)((quad >> 4) & 3u) * act_data[slot + 2];
      total += (float)((quad >> 6) & 3u) * act_data[slot + 3];
    }
    return total;
  }
  for (slot = 0; slot < span_count; ++slot)
    total += (float)pack_read(code_row, (size_t)(from_index + slot), bit_count) * act_data[slot];
  return total;
}

static float kern_row_code(const plane *sheet, int row_index, const float *act_data,
                           const float *sum_data) {
  const uint8_t *code_row = sheet->code_data + (size_t)row_index * sheet->row_stride;
  size_t gain_base = (size_t)row_index * (size_t)sheet->group_count;
  float total = 0.0f;
  int group_index;
  for (group_index = 0; group_index < sheet->group_count; ++group_index) {
    int from_index = group_index * sheet->group_size;
    int span_count = sheet->col_count - from_index;
    float gain_value = plane_gain(sheet, gain_base + (size_t)group_index);
    float bias_value = (float)(sheet->code_bias +
                               (sheet->bias_data ? sheet->bias_data[gain_base + (size_t)group_index] : 0));
    float part_value;
    if (span_count > sheet->group_size) span_count = sheet->group_size;
    part_value = kern_dot_code(code_row, from_index, span_count, act_data + from_index,
                               sheet->bit_count);
    total += gain_value * (part_value - bias_value * sum_data[group_index]);
  }
  return total;
}

typedef struct kern_job {
  const plane *sheet;
  const float *act_data;
  const float *sum_data;
  float       *out_data;
} kern_job;

static void kern_mat_vec_band(void *state, int slice_index, int slice_count) {
  kern_job *job = (kern_job *)state;
  const plane *sheet = job->sheet;
  int row_from, row_upto, row_index;
  slice_span(sheet->row_count, slice_index, slice_count, &row_from, &row_upto);
  if (sheet->form == PLANE_REAL) {
    size_t row_span = (size_t)sheet->col_count * store_type_bytes(sheet->real_type);
    for (row_index = row_from; row_index < row_upto; ++row_index)
      job->out_data[row_index] =
          kern_dot_real((const uint8_t *)sheet->real_data + (size_t)row_index * row_span,
                        sheet->real_type, job->act_data, sheet->col_count);
  } else {
    for (row_index = row_from; row_index < row_upto; ++row_index)
      job->out_data[row_index] = kern_row_code(sheet, row_index, job->act_data, job->sum_data);
  }
}

/* Per-group sums of the activation vector, shared by every row of a code plane. */
static void kern_group_sum(const plane *sheet, const float *act_data, float *sum_data) {
  int group_index;
  for (group_index = 0; group_index < sheet->group_count; ++group_index) {
    int from_index = group_index * sheet->group_size;
    int span_count = sheet->col_count - from_index;
    float total = 0.0f;
    int slot;
    if (span_count > sheet->group_size) span_count = sheet->group_size;
    for (slot = 0; slot < span_count; ++slot) total += act_data[from_index + slot];
    sum_data[group_index] = total;
  }
}

static void kern_norm_rms(const float *value_list, const float *gain_list, int value_count,
                          float eps_value, float *value_out) {
  double square_total = 0.0;
  float shrink_value;
  int value_index;
  for (value_index = 0; value_index < value_count; ++value_index)
    square_total += (double)value_list[value_index] * (double)value_list[value_index];
  shrink_value = (float)pow(square_total / (double)value_count + (double)eps_value, -0.5);
  if (gain_list) {
    for (value_index = 0; value_index < value_count; ++value_index)
      value_out[value_index] = value_list[value_index] * shrink_value * gain_list[value_index];
  } else {
    for (value_index = 0; value_index < value_count; ++value_index)
      value_out[value_index] = value_list[value_index] * shrink_value;
  }
}

static float kern_gelu_tanh(float value) {
  const float root_value = 0.7978845608028654f; /* sqrt(2/pi) */
  float cube_value = value * value * value;
  return 0.5f * value * (1.0f + tanhf(root_value * (value + 0.044715f * cube_value)));
}

static void kern_gelu_gate(float *gate_list, const float *rise_list, int value_count) {
  int value_index;
  for (value_index = 0; value_index < value_count; ++value_index)
    gate_list[value_index] = kern_gelu_tanh(gate_list[value_index]) * rise_list[value_index];
}

static void kern_soft_max(float *value_list, int value_count) {
  float peak_value = -FLT_MAX;
  float total_value = 0.0f;
  int value_index;
  for (value_index = 0; value_index < value_count; ++value_index)
    if (value_list[value_index] > peak_value) peak_value = value_list[value_index];
  for (value_index = 0; value_index < value_count; ++value_index) {
    value_list[value_index] = expf(value_list[value_index] - peak_value);
    total_value += value_list[value_index];
  }
  if (total_value > 0.0f) {
    float shrink_value = 1.0f / total_value;
    for (value_index = 0; value_index < value_count; ++value_index)
      value_list[value_index] *= shrink_value;
  }
}

/* Rotary turn using the reference rotate-half convention. */
static void kern_rope_turn(float *value_list, int head_size, const float *cos_list,
                           const float *sin_list) {
  int half_size = head_size / 2;
  int value_index;
  for (value_index = 0; value_index < half_size; ++value_index) {
    float low_value = value_list[value_index];
    float high_value = value_list[value_index + half_size];
    float cos_value = cos_list[value_index];
    float sin_value = sin_list[value_index];
    value_list[value_index] = low_value * cos_value - high_value * sin_value;
    value_list[value_index + half_size] = high_value * cos_value + low_value * sin_value;
  }
}

static void kern_add(float *value_list, const float *other_list, int value_count) {
  int value_index;
  for (value_index = 0; value_index < value_count; ++value_index)
    value_list[value_index] += other_list[value_index];
}

static void kern_scale(float *value_list, float gain_value, int value_count) {
  int value_index;
  for (value_index = 0; value_index < value_count; ++value_index)
    value_list[value_index] *= gain_value;
}

/* Compute is reached only through this table so an accelerator can replace it. */
typedef struct back_desk {
  const char *name_text;
  pool_group *pool_ref;
  float      *sum_room; /* scratch for per-group activation sums */
  int         sum_limit;
  void (*mat_vec)(struct back_desk *desk, const plane *sheet, const float *act_data, float *out_data);
  void (*norm_rms)(struct back_desk *desk, const float *value_list, const float *gain_list,
                   int value_count, float eps_value, float *value_out);
  void (*soft_max)(struct back_desk *desk, float *value_list, int value_count);
  void (*gelu_gate)(struct back_desk *desk, float *gate_list, const float *rise_list, int value_count);
  void (*rope_turn)(struct back_desk *desk, float *value_list, int head_size, const float *cos_list,
                    const float *sin_list);
} back_desk;

static void back_mat_vec(back_desk *desk, const plane *sheet, const float *act_data,
                         float *out_data) {
  kern_job job;
  job.sheet = sheet;
  job.act_data = act_data;
  job.out_data = out_data;
  job.sum_data = NULL;
  if (sheet->form == PLANE_CODE) {
    kern_group_sum(sheet, act_data, desk->sum_room);
    job.sum_data = desk->sum_room;
  }
  if (desk->pool_ref && sheet->row_count >= 64)
    pool_run(desk->pool_ref, kern_mat_vec_band, &job);
  else
    kern_mat_vec_band(&job, 0, 1);
}

static void back_norm_rms(back_desk *desk, const float *value_list, const float *gain_list,
                          int value_count, float eps_value, float *value_out) {
  (void)desk;
  kern_norm_rms(value_list, gain_list, value_count, eps_value, value_out);
}

static void back_soft_max(back_desk *desk, float *value_list, int value_count) {
  (void)desk;
  kern_soft_max(value_list, value_count);
}

static void back_gelu_gate(back_desk *desk, float *gate_list, const float *rise_list,
                           int value_count) {
  (void)desk;
  kern_gelu_gate(gate_list, rise_list, value_count);
}

static void back_rope_turn(back_desk *desk, float *value_list, int head_size, const float *cos_list,
                           const float *sin_list) {
  (void)desk;
  kern_rope_turn(value_list, head_size, cos_list, sin_list);
}

static const char *back_flavor(void) {
#if defined(APP_SIMD_AVX2)
  return "cpu/avx2";
#elif defined(APP_SIMD_SSE2)
  return "cpu/sse2";
#elif defined(APP_SIMD_NEON)
  return "cpu/neon";
#else
  return "cpu/plain";
#endif
}

static app_code back_open(back_desk *desk, pool_group *pool_ref, int sum_limit) {
  memset(desk, 0, sizeof(*desk));
  desk->name_text = back_flavor();
  desk->pool_ref = pool_ref;
  desk->sum_limit = sum_limit > 0 ? sum_limit : 1;
  desk->sum_room = (float *)mem_clear(sizeof(float) * (size_t)desk->sum_limit);
  if (!desk->sum_room) return APP_FAIL_MEMORY;
  desk->mat_vec = back_mat_vec;
  desk->norm_rms = back_norm_rms;
  desk->soft_max = back_soft_max;
  desk->gelu_gate = back_gelu_gate;
  desk->rope_turn = back_rope_turn;
  return APP_OKAY;
}

static void back_close(back_desk *desk) {
  if (!desk) return;
  mem_free(desk->sum_room);
  desk->sum_room = NULL;
}

/* ======================================================================== */
/* 7. model layer                                                           */
/* ======================================================================== */

#define MODEL_LAYER_LIMIT 128
#define MODEL_KIND_SLIDE  0
#define MODEL_KIND_WHOLE  1
#define MODEL_KIND_COUNT  2

typedef struct rope_form {
  double theta_value;
  double part_share; /* partial rotary factor */
  int    turn_count; /* rotated angle pairs */
  int    half_count; /* head_size / 2 */
  float *step_list;  /* inverse frequencies, half_count entries */
} rope_form;

typedef struct model_form {
  int vocab_count;
  int state_size;
  int inner_size;
  int layer_count;
  int head_count;
  int kv_count;
  int head_size;
  int whole_head_size;
  int whole_kv_count;
  int slide_span;
  int ple_vocab;
  int ple_size;
  int share_count;
  int twin_flag;      /* attention_k_eq_v */
  int wide_flag;      /* use_double_wide_mlp */
  int moe_flag;
  int window_limit;
  float norm_eps;
  float logit_cap;
  int start_id;
  int close_id;
  int pad_id;
  char kind_list[MODEL_LAYER_LIMIT];
  rope_form rope_list[MODEL_KIND_COUNT];
} model_form;

typedef struct layer_wing {
  int kind_mark;    /* MODEL_KIND_SLIDE or MODEL_KIND_WHOLE */
  int head_size;
  int kv_count;
  int group_share;  /* head_count / kv_count */
  int share_flag;   /* reads keys and values from another layer */
  int source_slot;  /* layer that owns those keys and values */
  int keep_flag;    /* stores full length keys and values for sharers */
  int inner_size;
  int cache_span;

  plane query_sheet, key_sheet, value_sheet, exit_sheet;
  plane gate_sheet, rise_sheet, drop_sheet;
  plane ple_gate_sheet, ple_lift_sheet;

  float *query_norm, *key_norm;
  float *enter_norm, *after_attn_norm, *before_feed_norm, *after_feed_norm, *after_ple_norm;
  float  layer_gain;
} layer_wing;

typedef struct token_book token_book; /* defined in the token layer */

struct app_model {
  app_setup  setup;
  model_form form;
  store_set  store;
  char      *name_text;
  char      *prefix_text;
  quant_book book;

  plane embed_sheet;      /* token embeddings, also the tied output head */
  plane ple_embed_sheet;  /* per layer embeddings */
  plane ple_lift_sheet;   /* per layer model projection */
  float *ple_norm;
  float *final_norm;
  plane  head_sheet;      /* untied output head when present */
  int    head_own_flag;

  layer_wing *wing_list;
  token_book *book_ref;

  pool_group pool;
  back_desk  desk;
  size_t     weight_bytes;
};

/* -- plane binding ------------------------------------------------------- */

static int plane_bits_of(int col_count, int word_count, int hint_bits) {
  int bit_count;
  if (hint_bits >= 1 && hint_bits <= 8 &&
      (int)(((int64_t)col_count * hint_bits + 31) / 32) == word_count)
    return hint_bits;
  for (bit_count = 1; bit_count <= 8; ++bit_count)
    if ((int)(((int64_t)col_count * bit_count + 31) / 32) == word_count) return bit_count;
  return 0;
}

static int zero_read(const uint32_t *word_list, int group_count, int group_index, int row_index,
                     int bit_count) {
  size_t bit_start = (size_t)row_index * (size_t)bit_count;
  size_t word_index = bit_start >> 5;
  int shift_count = (int)(bit_start & 31u);
  uint32_t window = word_list[word_index * (size_t)group_count + (size_t)group_index] >> shift_count;
  if (shift_count + bit_count > 32)
    window |= word_list[(word_index + 1) * (size_t)group_count + (size_t)group_index]
              << (32 - shift_count);
  return (int)(window & ((1u << bit_count) - 1u)) - (1 << (bit_count - 1));
}

/* Resolves `prefix.name` to either a real or a code plane. */
static app_code plane_bind(app_model *model, const char *stem, plane *sheet_out) {
  char name_text[512];
  const store_span *span;
  memset(sheet_out, 0, sizeof(*sheet_out));

  snprintf(name_text, sizeof(name_text), "%s.weight", stem);
  span = store_find(&model->store, name_text);
  if (span) {
    if (span->rank_count != 2) return APP_FAIL_FORMAT;
    sheet_out->form = PLANE_REAL;
    sheet_out->row_count = (int)span->size_list[0];
    sheet_out->col_count = (int)span->size_list[1];
    sheet_out->real_data = span->data_base;
    sheet_out->real_type = span->type_kind;
    model->weight_bytes += span->data_bytes;
    return APP_OKAY;
  }

  snprintf(name_text, sizeof(name_text), "%s.weight_packed", stem);
  span = store_find(&model->store, name_text);
  if (!span) return APP_FAIL_MISSING;
  {
    const store_span *gain_span;
    const store_span *zero_span;
    const store_span *shape_span;
    int row_count, col_count, word_count, group_count, bit_count;
    char other_text[512];

    if (span->rank_count != 2 || span->type_kind != STORE_I32) return APP_FAIL_SUPPORT;
    row_count = (int)span->size_list[0];
    word_count = (int)span->size_list[1];

    snprintf(other_text, sizeof(other_text), "%s.weight_scale", stem);
    gain_span = store_find(&model->store, other_text);
    if (!gain_span) return APP_FAIL_MISSING;
    snprintf(other_text, sizeof(other_text), "%s.weight_shape", stem);
    shape_span = store_find(&model->store, other_text);
    if (shape_span && shape_span->rank_count >= 1 && shape_span->size_list[0] >= 2)
      col_count = (int)whole_read(shape_span->data_base, shape_span->type_kind, 1);
    else
      col_count = (int)((int64_t)word_count * 32 /
                        (model->book.weight_rule.bit_count ? model->book.weight_rule.bit_count : 4));

    group_count = gain_span->rank_count >= 2 ? (int)gain_span->size_list[1] : 1;
    if (group_count < 1) group_count = 1;
    bit_count = plane_bits_of(col_count, word_count, model->book.weight_rule.bit_count);
    if (bit_count == 0) return APP_FAIL_FORMAT;

    sheet_out->form = PLANE_CODE;
    sheet_out->row_count = row_count;
    sheet_out->col_count = col_count;
    sheet_out->code_data = span->data_base;
    sheet_out->row_stride = (size_t)word_count * 4u;
    sheet_out->gain_data = gain_span->data_base;
    sheet_out->gain_type = gain_span->type_kind;
    sheet_out->bit_count = bit_count;
    sheet_out->code_bias = 1 << (bit_count - 1);
    sheet_out->group_count = group_count;
    sheet_out->group_size = (col_count + group_count - 1) / group_count;
    model->weight_bytes += span->data_bytes + gain_span->data_bytes;

    snprintf(other_text, sizeof(other_text), "%s.weight_zero_point", stem);
    zero_span = store_find(&model->store, other_text);
    if (zero_span && zero_span->type_kind == STORE_I32) {
      int8_t *bias_room = (int8_t *)mem_clear((size_t)row_count * (size_t)group_count);
      int row_index, group_index;
      if (!bias_room) return APP_FAIL_MEMORY;
      for (row_index = 0; row_index < row_count; ++row_index)
        for (group_index = 0; group_index < group_count; ++group_index)
          bias_room[(size_t)row_index * (size_t)group_count + (size_t)group_index] =
              (int8_t)zero_read((const uint32_t *)zero_span->data_base, group_count, group_index,
                                row_index, bit_count);
      sheet_out->bias_data = bias_room;
      sheet_out->own_block = bias_room;
    }
  }
  return APP_OKAY;
}

static app_code plane_bind_at(app_model *model, const char *shape_text, int layer_index,
                              const char *leaf, plane *sheet_out) {
  char stem_text[512];
  snprintf(stem_text, sizeof(stem_text), shape_text, model->prefix_text, layer_index, leaf);
  return plane_bind(model, stem_text, sheet_out);
}

/* Materializes a one dimensional weight, such as a norm gain, into f32. */
static float *vec_bind(app_model *model, const char *stem, int span_hint) {
  char name_text[512];
  const store_span *span;
  float *value_list;
  int value_index, value_count;
  snprintf(name_text, sizeof(name_text), "%s.weight", stem);
  span = store_find(&model->store, name_text);
  if (!span) {
    snprintf(name_text, sizeof(name_text), "%s", stem);
    span = store_find(&model->store, name_text);
  }
  if (!span) return NULL;
  value_count = 1;
  for (value_index = 0; value_index < span->rank_count; ++value_index)
    value_count *= (int)span->size_list[value_index];
  if (span_hint > 0 && value_count != span_hint) return NULL;
  value_list = (float *)mem_alloc(sizeof(float) * (size_t)value_count);
  if (!value_list) return NULL;
  for (value_index = 0; value_index < value_count; ++value_index)
    value_list[value_index] = real_read(span->data_base, span->type_kind, (size_t)value_index);
  model->weight_bytes += span->data_bytes;
  return value_list;
}

/* -- configuration ------------------------------------------------------- */

static void rope_build(rope_form *rope, int head_size, const char *type_text) {
  int angle_index;
  int half_count = head_size / 2;
  int turn_count = half_count;
  if (type_text && strcmp(type_text, "proportional") == 0)
    turn_count = (int)((double)rope->part_share * head_size / 2.0);
  if (turn_count > half_count) turn_count = half_count;
  if (turn_count < 0) turn_count = 0;
  rope->half_count = half_count;
  rope->turn_count = turn_count;
  rope->step_list = (float *)mem_clear(sizeof(float) * (size_t)(half_count > 0 ? half_count : 1));
  if (!rope->step_list) return;
  if (type_text && strcmp(type_text, "proportional") == 0) {
    for (angle_index = 0; angle_index < turn_count; ++angle_index)
      rope->step_list[angle_index] =
          (float)(1.0 / pow(rope->theta_value, (double)(2 * angle_index) / (double)head_size));
  } else {
    for (angle_index = 0; angle_index < half_count; ++angle_index)
      rope->step_list[angle_index] =
          (float)(1.0 / pow(rope->theta_value, (double)(2 * angle_index) / (double)head_size));
    rope->turn_count = half_count;
  }
}

static void rope_wave(const rope_form *rope, int place_index, float *cos_out, float *sin_out) {
  int angle_index;
  for (angle_index = 0; angle_index < rope->half_count; ++angle_index) {
    double angle_value = (double)place_index * (double)rope->step_list[angle_index];
    cos_out[angle_index] = (float)cos(angle_value);
    sin_out[angle_index] = (float)sin(angle_value);
  }
}

static int32_t config_text_node(const json_tree *tree) {
  int32_t node_index = json_field(tree, 0, "text_config");
  return node_index >= 0 ? node_index : 0;
}

static void config_rope_read(const json_tree *tree, int32_t text_index, const char *kind_text,
                             rope_form *rope_out, double spare_theta, const char **type_out) {
  int32_t rope_index = json_field(tree, text_index, "rope_parameters");
  int32_t kind_index = rope_index >= 0 ? json_field(tree, rope_index, kind_text) : -1;
  const char *type_text = NULL;
  memset(rope_out, 0, sizeof(*rope_out));
  rope_out->part_share = 1.0;
  rope_out->theta_value = spare_theta;
  if (kind_index >= 0) {
    type_text = json_field_text(tree, kind_index, "rope_type");
    rope_out->theta_value = json_field_number(tree, kind_index, "rope_theta", spare_theta);
    rope_out->part_share = json_field_number(tree, kind_index, "partial_rotary_factor", 1.0);
  } else if (rope_index >= 0 && json_field(tree, rope_index, "rope_theta") >= 0) {
    type_text = json_field_text(tree, rope_index, "rope_type");
    rope_out->theta_value = json_field_number(tree, rope_index, "rope_theta", spare_theta);
    rope_out->part_share = json_field_number(tree, rope_index, "partial_rotary_factor", 1.0);
  } else {
    /* Fall back to the family defaults recorded in the reference configuration. */
    if (strcmp(kind_text, "full_attention") == 0) {
      type_text = "proportional";
      rope_out->part_share = 0.25;
    } else {
      type_text = "default";
    }
  }
  *type_out = type_text ? type_text : "default";
}

static app_code config_read(app_model *model, const char *folder_path) {
  char path_text[1024];
  size_t text_size = 0;
  char *text_data;
  json_tree *tree;
  int32_t text_index;
  model_form *form = &model->form;
  int layer_index;

  path_join(path_text, sizeof(path_text), folder_path, "config.json");
  text_data = file_slurp(path_text, &text_size);
  if (!text_data) return APP_FAIL_MISSING;
  tree = json_read(text_data, text_size);
  mem_free(text_data);
  if (!tree) return APP_FAIL_FORMAT;

  text_index = config_text_node(tree);
  memset(form, 0, sizeof(*form));
  form->vocab_count = (int)json_field_number(tree, text_index, "vocab_size", 262144);
  form->state_size = (int)json_field_number(tree, text_index, "hidden_size", 2304);
  form->inner_size = (int)json_field_number(tree, text_index, "intermediate_size", 9216);
  form->layer_count = (int)json_field_number(tree, text_index, "num_hidden_layers", 30);
  form->head_count = (int)json_field_number(tree, text_index, "num_attention_heads", 8);
  form->kv_count = (int)json_field_number(tree, text_index, "num_key_value_heads", 4);
  form->head_size = (int)json_field_number(tree, text_index, "head_dim", 256);
  form->whole_head_size = (int)json_field_number(tree, text_index, "global_head_dim", 512);
  form->whole_kv_count =
      (int)json_field_number(tree, text_index, "num_global_key_value_heads", form->kv_count);
  form->slide_span = (int)json_field_number(tree, text_index, "sliding_window", 512);
  form->ple_vocab = (int)json_field_number(tree, text_index, "vocab_size_per_layer_input", 262144);
  form->ple_size = (int)json_field_number(tree, text_index, "hidden_size_per_layer_input", 256);
  form->share_count = (int)json_field_number(tree, text_index, "num_kv_shared_layers", 0);
  form->twin_flag = json_field_flag(tree, text_index, "attention_k_eq_v", 0);
  form->wide_flag = json_field_flag(tree, text_index, "use_double_wide_mlp", 0);
  form->moe_flag = json_field_flag(tree, text_index, "enable_moe_block", 0);
  form->norm_eps = (float)json_field_number(tree, text_index, "rms_norm_eps", 1e-6);
  form->logit_cap = (float)json_field_number(tree, text_index, "final_logit_softcapping", 0.0);
  form->window_limit = (int)json_field_number(tree, text_index, "max_position_embeddings", 131072);
  form->start_id = (int)json_field_number(tree, text_index, "bos_token_id", 2);
  form->pad_id = (int)json_field_number(tree, text_index, "pad_token_id", 0);
  {
    int32_t close_index = json_field(tree, text_index, "eos_token_id");
    if (close_index >= 0 && tree->node_list[close_index].kind == JSON_LIST)
      form->close_id = (int)json_number(tree, json_item(tree, close_index, 0), 1.0);
    else
      form->close_id = (int)json_number(tree, close_index, 1.0);
  }
  if (form->layer_count > MODEL_LAYER_LIMIT) { json_free(tree); return APP_FAIL_SUPPORT; }
  if (form->moe_flag) { json_free(tree); return APP_FAIL_SUPPORT; }

  {
    int32_t kind_index = json_field(tree, text_index, "layer_types");
    for (layer_index = 0; layer_index < form->layer_count; ++layer_index) {
      const char *kind_text = kind_index >= 0 ? json_text(tree, json_item(tree, kind_index, layer_index)) : NULL;
      if (kind_text)
        form->kind_list[layer_index] =
            strcmp(kind_text, "sliding_attention") == 0 ? MODEL_KIND_SLIDE : MODEL_KIND_WHOLE;
      else
        form->kind_list[layer_index] = ((layer_index + 1) % 6) ? MODEL_KIND_SLIDE : MODEL_KIND_WHOLE;
    }
    form->kind_list[form->layer_count - 1] = MODEL_KIND_WHOLE;
  }

  {
    const char *type_text = NULL;
    config_rope_read(tree, text_index, "sliding_attention", &form->rope_list[MODEL_KIND_SLIDE],
                     10000.0, &type_text);
    rope_build(&form->rope_list[MODEL_KIND_SLIDE], form->head_size, type_text);
    config_rope_read(tree, text_index, "full_attention", &form->rope_list[MODEL_KIND_WHOLE],
                     1000000.0, &type_text);
    rope_build(&form->rope_list[MODEL_KIND_WHOLE], form->whole_head_size, type_text);
  }

  {
    int32_t quant_index = json_field(tree, 0, "quantization_config");
    memset(&model->book, 0, sizeof(model->book));
    if (quant_index >= 0) {
      int32_t group_index = json_field(tree, quant_index, "config_groups");
      int32_t first_index = json_item(tree, group_index, 0);
      const char *form_text = json_field_text(tree, quant_index, "format");
      model->book.pack_flag = form_text && strstr(form_text, "pack") != NULL;
      if (first_index >= 0) {
        quant_rule_read(tree, json_field(tree, first_index, "weights"), &model->book.weight_rule);
        quant_rule_read(tree, json_field(tree, first_index, "input_activations"),
                        &model->book.input_rule);
      }
    }
  }

  json_free(tree);
  if (model->setup.window_limit > 0 && model->setup.window_limit < form->window_limit)
    form->window_limit = model->setup.window_limit;
  if (form->window_limit < 8) form->window_limit = 8;
  return APP_OKAY;
}

/* -- weight binding ------------------------------------------------------ */

static const char *model_prefix_pick(app_model *model) {
  static const char *candidate_list[] = {"model.language_model.", "language_model.model.", "model.",
                                         ""};
  char name_text[512];
  size_t candidate_index;
  for (candidate_index = 0; candidate_index < sizeof(candidate_list) / sizeof(candidate_list[0]);
       ++candidate_index) {
    snprintf(name_text, sizeof(name_text), "%sembed_tokens.weight", candidate_list[candidate_index]);
    if (store_find(&model->store, name_text)) return candidate_list[candidate_index];
    snprintf(name_text, sizeof(name_text), "%sembed_tokens.weight_packed",
             candidate_list[candidate_index]);
    if (store_find(&model->store, name_text)) return candidate_list[candidate_index];
  }
  return NULL;
}

static app_code model_bind(app_model *model) {
  model_form *form = &model->form;
  char stem_text[512];
  int layer_index;
  int first_share = form->layer_count - form->share_count;
  app_code code;

  snprintf(stem_text, sizeof(stem_text), "%sembed_tokens", model->prefix_text);
  code = plane_bind(model, stem_text, &model->embed_sheet);
  if (code != APP_OKAY) return code;
  if (model->embed_sheet.row_count > 0) form->vocab_count = model->embed_sheet.row_count;
  if (model->embed_sheet.col_count > 0) form->state_size = model->embed_sheet.col_count;

  snprintf(stem_text, sizeof(stem_text), "%sembed_tokens_per_layer", model->prefix_text);
  code = plane_bind(model, stem_text, &model->ple_embed_sheet);
  if (code != APP_OKAY) return code;
  form->ple_size = model->ple_embed_sheet.col_count / form->layer_count;

  snprintf(stem_text, sizeof(stem_text), "%sper_layer_model_projection", model->prefix_text);
  code = plane_bind(model, stem_text, &model->ple_lift_sheet);
  if (code != APP_OKAY) return code;

  snprintf(stem_text, sizeof(stem_text), "%sper_layer_projection_norm", model->prefix_text);
  model->ple_norm = vec_bind(model, stem_text, form->ple_size);
  snprintf(stem_text, sizeof(stem_text), "%snorm", model->prefix_text);
  model->final_norm = vec_bind(model, stem_text, form->state_size);
  if (!model->ple_norm || !model->final_norm) return APP_FAIL_MISSING;

  {
    static const char *head_list[] = {"lm_head", "model.lm_head", "language_model.lm_head"};
    size_t head_index;
    model->head_own_flag = 0;
    for (head_index = 0; head_index < sizeof(head_list) / sizeof(head_list[0]); ++head_index) {
      if (plane_bind(model, head_list[head_index], &model->head_sheet) == APP_OKAY) {
        model->head_own_flag = 1;
        break;
      }
    }
    if (!model->head_own_flag) model->head_sheet = model->embed_sheet;
  }

  model->wing_list = (layer_wing *)mem_clear(sizeof(layer_wing) * (size_t)form->layer_count);
  if (!model->wing_list) return APP_FAIL_MEMORY;

  for (layer_index = 0; layer_index < form->layer_count; ++layer_index) {
    layer_wing *wing = &model->wing_list[layer_index];
    int kind_mark = form->kind_list[layer_index];
    int probe_index;

    wing->kind_mark = kind_mark;
    wing->share_flag = form->share_count > 0 && layer_index >= first_share;
    wing->source_slot = -1;
    wing->inner_size = form->inner_size;

    if (wing->share_flag) {
      for (probe_index = first_share - 1; probe_index >= 0; --probe_index)
        if (form->kind_list[probe_index] == kind_mark) { wing->source_slot = probe_index; break; }
      if (wing->source_slot < 0) return APP_FAIL_FORMAT;
      if (form->wide_flag) wing->inner_size = form->inner_size * 2;
    } else {
      int last_slot = -1;
      for (probe_index = 0; probe_index < first_share; ++probe_index)
        if (form->kind_list[probe_index] == kind_mark) last_slot = probe_index;
      wing->keep_flag = form->share_count > 0 && last_slot == layer_index;
    }

    code = plane_bind_at(model, "%slayers.%d.self_attn.%s", layer_index, "q_proj", &wing->query_sheet);
    if (code != APP_OKAY) return code;
    code = plane_bind_at(model, "%slayers.%d.%s", layer_index, "mlp.gate_proj", &wing->gate_sheet);
    if (code != APP_OKAY) return code;
    code = plane_bind_at(model, "%slayers.%d.%s", layer_index, "mlp.up_proj", &wing->rise_sheet);
    if (code != APP_OKAY) return code;
    code = plane_bind_at(model, "%slayers.%d.%s", layer_index, "mlp.down_proj", &wing->drop_sheet);
    if (code != APP_OKAY) return code;
    code = plane_bind_at(model, "%slayers.%d.self_attn.%s", layer_index, "o_proj", &wing->exit_sheet);
    if (code != APP_OKAY) return code;

    wing->head_size = wing->query_sheet.row_count / form->head_count;
    wing->inner_size = wing->gate_sheet.row_count;

    if (!wing->share_flag) {
      code = plane_bind_at(model, "%slayers.%d.self_attn.%s", layer_index, "k_proj", &wing->key_sheet);
      if (code != APP_OKAY) return code;
      wing->kv_count = wing->key_sheet.row_count / wing->head_size;
      if (plane_bind_at(model, "%slayers.%d.self_attn.%s", layer_index, "v_proj",
                        &wing->value_sheet) != APP_OKAY)
        memset(&wing->value_sheet, 0, sizeof(wing->value_sheet)); /* keys double as values */
      snprintf(stem_text, sizeof(stem_text), "%slayers.%d.self_attn.k_norm", model->prefix_text,
               layer_index);
      wing->key_norm = vec_bind(model, stem_text, wing->head_size);
      if (!wing->key_norm) return APP_FAIL_MISSING;
    } else {
      wing->kv_count = model->wing_list[wing->source_slot].kv_count;
      wing->head_size = model->wing_list[wing->source_slot].head_size;
    }
    if (wing->kv_count < 1) wing->kv_count = 1;
    wing->group_share = form->head_count / wing->kv_count;

    snprintf(stem_text, sizeof(stem_text), "%slayers.%d.self_attn.q_norm", model->prefix_text,
             layer_index);
    wing->query_norm = vec_bind(model, stem_text, wing->head_size);
    snprintf(stem_text, sizeof(stem_text), "%slayers.%d.input_layernorm", model->prefix_text,
             layer_index);
    wing->enter_norm = vec_bind(model, stem_text, form->state_size);
    snprintf(stem_text, sizeof(stem_text), "%slayers.%d.post_attention_layernorm",
             model->prefix_text, layer_index);
    wing->after_attn_norm = vec_bind(model, stem_text, form->state_size);
    snprintf(stem_text, sizeof(stem_text), "%slayers.%d.pre_feedforward_layernorm",
             model->prefix_text, layer_index);
    wing->before_feed_norm = vec_bind(model, stem_text, form->state_size);
    snprintf(stem_text, sizeof(stem_text), "%slayers.%d.post_feedforward_layernorm",
             model->prefix_text, layer_index);
    wing->after_feed_norm = vec_bind(model, stem_text, form->state_size);
    if (!wing->query_norm || !wing->enter_norm || !wing->after_attn_norm ||
        !wing->before_feed_norm || !wing->after_feed_norm)
      return APP_FAIL_MISSING;

    if (form->ple_size > 0) {
      code = plane_bind_at(model, "%slayers.%d.%s", layer_index, "per_layer_input_gate",
                           &wing->ple_gate_sheet);
      if (code != APP_OKAY) return code;
      code = plane_bind_at(model, "%slayers.%d.%s", layer_index, "per_layer_projection",
                           &wing->ple_lift_sheet);
      if (code != APP_OKAY) return code;
      snprintf(stem_text, sizeof(stem_text), "%slayers.%d.post_per_layer_input_norm",
               model->prefix_text, layer_index);
      wing->after_ple_norm = vec_bind(model, stem_text, form->state_size);
      if (!wing->after_ple_norm) return APP_FAIL_MISSING;
    }

    wing->layer_gain = 1.0f;
    {
      char gain_text[512];
      const store_span *gain_span;
      snprintf(gain_text, sizeof(gain_text), "%slayers.%d.layer_scalar", model->prefix_text,
               layer_index);
      gain_span = store_find(&model->store, gain_text);
      if (gain_span) wing->layer_gain = real_read(gain_span->data_base, gain_span->type_kind, 0);
    }
  }

  /* Cache depth: full length where another layer reads these keys, window otherwise. */
  for (layer_index = 0; layer_index < form->layer_count; ++layer_index) {
    layer_wing *wing = &model->wing_list[layer_index];
    if (wing->share_flag) { wing->cache_span = 0; continue; }
    if (wing->kind_mark == MODEL_KIND_WHOLE || wing->keep_flag)
      wing->cache_span = form->window_limit;
    else
      wing->cache_span = form->slide_span < form->window_limit ? form->slide_span : form->window_limit;
    if (wing->cache_span < 1) wing->cache_span = 1;
  }
  return APP_OKAY;
}

/* ======================================================================== */
/* 8. token layer                                                           */
/* ======================================================================== */

#define TOKEN_SPACE_MARK "\xE2\x96\x81" /* U+2581 lower one eighth block */

typedef struct token_pair {
  const char *left_text;
  const char *right_text;
  int32_t     rank_value;
} token_pair;

struct token_book {
  int      size_count;
  char   **text_list;
  int32_t *slot_list; /* text -> id */
  int      slot_mask;

  token_pair *pair_list;
  int32_t    *pair_slot;
  int         pair_mask;
  int         pair_count;
  char       *pair_pool;
  size_t      pair_size;

  int start_id, close_id, pad_id, unk_id;
  int turn_open_id, turn_shut_id;
  int byte_list[256];
  int prefix_mode; /* 0 never, 1 always, 2 first */
  int start_flag;  /* prepend the sequence marker */
};

static int32_t token_find(const token_book *book, const char *text) {
  int slot;
  if (!book->slot_list) return -1;
  slot = (int)(store_hash(text) & (uint64_t)book->slot_mask);
  while (book->slot_list[slot] >= 0) {
    if (strcmp(book->text_list[book->slot_list[slot]], text) == 0) return book->slot_list[slot];
    slot = (slot + 1) & book->slot_mask;
  }
  return -1;
}

static uint64_t token_pair_hash(const char *left_text, const char *right_text) {
  uint64_t value = store_hash(left_text);
  value ^= store_hash(right_text) * 1099511628211ULL;
  return value;
}

static int32_t token_rank(const token_book *book, const char *left_text, const char *right_text) {
  int slot;
  if (!book->pair_slot) return -1;
  slot = (int)(token_pair_hash(left_text, right_text) & (uint64_t)book->pair_mask);
  while (book->pair_slot[slot] >= 0) {
    const token_pair *pair = &book->pair_list[book->pair_slot[slot]];
    if (strcmp(pair->left_text, left_text) == 0 && strcmp(pair->right_text, right_text) == 0)
      return pair->rank_value;
    slot = (slot + 1) & book->pair_mask;
  }
  return -1;
}

static void token_free(token_book *book) {
  int text_index;
  if (!book) return;
  if (book->text_list) {
    for (text_index = 0; text_index < book->size_count; ++text_index) mem_free(book->text_list[text_index]);
    mem_free(book->text_list);
  }
  mem_free(book->slot_list);
  mem_free(book->pair_list);
  mem_free(book->pair_slot);
  mem_free(book->pair_pool);
  mem_free(book);
}

static int token_hex_value(char letter) {
  if (letter >= '0' && letter <= '9') return letter - '0';
  if (letter >= 'a' && letter <= 'f') return letter - 'a' + 10;
  if (letter >= 'A' && letter <= 'F') return letter - 'A' + 10;
  return -1;
}

static app_code token_load(const char *folder_path, int vocab_hint, token_book **book_out) {
  char path_text[1024];
  size_t text_size = 0;
  char *text_data;
  json_tree *tree;
  token_book *book;
  int32_t model_index, vocab_index, merge_index, added_index, child_index;
  int slot_size, text_index;

  *book_out = NULL;
  path_join(path_text, sizeof(path_text), folder_path, "tokenizer.json");
  text_data = file_slurp(path_text, &text_size);
  if (!text_data) return APP_FAIL_MISSING;
  tree = json_read(text_data, text_size);
  mem_free(text_data);
  if (!tree) return APP_FAIL_FORMAT;

  book = (token_book *)mem_clear(sizeof(token_book));
  if (!book) { json_free(tree); return APP_FAIL_MEMORY; }
  for (text_index = 0; text_index < 256; ++text_index) book->byte_list[text_index] = -1;
  book->start_id = book->close_id = book->pad_id = book->unk_id = -1;
  book->turn_open_id = book->turn_shut_id = -1;

  model_index = json_field(tree, 0, "model");
  vocab_index = json_field(tree, model_index, "vocab");
  if (vocab_index < 0) { json_free(tree); token_free(book); return APP_FAIL_FORMAT; }

  book->size_count = json_count(tree, vocab_index);
  if (book->size_count < vocab_hint) book->size_count = vocab_hint;
  book->text_list = (char **)mem_clear(sizeof(char *) * (size_t)book->size_count);
  if (!book->text_list) { json_free(tree); token_free(book); return APP_FAIL_MEMORY; }
  for (child_index = tree->node_list[vocab_index].head_child; child_index >= 0;
       child_index = tree->node_list[child_index].next_peer) {
    const char *entry_text = json_text_at(tree, tree->node_list[child_index].name_start);
    int entry_id = (int)json_number(tree, child_index, -1.0);
    if (!entry_text || entry_id < 0 || entry_id >= book->size_count) continue;
    if (book->text_list[entry_id]) mem_free(book->text_list[entry_id]);
    book->text_list[entry_id] = text_copy(entry_text, strlen(entry_text));
  }

  added_index = json_field(tree, 0, "added_tokens");
  for (child_index = added_index >= 0 ? tree->node_list[added_index].head_child : -1;
       child_index >= 0; child_index = tree->node_list[child_index].next_peer) {
    const char *entry_text = json_field_text(tree, child_index, "content");
    int entry_id = (int)json_field_number(tree, child_index, "id", -1.0);
    if (!entry_text || entry_id < 0 || entry_id >= book->size_count) continue;
    if (!book->text_list[entry_id]) book->text_list[entry_id] = text_copy(entry_text, strlen(entry_text));
  }

  slot_size = 64;
  while (slot_size < book->size_count * 2) slot_size *= 2;
  book->slot_list = (int32_t *)mem_alloc(sizeof(int32_t) * (size_t)slot_size);
  if (!book->slot_list) { json_free(tree); token_free(book); return APP_FAIL_MEMORY; }
  memset(book->slot_list, 0xFF, sizeof(int32_t) * (size_t)slot_size);
  book->slot_mask = slot_size - 1;
  for (text_index = 0; text_index < book->size_count; ++text_index) {
    int slot;
    if (!book->text_list[text_index]) continue;
    slot = (int)(store_hash(book->text_list[text_index]) & (uint64_t)book->slot_mask);
    while (book->slot_list[slot] >= 0) slot = (slot + 1) & book->slot_mask;
    book->slot_list[slot] = text_index;
  }

  merge_index = json_field(tree, model_index, "merges");
  book->pair_count = json_count(tree, merge_index);
  if (book->pair_count > 0) {
    int pair_slot_size = 64;
    int pair_index = 0;
    size_t pool_size = 0;
    book->pair_list = (token_pair *)mem_clear(sizeof(token_pair) * (size_t)book->pair_count);
    while (pair_slot_size < book->pair_count * 2) pair_slot_size *= 2;
    book->pair_slot = (int32_t *)mem_alloc(sizeof(int32_t) * (size_t)pair_slot_size);
    if (!book->pair_list || !book->pair_slot) { json_free(tree); token_free(book); return APP_FAIL_MEMORY; }
    memset(book->pair_slot, 0xFF, sizeof(int32_t) * (size_t)pair_slot_size);
    book->pair_mask = pair_slot_size - 1;

    for (child_index = tree->node_list[merge_index].head_child; child_index >= 0;
         child_index = tree->node_list[child_index].next_peer) {
      if (tree->node_list[child_index].kind == JSON_TEXT) {
        const char *entry_text = json_text(tree, child_index);
        pool_size += strlen(entry_text) + 2;
      } else {
        const char *left_text = json_text(tree, json_item(tree, child_index, 0));
        const char *right_text = json_text(tree, json_item(tree, child_index, 1));
        pool_size += (left_text ? strlen(left_text) : 0) + (right_text ? strlen(right_text) : 0) + 2;
      }
    }
    book->pair_pool = (char *)mem_alloc(pool_size + 4);
    if (!book->pair_pool) { json_free(tree); token_free(book); return APP_FAIL_MEMORY; }

    for (child_index = tree->node_list[merge_index].head_child; child_index >= 0;
         child_index = tree->node_list[child_index].next_peer) {
      const char *left_text = NULL;
      const char *right_text = NULL;
      char *left_copy;
      char *right_copy;
      size_t left_size, right_size;
      if (tree->node_list[child_index].kind == JSON_TEXT) {
        const char *entry_text = json_text(tree, child_index);
        const char *break_mark = entry_text ? strchr(entry_text, ' ') : NULL;
        if (!break_mark) continue;
        left_size = (size_t)(break_mark - entry_text);
        right_text = break_mark + 1;
        right_size = strlen(right_text);
        left_text = entry_text;
      } else {
        left_text = json_text(tree, json_item(tree, child_index, 0));
        right_text = json_text(tree, json_item(tree, child_index, 1));
        if (!left_text || !right_text) continue;
        left_size = strlen(left_text);
        right_size = strlen(right_text);
      }
      left_copy = book->pair_pool + book->pair_size;
      memcpy(left_copy, left_text, left_size);
      left_copy[left_size] = '\0';
      book->pair_size += left_size + 1;
      right_copy = book->pair_pool + book->pair_size;
      memcpy(right_copy, right_text, right_size);
      right_copy[right_size] = '\0';
      book->pair_size += right_size + 1;

      book->pair_list[pair_index].left_text = left_copy;
      book->pair_list[pair_index].right_text = right_copy;
      book->pair_list[pair_index].rank_value = pair_index;
      {
        int slot = (int)(token_pair_hash(left_copy, right_copy) & (uint64_t)book->pair_mask);
        while (book->pair_slot[slot] >= 0) slot = (slot + 1) & book->pair_mask;
        book->pair_slot[slot] = pair_index;
      }
      pair_index += 1;
    }
    book->pair_count = pair_index;
  }

  /* Metaspace behaviour and special identifiers. */
  book->prefix_mode = 0;
  {
    int32_t pre_index = json_field(tree, 0, "pre_tokenizer");
    const char *type_text = json_field_text(tree, pre_index, "type");
    const char *scheme_text = json_field_text(tree, pre_index, "prepend_scheme");
    if (type_text && strcmp(type_text, "Metaspace") == 0) {
      if (scheme_text && strcmp(scheme_text, "always") == 0) book->prefix_mode = 1;
      else if (scheme_text && strcmp(scheme_text, "first") == 0) book->prefix_mode = 2;
    }
    if (json_field(tree, 0, "normalizer") >= 0) {
      const char *norm_text = json_field_text(tree, json_field(tree, 0, "normalizer"), "type");
      if (norm_text && strcmp(norm_text, "Prepend") == 0) book->prefix_mode = 1;
    }
  }
  json_free(tree);

  for (text_index = 0; text_index < book->size_count; ++text_index) {
    const char *entry_text = book->text_list[text_index];
    if (!entry_text) continue;
    if (entry_text[0] == '<' && entry_text[1] == '0' && entry_text[2] == 'x' &&
        entry_text[3] && entry_text[4] && entry_text[5] == '>' && entry_text[6] == '\0') {
      int high_part = token_hex_value(entry_text[3]);
      int low_part = token_hex_value(entry_text[4]);
      if (high_part >= 0 && low_part >= 0) book->byte_list[high_part * 16 + low_part] = text_index;
    }
  }
  book->start_id = token_find(book, "<bos>");
  book->close_id = token_find(book, "<eos>");
  book->pad_id = token_find(book, "<pad>");
  book->unk_id = token_find(book, "<unk>");
  book->turn_open_id = token_find(book, "<start_of_turn>");
  book->turn_shut_id = token_find(book, "<end_of_turn>");
  book->start_flag = 1;

  *book_out = book;
  return APP_OKAY;
}

/* Splits text into UTF-8 runes with spaces mapped onto the metaspace mark. */
static int token_split(const token_book *book, const char *text, int lead_marker, char *pool_data,
                       size_t pool_limit, char **piece_list, int piece_limit) {
  size_t pool_head = 0;
  int piece_count = 0;
  size_t text_head = 0;
  size_t text_size = strlen(text);
  int want_prefix = book->prefix_mode == 1 || (book->prefix_mode == 2 && lead_marker);

  while (text_head < text_size && piece_count < piece_limit) {
    unsigned char lead_byte = (unsigned char)text[text_head];
    size_t rune_size = 1;
    const char *rune_text = text + text_head;
    if (lead_byte >= 0xF0) rune_size = 4;
    else if (lead_byte >= 0xE0) rune_size = 3;
    else if (lead_byte >= 0xC0) rune_size = 2;
    if (text_head + rune_size > text_size) rune_size = 1;

    if (want_prefix && piece_count == 0 && lead_byte != ' ') {
      if (pool_head + sizeof(TOKEN_SPACE_MARK) > pool_limit) break;
      memcpy(pool_data + pool_head, TOKEN_SPACE_MARK, sizeof(TOKEN_SPACE_MARK));
      piece_list[piece_count++] = pool_data + pool_head;
      pool_head += sizeof(TOKEN_SPACE_MARK);
    }
    if (lead_byte == ' ') {
      if (pool_head + sizeof(TOKEN_SPACE_MARK) > pool_limit) break;
      memcpy(pool_data + pool_head, TOKEN_SPACE_MARK, sizeof(TOKEN_SPACE_MARK));
      piece_list[piece_count++] = pool_data + pool_head;
      pool_head += sizeof(TOKEN_SPACE_MARK);
    } else {
      if (pool_head + rune_size + 1 > pool_limit) break;
      memcpy(pool_data + pool_head, rune_text, rune_size);
      pool_data[pool_head + rune_size] = '\0';
      piece_list[piece_count++] = pool_data + pool_head;
      pool_head += rune_size + 1;
    }
    text_head += rune_size;
  }
  return piece_count;
}

static int token_encode_book(const token_book *book, const char *text, int lead_marker,
                             int32_t *id_list, int id_limit) {
  size_t text_size = strlen(text);
  size_t pool_limit = text_size * 6 + 64;
  char *pool_data = (char *)mem_alloc(pool_limit);
  char *join_pool = (char *)mem_alloc(pool_limit * 2 + 64);
  size_t join_head = 0;
  char **piece_list = (char **)mem_alloc(sizeof(char *) * (text_size + 8));
  int piece_count, piece_index, id_count = 0;

  if (!pool_data || !join_pool || !piece_list) {
    mem_free(pool_data); mem_free(join_pool); mem_free(piece_list);
    return -1;
  }
  piece_count = token_split(book, text, lead_marker, pool_data, pool_limit, piece_list,
                            (int)text_size + 4);

  for (;;) {
    int best_rank = -1, best_slot = -1;
    for (piece_index = 0; piece_index + 1 < piece_count; ++piece_index) {
      int32_t rank_value = token_rank(book, piece_list[piece_index], piece_list[piece_index + 1]);
      if (rank_value >= 0 && (best_rank < 0 || rank_value < best_rank)) {
        best_rank = rank_value;
        best_slot = piece_index;
      }
    }
    if (best_slot < 0) break;
    {
      size_t left_size = strlen(piece_list[best_slot]);
      size_t right_size = strlen(piece_list[best_slot + 1]);
      char *join_text;
      if (join_head + left_size + right_size + 1 > pool_limit * 2 + 64) break;
      join_text = join_pool + join_head;
      memcpy(join_text, piece_list[best_slot], left_size);
      memcpy(join_text + left_size, piece_list[best_slot + 1], right_size);
      join_text[left_size + right_size] = '\0';
      join_head += left_size + right_size + 1;
      piece_list[best_slot] = join_text;
      for (piece_index = best_slot + 1; piece_index + 1 < piece_count; ++piece_index)
        piece_list[piece_index] = piece_list[piece_index + 1];
      piece_count -= 1;
    }
  }

  for (piece_index = 0; piece_index < piece_count && id_count < id_limit; ++piece_index) {
    int32_t id_value = token_find(book, piece_list[piece_index]);
    if (id_value >= 0) {
      id_list[id_count++] = id_value;
      continue;
    }
    {
      const unsigned char *raw_text = (const unsigned char *)piece_list[piece_index];
      size_t raw_index;
      for (raw_index = 0; raw_text[raw_index] && id_count < id_limit; ++raw_index) {
        int byte_id = book->byte_list[raw_text[raw_index]];
        if (byte_id >= 0) id_list[id_count++] = byte_id;
        else if (book->unk_id >= 0) id_list[id_count++] = book->unk_id;
      }
    }
  }

  mem_free(pool_data);
  mem_free(join_pool);
  mem_free(piece_list);
  return id_count;
}

static int token_decode_book(const token_book *book, int32_t id_value, char *text_out,
                             int text_limit) {
  const char *entry_text;
  int out_count = 0;
  size_t text_index = 0;
  if (id_value < 0 || id_value >= book->size_count || !book->text_list[id_value]) return 0;
  entry_text = book->text_list[id_value];
  if (entry_text[0] == '<' && entry_text[1] == '0' && entry_text[2] == 'x' && entry_text[3] &&
      entry_text[4] && entry_text[5] == '>' && entry_text[6] == '\0') {
    int high_part = token_hex_value(entry_text[3]);
    int low_part = token_hex_value(entry_text[4]);
    if (high_part >= 0 && low_part >= 0 && text_limit > 1) {
      text_out[0] = (char)(high_part * 16 + low_part);
      text_out[1] = '\0';
      return 1;
    }
  }
  while (entry_text[text_index] && out_count + 1 < text_limit) {
    if ((unsigned char)entry_text[text_index] == 0xE2 &&
        (unsigned char)entry_text[text_index + 1] == 0x96 &&
        (unsigned char)entry_text[text_index + 2] == 0x81) {
      text_out[out_count++] = ' ';
      text_index += 3;
    } else {
      text_out[out_count++] = entry_text[text_index++];
    }
  }
  text_out[out_count] = '\0';
  return out_count;
}

/* ======================================================================== */
/* 9. session layer                                                         */
/* ======================================================================== */

struct app_session {
  app_model *model;
  int        fill_count;
  app_tally  tally;
  uint64_t   draw_state;

  float **key_store;
  float **value_store;

  float *state_room;
  float *scrap_room;
  float *lift_room;
  float *quant_room;
  float *query_room;
  float *key_room;
  float *value_room;
  float *blend_room;
  float *score_room;
  float *gate_room;
  float *rise_room;
  float *cos_room;
  float *sin_room;
  float *ple_seed;
  float *ple_room;
  float *logit_room;
  void  *pick_room;
  int32_t *echo_room;
  int      echo_count;
  int      echo_limit;
};

/* Applies the activation rule declared by the checkpoint, then multiplies. */
static void session_lift(app_session *session, const plane *sheet, const float *act_data,
                         float *out_data) {
  app_model *model = session->model;
  const float *use_data = act_data;
  if (model->book.input_rule.live_flag && sheet->form == PLANE_CODE) {
    memcpy(session->quant_room, act_data, sizeof(float) * (size_t)sheet->col_count);
    quant_act(session->quant_room, sheet->col_count, &model->book.input_rule);
    use_data = session->quant_room;
  }
  model->desk.mat_vec(&model->desk, sheet, use_data, out_data);
}

static void session_free_rooms(app_session *session) {
  int layer_index;
  if (session->key_store) {
    for (layer_index = 0; layer_index < session->model->form.layer_count; ++layer_index) {
      mem_free(session->key_store[layer_index]);
      mem_free(session->value_store[layer_index]);
    }
    mem_free(session->key_store);
    mem_free(session->value_store);
  }
  mem_free(session->state_room);
  mem_free(session->scrap_room);
  mem_free(session->lift_room);
  mem_free(session->quant_room);
  mem_free(session->query_room);
  mem_free(session->key_room);
  mem_free(session->value_room);
  mem_free(session->blend_room);
  mem_free(session->score_room);
  mem_free(session->gate_room);
  mem_free(session->rise_room);
  mem_free(session->cos_room);
  mem_free(session->sin_room);
  mem_free(session->ple_seed);
  mem_free(session->ple_room);
  mem_free(session->logit_room);
  mem_free(session->pick_room);
  mem_free(session->echo_room);
}

static void session_attend(app_session *session, int layer_index, int place_index,
                           const float *enter_data, float *exit_data) {
  app_model *model = session->model;
  model_form *form = &model->form;
  layer_wing *wing = &model->wing_list[layer_index];
  layer_wing *owner = wing->share_flag ? &model->wing_list[wing->source_slot] : wing;
  int owner_slot = wing->share_flag ? wing->source_slot : layer_index;
  int head_size = wing->head_size;
  int half_size = head_size / 2;
  int kv_width = wing->kv_count * head_size;
  int head_index, place_from, place_upto;

  rope_wave(&form->rope_list[wing->kind_mark], place_index, session->cos_room, session->sin_room);

  session_lift(session, &wing->query_sheet, enter_data, session->query_room);
  for (head_index = 0; head_index < form->head_count; ++head_index) {
    float *head_data = session->query_room + (size_t)head_index * head_size;
    model->desk.norm_rms(&model->desk, head_data, wing->query_norm, head_size, form->norm_eps,
                         head_data);
    model->desk.rope_turn(&model->desk, head_data, head_size, session->cos_room, session->sin_room);
  }

  if (!wing->share_flag) {
    float *key_slot = session->key_store[layer_index] +
                      (size_t)(place_index % wing->cache_span) * (size_t)kv_width;
    float *value_slot = session->value_store[layer_index] +
                        (size_t)(place_index % wing->cache_span) * (size_t)kv_width;
    session_lift(session, &wing->key_sheet, enter_data, session->key_room);
    if (wing->value_sheet.form != PLANE_VOID)
      session_lift(session, &wing->value_sheet, enter_data, session->value_room);
    else
      memcpy(session->value_room, session->key_room, sizeof(float) * (size_t)kv_width);

    for (head_index = 0; head_index < wing->kv_count; ++head_index) {
      float *key_head = session->key_room + (size_t)head_index * head_size;
      float *value_head = session->value_room + (size_t)head_index * head_size;
      model->desk.norm_rms(&model->desk, key_head, wing->key_norm, head_size, form->norm_eps,
                           key_head);
      model->desk.rope_turn(&model->desk, key_head, head_size, session->cos_room, session->sin_room);
      model->desk.norm_rms(&model->desk, value_head, NULL, head_size, form->norm_eps, value_head);
    }
    memcpy(key_slot, session->key_room, sizeof(float) * (size_t)kv_width);
    memcpy(value_slot, session->value_room, sizeof(float) * (size_t)kv_width);
  }

  place_upto = place_index;
  place_from = 0;
  if (wing->kind_mark == MODEL_KIND_SLIDE) {
    place_from = place_index - form->slide_span + 1;
    if (place_from < 0) place_from = 0;
  }
  (void)half_size;

  for (head_index = 0; head_index < form->head_count; ++head_index) {
    const float *query_head = session->query_room + (size_t)head_index * head_size;
    int kv_index = head_index / wing->group_share;
    float *blend_head = session->blend_room + (size_t)head_index * head_size;
    int span_count = place_upto - place_from + 1;
    int span_index, value_index;

    for (span_index = 0; span_index < span_count; ++span_index) {
      int slot_index = (place_from + span_index) % owner->cache_span;
      const float *key_head = session->key_store[owner_slot] +
                              (size_t)slot_index * (size_t)kv_width + (size_t)kv_index * head_size;
      session->score_room[span_index] =
          kern_dot_real(key_head, STORE_F32, query_head, head_size);
    }
    model->desk.soft_max(&model->desk, session->score_room, span_count);

    for (value_index = 0; value_index < head_size; ++value_index) blend_head[value_index] = 0.0f;
    for (span_index = 0; span_index < span_count; ++span_index) {
      int slot_index = (place_from + span_index) % owner->cache_span;
      const float *value_head = session->value_store[owner_slot] +
                                (size_t)slot_index * (size_t)kv_width + (size_t)kv_index * head_size;
      float weight_value = session->score_room[span_index];
      for (value_index = 0; value_index < head_size; ++value_index)
        blend_head[value_index] += weight_value * value_head[value_index];
    }
  }
  session_lift(session, &wing->exit_sheet, session->blend_room, exit_data);
}

static void session_layer(app_session *session, int layer_index, int place_index,
                          const float *ple_data) {
  app_model *model = session->model;
  model_form *form = &model->form;
  layer_wing *wing = &model->wing_list[layer_index];
  int state_size = form->state_size;
  float *state_data = session->state_room;
  float *scrap_data = session->scrap_room;
  float *lift_data = session->lift_room;

  model->desk.norm_rms(&model->desk, state_data, wing->enter_norm, state_size, form->norm_eps,
                       scrap_data);
  session_attend(session, layer_index, place_index, scrap_data, lift_data);
  model->desk.norm_rms(&model->desk, lift_data, wing->after_attn_norm, state_size, form->norm_eps,
                       lift_data);
  kern_add(state_data, lift_data, state_size);

  model->desk.norm_rms(&model->desk, state_data, wing->before_feed_norm, state_size, form->norm_eps,
                       scrap_data);
  session_lift(session, &wing->gate_sheet, scrap_data, session->gate_room);
  session_lift(session, &wing->rise_sheet, scrap_data, session->rise_room);
  model->desk.gelu_gate(&model->desk, session->gate_room, session->rise_room, wing->inner_size);
  session_lift(session, &wing->drop_sheet, session->gate_room, lift_data);
  model->desk.norm_rms(&model->desk, lift_data, wing->after_feed_norm, state_size, form->norm_eps,
                       lift_data);
  kern_add(state_data, lift_data, state_size);

  if (form->ple_size > 0) {
    int value_index;
    session_lift(session, &wing->ple_gate_sheet, state_data, session->gate_room);
    for (value_index = 0; value_index < form->ple_size; ++value_index)
      session->gate_room[value_index] =
          kern_gelu_tanh(session->gate_room[value_index]) * ple_data[value_index];
    session_lift(session, &wing->ple_lift_sheet, session->gate_room, lift_data);
    model->desk.norm_rms(&model->desk, lift_data, wing->after_ple_norm, state_size, form->norm_eps,
                         lift_data);
    kern_add(state_data, lift_data, state_size);
  }
  if (wing->layer_gain != 1.0f) kern_scale(state_data, wing->layer_gain, state_size);
}

/* One token through the whole graph.  `want_logits` skips the output head during prefill. */
static const float *session_pass(app_session *session, int32_t id_value, int want_logits) {
  app_model *model = session->model;
  model_form *form = &model->form;
  int place_index = session->fill_count;
  int layer_index;

  if (id_value < 0 || id_value >= model->embed_sheet.row_count) return NULL;
  if (place_index >= form->window_limit) return NULL;

  plane_row(&model->embed_sheet, id_value, session->state_room);
  kern_scale(session->state_room, (float)sqrt((double)form->state_size), form->state_size);

  if (form->ple_size > 0) {
    int chunk_index;
    float blend_gain = (float)(1.0 / sqrt(2.0));
    plane_row(&model->ple_embed_sheet, id_value % model->ple_embed_sheet.row_count,
              session->ple_seed);
    kern_scale(session->ple_seed, (float)sqrt((double)form->ple_size),
               form->layer_count * form->ple_size);
    session_lift(session, &model->ple_lift_sheet, session->state_room, session->ple_room);
    kern_scale(session->ple_room, (float)(1.0 / sqrt((double)form->state_size)),
               form->layer_count * form->ple_size);
    for (chunk_index = 0; chunk_index < form->layer_count; ++chunk_index) {
      float *chunk_data = session->ple_room + (size_t)chunk_index * form->ple_size;
      const float *seed_data = session->ple_seed + (size_t)chunk_index * form->ple_size;
      int value_index;
      model->desk.norm_rms(&model->desk, chunk_data, model->ple_norm, form->ple_size,
                           form->norm_eps, chunk_data);
      for (value_index = 0; value_index < form->ple_size; ++value_index)
        chunk_data[value_index] = (chunk_data[value_index] + seed_data[value_index]) * blend_gain;
    }
  }

  for (layer_index = 0; layer_index < form->layer_count; ++layer_index)
    session_layer(session, layer_index, place_index,
                  session->ple_room + (size_t)layer_index * form->ple_size);

  session->fill_count += 1;
  if (session->echo_count < session->echo_limit) session->echo_room[session->echo_count++] = id_value;

  if (!want_logits) return NULL;
  model->desk.norm_rms(&model->desk, session->state_room, model->final_norm, form->state_size,
                       form->norm_eps, session->scrap_room);
  session_lift(session, &model->head_sheet, session->scrap_room, session->logit_room);
  if (form->logit_cap > 0.0f) {
    int value_index;
    for (value_index = 0; value_index < model->head_sheet.row_count; ++value_index)
      session->logit_room[value_index] =
          tanhf(session->logit_room[value_index] / form->logit_cap) * form->logit_cap;
  }
  return session->logit_room;
}

static uint64_t draw_next(uint64_t *state) {
  uint64_t value = *state;
  value ^= value << 13;
  value ^= value >> 7;
  value ^= value << 17;
  *state = value;
  return value;
}

static float draw_unit(uint64_t *state) {
  return (float)((draw_next(state) >> 40) / 16777216.0);
}

typedef struct pick_slot {
  float   weight_value;
  int32_t id_value;
} pick_slot;

static int pick_order(const void *left_data, const void *right_data) {
  const pick_slot *left = (const pick_slot *)left_data;
  const pick_slot *right = (const pick_slot *)right_data;
  if (left->weight_value < right->weight_value) return 1;
  if (left->weight_value > right->weight_value) return -1;
  return 0;
}

/* ======================================================================== */
/* 10. public layer                                                         */
/* ======================================================================== */

const char *app_code_text(app_code code) {
  switch (code) {
    case APP_OKAY: return "okay";
    case APP_FAIL_ARGUMENT: return "bad argument";
    case APP_FAIL_MEMORY: return "out of memory";
    case APP_FAIL_FILE: return "file not readable";
    case APP_FAIL_FORMAT: return "malformed data";
    case APP_FAIL_MISSING: return "required piece missing";
    case APP_FAIL_SUPPORT: return "unsupported configuration";
    case APP_FAIL_STATE: return "invalid state";
    default: return "unknown";
  }
}

app_setup app_setup_plain(void) {
  app_setup setup;
  setup.thread_count = 0;
  setup.window_limit = 0;
  setup.verbose_level = 0;
  return setup;
}

app_taste app_taste_plain(void) {
  app_taste taste;
  taste.heat_value = 1.0f;
  taste.top_count = 64;
  taste.top_portion = 0.95f;
  taste.echo_penalty = 1.0f;
  taste.echo_window = 64;
  taste.seed_value = 0x2545F4914F6CDD1DULL;
  return taste;
}

app_code model_load(const char *folder_path, const app_setup *setup, app_model **model_out) {
  app_model *model;
  app_code code;
  const char *prefix_text;
  int thread_count;
  int group_peak = 1;
  int layer_index;

  if (!folder_path || !model_out) return APP_FAIL_ARGUMENT;
  *model_out = NULL;
  model = (app_model *)mem_clear(sizeof(app_model));
  if (!model) return APP_FAIL_MEMORY;
  model->setup = setup ? *setup : app_setup_plain();

  code = config_read(model, folder_path);
  if (code != APP_OKAY) { model_free(model); return code; }
  code = store_open(&model->store, folder_path);
  if (code != APP_OKAY) { model_free(model); return code; }

  prefix_text = model_prefix_pick(model);
  if (!prefix_text) { model_free(model); return APP_FAIL_MISSING; }
  model->prefix_text = text_copy(prefix_text, strlen(prefix_text));
  model->name_text = text_copy(folder_path, strlen(folder_path));
  if (!model->prefix_text || !model->name_text) { model_free(model); return APP_FAIL_MEMORY; }

  code = model_bind(model);
  if (code != APP_OKAY) { model_free(model); return code; }

  code = token_load(folder_path, model->form.vocab_count, &model->book_ref);
  if (code != APP_OKAY) { model_free(model); return code; }
  if (model->book_ref->start_id >= 0) model->form.start_id = model->book_ref->start_id;
  if (model->book_ref->close_id >= 0) model->form.close_id = model->book_ref->close_id;

  thread_count = model->setup.thread_count > 0 ? model->setup.thread_count : host_thread_count();
  if (thread_count > 64) thread_count = 64;
  code = pool_open(&model->pool, thread_count);
  if (code != APP_OKAY) { model_free(model); return code; }

  if (model->embed_sheet.group_count > group_peak) group_peak = model->embed_sheet.group_count;
  if (model->head_sheet.group_count > group_peak) group_peak = model->head_sheet.group_count;
  if (model->ple_lift_sheet.group_count > group_peak) group_peak = model->ple_lift_sheet.group_count;
  for (layer_index = 0; layer_index < model->form.layer_count; ++layer_index) {
    layer_wing *wing = &model->wing_list[layer_index];
    const plane *sheet_list[9];
    int sheet_index;
    sheet_list[0] = &wing->query_sheet; sheet_list[1] = &wing->key_sheet;
    sheet_list[2] = &wing->value_sheet; sheet_list[3] = &wing->exit_sheet;
    sheet_list[4] = &wing->gate_sheet;  sheet_list[5] = &wing->rise_sheet;
    sheet_list[6] = &wing->drop_sheet;  sheet_list[7] = &wing->ple_gate_sheet;
    sheet_list[8] = &wing->ple_lift_sheet;
    for (sheet_index = 0; sheet_index < 9; ++sheet_index)
      if (sheet_list[sheet_index]->group_count > group_peak)
        group_peak = sheet_list[sheet_index]->group_count;
  }
  code = back_open(&model->desk, &model->pool, group_peak);
  if (code != APP_OKAY) { model_free(model); return code; }

  *model_out = model;
  return APP_OKAY;
}

void model_free(app_model *model) {
  int layer_index;
  if (!model) return;
  back_close(&model->desk);
  pool_close(&model->pool);
  token_free(model->book_ref);
  if (model->wing_list) {
    for (layer_index = 0; layer_index < model->form.layer_count; ++layer_index) {
      layer_wing *wing = &model->wing_list[layer_index];
      mem_free(wing->query_sheet.own_block);
      mem_free(wing->key_sheet.own_block);
      mem_free(wing->value_sheet.own_block);
      mem_free(wing->exit_sheet.own_block);
      mem_free(wing->gate_sheet.own_block);
      mem_free(wing->rise_sheet.own_block);
      mem_free(wing->drop_sheet.own_block);
      mem_free(wing->ple_gate_sheet.own_block);
      mem_free(wing->ple_lift_sheet.own_block);
      mem_free(wing->query_norm);
      mem_free(wing->key_norm);
      mem_free(wing->enter_norm);
      mem_free(wing->after_attn_norm);
      mem_free(wing->before_feed_norm);
      mem_free(wing->after_feed_norm);
      mem_free(wing->after_ple_norm);
    }
    mem_free(model->wing_list);
  }
  mem_free(model->embed_sheet.own_block);
  mem_free(model->ple_embed_sheet.own_block);
  mem_free(model->ple_lift_sheet.own_block);
  if (model->head_own_flag) mem_free(model->head_sheet.own_block);
  mem_free(model->ple_norm);
  mem_free(model->final_norm);
  mem_free(model->form.rope_list[MODEL_KIND_SLIDE].step_list);
  mem_free(model->form.rope_list[MODEL_KIND_WHOLE].step_list);
  store_close(&model->store);
  mem_free(model->prefix_text);
  mem_free(model->name_text);
  mem_free(model);
}

const char *model_name(const app_model *model) { return model ? model->name_text : NULL; }
int model_layer_count(const app_model *model) { return model ? model->form.layer_count : 0; }
int model_vocab_count(const app_model *model) { return model ? model->form.vocab_count : 0; }
int model_window_limit(const app_model *model) { return model ? model->form.window_limit : 0; }
size_t model_memory_bytes(const app_model *model) { return model ? model->weight_bytes : 0; }

int token_encode(const app_model *model, const char *text, int lead_marker, int32_t *id_list,
                 int id_limit) {
  if (!model || !model->book_ref || !text || !id_list) return -1;
  return token_encode_book(model->book_ref, text, lead_marker, id_list, id_limit);
}

int token_decode(const app_model *model, int32_t id_value, char *text_out, int text_limit) {
  if (!model || !model->book_ref || !text_out || text_limit < 2) return 0;
  return token_decode_book(model->book_ref, id_value, text_out, text_limit);
}

int token_start_id(const app_model *model) { return model ? model->form.start_id : -1; }
int token_close_id(const app_model *model) { return model ? model->form.close_id : -1; }

int token_is_close(const app_model *model, int32_t id_value) {
  if (!model) return 0;
  if (id_value == model->form.close_id) return 1;
  if (model->book_ref && id_value == model->book_ref->turn_shut_id) return 1;
  return 0;
}

/* Wraps one user turn in the instruction-tuned chat frame. */
int token_frame(const app_model *model, const char *user_text, int32_t *id_list, int id_limit) {
  const token_book *book;
  int id_count = 0;
  int part_count;
  if (!model || !model->book_ref || !user_text || !id_list) return -1;
  book = model->book_ref;
  if (book->start_id >= 0 && id_count < id_limit) id_list[id_count++] = book->start_id;
  if (book->turn_open_id >= 0) {
    if (id_count < id_limit) id_list[id_count++] = book->turn_open_id;
    part_count = token_encode_book(book, "user\n", 0, id_list + id_count, id_limit - id_count);
    if (part_count > 0) id_count += part_count;
    part_count = token_encode_book(book, user_text, 0, id_list + id_count, id_limit - id_count);
    if (part_count > 0) id_count += part_count;
    if (id_count < id_limit && book->turn_shut_id >= 0) id_list[id_count++] = book->turn_shut_id;
    part_count = token_encode_book(book, "\n", 0, id_list + id_count, id_limit - id_count);
    if (part_count > 0) id_count += part_count;
    if (id_count < id_limit) id_list[id_count++] = book->turn_open_id;
    part_count = token_encode_book(book, "model\n", 0, id_list + id_count, id_limit - id_count);
    if (part_count > 0) id_count += part_count;
  } else {
    part_count = token_encode_book(book, user_text, 1, id_list + id_count, id_limit - id_count);
    if (part_count > 0) id_count += part_count;
  }
  return id_count;
}

app_code session_open(app_model *model, app_session **session_out) {
  app_session *session;
  model_form *form;
  int layer_index;
  int head_peak = 0, inner_peak = 0, half_peak = 0, wide_peak;

  if (!model || !session_out) return APP_FAIL_ARGUMENT;
  *session_out = NULL;
  session = (app_session *)mem_clear(sizeof(app_session));
  if (!session) return APP_FAIL_MEMORY;
  session->model = model;
  form = &model->form;

  for (layer_index = 0; layer_index < form->layer_count; ++layer_index) {
    layer_wing *wing = &model->wing_list[layer_index];
    if (wing->head_size > head_peak) head_peak = wing->head_size;
    if (wing->inner_size > inner_peak) inner_peak = wing->inner_size;
  }
  half_peak = head_peak / 2 + 1;
  wide_peak = form->head_count * head_peak;
  if (form->state_size > wide_peak) wide_peak = form->state_size;

  session->key_store = (float **)mem_clear(sizeof(float *) * (size_t)form->layer_count);
  session->value_store = (float **)mem_clear(sizeof(float *) * (size_t)form->layer_count);
  if (!session->key_store || !session->value_store) { session_close(session); return APP_FAIL_MEMORY; }
  for (layer_index = 0; layer_index < form->layer_count; ++layer_index) {
    layer_wing *wing = &model->wing_list[layer_index];
    size_t slot_count;
    if (wing->share_flag) continue;
    slot_count = (size_t)wing->cache_span * (size_t)wing->kv_count * (size_t)wing->head_size;
    session->key_store[layer_index] = (float *)mem_clear(sizeof(float) * slot_count);
    session->value_store[layer_index] = (float *)mem_clear(sizeof(float) * slot_count);
    if (!session->key_store[layer_index] || !session->value_store[layer_index]) {
      session_close(session);
      return APP_FAIL_MEMORY;
    }
  }

  session->state_room = (float *)mem_clear(sizeof(float) * (size_t)form->state_size);
  session->scrap_room = (float *)mem_clear(sizeof(float) * (size_t)form->state_size);
  session->lift_room = (float *)mem_clear(sizeof(float) * (size_t)wide_peak);
  session->quant_room = (float *)mem_clear(sizeof(float) * (size_t)(wide_peak + inner_peak));
  session->query_room = (float *)mem_clear(sizeof(float) * (size_t)(form->head_count * head_peak));
  session->key_room = (float *)mem_clear(sizeof(float) * (size_t)(form->head_count * head_peak));
  session->value_room = (float *)mem_clear(sizeof(float) * (size_t)(form->head_count * head_peak));
  session->blend_room = (float *)mem_clear(sizeof(float) * (size_t)(form->head_count * head_peak));
  session->score_room = (float *)mem_clear(sizeof(float) * (size_t)(form->window_limit + 1));
  session->gate_room = (float *)mem_clear(sizeof(float) * (size_t)(inner_peak + form->ple_size + 1));
  session->rise_room = (float *)mem_clear(sizeof(float) * (size_t)(inner_peak + 1));
  session->cos_room = (float *)mem_clear(sizeof(float) * (size_t)half_peak);
  session->sin_room = (float *)mem_clear(sizeof(float) * (size_t)half_peak);
  session->ple_seed =
      (float *)mem_clear(sizeof(float) * (size_t)(form->layer_count * form->ple_size + 1));
  session->ple_room =
      (float *)mem_clear(sizeof(float) * (size_t)(form->layer_count * form->ple_size + 1));
  session->logit_room = (float *)mem_clear(sizeof(float) * (size_t)model->head_sheet.row_count);
  session->pick_room = mem_clear(sizeof(pick_slot) * (size_t)model->head_sheet.row_count);
  session->echo_limit = form->window_limit;
  session->echo_room = (int32_t *)mem_clear(sizeof(int32_t) * (size_t)session->echo_limit);
  session->draw_state = 0x2545F4914F6CDD1DULL;

  if (!session->state_room || !session->scrap_room || !session->lift_room || !session->quant_room ||
      !session->query_room || !session->key_room || !session->value_room || !session->blend_room ||
      !session->score_room || !session->gate_room || !session->rise_room || !session->cos_room ||
      !session->sin_room || !session->ple_seed || !session->ple_room || !session->logit_room ||
      !session->pick_room || !session->echo_room) {
    session_close(session);
    return APP_FAIL_MEMORY;
  }
  *session_out = session;
  return APP_OKAY;
}

void session_close(app_session *session) {
  if (!session) return;
  session_free_rooms(session);
  mem_free(session);
}

void session_reset(app_session *session) {
  int layer_index;
  if (!session) return;
  session->fill_count = 0;
  session->echo_count = 0;
  memset(&session->tally, 0, sizeof(session->tally));
  for (layer_index = 0; layer_index < session->model->form.layer_count; ++layer_index) {
    layer_wing *wing = &session->model->wing_list[layer_index];
    size_t slot_count;
    if (wing->share_flag || !session->key_store[layer_index]) continue;
    slot_count = (size_t)wing->cache_span * (size_t)wing->kv_count * (size_t)wing->head_size;
    memset(session->key_store[layer_index], 0, sizeof(float) * slot_count);
    memset(session->value_store[layer_index], 0, sizeof(float) * slot_count);
  }
}

/* Consumes every token but the last, which the caller feeds to session_step. */
app_code session_prime(app_session *session, const int32_t *id_list, int id_count) {
  double from_time;
  int id_index;
  if (!session || !id_list || id_count < 1) return APP_FAIL_ARGUMENT;
  if (session->fill_count + id_count > session->model->form.window_limit) return APP_FAIL_STATE;
  from_time = time_now();
  for (id_index = 0; id_index + 1 < id_count; ++id_index)
    if (!session_pass(session, id_list[id_index], 0) && 0) return APP_FAIL_STATE;
  session->tally.prime_seconds += time_now() - from_time;
  session->tally.prime_tokens += (size_t)(id_count > 0 ? id_count - 1 : 0);
  return APP_OKAY;
}

const float *session_step(app_session *session, int32_t id_value) {
  const float *logit_list;
  double from_time;
  if (!session) return NULL;
  from_time = time_now();
  logit_list = session_pass(session, id_value, 1);
  session->tally.serve_seconds += time_now() - from_time;
  session->tally.serve_tokens += 1;
  session->tally.memory_bytes = app_total_bytes;
  return logit_list;
}

int session_fill(const app_session *session) { return session ? session->fill_count : 0; }

app_tally session_tally(const app_session *session) {
  app_tally tally;
  if (!session) { memset(&tally, 0, sizeof(tally)); return tally; }
  tally = session->tally;
  tally.memory_bytes = app_total_bytes;
  return tally;
}

int32_t session_pick(app_session *session, const float *logit_list, const app_taste *taste) {
  app_taste rule;
  int vocab_count;
  pick_slot *slot_list;
  int slot_index, keep_count;
  float total_value = 0.0f, draw_value;

  if (!session || !logit_list) return -1;
  rule = taste ? *taste : app_taste_plain();
  vocab_count = session->model->head_sheet.row_count;
  slot_list = (pick_slot *)session->pick_room;
  if (rule.seed_value) session->draw_state = rule.seed_value ^ (uint64_t)session->fill_count;

  if (rule.heat_value <= 0.0f) {
    int best_slot = 0;
    for (slot_index = 1; slot_index < vocab_count; ++slot_index)
      if (logit_list[slot_index] > logit_list[best_slot]) best_slot = slot_index;
    return best_slot;
  }

  for (slot_index = 0; slot_index < vocab_count; ++slot_index) {
    slot_list[slot_index].weight_value = logit_list[slot_index];
    slot_list[slot_index].id_value = slot_index;
  }
  if (rule.echo_penalty != 1.0f && rule.echo_window > 0) {
    int echo_from = session->echo_count - rule.echo_window;
    int echo_index;
    if (echo_from < 0) echo_from = 0;
    for (echo_index = echo_from; echo_index < session->echo_count; ++echo_index) {
      int32_t id_value = session->echo_room[echo_index];
      if (id_value < 0 || id_value >= vocab_count) continue;
      if (slot_list[id_value].weight_value > 0.0f)
        slot_list[id_value].weight_value /= rule.echo_penalty;
      else
        slot_list[id_value].weight_value *= rule.echo_penalty;
    }
  }

  keep_count = rule.top_count > 0 && rule.top_count < vocab_count ? rule.top_count : vocab_count;
  if (keep_count < vocab_count) {
    /* Partial selection: keep the strongest `keep_count` logits. */
    float bar_value = -FLT_MAX;
    int fill_count = 0;
    for (slot_index = 0; slot_index < vocab_count; ++slot_index) {
      if (fill_count < keep_count) {
        slot_list[fill_count++] = slot_list[slot_index];
        if (fill_count == keep_count) {
          qsort(slot_list, (size_t)keep_count, sizeof(pick_slot), pick_order);
          bar_value = slot_list[keep_count - 1].weight_value;
        }
      } else if (slot_list[slot_index].weight_value > bar_value) {
        slot_list[keep_count - 1] = slot_list[slot_index];
        qsort(slot_list, (size_t)keep_count, sizeof(pick_slot), pick_order);
        bar_value = slot_list[keep_count - 1].weight_value;
      }
    }
  }
  qsort(slot_list, (size_t)keep_count, sizeof(pick_slot), pick_order);

  {
    float peak_value = slot_list[0].weight_value;
    for (slot_index = 0; slot_index < keep_count; ++slot_index) {
      slot_list[slot_index].weight_value =
          expf((slot_list[slot_index].weight_value - peak_value) / rule.heat_value);
      total_value += slot_list[slot_index].weight_value;
    }
  }
  if (total_value <= 0.0f) return slot_list[0].id_value;
  for (slot_index = 0; slot_index < keep_count; ++slot_index)
    slot_list[slot_index].weight_value /= total_value;

  if (rule.top_portion > 0.0f && rule.top_portion < 1.0f) {
    float run_value = 0.0f;
    for (slot_index = 0; slot_index < keep_count; ++slot_index) {
      run_value += slot_list[slot_index].weight_value;
      if (run_value >= rule.top_portion) { keep_count = slot_index + 1; break; }
    }
    total_value = 0.0f;
    for (slot_index = 0; slot_index < keep_count; ++slot_index)
      total_value += slot_list[slot_index].weight_value;
    for (slot_index = 0; slot_index < keep_count; ++slot_index)
      slot_list[slot_index].weight_value /= total_value;
  }

  draw_value = draw_unit(&session->draw_state);
  for (slot_index = 0; slot_index < keep_count; ++slot_index) {
    draw_value -= slot_list[slot_index].weight_value;
    if (draw_value <= 0.0f) return slot_list[slot_index].id_value;
  }
  return slot_list[keep_count - 1].id_value;
}

#endif /* APP_CORE_IMPLEMENTED */

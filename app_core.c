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
 *   7. media layer          - image and audio decode, resize, filterbank
 *   8. model layer          - config, weights, graph, vision and audio towers
 *   9. token layer          - vocabulary, encode, decode
 *  10. session layer        - cache, forward, sample
 *  11. public layer         - entry point bodies
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
  int   cache_bits;    /* key and value cache storage: 0 keeps float, 8 quantizes */
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
  size_t memory_bytes; /* bytes handed out by the engine's allocator */
  size_t serve_bytes;  /* weights and cache read by the decode steps counted */
} app_tally;

/* One image or one clip, after a tower has turned it into embedding rows.  Each
 * row stands in for one placeholder token in the prompt. */
typedef struct app_media {
  float *state_data; /* [row_count][state_size], owned */
  int    row_count;
  int    state_size;
  int    cut_flag;   /* set when the source ran past what the budget allows */
} app_media;

/* Where one image's or one clip's soft tokens sit in a prompt.  The caller says
 * which tower made the rows and how many there are; the frame builder lays the
 * run down with the opener and the closer the processor puts around it, and
 * reports where the run itself begins so the rows can be laid on top of it. */
#define APP_MEDIA_IMAGE 0
#define APP_MEDIA_AUDIO 1

typedef struct app_media_span {
  int kind_mark;  /* APP_MEDIA_IMAGE or APP_MEDIA_AUDIO */
  int row_count;  /* soft tokens the tower produced */
  int place_from; /* set by the frame builder: the first id of the run */
} app_media_span;

typedef struct app_model   app_model;
typedef struct app_session app_session;

/* model layer -------------------------------------------------------------*/
app_code    model_load(const char *folder_path, const app_setup *setup, app_model **model_out);
void        model_free(app_model *model);
const char *model_name(const app_model *model);
int         model_layer_count(const app_model *model);
int         model_vocab_count(const app_model *model);
int         model_window_limit(const app_model *model);
int         model_state_size(const app_model *model);
size_t      model_memory_bytes(const app_model *model);
/* Weight bytes one decode step reads, which on an export with large embedding
 * tables is a fraction of the mapped total `model_memory_bytes` reports. */
size_t      model_decode_bytes(const app_model *model);

/* token layer -------------------------------------------------------------*/
/* `lead_marker` says this text begins a chunk rather than continuing one.  A
 * tokenizer that marks the start of a chunk — a metaspace prefix, a prepend
 * normalizer — marks it only then, which is what the reference does: it splits
 * its input on the special tokens and normalizes each chunk on its own. */
int  token_encode(const app_model *model, const char *text, int lead_marker,
                  int32_t *id_list, int id_limit);
int  token_decode(const app_model *model, int32_t id_value, char *text_out, int text_limit);
int  token_frame(const app_model *model, const char *user_text,
                 int32_t *id_list, int id_limit); /* chat framing for IT models */
/* The same frame with a run of media placeholders between the turn opener and
 * the user's text, which is where a multi-modal template puts them.  Each span
 * is bracketed the way the processor brackets it, and has its `place_from`
 * filled in, so the caller can lay its embedding rows down against the run
 * rather than against the bracket. */
int  token_frame_media(const app_model *model, const char *user_text,
                       app_media_span *span_list, int span_count, int32_t *id_list, int id_limit);
/* Just the media run: for every span an opener, one placeholder per row, and a
 * closer.  `place_base` is what the caller has already written, and is added to
 * each span's `place_from`. */
int  token_media_run(const app_model *model, app_media_span *span_list, int span_count,
                     int place_base, int32_t *id_list, int id_limit);

/* One piece of a turn, for a prompt whose attachments do not all sit in front
 * of the words: either a stretch of the user's text or one of the spans above.
 * A content list is exactly this — a picture, some words, another picture — and
 * the reference's template lays one down in the order it is given. */
#define APP_PART_TEXT  0
#define APP_PART_MEDIA 1

typedef struct app_part {
  int         kind_mark;  /* APP_PART_TEXT or APP_PART_MEDIA */
  const char *text_ref;   /* the words, when this is a text part */
  int         span_index; /* which of the caller's spans, when it is a media part */
} app_part;

/* The same chat frame around an ordered list of parts.  `token_frame_media` is
 * this with every span first and the words last, which is the common case and
 * the only one the command line could express before. */
/* The same frame for a turn that follows one the model has already answered.
 * A session carries the turns before it in its cache, so this lays down the
 * close of the model's turn and then the new one, without opening the document
 * a second time. */
int  token_frame_next(const app_model *model, const app_part *part_list, int part_count,
                      app_media_span *span_list, int span_count, int32_t *id_list, int id_limit);

int  token_frame_parts(const app_model *model, const app_part *part_list, int part_count,
                       app_media_span *span_list, int span_count,
                       int32_t *id_list, int id_limit);
int  token_start_id(const app_model *model);
int  token_close_id(const app_model *model);
int  token_is_close(const app_model *model, int32_t id_value);

/* media layer -------------------------------------------------------------*/
int      model_vision_ready(const app_model *model);
int      model_audio_ready(const app_model *model);
int      model_image_token(const app_model *model);  /* placeholder id, -1 when absent */
int      model_audio_token(const app_model *model);
/* The ids the processor puts either side of a soft token run, and zero when the
 * checkpoint records no pair: an export without them is read as bracketing
 * nothing, and half a pair is treated as none. */
int      model_image_wrap(const app_model *model, int *open_out, int *shut_out);
int      model_audio_wrap(const app_model *model, int *open_out, int *shut_out);
int      model_image_rows(const app_model *model);   /* rows one image produces */
/* The processor gives a clip a budget rather than reading all of it: so many
 * soft tokens, each standing for so many milliseconds.  A longer clip is cut to
 * fit, so these two are the ceiling on what one clip can contribute. */
int      model_audio_rows(const app_model *model);    /* rows one clip may produce */
int      model_audio_span_ms(const app_model *model); /* milliseconds one row stands for */
app_code media_image(app_model *model, const char *path_text, app_media *media_out);
app_code media_audio(app_model *model, const char *path_text, app_media *media_out);
void     media_free(app_media *media);

/* session layer -----------------------------------------------------------*/
app_code     session_open(app_model *model, app_session **session_out);
void         session_close(app_session *session);
void         session_reset(app_session *session);
app_code     session_prime(app_session *session, const int32_t *id_list, int id_count);
/* The same, with an embedding row supplied for every id a tower filled. */
app_code     session_prime_media(app_session *session, const int32_t *id_list, int id_count,
                                 const float *state_list, const uint8_t *state_flag);
const float *session_step(app_session *session, int32_t id_value);
const float *session_step_state(app_session *session, int32_t id_value, const float *state_data);
int32_t      session_pick(app_session *session, const float *logit_list, const app_taste *taste);
int          session_fill(const app_session *session); /* tokens held in cache */

/* The static range the export calibrates for a layer's key or value cache, and
 * the largest magnitude a session has actually put there.  `value_side` picks
 * the value cache over the key cache.  A scale of zero means the checkpoint
 * ships none for that layer. */
float        model_cache_scale(const app_model *model, int layer_index, int value_side);
float        session_cache_peak(const app_session *session, int layer_index, int value_side);

/* Bytes the key and value cache occupies at full span, as this session stores
 * it, and what it would occupy at the storage `cache_bits` names.  Eight counts
 * a byte a value for every layer the export ships a scale for and a float for
 * the rest; zero counts floats throughout. */
/* A session's cache written to a file and read back, so a prompt that took a
 * long time to prime need not be primed again.  What is written is the ids the
 * session has been fed and the rows of its cache that carry anything, so the
 * file is the size of the conversation rather than of the window.  It is host
 * native, and it is refused rather than restored where the shapes it was
 * written from disagree with the session reading it.
 *
 * `kind_mark` says which of two things the file is, because a session cannot
 * tell from the cache alone and the two are resumed differently.  A prompt is
 * saved before its first token is sampled, so it is one id short of the prompt
 * it holds — the id `session_step` was about to be fed — and what it is for is
 * to be matched against the front of a later prompt and primed onto.  A
 * conversation is saved after an answer, holds every id of it, and what it is
 * for is to have the next turn framed onto it.  A caller that read one where it
 * wanted the other would drop an id or repeat a turn, so `session_load` hands
 * the mark back and the caller checks it.
 *
 * `stamp_value` is written beside the cache and handed back unread.  The ids
 * alone cannot say that a picture in a prompt is the same picture — two
 * pictures lay down the same placeholder ids — so what identifies the rest of
 * a prompt is the caller's to decide and to put here.
 *
 * `session_ids` hands back the ids a session holds, which is how a caller finds
 * out whether the prompt it is about to run begins with the one in the file.
 * It returns the count it needs when the room given is too small. */
#define APP_KEEP_PROMPT 0 /* a prompt primed and not yet answered */
#define APP_KEEP_TALK   1 /* a conversation the model has answered in */
app_code    session_save(const app_session *session, const char *path_text, uint64_t stamp_value,
                         int kind_mark);
app_code    session_load(app_session *session, const char *path_text, uint64_t *stamp_out,
                         int *kind_out);
int         session_ids(const app_session *session, int32_t *id_list, int id_limit);

size_t       session_cache_room(const app_session *session);
size_t       session_cache_room_at(const app_session *session, int cache_bits);
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

/* Copies a name into a fixed slot, truncating rather than overrunning. */
static void text_fill(char *slot_data, size_t slot_limit, const char *text) {
  size_t length = text ? strlen(text) : 0;
  if (length >= slot_limit) length = slot_limit - 1;
  if (length > 0) memcpy(slot_data, text, length);
  slot_data[length] = '\0';
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

/* How many slices `pool_run` will hand out: the workers, and the thread that
 * called it, which takes one itself rather than waiting idle. Scratch indexed
 * by slice has to be this long, so the rule lives here rather than at each
 * caller that has to guess it. */
static int pool_bands(const pool_group *group) {
  return group->worker_count > 0 ? group->worker_count + 1 : 1;
}

static void pool_run(pool_group *group, pool_task task_call, void *task_state) {
  int slice_count = pool_bands(group);
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

/* This engine is little-endian by decision rather than by accident.  The
 * checkpoint is mapped and read where it lies: `real_read` casts the mapped
 * bytes to a `float` or a `uint16_t`, and the packed code stream below is a
 * dense little-endian bit field the kernels index directly.  A big-endian host
 * would not read the weights wrongly in one place but in every one of them, so
 * there is nothing for a kernel to gain by being careful about it on its own.
 * Where the compiler will say which it is, this says so before anything is
 * built against the wrong assumption. */
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#  error "igllm maps the checkpoint where it lies and wants a little-endian host"
#endif

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
  int            code_flip;  /* xor taken before the offset, for signed byte codes */
  int            group_size;
  int            group_count;
  float          enter_gain; /* static activation step taken before the product */
  float          leave_gain; /* and after it; zero where the export left it uncalibrated */

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
        int code_value = (int)(pack_read(code_row, (size_t)col_index, sheet->bit_count) ^
                               (uint32_t)sheet->code_flip);
        row_out[col_index] = (float)(code_value - sheet->code_bias - bias_value) * gain_value;
      }
    }
  }
}

/* Bytes a product against this plane reads: the payload, and for a code plane
 * the gains and zero points it is decoded with.  This is what the kernel
 * streams, not what the tensor occupies on disk — a plane bound as a view of a
 * stacked tensor counts only its own slice. */
static size_t plane_bytes(const plane *sheet) {
  size_t total;
  if (!sheet || sheet->form == PLANE_VOID) return 0;
  if (sheet->form == PLANE_REAL)
    return (size_t)sheet->row_count * (size_t)sheet->col_count *
           store_type_bytes(sheet->real_type);
  total = (size_t)sheet->row_count * sheet->row_stride;
  total += (size_t)sheet->row_count * (size_t)sheet->group_count *
           store_type_bytes(sheet->gain_type);
  if (sheet->bias_data) total += (size_t)sheet->row_count * (size_t)sheet->group_count;
  return total;
}

/* And what one row of it costs, which is what a table read by index reads. */
static size_t plane_row_bytes(const plane *sheet) {
  if (!sheet || sheet->form == PLANE_VOID || sheet->row_count < 1) return 0;
  return plane_bytes(sheet) / (size_t)sheet->row_count;
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

/* Rounds to the static grid the export calibrated, on eight bit levels.
 *
 * This is not the dynamic rule below: the step is a single number stored beside
 * the weights, one for what goes into a projection and one for what comes out.
 * A step of zero means the layer was never calibrated, and nothing happens.
 *
 * The rounding is to even on a tie, which is what the reference does, and the
 * ties are not rare here: a product of two quantized planes is an exact
 * multiple of the two steps, so landing on a half is common enough that
 * rounding up instead moved whole activations by a step. */
static void quant_step(float *value_list, int value_count, float step_value) {
  int value_index;
  if (!(step_value > 0.0f)) return;
  for (value_index = 0; value_index < value_count; ++value_index) {
    float level = rintf(value_list[value_index] / step_value);
    if (level < -128.0f) level = -128.0f;
    if (level > 127.0f) level = 127.0f;
    value_list[value_index] = level * step_value;
  }
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

/* How many tokens one batched projection carries, and how many codes are
 * unpacked into scratch at a time while it runs. */
#define KERN_LANE_LIMIT   16
#define KERN_SPREAD_LIMIT 256

static float kern_dot_real(const void *row_data, store_type row_type, const float *act_data,
                           int span_count) {
  int slot;
  float total = 0.0f;
  if (row_type == STORE_F32) {
    const float *row_real = (const float *)row_data;
#if defined(APP_SIMD_AVX2)
    /* Two accumulators rather than one: a single chain makes the loop wait a
     * whole multiply-add latency per iteration, where two let the pipeline keep
     * one in flight while the other lands. */
    __m256 part_a = _mm256_setzero_ps(), part_b = _mm256_setzero_ps();
    __m256 wide_total;
    for (slot = 0; slot + 16 <= span_count; slot += 16) {
      part_a = _mm256_fmadd_ps(_mm256_loadu_ps(row_real + slot),
                               _mm256_loadu_ps(act_data + slot), part_a);
      part_b = _mm256_fmadd_ps(_mm256_loadu_ps(row_real + slot + 8),
                               _mm256_loadu_ps(act_data + slot + 8), part_b);
    }
    for (; slot + 8 <= span_count; slot += 8)
      part_a = _mm256_fmadd_ps(_mm256_loadu_ps(row_real + slot),
                               _mm256_loadu_ps(act_data + slot), part_a);
    wide_total = _mm256_add_ps(part_a, part_b);
    {
      __m128 half_total = _mm_add_ps(_mm256_castps256_ps128(wide_total),
                                     _mm256_extractf128_ps(wide_total, 1));
      half_total = _mm_hadd_ps(half_total, half_total);
      half_total = _mm_hadd_ps(half_total, half_total);
      total = _mm_cvtss_f32(half_total);
    }
#elif defined(APP_SIMD_SSE2)
    __m128 part_a = _mm_setzero_ps(), part_b = _mm_setzero_ps();
    __m128 wide_total;
    for (slot = 0; slot + 8 <= span_count; slot += 8) {
      part_a = _mm_add_ps(
          part_a, _mm_mul_ps(_mm_loadu_ps(row_real + slot), _mm_loadu_ps(act_data + slot)));
      part_b = _mm_add_ps(part_b, _mm_mul_ps(_mm_loadu_ps(row_real + slot + 4),
                                             _mm_loadu_ps(act_data + slot + 4)));
    }
    for (; slot + 4 <= span_count; slot += 4)
      part_a = _mm_add_ps(
          part_a, _mm_mul_ps(_mm_loadu_ps(row_real + slot), _mm_loadu_ps(act_data + slot)));
    wide_total = _mm_add_ps(part_a, part_b);
    {
      __m128 half_total = _mm_add_ps(wide_total, _mm_movehl_ps(wide_total, wide_total));
      half_total = _mm_add_ss(half_total, _mm_shuffle_ps(half_total, half_total, 0x55));
      total = _mm_cvtss_f32(half_total);
    }
#elif defined(APP_SIMD_NEON)
    float32x4_t part_a = vdupq_n_f32(0.0f), part_b = vdupq_n_f32(0.0f);
    float32x4_t wide_total;
    for (slot = 0; slot + 8 <= span_count; slot += 8) {
      part_a = vmlaq_f32(part_a, vld1q_f32(row_real + slot), vld1q_f32(act_data + slot));
      part_b = vmlaq_f32(part_b, vld1q_f32(row_real + slot + 4), vld1q_f32(act_data + slot + 4));
    }
    for (; slot + 4 <= span_count; slot += 4)
      part_a = vmlaq_f32(part_a, vld1q_f32(row_real + slot), vld1q_f32(act_data + slot));
    wide_total = vaddq_f32(part_a, part_b);
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

/* Horizontal sum of the widest accumulator the selected path uses. */
#if defined(APP_SIMD_AVX2)
static float kern_wide_total(__m256 wide_total) {
  __m128 half_total =
      _mm_add_ps(_mm256_castps256_ps128(wide_total), _mm256_extractf128_ps(wide_total, 1));
  half_total = _mm_add_ps(half_total, _mm_movehl_ps(half_total, half_total));
  half_total = _mm_add_ss(half_total, _mm_shuffle_ps(half_total, half_total, 0x55));
  return _mm_cvtss_f32(half_total);
}
#elif defined(APP_SIMD_SSE2)
static float kern_wide_total(__m128 wide_total) {
  __m128 half_total = _mm_add_ps(wide_total, _mm_movehl_ps(wide_total, wide_total));
  half_total = _mm_add_ss(half_total, _mm_shuffle_ps(half_total, half_total, 0x55));
  return _mm_cvtss_f32(half_total);
}
#elif defined(APP_SIMD_NEON)
static float kern_wide_total(float32x4_t wide_total) {
  return vgetq_lane_f32(wide_total, 0) + vgetq_lane_f32(wide_total, 1) +
         vgetq_lane_f32(wide_total, 2) + vgetq_lane_f32(wide_total, 3);
}
#endif

/* Four codes to four floats, indexed by the byte that holds them.
 *
 * Two bits a code means a byte is exactly four codes, so a decode that costs a
 * shift, a mask and a convert per code becomes one sixteen byte read of a table
 * that is four kilobytes and stays in the first level cache.  The values are
 * the codes themselves — zero to three — so a path that reads the table reaches
 * the same float in the same lane as the path that unpacks, and the two are
 * interchangeable to the last bit rather than merely close.
 *
 * It is built by the preprocessor rather than filled at startup, which keeps it
 * in read-only memory and keeps the kernels free of an initialization order to
 * get wrong.  It is read unaligned: C11 has no way to state the alignment of a
 * static array that every compiler this file targets agrees on, and on any part
 * that runs this path an unaligned load of an aligned address costs nothing. */
#define KERN_CODE_TWO_ROW(byte)                                                                  \
  {(float)((byte) & 3), (float)(((byte) >> 2) & 3), (float)(((byte) >> 4) & 3),                  \
   (float)(((byte) >> 6) & 3)}
#define KERN_CODE_TWO_4(byte)                                                                    \
  KERN_CODE_TWO_ROW(byte), KERN_CODE_TWO_ROW((byte) + 1),                                        \
      KERN_CODE_TWO_ROW((byte) + 2), KERN_CODE_TWO_ROW((byte) + 3)
#define KERN_CODE_TWO_16(byte)                                                                   \
  KERN_CODE_TWO_4(byte), KERN_CODE_TWO_4((byte) + 4), KERN_CODE_TWO_4((byte) + 8),               \
      KERN_CODE_TWO_4((byte) + 12)
#define KERN_CODE_TWO_64(byte)                                                                   \
  KERN_CODE_TWO_16(byte), KERN_CODE_TWO_16((byte) + 16),                                         \
      KERN_CODE_TWO_16((byte) + 32), KERN_CODE_TWO_16((byte) + 48)

static const float kern_code_two[256][4] = {KERN_CODE_TWO_64(0), KERN_CODE_TWO_64(64),
                                            KERN_CODE_TWO_64(128), KERN_CODE_TWO_64(192)};

/* Sum of activation times code over one quantization group.
 *
 * Two, four and eight bits are the widths the checkpoint leans on, and each has
 * a vector path behind the same macro layer the dense kernels use.  A group
 * starts on a byte boundary in the two narrow widths whenever `from_index` is a
 * multiple of the codes per byte; when it is not, the general bit-stream loop
 * at the bottom still produces the right answer.
 *
 * Every path carries two accumulators rather than one.  A single chain stalls
 * on the latency of its own add: the arithmetic here is cheap enough that the
 * dependency, not the work, was setting the pace. */
/* Eight codes of an odd width, out of one word rather than out of eight walks
 * of the bit stream.
 *
 * Three, five, six and seven bits share the property that eight codes occupy
 * exactly `bit_count` bytes, so a run of eight always begins on a byte boundary
 * where the run before it did.  That makes a block of eight one load of at most
 * seven bytes and eight shifts, against `pack_read`'s walk per code — and it
 * leaves the codes in the shape every other width's vector path wants.
 *
 * The word is one load where the run still has eight bytes ahead of it.  The
 * packing is a dense little-endian bit stream and the host is little-endian —
 * which is stated once, beside `pack_read`, rather than worked around here —
 * so the eight bytes are the word already, and the bits above the block's own
 * `bit_count` of them are never reached by a shift a block takes.  Assembling
 * it a byte at a time was costing a fifth of the loop at three bits and rather
 * more of it at seven.
 *
 * The tail is the exception, and the reason `run_end` is carried at all: a
 * block at the end of a row has as few as three bytes behind it, and the five
 * beyond them may be past the end of the mapping rather than merely past the
 * end of the row.  There the word is assembled from its own bytes as it always
 * was, which is a handful of blocks a row against the thousands that are
 * not. */
static uint64_t kern_code_word(const uint8_t *word_head, const uint8_t *run_end, int bit_count) {
  uint64_t word_value = 0;
  int part_index;
  if (word_head + 8 <= run_end) {
    memcpy(&word_value, word_head, sizeof(word_value));
    return word_value;
  }
  for (part_index = 0; part_index < bit_count; ++part_index)
    word_value |= (uint64_t)word_head[part_index] << (8 * part_index);
  return word_value;
}

#if !defined(APP_SIMD_AVX2)
/* The block written out into scratch, which is what every host without the
 * variable shift still reads it through. */
static void kern_code_eight(const uint8_t *word_head, const uint8_t *run_end, int bit_count,
                            uint32_t code_mask, int code_flip, int32_t *code_out) {
  uint64_t word_value = kern_code_word(word_head, run_end, bit_count);
  int part_index;
  for (part_index = 0; part_index < 8; ++part_index)
    code_out[part_index] = (int32_t)((((uint32_t)(word_value >> (bit_count * part_index))) &
                                      code_mask) ^
                                     (uint32_t)code_flip);
}
#endif

#if defined(APP_SIMD_AVX2)
/* The same block of eight, decoded inside the vector rather than through
 * scratch.  Both readers of a block want exactly this and neither can afford
 * the other shape: eight four byte stores feeding one thirty-two byte load is a
 * forwarding stall, and it costs the fused dot nearly half of its loop.
 *
 * There are two ways in, and which is taken is decided by how many bytes of the
 * row are certainly behind the block rather than by the width.
 *
 * Where sixteen bytes are, the block is a shuffle over one wide load.  Eight
 * codes are exactly `bit_count` bytes, so a block always begins on a byte
 * boundary and every block of a width picks the same bytes at the same shifts:
 * the tables are the width's own and are built once for a run.  Each lane is
 * handed the four bytes its code starts in — a code of seven bits or fewer
 * straddles two of them at most, so four is more than enough — and shifts its
 * own code down.  The bytes a block reaches are all within the first ten, so
 * the low half of the load is broadcast to both halves and one `shuffle_epi8`
 * serves all eight lanes despite picking across what would otherwise be a lane
 * boundary.
 *
 * Where they are not — the last block or two of a row, where the bytes beyond
 * may be past the mapping rather than merely past the row — the word is
 * broadcast to four sixty-four bit lanes instead and gathered back into one
 * vector by two permutes and a join.  That is three ops a block more and is
 * paid on a handful of blocks a row against the thousands that are not. */
typedef struct {
  __m256i byte_pick;  /* the four bytes each code starts in */
  __m256i step_wide;  /* how far down its own lane shifts it */
  __m256i mask_slim;  /* the width, thirty-two bits a lane */
  __m256i flip_wide;
  __m256i shift_low;  /* the tail's path: the first four codes, one shift a lane */
  __m256i shift_high; /* the second four */
  __m256i mask_wide;  /* the width, sixty-four bits a lane */
  __m256i pick_data;
} kern_code_plan;

static kern_code_plan kern_code_plan_make(int bit_count, uint32_t code_mask, int code_flip) {
  kern_code_plan plan;
  int8_t pick_room[32];
  int32_t step_room[8];
  int part_index, byte_index;
  for (part_index = 0; part_index < 8; ++part_index) {
    int bit_place = part_index * bit_count;
    for (byte_index = 0; byte_index < 4; ++byte_index)
      pick_room[part_index * 4 + byte_index] = (int8_t)(bit_place / 8 + byte_index);
    step_room[part_index] = bit_place % 8;
  }
  plan.byte_pick = _mm256_loadu_si256((const __m256i *)(const void *)pick_room);
  plan.step_wide = _mm256_loadu_si256((const __m256i *)(const void *)step_room);
  plan.mask_slim = _mm256_set1_epi32((int)code_mask);
  plan.flip_wide = _mm256_set1_epi32(code_flip);
  plan.shift_low = _mm256_setr_epi64x(0, bit_count, 2 * bit_count, 3 * bit_count);
  plan.shift_high = _mm256_setr_epi64x(4 * bit_count, 5 * bit_count, 6 * bit_count, 7 * bit_count);
  plan.mask_wide = _mm256_set1_epi64x((long long)code_mask);
  /* Four codes a vector sit in the low half of each sixty-four bit lane. */
  plan.pick_data = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
  return plan;
}

/* How many codes from the head of a run have sixteen bytes of the row behind
 * every block they fall in, and so can be taken by the shuffle.  The two are
 * split into two loops rather than branched between a block at a time: the
 * bound is the same for a whole run, and asking it per block costs the spread
 * everything the shuffle wins it. */
static int kern_code_wide_span(const uint8_t *byte_head, const uint8_t *run_end, int bit_count) {
  ptrdiff_t room = (run_end - byte_head) - 16;
  if (room < 0) return 0;
  return ((int)(room / bit_count) + 1) * 8;
}

static __m256i kern_code_wide(const kern_code_plan *plan, const uint8_t *word_head) {
  __m256i load_data =
      _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(const void *)word_head));
  __m256i lane_data = _mm256_shuffle_epi8(load_data, plan->byte_pick);
  return _mm256_xor_si256(
      _mm256_and_si256(_mm256_srlv_epi32(lane_data, plan->step_wide), plan->mask_slim),
      plan->flip_wide);
}

static __m256i kern_code_wide_edge(const kern_code_plan *plan, const uint8_t *word_head,
                                   const uint8_t *run_end, int bit_count) {
  __m256i base_data = _mm256_set1_epi64x((long long)kern_code_word(word_head, run_end, bit_count));
  __m256i part_low =
      _mm256_and_si256(_mm256_srlv_epi64(base_data, plan->shift_low), plan->mask_wide);
  __m256i part_high =
      _mm256_and_si256(_mm256_srlv_epi64(base_data, plan->shift_high), plan->mask_wide);
  return _mm256_xor_si256(
      _mm256_permute2x128_si256(_mm256_permutevar8x32_epi32(part_low, plan->pick_data),
                                _mm256_permutevar8x32_epi32(part_high, plan->pick_data), 0x20),
      plan->flip_wide);
}
#endif

/* Whether a width is one of those, and whether the run starts where a block
 * of eight does. */
static int kern_code_odd(int bit_count, int from_index) {
  if (bit_count != 3 && bit_count != 5 && bit_count != 6 && bit_count != 7) return 0;
  return (from_index & 7) == 0;
}

static float kern_dot_code(const uint8_t *code_row, int from_index, int span_count,
                           const float *act_data, int bit_count, int code_flip) {
  float total = 0.0f;
  int slot = 0;

  /* The two narrow paths read the packed bytes as they lie, so they are taken
   * only when the codes need no flip.  A signed plane is eight bits wide, and
   * that path applies the flip a whole vector at a time. */
  if (bit_count == 4 && code_flip == 0 && (from_index & 1) == 0) {
    const uint8_t *byte_head = code_row + (size_t)from_index / 2;
#if defined(APP_SIMD_AVX2)
    {
      const __m128i low_mask = _mm_set1_epi8(0x0F);
      __m256 part_a = _mm256_setzero_ps(), part_b = _mm256_setzero_ps();
      for (; slot + 16 <= span_count; slot += 16) {
        __m128i pack_data = _mm_loadl_epi64((const __m128i *)(const void *)(byte_head + slot / 2));
        __m128i low_part = _mm_and_si128(pack_data, low_mask);
        __m128i high_part = _mm_and_si128(_mm_srli_epi16(pack_data, 4), low_mask);
        __m128i code_byte = _mm_unpacklo_epi8(low_part, high_part);
        part_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(code_byte)),
                                 _mm256_loadu_ps(act_data + slot), part_a);
        part_b = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(code_byte, 8))),
            _mm256_loadu_ps(act_data + slot + 8), part_b);
      }
      total = kern_wide_total(_mm256_add_ps(part_a, part_b));
    }
#elif defined(APP_SIMD_SSE2)
    {
      const __m128i low_mask = _mm_set1_epi8(0x0F);
      const __m128i zero_data = _mm_setzero_si128();
      __m128 part_a = _mm_setzero_ps(), part_b = _mm_setzero_ps();
      for (; slot + 16 <= span_count; slot += 16) {
        __m128i pack_data = _mm_loadl_epi64((const __m128i *)(const void *)(byte_head + slot / 2));
        __m128i low_part = _mm_and_si128(pack_data, low_mask);
        __m128i high_part = _mm_and_si128(_mm_srli_epi16(pack_data, 4), low_mask);
        __m128i code_byte = _mm_unpacklo_epi8(low_part, high_part);
        __m128i word_low = _mm_unpacklo_epi8(code_byte, zero_data);
        __m128i word_high = _mm_unpackhi_epi8(code_byte, zero_data);
        part_a = _mm_add_ps(
            part_a, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(word_low, zero_data)),
                               _mm_loadu_ps(act_data + slot)));
        part_b = _mm_add_ps(
            part_b, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(word_low, zero_data)),
                               _mm_loadu_ps(act_data + slot + 4)));
        part_a = _mm_add_ps(
            part_a, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(word_high, zero_data)),
                               _mm_loadu_ps(act_data + slot + 8)));
        part_b = _mm_add_ps(
            part_b, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(word_high, zero_data)),
                               _mm_loadu_ps(act_data + slot + 12)));
      }
      total = kern_wide_total(_mm_add_ps(part_a, part_b));
    }
#elif defined(APP_SIMD_NEON)
    {
      float32x4_t part_a = vdupq_n_f32(0.0f), part_b = vdupq_n_f32(0.0f);
      for (; slot + 16 <= span_count; slot += 16) {
        uint8x8_t pack_data = vld1_u8(byte_head + slot / 2);
        uint8x8x2_t code_byte =
            vzip_u8(vand_u8(pack_data, vdup_n_u8(0x0F)), vshr_n_u8(pack_data, 4));
        int part_index;
        for (part_index = 0; part_index < 2; ++part_index) {
          uint16x8_t code_word = vmovl_u8(code_byte.val[part_index]);
          part_a = vmlaq_f32(part_a, vcvtq_f32_u32(vmovl_u16(vget_low_u16(code_word))),
                             vld1q_f32(act_data + slot + part_index * 8));
          part_b = vmlaq_f32(part_b, vcvtq_f32_u32(vmovl_u16(vget_high_u16(code_word))),
                             vld1q_f32(act_data + slot + part_index * 8 + 4));
        }
      }
      total = kern_wide_total(vaddq_f32(part_a, part_b));
    }
#endif
    for (; slot + 2 <= span_count; slot += 2) {
      uint8_t pair = byte_head[slot / 2];
      total += (float)(pair & 0x0Fu) * act_data[slot];
      total += (float)(pair >> 4) * act_data[slot + 1];
    }
    if (slot < span_count) total += (float)(byte_head[slot / 2] & 0x0Fu) * act_data[slot];
    return total;
  }

  if (bit_count == 8) {
    const uint8_t *byte_head = code_row + (size_t)from_index;
    uint8_t flip_byte = (uint8_t)code_flip;
#if defined(APP_SIMD_AVX2)
    {
      const __m128i flip_data = _mm_set1_epi8((char)flip_byte);
      __m256 part_a = _mm256_setzero_ps(), part_b = _mm256_setzero_ps();
      for (; slot + 16 <= span_count; slot += 16) {
        __m128i code_byte = _mm_xor_si128(
            _mm_loadu_si128((const __m128i *)(const void *)(byte_head + slot)), flip_data);
        part_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(code_byte)),
                                 _mm256_loadu_ps(act_data + slot), part_a);
        part_b = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(code_byte, 8))),
            _mm256_loadu_ps(act_data + slot + 8), part_b);
      }
      total = kern_wide_total(_mm256_add_ps(part_a, part_b));
    }
#elif defined(APP_SIMD_SSE2)
    {
      const __m128i flip_data = _mm_set1_epi8((char)flip_byte);
      const __m128i zero_data = _mm_setzero_si128();
      __m128 part_a = _mm_setzero_ps(), part_b = _mm_setzero_ps();
      for (; slot + 16 <= span_count; slot += 16) {
        __m128i code_byte = _mm_xor_si128(
            _mm_loadu_si128((const __m128i *)(const void *)(byte_head + slot)), flip_data);
        __m128i word_low = _mm_unpacklo_epi8(code_byte, zero_data);
        __m128i word_high = _mm_unpackhi_epi8(code_byte, zero_data);
        part_a = _mm_add_ps(
            part_a, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(word_low, zero_data)),
                               _mm_loadu_ps(act_data + slot)));
        part_b = _mm_add_ps(
            part_b, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(word_low, zero_data)),
                               _mm_loadu_ps(act_data + slot + 4)));
        part_a = _mm_add_ps(
            part_a, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(word_high, zero_data)),
                               _mm_loadu_ps(act_data + slot + 8)));
        part_b = _mm_add_ps(
            part_b, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(word_high, zero_data)),
                               _mm_loadu_ps(act_data + slot + 12)));
      }
      total = kern_wide_total(_mm_add_ps(part_a, part_b));
    }
#elif defined(APP_SIMD_NEON)
    {
      const uint8x16_t flip_data = vdupq_n_u8(flip_byte);
      float32x4_t part_a = vdupq_n_f32(0.0f), part_b = vdupq_n_f32(0.0f);
      for (; slot + 16 <= span_count; slot += 16) {
        uint8x16_t code_byte = veorq_u8(vld1q_u8(byte_head + slot), flip_data);
        uint16x8_t word_low = vmovl_u8(vget_low_u8(code_byte));
        uint16x8_t word_high = vmovl_u8(vget_high_u8(code_byte));
        part_a = vmlaq_f32(part_a, vcvtq_f32_u32(vmovl_u16(vget_low_u16(word_low))),
                           vld1q_f32(act_data + slot));
        part_b = vmlaq_f32(part_b, vcvtq_f32_u32(vmovl_u16(vget_high_u16(word_low))),
                           vld1q_f32(act_data + slot + 4));
        part_a = vmlaq_f32(part_a, vcvtq_f32_u32(vmovl_u16(vget_low_u16(word_high))),
                           vld1q_f32(act_data + slot + 8));
        part_b = vmlaq_f32(part_b, vcvtq_f32_u32(vmovl_u16(vget_high_u16(word_high))),
                           vld1q_f32(act_data + slot + 12));
      }
      total = kern_wide_total(vaddq_f32(part_a, part_b));
    }
#endif
    for (; slot < span_count; ++slot)
      total += (float)(uint8_t)(byte_head[slot] ^ flip_byte) * act_data[slot];
    return total;
  }

  if (bit_count == 2 && code_flip == 0 && (from_index & 3) == 0) {
    const uint8_t *byte_head = code_row + (size_t)from_index / 4;
#if defined(APP_SIMD_AVX2)
    {
      /* One broadcast of the whole dword rather than one of each half.
       *
       * A variable shift reaches bit thirty-one, so the eight codes of the
       * upper half are the same broadcast shifted by sixteen more places, and
       * the mask below keeps only the two bits meant either way.  The lanes,
       * the codes and the order are what they were — this is the same sum to
       * the last bit — and the loop spends one broadcast per sixteen codes
       * where it spent two.  On the export's widest two bit row, 12288 by
       * 1536: 0.0027 s against 0.0019 s. */
      const __m256i step_low = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
      const __m256i step_high = _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30);
      const __m256i code_mask = _mm256_set1_epi32(3);
      __m256 part_a = _mm256_setzero_ps(), part_b = _mm256_setzero_ps();
      for (; slot + 16 <= span_count; slot += 16) {
        uint32_t word_value;
        __m256i base_data;
        memcpy(&word_value, byte_head + slot / 4, 4);
        base_data = _mm256_set1_epi32((int)word_value);
        part_a = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(
                _mm256_and_si256(_mm256_srlv_epi32(base_data, step_low), code_mask)),
            _mm256_loadu_ps(act_data + slot), part_a);
        part_b = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(
                _mm256_and_si256(_mm256_srlv_epi32(base_data, step_high), code_mask)),
            _mm256_loadu_ps(act_data + slot + 8), part_b);
      }
      total = kern_wide_total(_mm256_add_ps(part_a, part_b));
    }
#elif defined(APP_SIMD_SSE2)
    {
      /* One table read a byte, multiplied into the same accumulator the
       * unpacking path put that lane in, so the sum is the same to the last
       * bit and only the way the codes were reached has changed.  Measured on
       * the export's widest row, 12288 by 1536 in one thread: 0.0039 s
       * unpacking, 0.0023 s reading the table.
       *
       * The wide path above keeps its shifts.  The same table there costs two
       * narrow loads and an insert to fill one 256 bit vector, against one
       * broadcast, one variable shift and one mask, and it measures slower —
       * 0.0018 s against 0.0016 s on the same row. */
      __m128 part_a = _mm_setzero_ps(), part_b = _mm_setzero_ps();
      for (; slot + 16 <= span_count; slot += 16) {
        const uint8_t *quad_head = byte_head + slot / 4;
        part_a = _mm_add_ps(part_a, _mm_mul_ps(_mm_loadu_ps(kern_code_two[quad_head[0]]),
                                               _mm_loadu_ps(act_data + slot)));
        part_b = _mm_add_ps(part_b, _mm_mul_ps(_mm_loadu_ps(kern_code_two[quad_head[1]]),
                                               _mm_loadu_ps(act_data + slot + 4)));
        part_a = _mm_add_ps(part_a, _mm_mul_ps(_mm_loadu_ps(kern_code_two[quad_head[2]]),
                                               _mm_loadu_ps(act_data + slot + 8)));
        part_b = _mm_add_ps(part_b, _mm_mul_ps(_mm_loadu_ps(kern_code_two[quad_head[3]]),
                                               _mm_loadu_ps(act_data + slot + 12)));
      }
      total = kern_wide_total(_mm_add_ps(part_a, part_b));
    }
#elif defined(APP_SIMD_NEON)
    {
      static const uint8_t low_pick[8] = {0, 0, 0, 0, 1, 1, 1, 1};
      static const uint8_t high_pick[8] = {2, 2, 2, 2, 3, 3, 3, 3};
      static const int8_t step_list[8] = {0, -2, -4, -6, 0, -2, -4, -6};
      const int8x8_t step_data = vld1_s8(step_list);
      float32x4_t part_a = vdupq_n_f32(0.0f), part_b = vdupq_n_f32(0.0f);
      for (; slot + 16 <= span_count; slot += 16) {
        uint32_t word_value;
        uint8x8_t pack_data;
        uint8x8_t part_list[2];
        int part_index;
        memcpy(&word_value, byte_head + slot / 4, 4);
        pack_data = vreinterpret_u8_u32(vdup_n_u32(word_value));
        part_list[0] = vtbl1_u8(pack_data, vld1_u8(low_pick));
        part_list[1] = vtbl1_u8(pack_data, vld1_u8(high_pick));
        for (part_index = 0; part_index < 2; ++part_index) {
          uint16x8_t code_word = vmovl_u8(
              vand_u8(vshl_u8(part_list[part_index], step_data), vdup_n_u8(3)));
          part_a = vmlaq_f32(part_a, vcvtq_f32_u32(vmovl_u16(vget_low_u16(code_word))),
                             vld1q_f32(act_data + slot + part_index * 8));
          part_b = vmlaq_f32(part_b, vcvtq_f32_u32(vmovl_u16(vget_high_u16(code_word))),
                             vld1q_f32(act_data + slot + part_index * 8 + 4));
        }
      }
      total = kern_wide_total(vaddq_f32(part_a, part_b));
    }
#endif
    for (; slot + 4 <= span_count; slot += 4) {
      uint8_t quad = byte_head[slot / 4];
      total += (float)(quad & 3u) * act_data[slot];
      total += (float)((quad >> 2) & 3u) * act_data[slot + 1];
      total += (float)((quad >> 4) & 3u) * act_data[slot + 2];
      total += (float)((quad >> 6) & 3u) * act_data[slot + 3];
    }
    for (; slot < span_count; ++slot)
      total += (float)((byte_head[slot / 4] >> (2 * (slot & 3))) & 3u) * act_data[slot];
    return total;
  }

  /* Three, five, six and seven bits.  Eight codes are exactly `bit_count`
   * bytes, so a run of them comes out of one word rather than out of eight
   * walks of the bit stream, and the width stops deciding how the loop is
   * shaped.  On a five bit row of 12288, one thread: 0.55 G codes a second
   * against 3.97 tuned and 1.62 default. */
  if (kern_code_odd(bit_count, from_index)) {
    const uint8_t *byte_head = code_row + (size_t)from_index * (size_t)bit_count / 8u;
    /* The bytes this run is certainly inside: the codes begin on a byte
     * boundary, so the whole ones of them are the run's own. */
    const uint8_t *run_end = byte_head + (size_t)span_count * (size_t)bit_count / 8u;
    const uint32_t code_mask = (1u << bit_count) - 1u;
#if defined(APP_SIMD_AVX2)
    {
      /* The codes come out of the row inside the vector rather than through
       * scratch — `kern_code_wide` above, which the spread reads the same
       * blocks through.  On a five bit row of 12288, one thread, tuned: 6.29 G
       * codes a second, against 3.34 when the block was a broadcast word and
       * two permutes and 0.94 when it went through scratch.  Two bits reaches
       * 10.52 on that row and four 8.32, so the width still shapes this loop,
       * but it is 60% of them rather than a third. */
      const kern_code_plan plan = kern_code_plan_make(bit_count, code_mask, code_flip);
      int wide_count = kern_code_wide_span(byte_head, run_end, bit_count);
      __m256 part_a = _mm256_setzero_ps(), part_b = _mm256_setzero_ps();
      if (wide_count > span_count) wide_count = span_count;
      for (; slot + 16 <= wide_count; slot += 16) {
        const uint8_t *word_head = byte_head + (size_t)(slot / 8) * (size_t)bit_count;
        part_a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(kern_code_wide(&plan, word_head)),
                                 _mm256_loadu_ps(act_data + slot), part_a);
        part_b = _mm256_fmadd_ps(_mm256_cvtepi32_ps(kern_code_wide(&plan, word_head + bit_count)),
                                 _mm256_loadu_ps(act_data + slot + 8), part_b);
      }
      /* The blocks at the end of the row, which have fewer than sixteen bytes
       * behind them. */
      for (; slot + 8 <= span_count; slot += 8)
        part_a = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(kern_code_wide_edge(
                &plan, byte_head + (size_t)(slot / 8) * (size_t)bit_count, run_end, bit_count)),
            _mm256_loadu_ps(act_data + slot), part_a);
      total = kern_wide_total(_mm256_add_ps(part_a, part_b));
    }
#else
    {
      /* Everywhere else the block is written to scratch and summed from it a
       * value at a time.  Reading it back a vector at a time was measured on
       * SSE2 and is worth nothing — the same forwarding stall — where reading
       * it scalar is worth twice the bit stream walk: 1.62 G codes a second
       * against 0.75 on that row.  A NEON host has the variable shift the AVX2
       * path uses and might do better still with it; that is unmeasured and so
       * it takes this path with the rest. */
      int32_t code_room[16];
      float sum_a = 0.0f, sum_b = 0.0f;
      for (; slot + 16 <= span_count; slot += 16) {
        const uint8_t *word_head = byte_head + (size_t)(slot / 8) * (size_t)bit_count;
        int part_index;
        kern_code_eight(word_head, run_end, bit_count, code_mask, code_flip, code_room);
        kern_code_eight(word_head + bit_count, run_end, bit_count, code_mask, code_flip,
                        code_room + 8);
        for (part_index = 0; part_index < 16; part_index += 2) {
          sum_a += (float)code_room[part_index] * act_data[slot + part_index];
          sum_b += (float)code_room[part_index + 1] * act_data[slot + part_index + 1];
        }
      }
      total = sum_a + sum_b;
    }
#endif
    /* What is left of a row is under eight codes, which is under one block. */
    for (; slot < span_count; ++slot)
      total += (float)(pack_read(code_row, (size_t)(from_index + slot), bit_count) ^
                       (uint32_t)code_flip) * act_data[slot];
    return total;
  }

  for (; slot < span_count; ++slot)
    total += (float)(pack_read(code_row, (size_t)(from_index + slot), bit_count) ^
                     (uint32_t)code_flip) * act_data[slot];
  return total;
}

/* The same decode `kern_dot_code` performs, written out into floats instead of
 * summed against one activation vector.  A batch pays it once and every lane
 * reads what it leaves, so the two widths the checkpoint leans on get the same
 * vector paths here that they get there.  Anything else walks the bit stream,
 * which is what the whole of this did before there was a reason to hurry it:
 * with the decode scalar, a sixteen lane batch cost three and a half times what
 * the same sixteen lanes cost one at a time, and batching lost to the thing it
 * exists to beat. */
static void kern_code_spread(const uint8_t *code_row, int from_index, int span_count,
                             int bit_count, int code_flip, float *out_data) {
  int slot = 0;

  if (bit_count == 8) {
    const uint8_t *byte_head = code_row + (size_t)from_index;
    uint8_t flip_byte = (uint8_t)code_flip;
    for (; slot < span_count; ++slot)
      out_data[slot] = (float)(uint8_t)(byte_head[slot] ^ flip_byte);
    return;
  }

  if (bit_count == 4 && code_flip == 0 && (from_index & 1) == 0) {
    const uint8_t *byte_head = code_row + (size_t)from_index / 2;
#if defined(APP_SIMD_AVX2)
    {
      const __m128i low_mask = _mm_set1_epi8(0x0F);
      for (; slot + 16 <= span_count; slot += 16) {
        __m128i pack_data = _mm_loadl_epi64((const __m128i *)(const void *)(byte_head + slot / 2));
        __m128i low_part = _mm_and_si128(pack_data, low_mask);
        __m128i high_part = _mm_and_si128(_mm_srli_epi16(pack_data, 4), low_mask);
        __m128i code_byte = _mm_unpacklo_epi8(low_part, high_part);
        _mm256_storeu_ps(out_data + slot, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(code_byte)));
        _mm256_storeu_ps(out_data + slot + 8,
                         _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(code_byte, 8))));
      }
    }
#elif defined(APP_SIMD_SSE2)
    {
      const __m128i low_mask = _mm_set1_epi8(0x0F);
      const __m128i zero_data = _mm_setzero_si128();
      for (; slot + 16 <= span_count; slot += 16) {
        __m128i pack_data = _mm_loadl_epi64((const __m128i *)(const void *)(byte_head + slot / 2));
        __m128i low_part = _mm_and_si128(pack_data, low_mask);
        __m128i high_part = _mm_and_si128(_mm_srli_epi16(pack_data, 4), low_mask);
        __m128i code_byte = _mm_unpacklo_epi8(low_part, high_part);
        __m128i word_low = _mm_unpacklo_epi8(code_byte, zero_data);
        __m128i word_high = _mm_unpackhi_epi8(code_byte, zero_data);
        _mm_storeu_ps(out_data + slot, _mm_cvtepi32_ps(_mm_unpacklo_epi16(word_low, zero_data)));
        _mm_storeu_ps(out_data + slot + 4,
                      _mm_cvtepi32_ps(_mm_unpackhi_epi16(word_low, zero_data)));
        _mm_storeu_ps(out_data + slot + 8,
                      _mm_cvtepi32_ps(_mm_unpacklo_epi16(word_high, zero_data)));
        _mm_storeu_ps(out_data + slot + 12,
                      _mm_cvtepi32_ps(_mm_unpackhi_epi16(word_high, zero_data)));
      }
    }
#elif defined(APP_SIMD_NEON)
    {
      for (; slot + 16 <= span_count; slot += 16) {
        uint8x8_t pack_data = vld1_u8(byte_head + slot / 2);
        uint8x8x2_t code_byte =
            vzip_u8(vand_u8(pack_data, vdup_n_u8(0x0F)), vshr_n_u8(pack_data, 4));
        int part_index;
        for (part_index = 0; part_index < 2; ++part_index) {
          uint16x8_t code_word = vmovl_u8(code_byte.val[part_index]);
          vst1q_f32(out_data + slot + part_index * 8,
                    vcvtq_f32_u32(vmovl_u16(vget_low_u16(code_word))));
          vst1q_f32(out_data + slot + part_index * 8 + 4,
                    vcvtq_f32_u32(vmovl_u16(vget_high_u16(code_word))));
        }
      }
    }
#endif
    for (; slot + 2 <= span_count; slot += 2) {
      uint8_t pair = byte_head[slot / 2];
      out_data[slot] = (float)(pair & 0x0Fu);
      out_data[slot + 1] = (float)(pair >> 4);
    }
    if (slot < span_count) out_data[slot] = (float)(byte_head[slot / 2] & 0x0Fu);
    return;
  }

  if (bit_count == 2 && code_flip == 0 && (from_index & 3) == 0) {
    const uint8_t *byte_head = code_row + (size_t)from_index / 4;
#if defined(APP_SIMD_AVX2)
    {
      /* One broadcast of the whole dword, as the fused dot above takes it. */
      const __m256i step_low = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
      const __m256i step_high = _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30);
      const __m256i code_mask = _mm256_set1_epi32(3);
      for (; slot + 16 <= span_count; slot += 16) {
        uint32_t word_value;
        __m256i base_data;
        memcpy(&word_value, byte_head + slot / 4, 4);
        base_data = _mm256_set1_epi32((int)word_value);
        _mm256_storeu_ps(out_data + slot,
                         _mm256_cvtepi32_ps(_mm256_and_si256(
                             _mm256_srlv_epi32(base_data, step_low), code_mask)));
        _mm256_storeu_ps(out_data + slot + 8,
                         _mm256_cvtepi32_ps(_mm256_and_si256(
                             _mm256_srlv_epi32(base_data, step_high), code_mask)));
      }
    }
#elif defined(APP_SIMD_SSE2)
    {
      /* Written out, a byte of codes is a row of the table and nothing else,
       * so the whole decode is a sixteen byte copy: 0.0031 s unpacking against
       * 0.0023 s reading the table, on the row measured above. */
      for (; slot + 16 <= span_count; slot += 16) {
        const uint8_t *quad_head = byte_head + slot / 4;
        _mm_storeu_ps(out_data + slot, _mm_loadu_ps(kern_code_two[quad_head[0]]));
        _mm_storeu_ps(out_data + slot + 4, _mm_loadu_ps(kern_code_two[quad_head[1]]));
        _mm_storeu_ps(out_data + slot + 8, _mm_loadu_ps(kern_code_two[quad_head[2]]));
        _mm_storeu_ps(out_data + slot + 12, _mm_loadu_ps(kern_code_two[quad_head[3]]));
      }
    }
#elif defined(APP_SIMD_NEON)
    {
      static const uint8_t low_pick[8] = {0, 0, 0, 0, 1, 1, 1, 1};
      static const uint8_t high_pick[8] = {2, 2, 2, 2, 3, 3, 3, 3};
      static const int8_t step_list[8] = {0, -2, -4, -6, 0, -2, -4, -6};
      const int8x8_t step_data = vld1_s8(step_list);
      for (; slot + 16 <= span_count; slot += 16) {
        uint32_t word_value;
        uint8x8_t pack_data;
        uint8x8_t part_list[2];
        int part_index;
        memcpy(&word_value, byte_head + slot / 4, 4);
        pack_data = vreinterpret_u8_u32(vdup_n_u32(word_value));
        part_list[0] = vtbl1_u8(pack_data, vld1_u8(low_pick));
        part_list[1] = vtbl1_u8(pack_data, vld1_u8(high_pick));
        for (part_index = 0; part_index < 2; ++part_index) {
          uint16x8_t code_word =
              vmovl_u8(vand_u8(vshl_u8(part_list[part_index], step_data), vdup_n_u8(3)));
          vst1q_f32(out_data + slot + part_index * 8,
                    vcvtq_f32_u32(vmovl_u16(vget_low_u16(code_word))));
          vst1q_f32(out_data + slot + part_index * 8 + 4,
                    vcvtq_f32_u32(vmovl_u16(vget_high_u16(code_word))));
        }
      }
    }
#endif
    /* The remainder, and every host without a vector path, reads the table
     * too: written out, four codes are a row of it and nothing else.  On the
     * row measured above that loop alone goes from 0.0092 s to 0.0023 s.  The
     * fused dot's remainder is left unpacking, because there the cost is the
     * chain of dependent adds rather than the decode and the same swap moves
     * it by nothing. */
    for (; slot + 4 <= span_count; slot += 4)
      memcpy(out_data + slot, kern_code_two[byte_head[slot / 4]], 4 * sizeof(float));
    for (; slot < span_count; ++slot)
      out_data[slot] = (float)((byte_head[slot / 4] >> (2 * (slot & 3))) & 3u);
    return;
  }

  /* Three, five, six and seven bits.  A block of eight is decoded inside the
   * vector and stored as eight floats — `kern_code_wide` above, which the fused
   * dot reads its blocks through as well — rather than written to scratch and
   * read back a value at a time.  That round trip was the whole of what stood
   * between these widths and the two the checkpoint leans on.  On a five bit
   * row of 12288, one thread, tuned: 8.90 G codes a second against 1.19 before
   * it, and against 9.89 at two bits and 9.79 at four on the same row.  The
   * width no longer decides the rate of the spread either. */
  if (kern_code_odd(bit_count, from_index)) {
    const uint8_t *byte_head = code_row + (size_t)from_index * (size_t)bit_count / 8u;
    const uint8_t *run_end = byte_head + (size_t)span_count * (size_t)bit_count / 8u;
    const uint32_t code_mask = (1u << bit_count) - 1u;
#if defined(APP_SIMD_AVX2)
    {
      const kern_code_plan plan = kern_code_plan_make(bit_count, code_mask, code_flip);
      int wide_count = kern_code_wide_span(byte_head, run_end, bit_count);
      if (wide_count > span_count) wide_count = span_count;
      for (; slot + 16 <= wide_count; slot += 16) {
        const uint8_t *word_head = byte_head + (size_t)(slot / 8) * (size_t)bit_count;
        _mm256_storeu_ps(out_data + slot, _mm256_cvtepi32_ps(kern_code_wide(&plan, word_head)));
        _mm256_storeu_ps(out_data + slot + 8,
                         _mm256_cvtepi32_ps(kern_code_wide(&plan, word_head + bit_count)));
      }
      for (; slot + 8 <= span_count; slot += 8)
        _mm256_storeu_ps(
            out_data + slot,
            _mm256_cvtepi32_ps(kern_code_wide_edge(
                &plan, byte_head + (size_t)(slot / 8) * (size_t)bit_count, run_end, bit_count)));
    }
#else
    {
      /* Everywhere else the block still goes through scratch, which is what
       * the width had before either reader hurried it, and is still twice the
       * bit stream walk below. */
      int32_t code_room[8];
      int part_index;
      for (; slot + 8 <= span_count; slot += 8) {
        kern_code_eight(byte_head + (size_t)(slot / 8) * (size_t)bit_count, run_end, bit_count,
                        code_mask, code_flip, code_room);
        for (part_index = 0; part_index < 8; ++part_index)
          out_data[slot + part_index] = (float)code_room[part_index];
      }
    }
#endif
  }
  for (; slot < span_count; ++slot)
    out_data[slot] = (float)(pack_read(code_row, (size_t)(from_index + slot), bit_count) ^
                             (uint32_t)code_flip);
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
                               sheet->bit_count, sheet->code_flip);
    total += gain_value * (part_value - bias_value * sum_data[group_index]);
  }
  return total;
}

typedef struct kern_job {
  const plane *sheet;
  const float *act_data;  /* lane_count rows of col_count, act_stride apart */
  const float *sum_data;  /* lane_count rows of group_count, sum_stride apart */
  float       *out_data;  /* lane_count rows of row_count, out_stride apart */
  int          act_stride;
  int          sum_stride;
  int          out_stride;
  int          lane_count;
} kern_job;

/* Codes are unpacked once per group and reused by every lane, so a batch pays
 * the decode cost of a single vector and turns the projection into a product. */
static void kern_row_code_many(const plane *sheet, int row_index, const kern_job *job) {
  const uint8_t *code_row = sheet->code_data + (size_t)row_index * sheet->row_stride;
  size_t gain_base = (size_t)row_index * (size_t)sheet->group_count;
  float part_list[KERN_LANE_LIMIT];
  float code_room[KERN_SPREAD_LIMIT];
  int lane_index, group_index;
  for (lane_index = 0; lane_index < job->lane_count; ++lane_index)
    job->out_data[(size_t)lane_index * (size_t)job->out_stride + (size_t)row_index] = 0.0f;
  for (group_index = 0; group_index < sheet->group_count; ++group_index) {
    int from_index = group_index * sheet->group_size;
    int span_count = sheet->col_count - from_index;
    float gain_value = plane_gain(sheet, gain_base + (size_t)group_index);
    float bias_value = (float)(sheet->code_bias +
                               (sheet->bias_data ? sheet->bias_data[gain_base + (size_t)group_index] : 0));
    int done_count = 0;
    if (span_count > sheet->group_size) span_count = sheet->group_size;
    for (lane_index = 0; lane_index < job->lane_count; ++lane_index) part_list[lane_index] = 0.0f;
    while (done_count < span_count) {
      int chunk_count = span_count - done_count;
      if (chunk_count > KERN_SPREAD_LIMIT) chunk_count = KERN_SPREAD_LIMIT;
      kern_code_spread(code_row, from_index + done_count, chunk_count, sheet->bit_count,
                       sheet->code_flip, code_room);
      for (lane_index = 0; lane_index < job->lane_count; ++lane_index)
        part_list[lane_index] += kern_dot_real(
            code_room, STORE_F32,
            job->act_data + (size_t)lane_index * (size_t)job->act_stride + from_index + done_count,
            chunk_count);
      done_count += chunk_count;
    }
    for (lane_index = 0; lane_index < job->lane_count; ++lane_index)
      job->out_data[(size_t)lane_index * (size_t)job->out_stride + (size_t)row_index] +=
          gain_value * (part_list[lane_index] -
                        bias_value *
                            job->sum_data[(size_t)lane_index * (size_t)job->sum_stride +
                                          (size_t)group_index]);
  }
}

static void kern_mat_vec_band(void *state, int slice_index, int slice_count) {
  kern_job *job = (kern_job *)state;
  const plane *sheet = job->sheet;
  int row_from, row_upto, row_index, lane_index;
  slice_span(sheet->row_count, slice_index, slice_count, &row_from, &row_upto);
  if (sheet->form == PLANE_REAL) {
    size_t row_span = (size_t)sheet->col_count * store_type_bytes(sheet->real_type);
    for (row_index = row_from; row_index < row_upto; ++row_index) {
      const uint8_t *row_head = (const uint8_t *)sheet->real_data + (size_t)row_index * row_span;
      for (lane_index = 0; lane_index < job->lane_count; ++lane_index)
        job->out_data[(size_t)lane_index * (size_t)job->out_stride + (size_t)row_index] =
            kern_dot_real(row_head, sheet->real_type,
                          job->act_data + (size_t)lane_index * (size_t)job->act_stride,
                          sheet->col_count);
    }
  } else if (job->lane_count == 1) {
    for (row_index = row_from; row_index < row_upto; ++row_index)
      job->out_data[row_index] = kern_row_code(sheet, row_index, job->act_data, job->sum_data);
  } else {
    for (row_index = row_from; row_index < row_upto; ++row_index)
      kern_row_code_many(sheet, row_index, job);
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

/* How many values of a blend are held in registers at once.  Thirty-two floats
 * are four of the widest vector this file uses and eight of the narrowest,
 * which leaves the narrow paths room for the row being read and the weight
 * being broadcast without spilling. */
#define KERN_BLEND_BLOCK 32

/* The weighted sum of `span_count` rows into one: `into[v] = sum_s w[s] * from[s][v]`.
 *
 * This is what an attention head does once its scores are softmaxed, and
 * written the obvious way — a row at a time, folded into the destination — it
 * reads and writes the destination once a span.  That is three memory
 * operations for every multiply-add on a loop whose whole content is the
 * multiply-add.  Blocked by value instead, the running sums stay in registers
 * across every span and only the rows are read.
 *
 * The order of the sum is untouched: value `v` still takes span 0 first and
 * span `span_count - 1` last, into the same accumulator it always did.  Only
 * where that accumulator lives has changed, and each path multiplies and adds
 * the way the loop it replaces did on that build, so the towers reach the same
 * numbers they reached before.
 *
 * `from_stride` is the distance between rows, which is the head width where the
 * rows are heads of a wider array and the head size where they have been
 * gathered into a run of their own. */
static void kern_blend_rows(const float *from_data, int from_stride, const float *weight_list,
                            int span_count, int value_count, float *into_data) {
  int base_index;
  for (base_index = 0; base_index < value_count; base_index += KERN_BLEND_BLOCK) {
    const float *from_head = from_data + base_index;
    int chunk_count = value_count - base_index;
    int span_index, value_index;
    if (chunk_count > KERN_BLEND_BLOCK) chunk_count = KERN_BLEND_BLOCK;
#if defined(APP_SIMD_AVX2)
    if (chunk_count == KERN_BLEND_BLOCK) {
      __m256 part_a = _mm256_setzero_ps(), part_b = _mm256_setzero_ps();
      __m256 part_c = _mm256_setzero_ps(), part_d = _mm256_setzero_ps();
      for (span_index = 0; span_index < span_count; ++span_index, from_head += from_stride) {
        __m256 weight_wide = _mm256_broadcast_ss(weight_list + span_index);
        part_a = _mm256_add_ps(part_a, _mm256_mul_ps(weight_wide, _mm256_loadu_ps(from_head)));
        part_b = _mm256_add_ps(part_b, _mm256_mul_ps(weight_wide, _mm256_loadu_ps(from_head + 8)));
        part_c = _mm256_add_ps(part_c, _mm256_mul_ps(weight_wide, _mm256_loadu_ps(from_head + 16)));
        part_d = _mm256_add_ps(part_d, _mm256_mul_ps(weight_wide, _mm256_loadu_ps(from_head + 24)));
      }
      _mm256_storeu_ps(into_data + base_index, part_a);
      _mm256_storeu_ps(into_data + base_index + 8, part_b);
      _mm256_storeu_ps(into_data + base_index + 16, part_c);
      _mm256_storeu_ps(into_data + base_index + 24, part_d);
      continue;
    }
#elif defined(APP_SIMD_SSE2)
    if (chunk_count == KERN_BLEND_BLOCK) {
      __m128 part_list[8];
      int part_index;
      for (part_index = 0; part_index < 8; ++part_index) part_list[part_index] = _mm_setzero_ps();
      for (span_index = 0; span_index < span_count; ++span_index, from_head += from_stride) {
        __m128 weight_wide = _mm_set1_ps(weight_list[span_index]);
        for (part_index = 0; part_index < 8; ++part_index)
          part_list[part_index] = _mm_add_ps(
              part_list[part_index], _mm_mul_ps(weight_wide, _mm_loadu_ps(from_head + part_index * 4)));
      }
      for (part_index = 0; part_index < 8; ++part_index)
        _mm_storeu_ps(into_data + base_index + part_index * 4, part_list[part_index]);
      continue;
    }
#elif defined(APP_SIMD_NEON)
    if (chunk_count == KERN_BLEND_BLOCK) {
      float32x4_t part_list[8];
      int part_index;
      for (part_index = 0; part_index < 8; ++part_index) part_list[part_index] = vdupq_n_f32(0.0f);
      for (span_index = 0; span_index < span_count; ++span_index, from_head += from_stride) {
        float32x4_t weight_wide = vdupq_n_f32(weight_list[span_index]);
        for (part_index = 0; part_index < 8; ++part_index)
          part_list[part_index] =
              vmlaq_f32(part_list[part_index], weight_wide, vld1q_f32(from_head + part_index * 4));
      }
      for (part_index = 0; part_index < 8; ++part_index)
        vst1q_f32(into_data + base_index + part_index * 4, part_list[part_index]);
      continue;
    }
#endif
    /* The tail, and the whole of it on a build with no vector path.  A block of
     * scratch rather than the destination keeps this the same arithmetic in the
     * same order as the paths above. */
    {
      float part_list[KERN_BLEND_BLOCK];
      for (value_index = 0; value_index < chunk_count; ++value_index) part_list[value_index] = 0.0f;
      for (span_index = 0; span_index < span_count; ++span_index, from_head += from_stride) {
        float weight_value = weight_list[span_index];
        for (value_index = 0; value_index < chunk_count; ++value_index)
          part_list[value_index] += weight_value * from_head[value_index];
      }
      for (value_index = 0; value_index < chunk_count; ++value_index)
        into_data[base_index + value_index] = part_list[value_index];
    }
  }
}

/* `into[v] += left[v] * right[v]`, the element-wise multiply-add a depthwise
 * convolution reduces to once its kernel is laid out tap-major. */
static void kern_fma_row(const float *left_data, const float *right_data, int value_count,
                         float *into_data) {
  int slot = 0;
#if defined(APP_SIMD_AVX2)
  for (; slot + 8 <= value_count; slot += 8)
    _mm256_storeu_ps(into_data + slot,
                     _mm256_fmadd_ps(_mm256_loadu_ps(left_data + slot),
                                     _mm256_loadu_ps(right_data + slot),
                                     _mm256_loadu_ps(into_data + slot)));
#elif defined(APP_SIMD_SSE2)
  for (; slot + 4 <= value_count; slot += 4)
    _mm_storeu_ps(into_data + slot,
                  _mm_add_ps(_mm_loadu_ps(into_data + slot),
                             _mm_mul_ps(_mm_loadu_ps(left_data + slot),
                                        _mm_loadu_ps(right_data + slot))));
#elif defined(APP_SIMD_NEON)
  for (; slot + 4 <= value_count; slot += 4)
    vst1q_f32(into_data + slot, vmlaq_f32(vld1q_f32(into_data + slot), vld1q_f32(left_data + slot),
                                          vld1q_f32(right_data + slot)));
#endif
  for (; slot < value_count; ++slot) into_data[slot] += left_data[slot] * right_data[slot];
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

/* Mean-subtracting normalization, which the audio subsampler uses where the
 * rest of the engine uses the root-mean-square form.  The reference gives it a
 * weight and no bias. */
static void kern_norm_layer(const float *value_list, const float *gain_list, int value_count,
                            float eps_value, float *value_out) {
  double mean_value = 0.0, square_total = 0.0;
  float shrink_value;
  int value_index;
  for (value_index = 0; value_index < value_count; ++value_index)
    mean_value += (double)value_list[value_index];
  mean_value /= (double)value_count;
  for (value_index = 0; value_index < value_count; ++value_index) {
    double gap_value = (double)value_list[value_index] - mean_value;
    square_total += gap_value * gap_value;
  }
  shrink_value = (float)pow(square_total / (double)value_count + (double)eps_value, -0.5);
  for (value_index = 0; value_index < value_count; ++value_index)
    value_out[value_index] = (float)((double)value_list[value_index] - mean_value) * shrink_value *
                             (gain_list ? gain_list[value_index] : 1.0f);
}

static float kern_silu(float value) { return value / (1.0f + expf(-value)); }

/* log(1 + e^x), evaluated so that a large argument does not overflow. */
static float kern_soft_plus(float value) {
  return value > 20.0f ? value : logf(1.0f + expf(value));
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

/* The exponential a softmax needs, to within a bit of the last, without a call
 * into libm.
 *
 * A picture's attention is a billion of these — a patch pair a head a layer, at
 * the full patch budget — and a call a value is what left the softmax the
 * largest thing in a picture that is not a matrix product.  What replaces the
 * call is the definition.  `e^x` is `2^k` times `e^r`, where `k` is the whole
 * number nearest `x / ln 2` and `r` is what is left over, never further from
 * zero than half of `ln 2`.  Over that range the series for `e^r` is finished
 * after eight terms — the ninth is five parts in a thousand million of the
 * value, an eighth of the last bit a float carries — and `2^k` is an exponent
 * field written straight into the word.
 *
 * `ln 2` is taken off in two pieces because `k * ln2` in one piece rounds away
 * the low bits of `x` that the remainder is made of.  The first piece is exact
 * in a float — its significand is eleven bits — so `x - k * high` is exact too,
 * and the second piece carries the rest of the constant.
 *
 * Only a softmax reaches this, and a softmax has already taken its own peak off
 * every value, so every argument is at most zero and the total is at least one.
 * An argument below where a float stops being normal is therefore floored
 * rather than allowed to reach a subnormal: either way it is a part in ten to
 * the thirty-eighth of a total of at least one.
 *
 * The four paths agree to the last bit except where one of them has a fused
 * multiply-add and another does not, which is the same difference the dot
 * product kernels already carry. */
#define KERN_EXP_FLOOR (-87.0f) /* below this `e^x` is under the smallest normal */
#define KERN_EXP_LOG2E 1.4426950408889634f
#define KERN_EXP_LN2_HIGH 0.693359375f /* 1419/2048, exact in a float */
#define KERN_EXP_LN2_LOW (-2.1219444005469057e-4f)

/* The series in Horner's order, over the eight terms the half-`ln 2` range
 * needs.  Written as reciprocals of the factorials the definition names rather
 * than as a fitted set, so there is nothing here to have copied wrong. */
#define KERN_EXP_TERM_7 (1.0f / 5040.0f)
#define KERN_EXP_TERM_6 (1.0f / 720.0f)
#define KERN_EXP_TERM_5 (1.0f / 120.0f)
#define KERN_EXP_TERM_4 (1.0f / 24.0f)
#define KERN_EXP_TERM_3 (1.0f / 6.0f)
#define KERN_EXP_TERM_2 (1.0f / 2.0f)

static float kern_exp_near(float value_now) {
  float step_value, whole_value, rest_value, poly_value, scale_value;
  uint32_t word_value;
  if (value_now < KERN_EXP_FLOOR) value_now = KERN_EXP_FLOOR;
  step_value = value_now * KERN_EXP_LOG2E + 0.5f;
  /* A floor without a call: truncation gives the ceiling of a negative, so the
   * one that overshot steps back.  The argument is never above a half here, so
   * the whole part always fits an int. */
  whole_value = (float)(int)step_value;
  if (whole_value > step_value) whole_value -= 1.0f;
  rest_value = value_now - whole_value * KERN_EXP_LN2_HIGH;
  rest_value -= whole_value * KERN_EXP_LN2_LOW;
  poly_value = KERN_EXP_TERM_7;
  poly_value = poly_value * rest_value + KERN_EXP_TERM_6;
  poly_value = poly_value * rest_value + KERN_EXP_TERM_5;
  poly_value = poly_value * rest_value + KERN_EXP_TERM_4;
  poly_value = poly_value * rest_value + KERN_EXP_TERM_3;
  poly_value = poly_value * rest_value + KERN_EXP_TERM_2;
  poly_value = poly_value * rest_value + 1.0f;
  poly_value = poly_value * rest_value + 1.0f;
  word_value = (uint32_t)((int)whole_value + 127) << 23;
  memcpy(&scale_value, &word_value, sizeof(scale_value));
  return poly_value * scale_value;
}

#if defined(APP_SIMD_AVX2)
static __m256 kern_exp_wide(__m256 wide_value) {
  __m256 step_value, whole_value, rest_value, poly_value, scale_value;
  wide_value = _mm256_max_ps(wide_value, _mm256_set1_ps(KERN_EXP_FLOOR));
  step_value = _mm256_fmadd_ps(wide_value, _mm256_set1_ps(KERN_EXP_LOG2E), _mm256_set1_ps(0.5f));
  whole_value = _mm256_cvtepi32_ps(_mm256_cvttps_epi32(step_value));
  whole_value = _mm256_sub_ps(
      whole_value, _mm256_and_ps(_mm256_cmp_ps(whole_value, step_value, _CMP_GT_OQ),
                                 _mm256_set1_ps(1.0f)));
  rest_value = _mm256_fnmadd_ps(whole_value, _mm256_set1_ps(KERN_EXP_LN2_HIGH), wide_value);
  rest_value = _mm256_fnmadd_ps(whole_value, _mm256_set1_ps(KERN_EXP_LN2_LOW), rest_value);
  poly_value = _mm256_set1_ps(KERN_EXP_TERM_7);
  poly_value = _mm256_fmadd_ps(poly_value, rest_value, _mm256_set1_ps(KERN_EXP_TERM_6));
  poly_value = _mm256_fmadd_ps(poly_value, rest_value, _mm256_set1_ps(KERN_EXP_TERM_5));
  poly_value = _mm256_fmadd_ps(poly_value, rest_value, _mm256_set1_ps(KERN_EXP_TERM_4));
  poly_value = _mm256_fmadd_ps(poly_value, rest_value, _mm256_set1_ps(KERN_EXP_TERM_3));
  poly_value = _mm256_fmadd_ps(poly_value, rest_value, _mm256_set1_ps(KERN_EXP_TERM_2));
  poly_value = _mm256_fmadd_ps(poly_value, rest_value, _mm256_set1_ps(1.0f));
  poly_value = _mm256_fmadd_ps(poly_value, rest_value, _mm256_set1_ps(1.0f));
  scale_value = _mm256_castsi256_ps(_mm256_slli_epi32(
      _mm256_add_epi32(_mm256_cvttps_epi32(whole_value), _mm256_set1_epi32(127)), 23));
  return _mm256_mul_ps(poly_value, scale_value);
}
#elif defined(APP_SIMD_SSE2)
static __m128 kern_exp_wide(__m128 wide_value) {
  __m128 step_value, whole_value, rest_value, poly_value, scale_value;
  wide_value = _mm_max_ps(wide_value, _mm_set1_ps(KERN_EXP_FLOOR));
  step_value = _mm_add_ps(_mm_mul_ps(wide_value, _mm_set1_ps(KERN_EXP_LOG2E)), _mm_set1_ps(0.5f));
  whole_value = _mm_cvtepi32_ps(_mm_cvttps_epi32(step_value));
  whole_value = _mm_sub_ps(
      whole_value, _mm_and_ps(_mm_cmpgt_ps(whole_value, step_value), _mm_set1_ps(1.0f)));
  rest_value =
      _mm_sub_ps(wide_value, _mm_mul_ps(whole_value, _mm_set1_ps(KERN_EXP_LN2_HIGH)));
  rest_value =
      _mm_sub_ps(rest_value, _mm_mul_ps(whole_value, _mm_set1_ps(KERN_EXP_LN2_LOW)));
  poly_value = _mm_set1_ps(KERN_EXP_TERM_7);
  poly_value = _mm_add_ps(_mm_mul_ps(poly_value, rest_value), _mm_set1_ps(KERN_EXP_TERM_6));
  poly_value = _mm_add_ps(_mm_mul_ps(poly_value, rest_value), _mm_set1_ps(KERN_EXP_TERM_5));
  poly_value = _mm_add_ps(_mm_mul_ps(poly_value, rest_value), _mm_set1_ps(KERN_EXP_TERM_4));
  poly_value = _mm_add_ps(_mm_mul_ps(poly_value, rest_value), _mm_set1_ps(KERN_EXP_TERM_3));
  poly_value = _mm_add_ps(_mm_mul_ps(poly_value, rest_value), _mm_set1_ps(KERN_EXP_TERM_2));
  poly_value = _mm_add_ps(_mm_mul_ps(poly_value, rest_value), _mm_set1_ps(1.0f));
  poly_value = _mm_add_ps(_mm_mul_ps(poly_value, rest_value), _mm_set1_ps(1.0f));
  scale_value = _mm_castsi128_ps(
      _mm_slli_epi32(_mm_add_epi32(_mm_cvttps_epi32(whole_value), _mm_set1_epi32(127)), 23));
  return _mm_mul_ps(poly_value, scale_value);
}
#elif defined(APP_SIMD_NEON)
static float32x4_t kern_exp_wide(float32x4_t wide_value) {
  float32x4_t step_value, whole_value, rest_value, poly_value, scale_value;
  wide_value = vmaxq_f32(wide_value, vdupq_n_f32(KERN_EXP_FLOOR));
  step_value = vaddq_f32(vmulq_f32(wide_value, vdupq_n_f32(KERN_EXP_LOG2E)), vdupq_n_f32(0.5f));
  whole_value = vcvtq_f32_s32(vcvtq_s32_f32(step_value));
  whole_value = vsubq_f32(
      whole_value, vreinterpretq_f32_u32(vandq_u32(vcgtq_f32(whole_value, step_value),
                                                   vreinterpretq_u32_f32(vdupq_n_f32(1.0f)))));
  rest_value = vmlsq_f32(wide_value, whole_value, vdupq_n_f32(KERN_EXP_LN2_HIGH));
  rest_value = vmlsq_f32(rest_value, whole_value, vdupq_n_f32(KERN_EXP_LN2_LOW));
  poly_value = vdupq_n_f32(KERN_EXP_TERM_7);
  poly_value = vmlaq_f32(vdupq_n_f32(KERN_EXP_TERM_6), poly_value, rest_value);
  poly_value = vmlaq_f32(vdupq_n_f32(KERN_EXP_TERM_5), poly_value, rest_value);
  poly_value = vmlaq_f32(vdupq_n_f32(KERN_EXP_TERM_4), poly_value, rest_value);
  poly_value = vmlaq_f32(vdupq_n_f32(KERN_EXP_TERM_3), poly_value, rest_value);
  poly_value = vmlaq_f32(vdupq_n_f32(KERN_EXP_TERM_2), poly_value, rest_value);
  poly_value = vmlaq_f32(vdupq_n_f32(1.0f), poly_value, rest_value);
  poly_value = vmlaq_f32(vdupq_n_f32(1.0f), poly_value, rest_value);
  scale_value = vreinterpretq_f32_s32(
      vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(whole_value), vdupq_n_s32(127)), 23));
  return vmulq_f32(poly_value, scale_value);
}
#endif

/* The peak, the exponentials and their total, and the scaling: three passes,
 * each of them a lane at a time where the host has lanes.  The total is
 * gathered in as many accumulators as the widest load, so a long row's sum is
 * associated differently from the scalar path's — which is the same trade the
 * dot product kernels make, and below what the last bit of a float carries. */
static void kern_soft_max(float *value_list, int value_count) {
  float peak_value = -FLT_MAX;
  float total_value = 0.0f;
  int value_index = 0;
#if defined(APP_SIMD_AVX2)
  if (value_count >= 8) {
    __m256 wide_peak = _mm256_loadu_ps(value_list);
    __m256 wide_total = _mm256_setzero_ps();
    for (value_index = 8; value_index + 8 <= value_count; value_index += 8)
      wide_peak = _mm256_max_ps(wide_peak, _mm256_loadu_ps(value_list + value_index));
    {
      __m128 half_peak = _mm_max_ps(_mm256_castps256_ps128(wide_peak),
                                    _mm256_extractf128_ps(wide_peak, 1));
      half_peak = _mm_max_ps(half_peak, _mm_movehl_ps(half_peak, half_peak));
      half_peak = _mm_max_ss(half_peak, _mm_shuffle_ps(half_peak, half_peak, 0x55));
      peak_value = _mm_cvtss_f32(half_peak);
    }
    for (; value_index < value_count; ++value_index)
      if (value_list[value_index] > peak_value) peak_value = value_list[value_index];
    wide_peak = _mm256_set1_ps(peak_value);
    for (value_index = 0; value_index + 8 <= value_count; value_index += 8) {
      __m256 wide_step =
          kern_exp_wide(_mm256_sub_ps(_mm256_loadu_ps(value_list + value_index), wide_peak));
      _mm256_storeu_ps(value_list + value_index, wide_step);
      wide_total = _mm256_add_ps(wide_total, wide_step);
    }
    total_value = kern_wide_total(wide_total);
  } else {
    for (; value_index < value_count; ++value_index)
      if (value_list[value_index] > peak_value) peak_value = value_list[value_index];
    value_index = 0;
  }
#elif defined(APP_SIMD_SSE2) || defined(APP_SIMD_NEON)
  if (value_count >= 4) {
#  if defined(APP_SIMD_SSE2)
    __m128 wide_peak = _mm_loadu_ps(value_list);
    __m128 wide_total = _mm_setzero_ps();
    for (value_index = 4; value_index + 4 <= value_count; value_index += 4)
      wide_peak = _mm_max_ps(wide_peak, _mm_loadu_ps(value_list + value_index));
    {
      __m128 half_peak = _mm_max_ps(wide_peak, _mm_movehl_ps(wide_peak, wide_peak));
      half_peak = _mm_max_ss(half_peak, _mm_shuffle_ps(half_peak, half_peak, 0x55));
      peak_value = _mm_cvtss_f32(half_peak);
    }
    for (; value_index < value_count; ++value_index)
      if (value_list[value_index] > peak_value) peak_value = value_list[value_index];
    wide_peak = _mm_set1_ps(peak_value);
    for (value_index = 0; value_index + 4 <= value_count; value_index += 4) {
      __m128 wide_step =
          kern_exp_wide(_mm_sub_ps(_mm_loadu_ps(value_list + value_index), wide_peak));
      _mm_storeu_ps(value_list + value_index, wide_step);
      wide_total = _mm_add_ps(wide_total, wide_step);
    }
    total_value = kern_wide_total(wide_total);
#  else
    float32x4_t wide_peak = vld1q_f32(value_list);
    float32x4_t wide_total = vdupq_n_f32(0.0f);
    for (value_index = 4; value_index + 4 <= value_count; value_index += 4)
      wide_peak = vmaxq_f32(wide_peak, vld1q_f32(value_list + value_index));
    peak_value = vgetq_lane_f32(wide_peak, 0);
    if (vgetq_lane_f32(wide_peak, 1) > peak_value) peak_value = vgetq_lane_f32(wide_peak, 1);
    if (vgetq_lane_f32(wide_peak, 2) > peak_value) peak_value = vgetq_lane_f32(wide_peak, 2);
    if (vgetq_lane_f32(wide_peak, 3) > peak_value) peak_value = vgetq_lane_f32(wide_peak, 3);
    for (; value_index < value_count; ++value_index)
      if (value_list[value_index] > peak_value) peak_value = value_list[value_index];
    wide_peak = vdupq_n_f32(peak_value);
    for (value_index = 0; value_index + 4 <= value_count; value_index += 4) {
      float32x4_t wide_step =
          kern_exp_wide(vsubq_f32(vld1q_f32(value_list + value_index), wide_peak));
      vst1q_f32(value_list + value_index, wide_step);
      wide_total = vaddq_f32(wide_total, wide_step);
    }
    total_value = kern_wide_total(wide_total);
#  endif
  } else {
    for (; value_index < value_count; ++value_index)
      if (value_list[value_index] > peak_value) peak_value = value_list[value_index];
    value_index = 0;
  }
#else
  for (; value_index < value_count; ++value_index)
    if (value_list[value_index] > peak_value) peak_value = value_list[value_index];
  value_index = 0;
#endif
  for (; value_index < value_count; ++value_index) {
    value_list[value_index] = kern_exp_near(value_list[value_index] - peak_value);
    total_value += value_list[value_index];
  }
  if (total_value > 0.0f) {
    float shrink_value = 1.0f / total_value;
    value_index = 0;
#if defined(APP_SIMD_AVX2)
    {
      __m256 wide_shrink = _mm256_set1_ps(shrink_value);
      for (; value_index + 8 <= value_count; value_index += 8)
        _mm256_storeu_ps(value_list + value_index,
                         _mm256_mul_ps(_mm256_loadu_ps(value_list + value_index), wide_shrink));
    }
#elif defined(APP_SIMD_SSE2)
    {
      __m128 wide_shrink = _mm_set1_ps(shrink_value);
      for (; value_index + 4 <= value_count; value_index += 4)
        _mm_storeu_ps(value_list + value_index,
                      _mm_mul_ps(_mm_loadu_ps(value_list + value_index), wide_shrink));
    }
#elif defined(APP_SIMD_NEON)
    {
      float32x4_t wide_shrink = vdupq_n_f32(shrink_value);
      for (; value_index + 4 <= value_count; value_index += 4)
        vst1q_f32(value_list + value_index,
                  vmulq_f32(vld1q_f32(value_list + value_index), wide_shrink));
    }
#endif
    for (; value_index < value_count; ++value_index) value_list[value_index] *= shrink_value;
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
  void (*mat_mat)(struct back_desk *desk, const plane *sheet, const float *act_data, int act_stride,
                  int lane_count, float *out_data, int out_stride);
  void (*norm_rms)(struct back_desk *desk, const float *value_list, const float *gain_list,
                   int value_count, float eps_value, float *value_out);
  void (*soft_max)(struct back_desk *desk, float *value_list, int value_count);
  void (*gelu_gate)(struct back_desk *desk, float *gate_list, const float *rise_list, int value_count);
  void (*rope_turn)(struct back_desk *desk, float *value_list, int head_size, const float *cos_list,
                    const float *sin_list);
} back_desk;

static void back_mat_mat(back_desk *desk, const plane *sheet, const float *act_data, int act_stride,
                         int lane_count, float *out_data, int out_stride) {
  int done_count = 0;
  while (done_count < lane_count) {
    int chunk_count = lane_count - done_count;
    kern_job job;
    if (chunk_count > KERN_LANE_LIMIT) chunk_count = KERN_LANE_LIMIT;
    job.sheet = sheet;
    job.act_data = act_data + (size_t)done_count * (size_t)act_stride;
    job.out_data = out_data + (size_t)done_count * (size_t)out_stride;
    job.act_stride = act_stride;
    job.out_stride = out_stride;
    job.sum_stride = sheet->group_count;
    job.lane_count = chunk_count;
    job.sum_data = NULL;
    if (sheet->form == PLANE_CODE) {
      int lane_index;
      for (lane_index = 0; lane_index < chunk_count; ++lane_index)
        kern_group_sum(sheet, job.act_data + (size_t)lane_index * (size_t)act_stride,
                       desk->sum_room + (size_t)lane_index * (size_t)sheet->group_count);
      job.sum_data = desk->sum_room;
    }
    if (desk->pool_ref && sheet->row_count >= 64)
      pool_run(desk->pool_ref, kern_mat_vec_band, &job);
    else
      kern_mat_vec_band(&job, 0, 1);
    done_count += chunk_count;
  }
}

static void back_mat_vec(back_desk *desk, const plane *sheet, const float *act_data,
                         float *out_data) {
  back_mat_mat(desk, sheet, act_data, 0, 1, out_data, 0);
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
  desk->sum_room = (float *)mem_clear(sizeof(float) * (size_t)desk->sum_limit * KERN_LANE_LIMIT);
  if (!desk->sum_room) return APP_FAIL_MEMORY;
  desk->mat_vec = back_mat_vec;
  desk->mat_mat = back_mat_mat;
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
/* 7. media layer                                                           */
/* ======================================================================== */

/* What a picture or a sound passes through before it reaches a tower: the
 * container formats, the resampler, and the filterbank.  Nothing here knows
 * about models — the layer turns bytes on disk into a rectangle of floats or a
 * run of samples, and the tower layer above decides what those mean.
 *
 * The deflate reader is part of it because a PNG cannot be read without one,
 * and a third party decoder is not an option in an engine with no
 * dependencies.  It has the same shape as the rest of the file: one pass over
 * the input, no allocation inside a loop, and a refusal rather than a guess
 * wherever the container says something this reader does not implement. */

#define MEDIA_BAND_LIMIT 4
#define MEDIA_SIDE_LIMIT 16384 /* refuse a raster wider or taller than this */

/* A decoded raster, band interleaved, laid out row by row.  A mel spectrogram
 * is one of these too, with one band, the frame as the row and the filter as
 * the column, which is what lets the tower layer treat both the same way. */
typedef struct flat_grid {
  float *value_data;
  int    wide_count;
  int    high_count;
  int    band_count;
} flat_grid;

static void grid_free(flat_grid *grid) {
  if (!grid) return;
  mem_free(grid->value_data);
  memset(grid, 0, sizeof(*grid));
}

static app_code grid_open(flat_grid *grid, int wide_count, int high_count, int band_count) {
  memset(grid, 0, sizeof(*grid));
  if (wide_count < 1 || high_count < 1 || band_count < 1 || band_count > MEDIA_BAND_LIMIT)
    return APP_FAIL_ARGUMENT;
  if (wide_count > MEDIA_SIDE_LIMIT || high_count > MEDIA_SIDE_LIMIT) return APP_FAIL_SUPPORT;
  grid->value_data = (float *)mem_clear(sizeof(float) * (size_t)wide_count * (size_t)high_count *
                                        (size_t)band_count);
  if (!grid->value_data) return APP_FAIL_MEMORY;
  grid->wide_count = wide_count;
  grid->high_count = high_count;
  grid->band_count = band_count;
  return APP_OKAY;
}

static float *grid_at(const flat_grid *grid, int high_index, int wide_index) {
  return grid->value_data + ((size_t)high_index * (size_t)grid->wide_count + (size_t)wide_index) *
                                (size_t)grid->band_count;
}

/* -- deflate ------------------------------------------------------------- */

/* A canonical Huffman table: how many codes carry each length, and the symbols
 * in code order.  Decoding walks the lengths one bit at a time, which needs no
 * table wider than the alphabet and no second pass over the stream. */
typedef struct puff_tree {
  short count_list[16];
  short sign_list[288];
} puff_tree;

typedef struct puff_state {
  const uint8_t *from_data;
  size_t         from_size;
  size_t         from_walk;
  uint32_t       bit_room;
  int            bit_count;
  uint8_t       *into_data;
  size_t         into_size;
  size_t         into_walk;
  int            fault_flag;
} puff_state;

static int puff_bits(puff_state *state, int need_count) {
  int value_out;
  if (need_count < 1) return 0;
  while (state->bit_count < need_count) {
    if (state->from_walk >= state->from_size) { state->fault_flag = 1; return 0; }
    state->bit_room |= (uint32_t)state->from_data[state->from_walk++] << state->bit_count;
    state->bit_count += 8;
  }
  value_out = (int)(state->bit_room & ((1u << need_count) - 1u));
  state->bit_room >>= need_count;
  state->bit_count -= need_count;
  return value_out;
}

static void puff_tree_build(puff_tree *tree, const short *length_list, int sign_count) {
  short offset_list[16];
  int length_index, sign_index;
  for (length_index = 0; length_index < 16; ++length_index) tree->count_list[length_index] = 0;
  for (sign_index = 0; sign_index < sign_count; ++sign_index)
    tree->count_list[length_list[sign_index]] += 1;
  tree->count_list[0] = 0;
  offset_list[0] = 0;
  offset_list[1] = 0;
  for (length_index = 1; length_index < 15; ++length_index)
    offset_list[length_index + 1] =
        (short)(offset_list[length_index] + tree->count_list[length_index]);
  for (sign_index = 0; sign_index < sign_count; ++sign_index)
    if (length_list[sign_index] != 0)
      tree->sign_list[offset_list[length_list[sign_index]]++] = (short)sign_index;
}

static int puff_sign(puff_state *state, const puff_tree *tree) {
  int code_value = 0, first_code = 0, index_base = 0, length_index;
  for (length_index = 1; length_index <= 15; ++length_index) {
    int count_value;
    code_value |= puff_bits(state, 1);
    if (state->fault_flag) return -1;
    count_value = tree->count_list[length_index];
    if (code_value - first_code < count_value)
      return tree->sign_list[index_base + (code_value - first_code)];
    index_base += count_value;
    first_code = (first_code + count_value) << 1;
    code_value <<= 1;
  }
  return -1;
}

static const short puff_span_base[29] = {3,  4,  5,   6,   7,   8,   9,   10,  11,  13,
                                         15, 17, 19,  23,  27,  31,  35,  43,  51,  59,
                                         67, 83, 99,  115, 131, 163, 195, 227, 258};
static const short puff_span_more[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                         2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
static const short puff_gap_base[30] = {1,    2,    3,     4,     5,     7,    9,    13,
                                        17,   25,   33,    49,    65,    97,   129,  193,
                                        257,  385,  513,   769,   1025,  1537, 2049, 3073,
                                        4097, 6145, 8193,  12289, 16385, 24577};
static const short puff_gap_more[30] = {0, 0, 0, 0, 1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,
                                        6, 7, 7, 8, 8,  9,  9,  10, 10, 11, 11, 12, 12, 13, 13};

static int puff_codes(puff_state *state, const puff_tree *span_tree, const puff_tree *gap_tree) {
  for (;;) {
    int sign_value = puff_sign(state, span_tree);
    if (sign_value < 0) return -1;
    if (sign_value < 256) {
      if (state->into_walk >= state->into_size) return -1;
      state->into_data[state->into_walk++] = (uint8_t)sign_value;
    } else if (sign_value == 256) {
      return 0;
    } else {
      int span_count, gap_index, gap_count, copy_index;
      sign_value -= 257;
      if (sign_value >= 29) return -1;
      span_count = puff_span_base[sign_value] + puff_bits(state, puff_span_more[sign_value]);
      gap_index = puff_sign(state, gap_tree);
      if (gap_index < 0 || gap_index >= 30) return -1;
      gap_count = puff_gap_base[gap_index] + puff_bits(state, puff_gap_more[gap_index]);
      if (state->fault_flag) return -1;
      if ((size_t)gap_count > state->into_walk) return -1;
      if (state->into_walk + (size_t)span_count > state->into_size) return -1;
      /* Overlapping copies are legal and common, so this is a byte loop rather
       * than a memcpy: the source may still be being written. */
      for (copy_index = 0; copy_index < span_count; ++copy_index) {
        state->into_data[state->into_walk] =
            state->into_data[state->into_walk - (size_t)gap_count];
        state->into_walk += 1;
      }
    }
  }
}

static void puff_fixed(puff_tree *span_tree, puff_tree *gap_tree) {
  short length_list[288];
  int sign_index;
  for (sign_index = 0; sign_index < 144; ++sign_index) length_list[sign_index] = 8;
  for (; sign_index < 256; ++sign_index) length_list[sign_index] = 9;
  for (; sign_index < 280; ++sign_index) length_list[sign_index] = 7;
  for (; sign_index < 288; ++sign_index) length_list[sign_index] = 8;
  puff_tree_build(span_tree, length_list, 288);
  for (sign_index = 0; sign_index < 30; ++sign_index) length_list[sign_index] = 5;
  puff_tree_build(gap_tree, length_list, 30);
}

static int puff_dynamic(puff_state *state, puff_tree *span_tree, puff_tree *gap_tree) {
  static const short order_list[19] = {16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                       11, 4,  12, 3, 13, 2, 14, 1, 15};
  short length_list[320];
  puff_tree code_tree;
  int lit_count, gap_count, code_count, slot_index, fill_index;
  lit_count = puff_bits(state, 5) + 257;
  gap_count = puff_bits(state, 5) + 1;
  code_count = puff_bits(state, 4) + 4;
  if (state->fault_flag || lit_count > 286 || gap_count > 30) return -1;
  for (slot_index = 0; slot_index < 19; ++slot_index) length_list[slot_index] = 0;
  for (slot_index = 0; slot_index < code_count; ++slot_index)
    length_list[order_list[slot_index]] = (short)puff_bits(state, 3);
  if (state->fault_flag) return -1;
  puff_tree_build(&code_tree, length_list, 19);
  fill_index = 0;
  while (fill_index < lit_count + gap_count) {
    int sign_value = puff_sign(state, &code_tree);
    short echo_value = 0;
    int echo_count;
    if (sign_value < 0) return -1;
    if (sign_value < 16) {
      length_list[fill_index++] = (short)sign_value;
      continue;
    }
    if (sign_value == 16) {
      if (fill_index == 0) return -1;
      echo_value = length_list[fill_index - 1];
      echo_count = 3 + puff_bits(state, 2);
    } else if (sign_value == 17) {
      echo_count = 3 + puff_bits(state, 3);
    } else {
      echo_count = 11 + puff_bits(state, 7);
    }
    if (state->fault_flag || fill_index + echo_count > lit_count + gap_count) return -1;
    while (echo_count-- > 0) length_list[fill_index++] = echo_value;
  }
  if (length_list[256] == 0) return -1; /* no end-of-block code */
  puff_tree_build(span_tree, length_list, lit_count);
  puff_tree_build(gap_tree, length_list + lit_count, gap_count);
  return 0;
}

/* Inflates one stream and returns the bytes written, or -1.  A zlib wrapper is
 * skipped when the first two bytes look like one, which is what a PNG payload
 * always carries and a bare deflate stream never does. */
static long puff_run(const uint8_t *from_data, size_t from_size, uint8_t *into_data,
                     size_t into_size) {
  puff_state state;
  int last_flag = 0;
  memset(&state, 0, sizeof(state));
  state.from_data = from_data;
  state.from_size = from_size;
  state.into_data = into_data;
  state.into_size = into_size;
  if (from_size >= 2 && (from_data[0] & 0x0Fu) == 8 &&
      (((uint32_t)from_data[0] << 8) | (uint32_t)from_data[1]) % 31u == 0)
    state.from_walk = 2;
  while (!last_flag) {
    int kind_mark;
    last_flag = puff_bits(&state, 1);
    kind_mark = puff_bits(&state, 2);
    if (state.fault_flag) return -1;
    if (kind_mark == 0) {
      unsigned span_count;
      state.bit_room = 0;
      state.bit_count = 0;
      if (state.from_walk + 4 > state.from_size) return -1;
      span_count = (unsigned)state.from_data[state.from_walk] |
                   ((unsigned)state.from_data[state.from_walk + 1] << 8);
      state.from_walk += 4; /* the length is followed by its own complement */
      if (state.from_walk + span_count > state.from_size) return -1;
      if (state.into_walk + span_count > state.into_size) return -1;
      memcpy(state.into_data + state.into_walk, state.from_data + state.from_walk, span_count);
      state.from_walk += span_count;
      state.into_walk += span_count;
    } else if (kind_mark == 3) {
      return -1;
    } else {
      puff_tree span_tree, gap_tree;
      if (kind_mark == 1) puff_fixed(&span_tree, &gap_tree);
      else if (puff_dynamic(&state, &span_tree, &gap_tree) != 0) return -1;
      if (puff_codes(&state, &span_tree, &gap_tree) != 0) return -1;
    }
  }
  return (long)state.into_walk;
}

/* -- png ------------------------------------------------------------------ */

static uint32_t png_word(const uint8_t *data) {
  return ((uint32_t)data[0] << 24) | ((uint32_t)data[1] << 16) | ((uint32_t)data[2] << 8) |
         (uint32_t)data[3];
}

static int png_paeth(int left_value, int over_value, int corner_value) {
  int guess_value = left_value + over_value - corner_value;
  int left_gap = guess_value - left_value;
  int over_gap = guess_value - over_value;
  int corner_gap = guess_value - corner_value;
  if (left_gap < 0) left_gap = -left_gap;
  if (over_gap < 0) over_gap = -over_gap;
  if (corner_gap < 0) corner_gap = -corner_gap;
  if (left_gap <= over_gap && left_gap <= corner_gap) return left_value;
  if (over_gap <= corner_gap) return over_value;
  return corner_value;
}

/* Undoes the five per-line filters in place.  Every one of them refers only to
 * bytes to its left and to the line above, both already reconstructed. */
static void png_unfilter(uint8_t *raw_data, int high_count, size_t line_bytes, int step_bytes) {
  int high_index;
  size_t byte_index;
  for (high_index = 0; high_index < high_count; ++high_index) {
    uint8_t *line_data = raw_data + (size_t)high_index * (line_bytes + 1);
    uint8_t *over_data = high_index > 0 ? line_data - line_bytes : NULL;
    int rule_mark = line_data[0];
    uint8_t *this_data = line_data + 1;
    for (byte_index = 0; byte_index < line_bytes; ++byte_index) {
      int left_value =
          byte_index >= (size_t)step_bytes ? this_data[byte_index - (size_t)step_bytes] : 0;
      int over_value = over_data ? over_data[byte_index] : 0;
      int corner_value = (over_data && byte_index >= (size_t)step_bytes)
                             ? over_data[byte_index - (size_t)step_bytes]
                             : 0;
      int value_now = this_data[byte_index];
      switch (rule_mark) {
        case 1: value_now += left_value; break;
        case 2: value_now += over_value; break;
        case 3: value_now += (left_value + over_value) / 2; break;
        case 4: value_now += png_paeth(left_value, over_value, corner_value); break;
        default: break;
      }
      this_data[byte_index] = (uint8_t)value_now;
    }
  }
}

/* The Adam7 lattice: where each of the seven passes of an interlaced file
 * starts, and how far apart the samples it carries are.  A file that is not
 * interlaced is one pass whose lattice is every pixel, which is why the walk
 * below has one shape rather than two. */
static const uint8_t png_pass_from_x[7] = {0, 4, 0, 2, 0, 1, 0};
static const uint8_t png_pass_from_y[7] = {0, 0, 4, 0, 2, 0, 1};
static const uint8_t png_pass_step_x[7] = {8, 8, 4, 4, 2, 2, 1};
static const uint8_t png_pass_step_y[7] = {8, 8, 8, 4, 4, 2, 2};

static void png_pass_shape(int weave_mark, int pass_index, int *from_x, int *from_y, int *step_x,
                           int *step_y) {
  *from_x = weave_mark ? png_pass_from_x[pass_index] : 0;
  *from_y = weave_mark ? png_pass_from_y[pass_index] : 0;
  *step_x = weave_mark ? png_pass_step_x[pass_index] : 1;
  *step_y = weave_mark ? png_pass_step_y[pass_index] : 1;
}

/* How many of a lattice's samples fall inside the picture, in each direction. */
static int png_pass_span(int total_count, int from_index, int step_count) {
  int span = (total_count - from_index + step_count - 1) / step_count;
  return span > 0 ? span : 0;
}

/* The bytes one row of a lattice occupies, rounded up to the byte where the
 * depth packs more than one sample into one. */
static size_t png_pass_bytes(int wide_count, int lane_count, int deep_count) {
  return ((size_t)wide_count * (size_t)lane_count * (size_t)deep_count + 7u) / 8u;
}

/* One sample out of an unfiltered row, at whatever the depth packs it: two
 * bytes big endian at sixteen, one byte at eight, and below that the high bits
 * of a byte before the low ones. */
static int png_sample(const uint8_t *line_data, int slot_index, int deep_count) {
  if (deep_count == 16)
    return ((int)line_data[(size_t)slot_index * 2] << 8) | line_data[(size_t)slot_index * 2 + 1];
  if (deep_count == 8) return line_data[slot_index];
  {
    int per_byte = 8 / deep_count;
    int shift_count = 8 - deep_count * (slot_index % per_byte + 1);
    return (line_data[slot_index / per_byte] >> shift_count) & ((1 << deep_count) - 1);
  }
}

static app_code png_read(const uint8_t *file_data, size_t file_size, flat_grid *grid_out) {
  static const uint8_t mark_list[8] = {137, 'P', 'N', 'G', 13, 10, 26, 10};
  uint8_t *palette_data = NULL;
  uint8_t *pack_data = NULL;
  uint8_t *raw_data = NULL;
  size_t pack_size = 0, pack_fill = 0, raw_size = 0, line_bytes = 0, walk;
  int wide_count = 0, high_count = 0, deep_count = 0, kind_mark = -1, weave_mark = 0;
  int lane_count = 0, step_bytes, band_count, pass_index, pass_count, turn_index;
  int high_index, wide_index, band_index;
  app_code code = APP_FAIL_FORMAT;
  float level_span;

  memset(grid_out, 0, sizeof(*grid_out));
  if (file_size < 8 || memcmp(file_data, mark_list, 8) != 0) return APP_FAIL_FORMAT;

  /* Two passes: the first reads the header and measures the payload, the
   * second copies it.  A PNG may split its data over any number of chunks and
   * nothing announces the total, so one pass would mean growing a buffer. */
  for (turn_index = 0; turn_index < 2; ++turn_index) {
    walk = 8;
    pack_fill = 0;
    while (walk + 12 <= file_size) {
      uint32_t body_size = png_word(file_data + walk);
      const uint8_t *name_text = file_data + walk + 4;
      const uint8_t *body_data = file_data + walk + 8;
      if ((size_t)body_size > file_size || walk + 12 + (size_t)body_size > file_size) goto png_done;
      if (memcmp(name_text, "IHDR", 4) == 0) {
        if (body_size < 13) goto png_done;
        wide_count = (int)png_word(body_data);
        high_count = (int)png_word(body_data + 4);
        deep_count = body_data[8];
        kind_mark = body_data[9];
        weave_mark = body_data[12];
      } else if (memcmp(name_text, "PLTE", 4) == 0 && turn_index == 1) {
        mem_free(palette_data);
        palette_data = (uint8_t *)mem_clear(768);
        if (!palette_data) { code = APP_FAIL_MEMORY; goto png_done; }
        memcpy(palette_data, body_data, body_size < 768u ? (size_t)body_size : 768u);
      } else if (memcmp(name_text, "IDAT", 4) == 0) {
        if (turn_index == 0) {
          pack_size += body_size;
        } else {
          if (pack_fill + body_size > pack_size) goto png_done;
          memcpy(pack_data + pack_fill, body_data, body_size);
          pack_fill += body_size;
        }
      } else if (memcmp(name_text, "IEND", 4) == 0) {
        break;
      }
      walk += 12 + (size_t)body_size;
    }
    if (turn_index == 0) {
      if (wide_count < 1 || high_count < 1 || pack_size == 0) goto png_done;
      if (wide_count > MEDIA_SIDE_LIMIT || high_count > MEDIA_SIDE_LIMIT) {
        code = APP_FAIL_SUPPORT;
        goto png_done;
      }
      if (weave_mark != 0 && weave_mark != 1) { code = APP_FAIL_SUPPORT; goto png_done; }
      /* Each colour kind allows its own set of depths and nothing else, so the
       * two are checked together rather than one and then the other. */
      switch (kind_mark) {
        case 0: lane_count = 1; break; /* grey: 1, 2, 4, 8, 16 */
        case 2: lane_count = 3; break;
        case 3: lane_count = 1; break; /* palette: 1, 2, 4, 8 */
        case 4: lane_count = 2; break;
        case 6: lane_count = 4; break;
        default: code = APP_FAIL_SUPPORT; goto png_done;
      }
      if (deep_count != 1 && deep_count != 2 && deep_count != 4 && deep_count != 8 &&
          deep_count != 16) {
        code = APP_FAIL_SUPPORT;
        goto png_done;
      }
      if (kind_mark != 0 && kind_mark != 3 && deep_count < 8) {
        code = APP_FAIL_SUPPORT;
        goto png_done;
      }
      if (kind_mark == 3 && deep_count == 16) { code = APP_FAIL_SUPPORT; goto png_done; }
      pack_data = (uint8_t *)mem_clear(pack_size);
      if (!pack_data) { code = APP_FAIL_MEMORY; goto png_done; }
    }
  }

  /* A byte of filter carries the width of one pixel, or one byte where a pixel
   * is narrower than that. */
  step_bytes = lane_count * deep_count / 8;
  if (step_bytes < 1) step_bytes = 1;

  /* Seven lattices where the file is interlaced and one where it is not.  Each
   * of them is a picture of its own in the stream: its own rows, its own filter
   * byte a row, and its filters looking back only within the lattice. */
  pass_count = weave_mark ? 7 : 1;
  raw_size = 0;
  for (pass_index = 0; pass_index < pass_count; ++pass_index) {
    int from_x, from_y, step_x, step_y, pass_wide, pass_high;
    png_pass_shape(weave_mark, pass_index, &from_x, &from_y, &step_x, &step_y);
    pass_wide = png_pass_span(wide_count, from_x, step_x);
    pass_high = png_pass_span(high_count, from_y, step_y);
    /* A lattice that catches no pixel is not in the stream at all, and a
     * lattice with rows but no columns catches none: a picture narrower than
     * the lattice is the case that tells the two apart. */
    if (pass_wide < 1 || pass_high < 1) continue;
    raw_size += (size_t)pass_high * (png_pass_bytes(pass_wide, lane_count, deep_count) + 1u);
  }
  if (raw_size == 0) goto png_done;
  raw_data = (uint8_t *)mem_clear(raw_size);
  if (!raw_data) { code = APP_FAIL_MEMORY; goto png_done; }
  if (puff_run(pack_data, pack_fill, raw_data, raw_size) != (long)raw_size) goto png_done;

  /* Alpha is dropped rather than composited: a tower is shown the colour the
   * file recorded, and inventing a background would be a preprocessing choice
   * this layer has no business making. */
  band_count = (kind_mark == 2 || kind_mark == 6 || kind_mark == 3) ? 3 : 1;
  code = grid_open(grid_out, wide_count, high_count, band_count);
  if (code != APP_OKAY) goto png_done;
  level_span = (float)((1 << deep_count) - 1);
  if (deep_count == 16) level_span = 65535.0f;
  walk = 0;
  for (pass_index = 0; pass_index < pass_count; ++pass_index) {
    int from_x, from_y, step_x, step_y, pass_wide, pass_high;
    png_pass_shape(weave_mark, pass_index, &from_x, &from_y, &step_x, &step_y);
    pass_wide = png_pass_span(wide_count, from_x, step_x);
    pass_high = png_pass_span(high_count, from_y, step_y);
    if (pass_wide < 1 || pass_high < 1) continue;
    line_bytes = png_pass_bytes(pass_wide, lane_count, deep_count);
    png_unfilter(raw_data + walk, pass_high, line_bytes, step_bytes);
    for (high_index = 0; high_index < pass_high; ++high_index) {
      const uint8_t *line_data = raw_data + walk + (size_t)high_index * (line_bytes + 1) + 1;
      for (wide_index = 0; wide_index < pass_wide; ++wide_index) {
        float *out_data =
            grid_at(grid_out, from_y + high_index * step_y, from_x + wide_index * step_x);
        if (kind_mark == 3) {
          int slot_index = png_sample(line_data, wide_index, deep_count);
          for (band_index = 0; band_index < 3; ++band_index)
            out_data[band_index] =
                palette_data ? (float)palette_data[slot_index * 3 + band_index] / 255.0f : 0.0f;
          continue;
        }
        for (band_index = 0; band_index < band_count; ++band_index)
          out_data[band_index] =
              (float)png_sample(line_data, wide_index * lane_count + band_index, deep_count) /
              level_span;
      }
    }
    walk += (line_bytes + 1) * (size_t)pass_high;
  }
  code = APP_OKAY;

png_done:
  mem_free(pack_data);
  mem_free(raw_data);
  mem_free(palette_data);
  if (code != APP_OKAY) grid_free(grid_out);
  return code;
}

/* -- portable any-map ----------------------------------------------------- */

/* Reads one decimal field, skipping the whitespace and `#` comments the format
 * allows between any two of them. */
static int pnm_field(const uint8_t *file_data, size_t file_size, size_t *walk_out, int *value_out) {
  size_t walk = *walk_out;
  int value_now = 0, digit_flag = 0;
  while (walk < file_size) {
    uint8_t letter = file_data[walk];
    if (letter == '#') {
      while (walk < file_size && file_data[walk] != '\n') walk += 1;
    } else if (letter == ' ' || letter == '\t' || letter == '\r' || letter == '\n') {
      if (digit_flag) break;
      walk += 1;
    } else if (letter >= '0' && letter <= '9') {
      digit_flag = 1;
      value_now = value_now * 10 + (letter - '0');
      walk += 1;
    } else {
      return 0;
    }
  }
  if (!digit_flag) return 0;
  *walk_out = walk;
  *value_out = value_now;
  return 1;
}

static app_code pnm_read(const uint8_t *file_data, size_t file_size, flat_grid *grid_out) {
  size_t walk = 2;
  int wide_count = 0, high_count = 0, level_peak = 0;
  int band_count, wide_index, high_index, band_index, deep_bytes;
  app_code code;

  memset(grid_out, 0, sizeof(*grid_out));
  if (file_size < 3 || file_data[0] != 'P') return APP_FAIL_FORMAT;
  if (file_data[1] != '5' && file_data[1] != '6') return APP_FAIL_SUPPORT;
  band_count = file_data[1] == '6' ? 3 : 1;
  if (!pnm_field(file_data, file_size, &walk, &wide_count)) return APP_FAIL_FORMAT;
  if (!pnm_field(file_data, file_size, &walk, &high_count)) return APP_FAIL_FORMAT;
  if (!pnm_field(file_data, file_size, &walk, &level_peak)) return APP_FAIL_FORMAT;
  if (level_peak < 1 || level_peak > 65535) return APP_FAIL_FORMAT;
  walk += 1; /* exactly one whitespace byte separates the header from the body */
  deep_bytes = level_peak > 255 ? 2 : 1;
  if (walk + (size_t)wide_count * (size_t)high_count * (size_t)band_count * (size_t)deep_bytes >
      file_size)
    return APP_FAIL_FORMAT;
  code = grid_open(grid_out, wide_count, high_count, band_count);
  if (code != APP_OKAY) return code;
  for (high_index = 0; high_index < high_count; ++high_index)
    for (wide_index = 0; wide_index < wide_count; ++wide_index) {
      float *out_data = grid_at(grid_out, high_index, wide_index);
      for (band_index = 0; band_index < band_count; ++band_index) {
        const uint8_t *cell_data =
            file_data + walk +
            (((size_t)high_index * (size_t)wide_count + (size_t)wide_index) * (size_t)band_count +
             (size_t)band_index) *
                (size_t)deep_bytes;
        int level_value = deep_bytes == 2 ? (((int)cell_data[0] << 8) | cell_data[1]) : cell_data[0];
        out_data[band_index] = (float)level_value / (float)level_peak;
      }
    }
  return APP_OKAY;
}

/* -- bitmap --------------------------------------------------------------- */

static uint32_t byte_word(const uint8_t *data) {
  return (uint32_t)data[0] | ((uint32_t)data[1] << 8) | ((uint32_t)data[2] << 16) |
         ((uint32_t)data[3] << 24);
}

static app_code bmp_read(const uint8_t *file_data, size_t file_size, flat_grid *grid_out) {
  uint32_t body_from, head_size, press_kind;
  int wide_count, high_count, deep_count, flip_flag = 0;
  size_t line_bytes;
  int wide_index, high_index;
  app_code code;

  memset(grid_out, 0, sizeof(*grid_out));
  if (file_size < 54 || file_data[0] != 'B' || file_data[1] != 'M') return APP_FAIL_FORMAT;
  body_from = byte_word(file_data + 10);
  head_size = byte_word(file_data + 14);
  if (head_size < 40) return APP_FAIL_SUPPORT;
  wide_count = (int)byte_word(file_data + 18);
  high_count = (int)byte_word(file_data + 22);
  deep_count = (int)(byte_word(file_data + 28) & 0xFFFFu);
  press_kind = byte_word(file_data + 30);
  if (high_count < 0) { high_count = -high_count; flip_flag = 1; }
  if (wide_count < 1 || high_count < 1) return APP_FAIL_FORMAT;
  if (press_kind != 0 && press_kind != 3) return APP_FAIL_SUPPORT;
  if (deep_count != 24 && deep_count != 32) return APP_FAIL_SUPPORT;
  line_bytes = (((size_t)wide_count * (size_t)(deep_count / 8) + 3u) / 4u) * 4u;
  if ((size_t)body_from + line_bytes * (size_t)high_count > file_size) return APP_FAIL_FORMAT;
  code = grid_open(grid_out, wide_count, high_count, 3);
  if (code != APP_OKAY) return code;
  for (high_index = 0; high_index < high_count; ++high_index) {
    /* Rows run bottom to top unless the height was recorded negative. */
    int source_row = flip_flag ? high_index : high_count - 1 - high_index;
    const uint8_t *line_data = file_data + body_from + (size_t)source_row * line_bytes;
    for (wide_index = 0; wide_index < wide_count; ++wide_index) {
      const uint8_t *cell_data = line_data + (size_t)wide_index * (size_t)(deep_count / 8);
      float *out_data = grid_at(grid_out, high_index, wide_index);
      out_data[0] = (float)cell_data[2] / 255.0f;
      out_data[1] = (float)cell_data[1] / 255.0f;
      out_data[2] = (float)cell_data[0] / 255.0f;
    }
  }
  return APP_OKAY;
}

/* -- jpeg ----------------------------------------------------------------- */

/* A jpeg reader: the marker walk, a canonical Huffman decode per component,
 * dequantization against the tables `DQT` carried, an eight by eight inverse
 * cosine transform, chroma upsampling at whatever the sampling factors say, and
 * the colour transform the frame's components and its `APP14` segment name.
 * Baseline, extended sequential and progressive frames are read, at eight or
 * twelve bits of precision, over one, three or four components.  Restart
 * markers are handled in every one of them.
 *
 * Sequential and progressive are two decoders sharing this marker walk.  A
 * sequential block is final when its scan has read it, so it is dequantized and
 * transformed where it is read and nothing outlives it.  A progressive block
 * arrives across several scans with successive approximation, so the frame's
 * whole coefficient store is held in `coef_data` until the last scan lands and
 * the transform is one pass over the store at the end.  What the two share
 * besides the walk is the Huffman decode, the bit reader, the transform, the
 * upsampling and the colour transform; what is progressive's alone is the
 * store, the four scan kinds and the end-of-band run.
 *
 * Nothing here is borrowed.  `stb_image.h` was read for where a decoder has to
 * make a choice rather than follow the specification — what to do with a marker
 * in the middle of the entropy stream, and where the transform rounds — and the
 * two places this reader decides differently are named where they are decided.
 *
 * The canonical decoder beside deflate's is not reused.  `DHT` already ships
 * the table in exactly the form `puff_tree` holds — a count per code length,
 * then the symbols in code order — so there is nothing to build; and the walk
 * over it reads bits the other way round, most significant first, out of a
 * stream where `FF 00` means a literal `FF`.  What could be shared is four
 * lines of arithmetic over a bit reader that cannot be. */

#define JPEG_PART_LIMIT  4 /* components in one frame */
#define JPEG_TREE_LIMIT  4 /* Huffman tables a class */
#define JPEG_QUANT_LIMIT 4

/* Where a coefficient in the order the entropy stream sends them belongs in the
 * eight by eight block: row major, the row being the vertical frequency. */
static const uint8_t jpeg_zig_list[64] = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

/* A canonical Huffman table as `DHT` ships it: how many codes carry each of
 * the sixteen lengths, and the symbols in code order. */
typedef struct jpeg_tree {
  short count_list[17];
  short sign_list[256];
  int   have_flag;
} jpeg_tree;

/* The entropy stream: bits most significant first, `FF 00` standing for a
 * literal `FF`, and any other `FF` being the marker that ends the run.
 *
 * `band_left` is the end-of-band run a progressive scan is inside — how many
 * more blocks the last end-of-band code stands for.  It is a property of the
 * scan rather than of a component because a run carries across the blocks of
 * the one component a band scan walks, and a restart marker ends it. */
typedef struct jpeg_scan {
  const uint8_t *from_data;
  size_t         from_size;
  size_t         from_walk;
  uint32_t       bit_room;
  int            bit_count;
  int            mark_flag; /* the run reached a marker and is feeding zeros */
  int32_t        band_left;
} jpeg_scan;

typedef struct jpeg_part {
  int       id_mark;
  int       wide_share, high_share; /* sampling factors */
  int       quant_slot;
  int       dc_slot, ac_slot;
  int       wide_size, high_size;   /* samples this component actually carries */
  int       wide_blocks, high_blocks; /* blocks the frame holds, edge padding in */
  int       line_step;              /* stride in samples of the plane below */
  int       high_room;              /* rows of the plane below */
  int       last_dc;                /* the running dc predictor */
  uint16_t *plane_data;
  int32_t  *coef_data;              /* progressive only, in the sent order */
  uint16_t  quant_own[64];          /* the table as it stood when a scan claimed it */
  int       quant_kept;
} jpeg_part;

/* One bit, or a zero once the run has reached its marker.
 *
 * A marker is not consumed and does not fail: an encoder ends a run on a byte
 * boundary, and a decoder that needed the last few bits of a block would rather
 * read zeros than refuse a file every other reader accepts.  What it does is
 * raise `mark_flag`, and the scan loop refuses a file that is still fabricating
 * bits with more than its last unit to go — which is the difference between a
 * stream that ended tidily and one that was truncated. */
static int jpeg_bit(jpeg_scan *scan) {
  if (scan->bit_count == 0) {
    int byte_value;
    if (scan->mark_flag || scan->from_walk >= scan->from_size) {
      scan->mark_flag = 1;
      return 0;
    }
    byte_value = scan->from_data[scan->from_walk];
    if (byte_value == 0xFF) {
      int next_value =
          scan->from_walk + 1 < scan->from_size ? scan->from_data[scan->from_walk + 1] : 0xD9;
      if (next_value != 0x00) {
        scan->mark_flag = 1;
        return 0;
      }
      scan->from_walk += 2;
    } else {
      scan->from_walk += 1;
    }
    scan->bit_room = (uint32_t)byte_value;
    scan->bit_count = 8;
  }
  scan->bit_count -= 1;
  return (int)((scan->bit_room >> scan->bit_count) & 1u);
}

static int jpeg_take(jpeg_scan *scan, int bit_count) {
  int value_now = 0, bit_index;
  for (bit_index = 0; bit_index < bit_count; ++bit_index)
    value_now = (value_now << 1) | jpeg_bit(scan);
  return value_now;
}

/* The signed value a run of `bit_count` bits stands for: the top half of the
 * range as it reads, the bottom half as a negative of the same magnitude. */
static int jpeg_wide(int value_now, int bit_count) {
  if (bit_count == 0) return 0;
  return value_now < (1 << (bit_count - 1)) ? value_now - (1 << bit_count) + 1 : value_now;
}

/* The canonical walk: one bit at a time, comparing against the first code of
 * each length.  Same shape as `puff_sign`, over sixteen lengths and this
 * stream's bit order. */
static int jpeg_sign(jpeg_scan *scan, const jpeg_tree *tree) {
  int code_value = 0, first_code = 0, index_base = 0, length_index;
  for (length_index = 1; length_index <= 16; ++length_index) {
    int count_value = tree->count_list[length_index];
    code_value |= jpeg_bit(scan);
    if (code_value - first_code < count_value)
      return tree->sign_list[index_base + (code_value - first_code)];
    index_base += count_value;
    first_code = (first_code + count_value) << 1;
    code_value <<= 1;
  }
  return -1;
}

/* The separable inverse transform, through an orthonormal basis built once for
 * the whole picture.
 *
 * Two eight by eight products rather than one sixty-four wide one, and the
 * rounding is a single round to nearest at the end because everything before it
 * is float.  A fixed point transform would round twice and land within a step
 * of this; the difference shows against libjpeg and is why `app_diff.py` holds
 * a jpeg to a looser floor than a png. */
static void jpeg_basis_fill(float *basis_data) {
  int freq_index, slot_index;
  for (freq_index = 0; freq_index < 8; ++freq_index) {
    double gain_value = freq_index == 0 ? sqrt(0.125) : 0.5;
    for (slot_index = 0; slot_index < 8; ++slot_index)
      basis_data[freq_index * 8 + slot_index] =
          (float)(gain_value *
                  cos((2.0 * slot_index + 1.0) * freq_index * 3.14159265358979323846 / 16.0));
  }
}

/* The level shift is half the range the frame's precision names, so the same
 * transform serves an eight bit frame and a twelve bit one. */
static void jpeg_block_turn(const float *basis_data, const float *coef_data, uint16_t *into_data,
                            int line_step, int level_mid, int peak_value) {
  float mid_list[64];
  int row_index, slot_index, freq_index;
  for (row_index = 0; row_index < 8; ++row_index)
    for (slot_index = 0; slot_index < 8; ++slot_index) {
      float total = 0.0f;
      for (freq_index = 0; freq_index < 8; ++freq_index)
        total += basis_data[freq_index * 8 + slot_index] * coef_data[row_index * 8 + freq_index];
      mid_list[row_index * 8 + slot_index] = total;
    }
  for (row_index = 0; row_index < 8; ++row_index)
    for (slot_index = 0; slot_index < 8; ++slot_index) {
      float total = 0.0f;
      int level_value;
      for (freq_index = 0; freq_index < 8; ++freq_index)
        total += basis_data[freq_index * 8 + row_index] * mid_list[freq_index * 8 + slot_index];
      level_value = (int)(total + (float)level_mid + 0.5f); /* the level shift and the one rounding */
      if (level_value < 0) level_value = 0;
      if (level_value > peak_value) level_value = peak_value;
      into_data[(size_t)row_index * (size_t)line_step + (size_t)slot_index] =
          (uint16_t)level_value;
    }
}

/* One sequential block: the dc difference against the component's running
 * predictor, then the ac coefficients as run-length pairs, dequantized where
 * they land and transformed straight into the component's plane.  Sequential
 * means a block is final when its scan has read it, so no coefficient buffer
 * outlives this. */
static int jpeg_block_read(jpeg_scan *scan, jpeg_part *part, const jpeg_tree *dc_tree,
                           const jpeg_tree *ac_tree, const uint16_t *quant_list,
                           const float *basis_data, uint16_t *into_data, int line_step,
                           int level_mid, int peak_value) {
  float coef_list[64];
  int sign_value, slot_index;

  for (slot_index = 0; slot_index < 64; ++slot_index) coef_list[slot_index] = 0.0f;
  sign_value = jpeg_sign(scan, dc_tree);
  if (sign_value < 0 || sign_value > 16) return 0;
  part->last_dc += jpeg_wide(jpeg_take(scan, sign_value), sign_value);
  coef_list[0] = (float)part->last_dc * (float)quant_list[0];

  slot_index = 1;
  while (slot_index < 64) {
    int run_value, size_value;
    sign_value = jpeg_sign(scan, ac_tree);
    if (sign_value < 0) return 0;
    run_value = (sign_value >> 4) & 15;
    size_value = sign_value & 15;
    if (size_value == 0) {
      if (run_value != 15) break; /* end of block */
      slot_index += 16;
      continue;
    }
    slot_index += run_value;
    if (slot_index > 63) return 0;
    coef_list[jpeg_zig_list[slot_index]] =
        (float)jpeg_wide(jpeg_take(scan, size_value), size_value) * (float)quant_list[slot_index];
    slot_index += 1;
  }
  jpeg_block_turn(basis_data, coef_list, into_data, line_step, level_mid, peak_value);
  return 1;
}

/* The four progressive scan kinds, each over one block of the coefficient
 * store, which holds the coefficients in the order the stream sends them —
 * the same order the quantization table is stored in, so the two are indexed
 * alike and the zigzag is undone once, at the transform.
 *
 * A first scan of a band writes the bits it carries at the position the scan
 * header names; a refining scan of the same band adds one bit below what is
 * already there.  Which of the two a scan is, is the scan header's `Ah`: zero
 * for the first, the position the last scan wrote for a refinement. */

static int jpeg_wave_dc_first(jpeg_scan *scan, jpeg_part *part, const jpeg_tree *dc_tree,
                              int32_t *block_data, int low_bit) {
  int size_value = jpeg_sign(scan, dc_tree);
  if (size_value < 0 || size_value > 16) return 0;
  part->last_dc += jpeg_wide(jpeg_take(scan, size_value), size_value);
  block_data[0] = (int32_t)((uint32_t)part->last_dc << low_bit);
  return 1;
}

static void jpeg_wave_dc_next(jpeg_scan *scan, int32_t *block_data, int low_bit) {
  if (jpeg_bit(scan)) block_data[0] |= (int32_t)(1u << low_bit);
}

/* A band's first scan.  Runs of zeros and a coefficient, as a sequential block
 * sends them, except that the run of zeros can also stand for a run of whole
 * blocks — the end-of-band code, whose length is the two to the power of its
 * run field plus that many appended bits. */
static int jpeg_wave_ac_first(jpeg_scan *scan, const jpeg_tree *ac_tree, int32_t *block_data,
                              int from_slot, int upto_slot, int low_bit) {
  int slot_index = from_slot;
  if (scan->band_left > 0) {
    scan->band_left -= 1;
    return 1;
  }
  while (slot_index <= upto_slot) {
    int sign_value = jpeg_sign(scan, ac_tree), run_value, size_value;
    if (sign_value < 0) return 0;
    run_value = (sign_value >> 4) & 15;
    size_value = sign_value & 15;
    if (size_value != 0) {
      slot_index += run_value;
      if (slot_index > upto_slot) return 0;
      block_data[slot_index] =
          (int32_t)((uint32_t)jpeg_wide(jpeg_take(scan, size_value), size_value) << low_bit);
      slot_index += 1;
    } else if (run_value != 15) {
      scan->band_left = (int32_t)(1u << run_value);
      if (run_value) scan->band_left += (int32_t)jpeg_take(scan, run_value);
      scan->band_left -= 1; /* this block is the first member of the run */
      break;
    } else {
      slot_index += 16;
    }
  }
  return 1;
}

/* A band's refinement.  Every coefficient the earlier scans left nonzero
 * carries one correction bit, in band order, wherever the walk passes it; the
 * run field counts only the coefficients they left zero, and a newly nonzero
 * coefficient lands after that many of them.  The magnitude a refinement can
 * introduce is one bit, so a size field other than one is a broken file. */
static int jpeg_wave_ac_next(jpeg_scan *scan, const jpeg_tree *ac_tree, int32_t *block_data,
                             int from_slot, int upto_slot, int low_bit) {
  int32_t step_up = (int32_t)(1u << low_bit);
  int slot_index = from_slot;
  if (scan->band_left == 0) {
    while (slot_index <= upto_slot) {
      int sign_value = jpeg_sign(scan, ac_tree), run_value, size_value;
      int32_t new_value = 0;
      if (sign_value < 0) return 0;
      run_value = (sign_value >> 4) & 15;
      size_value = sign_value & 15;
      if (size_value != 0) {
        if (size_value != 1) return 0;
        new_value = jpeg_bit(scan) ? step_up : -step_up;
      } else if (run_value != 15) {
        scan->band_left = (int32_t)(1u << run_value);
        if (run_value) scan->band_left += (int32_t)jpeg_take(scan, run_value);
        break; /* the tail below carries this block's corrections and counts it */
      }
      while (slot_index <= upto_slot) {
        if (block_data[slot_index] != 0) {
          if (jpeg_bit(scan) && (block_data[slot_index] & step_up) == 0)
            block_data[slot_index] += block_data[slot_index] >= 0 ? step_up : -step_up;
        } else {
          if (run_value == 0) break;
          run_value -= 1;
        }
        slot_index += 1;
      }
      if (new_value != 0) {
        if (slot_index > upto_slot) return 0;
        block_data[slot_index] = new_value;
      }
      slot_index += 1;
    }
  }
  if (scan->band_left > 0) {
    while (slot_index <= upto_slot) {
      if (block_data[slot_index] != 0) {
        if (jpeg_bit(scan) && (block_data[slot_index] & step_up) == 0)
          block_data[slot_index] += block_data[slot_index] >= 0 ? step_up : -step_up;
      }
      slot_index += 1;
    }
    scan->band_left -= 1;
  }
  return 1;
}

/* The pass a progressive frame ends with: every block of the store dequantized
 * against the table its component claimed, and transformed into the plane the
 * upsampler reads. */
static void jpeg_part_send(const jpeg_part *part, const float *basis_data, int level_mid,
                           int peak_value) {
  int block_x, block_y, slot_index;
  for (block_y = 0; block_y < part->high_blocks; ++block_y)
    for (block_x = 0; block_x < part->wide_blocks; ++block_x) {
      const int32_t *coef_from =
          part->coef_data +
          ((size_t)block_y * (size_t)part->wide_blocks + (size_t)block_x) * 64u;
      float coef_list[64];
      for (slot_index = 0; slot_index < 64; ++slot_index)
        coef_list[jpeg_zig_list[slot_index]] =
            (float)coef_from[slot_index] * (float)part->quant_own[slot_index];
      jpeg_block_turn(basis_data, coef_list,
                      part->plane_data + (size_t)block_y * 8u * (size_t)part->line_step +
                          (size_t)block_x * 8u,
                      part->line_step, level_mid, peak_value);
    }
}

/* Steps over a restart marker: the run is byte aligned again, the predictors
 * and any end-of-band run start from zero, and the two bytes must be `FF D0`
 * through `FF D7`. */
static int jpeg_restart(jpeg_scan *scan, jpeg_part *part_list, int part_count) {
  size_t walk = scan->from_walk;
  int part_index;
  scan->bit_count = 0;
  scan->mark_flag = 0;
  scan->band_left = 0;
  while (walk + 1 < scan->from_size && scan->from_data[walk] == 0xFF &&
         scan->from_data[walk + 1] == 0xFF)
    walk += 1; /* fill bytes before the marker */
  if (walk + 1 >= scan->from_size || scan->from_data[walk] != 0xFF) return 0;
  if (scan->from_data[walk + 1] < 0xD0 || scan->from_data[walk + 1] > 0xD7) return 0;
  scan->from_walk = walk + 2;
  for (part_index = 0; part_index < part_count; ++part_index) part_list[part_index].last_dc = 0;
  return 1;
}

/* One component sampled at a picture pixel's centre.
 *
 * A component at the picture's own sampling factor lands exactly on a sample
 * and this is a load.  A chroma plane at half the factor lands between two, and
 * what comes back is the linear blend of them — which on the two-to-one factors
 * every real file uses is the three-quarters-and-a-quarter of libjpeg's fancy
 * upsampling.  Repeating the nearest sample instead would put a visible step on
 * every chroma edge, and cost the same. */
static float jpeg_part_at(const jpeg_part *part, int wide_peak, int high_peak, int wide_index,
                          int high_index) {
  float wide_spot = ((float)wide_index + 0.5f) * (float)part->wide_share / (float)wide_peak - 0.5f;
  float high_spot = ((float)high_index + 0.5f) * (float)part->high_share / (float)high_peak - 0.5f;
  int wide_from = (int)(wide_spot + 1.0f) - 1; /* a floor, the spot never below -1 */
  int high_from = (int)(high_spot + 1.0f) - 1;
  float wide_part = wide_spot - (float)wide_from;
  float high_part = high_spot - (float)high_from;
  int wide_upto = wide_from + 1, high_upto = high_from + 1;
  float near_row, far_row;
  const uint16_t *plane_data = part->plane_data;

  if (wide_from < 0) wide_from = 0;
  if (high_from < 0) high_from = 0;
  if (wide_upto > part->wide_size - 1) wide_upto = part->wide_size - 1;
  if (high_upto > part->high_size - 1) high_upto = part->high_size - 1;
  if (wide_from > wide_upto) wide_from = wide_upto;
  if (high_from > high_upto) high_from = high_upto;
  {
    const uint16_t *near_data = plane_data + (size_t)high_from * (size_t)part->line_step;
    const uint16_t *far_data = plane_data + (size_t)high_upto * (size_t)part->line_step;
    near_row = (float)near_data[wide_from] +
               ((float)near_data[wide_upto] - (float)near_data[wide_from]) * wide_part;
    far_row = (float)far_data[wide_from] +
              ((float)far_data[wide_upto] - (float)far_data[wide_from]) * wide_part;
  }
  return near_row + (far_row - near_row) * high_part;
}

static app_code jpeg_read(const uint8_t *file_data, size_t file_size, flat_grid *grid_out) {
  jpeg_tree dc_list[JPEG_TREE_LIMIT], ac_list[JPEG_TREE_LIMIT];
  uint16_t quant_list[JPEG_QUANT_LIMIT][64];
  jpeg_part part_list[JPEG_PART_LIMIT];
  float basis_list[64];
  int wide_count = 0, high_count = 0, part_count = 0;
  int wide_peak = 1, high_peak = 1, mcu_wide = 0, mcu_high = 0;
  int rest_span = 0, frame_flag = 0, scan_flag = 0, band_count;
  int wave_flag = 0;                        /* the frame is progressive */
  int deep_count = 8, level_mid = 128, peak_value = 255;
  int adobe_flag = 0, adobe_turn = 0, turn_mark = 1;
  int quant_have[JPEG_QUANT_LIMIT];
  int part_index, wide_index, high_index, slot_index;
  size_t walk = 2;
  app_code code = APP_FAIL_FORMAT;

  memset(grid_out, 0, sizeof(*grid_out));
  memset(dc_list, 0, sizeof(dc_list));
  memset(ac_list, 0, sizeof(ac_list));
  memset(quant_list, 0, sizeof(quant_list));
  memset(quant_have, 0, sizeof(quant_have));
  memset(part_list, 0, sizeof(part_list));
  jpeg_basis_fill(basis_list);
  if (file_size < 4 || file_data[0] != 0xFF || file_data[1] != 0xD8) return APP_FAIL_FORMAT;

  while (walk + 1 < file_size) {
    int mark_value;
    size_t body_size, body_from;
    if (file_data[walk] != 0xFF) { walk += 1; continue; } /* fill between segments */
    mark_value = file_data[walk + 1];
    if (mark_value == 0xFF) { walk += 1; continue; }
    if (mark_value == 0xD8 || (mark_value >= 0xD0 && mark_value <= 0xD7) || mark_value == 0x01) {
      walk += 2;
      continue;
    }
    if (mark_value == 0xD9) break; /* end of image */
    if (walk + 4 > file_size) goto jpeg_done;
    body_size = ((size_t)file_data[walk + 2] << 8) | (size_t)file_data[walk + 3];
    if (body_size < 2 || walk + 2 + body_size > file_size) goto jpeg_done;
    body_from = walk + 4;
    body_size -= 2;

    if (mark_value == 0xDB) { /* quantization tables */
      size_t step = body_from;
      while (step < body_from + body_size) {
        int wide_flag = file_data[step] >> 4, slot_want = file_data[step] & 15;
        step += 1;
        if (slot_want >= JPEG_QUANT_LIMIT) goto jpeg_done;
        if (step + (wide_flag ? 128u : 64u) > body_from + body_size) goto jpeg_done;
        for (slot_index = 0; slot_index < 64; ++slot_index)
          quant_list[slot_want][slot_index] =
              wide_flag ? (uint16_t)(((int)file_data[step + (size_t)slot_index * 2] << 8) |
                                     file_data[step + (size_t)slot_index * 2 + 1])
                        : (uint16_t)file_data[step + (size_t)slot_index];
        quant_have[slot_want] = 1;
        step += wide_flag ? 128u : 64u;
      }
    } else if (mark_value == 0xC4) { /* Huffman tables */
      size_t step = body_from;
      while (step < body_from + body_size) {
        int class_mark = file_data[step] >> 4, slot_want = file_data[step] & 15;
        jpeg_tree *tree;
        int total_count = 0;
        step += 1;
        if (class_mark > 1 || slot_want >= JPEG_TREE_LIMIT) goto jpeg_done;
        if (step + 16 > body_from + body_size) goto jpeg_done;
        tree = class_mark ? &ac_list[slot_want] : &dc_list[slot_want];
        memset(tree, 0, sizeof(*tree));
        for (slot_index = 1; slot_index <= 16; ++slot_index) {
          tree->count_list[slot_index] = (short)file_data[step + (size_t)slot_index - 1];
          total_count += tree->count_list[slot_index];
        }
        step += 16;
        if (total_count > 256 || step + (size_t)total_count > body_from + body_size)
          goto jpeg_done;
        for (slot_index = 0; slot_index < total_count; ++slot_index)
          tree->sign_list[slot_index] = (short)file_data[step + (size_t)slot_index];
        tree->have_flag = 1;
        step += (size_t)total_count;
      }
    } else if (mark_value == 0xDD) { /* restart interval */
      if (body_size < 2) goto jpeg_done;
      rest_span = ((int)file_data[body_from] << 8) | file_data[body_from + 1];
    } else if (mark_value == 0xEE) { /* Adobe APP14: which colour space the bands are in */
      /* `Adobe`, a version, two flag words and the transform byte.  It is the
       * only thing in the file that says whether three bands are YCbCr or RGB
       * and whether four are CMYK or YCCK, and it is also what says the four
       * are written inverted, because every writer that emits it inverts. */
      if (body_size >= 12 && memcmp(file_data + body_from, "Adobe", 5) == 0) {
        adobe_flag = 1;
        adobe_turn = file_data[body_from + 11];
      }
    } else if (mark_value == 0xC0 || mark_value == 0xC1 || mark_value == 0xC2) {
      /* baseline, extended sequential, progressive — all Huffman coded */
      if (frame_flag || body_size < 6) goto jpeg_done;
      deep_count = file_data[body_from];
      if (deep_count != 8 && deep_count != 12) { code = APP_FAIL_SUPPORT; goto jpeg_done; }
      level_mid = 1 << (deep_count - 1);
      peak_value = (1 << deep_count) - 1;
      wave_flag = mark_value == 0xC2;
      high_count = ((int)file_data[body_from + 1] << 8) | file_data[body_from + 2];
      wide_count = ((int)file_data[body_from + 3] << 8) | file_data[body_from + 4];
      part_count = file_data[body_from + 5];
      if (wide_count < 1 || high_count < 1) goto jpeg_done;
      if (wide_count > MEDIA_SIDE_LIMIT || high_count > MEDIA_SIDE_LIMIT) {
        code = APP_FAIL_SUPPORT;
        goto jpeg_done;
      }
      /* One band, three, or four.  Four is CMYK or YCCK, which is read through
       * whatever `APP14` said and refused where it said nothing this reader
       * can act on. */
      if (part_count != 1 && part_count != 3 && part_count != 4) {
        code = APP_FAIL_SUPPORT;
        goto jpeg_done;
      }
      if (body_size < 6u + 3u * (size_t)part_count) goto jpeg_done;
      for (part_index = 0; part_index < part_count; ++part_index) {
        const uint8_t *note_data = file_data + body_from + 6 + (size_t)part_index * 3;
        jpeg_part *part = &part_list[part_index];
        part->id_mark = note_data[0];
        part->wide_share = note_data[1] >> 4;
        part->high_share = note_data[1] & 15;
        part->quant_slot = note_data[2];
        if (part->wide_share < 1 || part->wide_share > 4 || part->high_share < 1 ||
            part->high_share > 4 || part->quant_slot >= JPEG_QUANT_LIMIT) {
          code = APP_FAIL_SUPPORT;
          goto jpeg_done;
        }
        if (part->wide_share > wide_peak) wide_peak = part->wide_share;
        if (part->high_share > high_peak) high_peak = part->high_share;
      }
      mcu_wide = (wide_count + wide_peak * 8 - 1) / (wide_peak * 8);
      mcu_high = (high_count + high_peak * 8 - 1) / (high_peak * 8);
      for (part_index = 0; part_index < part_count; ++part_index) {
        jpeg_part *part = &part_list[part_index];
        part->wide_size = (wide_count * part->wide_share + wide_peak - 1) / wide_peak;
        part->high_size = (high_count * part->high_share + high_peak - 1) / high_peak;
        part->wide_blocks = mcu_wide * part->wide_share;
        part->high_blocks = mcu_high * part->high_share;
        part->line_step = part->wide_blocks * 8;
        part->high_room = part->high_blocks * 8;
        part->plane_data = (uint16_t *)mem_clear(sizeof(uint16_t) * (size_t)part->line_step *
                                                 (size_t)part->high_room);
        if (!part->plane_data) { code = APP_FAIL_MEMORY; goto jpeg_done; }
        if (wave_flag) {
          part->coef_data = (int32_t *)mem_clear(sizeof(int32_t) * (size_t)part->wide_blocks *
                                                 (size_t)part->high_blocks * 64u);
          if (!part->coef_data) { code = APP_FAIL_MEMORY; goto jpeg_done; }
        }
      }
      frame_flag = 1;
    } else if (mark_value >= 0xC3 && mark_value <= 0xCF && mark_value != 0xC4 &&
               mark_value != 0xC8 && mark_value != 0xCC) {
      /* Lossless, differential, arithmetic coded, hierarchical: each of them is
       * another decoder rather than another branch of this one. */
      code = APP_FAIL_SUPPORT;
      goto jpeg_done;
    } else if (mark_value == 0xDA) { /* start of scan */
      jpeg_scan scan;
      int scan_count, unit_wide, unit_high, unit_index, unit_total, rest_left;
      int band_from, band_upto, high_bit, low_bit;
      int scan_slot[JPEG_PART_LIMIT];
      if (!frame_flag || body_size < 1) goto jpeg_done;
      scan_count = file_data[body_from];
      if (scan_count < 1 || scan_count > part_count) goto jpeg_done;
      if (body_size < 1u + 2u * (size_t)scan_count + 3u) goto jpeg_done;
      for (slot_index = 0; slot_index < scan_count; ++slot_index) {
        const uint8_t *note_data = file_data + body_from + 1 + (size_t)slot_index * 2;
        int found_slot = -1;
        for (part_index = 0; part_index < part_count; ++part_index)
          if (part_list[part_index].id_mark == note_data[0]) found_slot = part_index;
        if (found_slot < 0) goto jpeg_done;
        scan_slot[slot_index] = found_slot;
        part_list[found_slot].dc_slot = note_data[1] >> 4;
        part_list[found_slot].ac_slot = note_data[1] & 15;
        if (part_list[found_slot].dc_slot >= JPEG_TREE_LIMIT ||
            part_list[found_slot].ac_slot >= JPEG_TREE_LIMIT)
          goto jpeg_done;
      }
      {
        const uint8_t *tail_data = file_data + body_from + 1 + (size_t)scan_count * 2;
        band_from = tail_data[0];
        band_upto = tail_data[1];
        high_bit = tail_data[2] >> 4;
        low_bit = tail_data[2] & 15;
        if (!wave_flag) {
          /* A sequential scan carries the whole spectrum at full precision.
           * Anything else is a progressive scan under a sequential frame
           * header, which is a file this reader has no frame shape for. */
          if (band_from != 0 || band_upto != 63 || tail_data[2] != 0) {
            code = APP_FAIL_SUPPORT;
            goto jpeg_done;
          }
        } else {
          if (band_upto > 63 || band_from > band_upto || low_bit > 13) goto jpeg_done;
          if (high_bit != 0 && high_bit != low_bit + 1) goto jpeg_done;
          /* The dc scan is the only one that may interleave components; a band
           * above it walks one component's own blocks. */
          if (band_from == 0) {
            if (band_upto != 0) goto jpeg_done;
          } else if (scan_count != 1) {
            goto jpeg_done;
          }
        }
      }
      /* A progressive component's coefficients are dequantized at the end of
       * the file rather than where they are read, so which table stood when its
       * first scan claimed it is the one that has to be kept. */
      if (wave_flag)
        for (slot_index = 0; slot_index < scan_count; ++slot_index) {
          jpeg_part *part = &part_list[scan_slot[slot_index]];
          if (part->quant_kept) continue;
          if (!quant_have[part->quant_slot]) goto jpeg_done;
          memcpy(part->quant_own, quant_list[part->quant_slot], sizeof(part->quant_own));
          part->quant_kept = 1;
        }
      memset(&scan, 0, sizeof(scan));
      scan.from_data = file_data;
      scan.from_size = file_size;
      scan.from_walk = body_from + body_size;
      /* An interleaved scan walks minimum coded units, each holding a
       * component's sampling factors worth of blocks; a scan of one component
       * walks that component's own blocks, which is what a file that sends its
       * planes one after another means. */
      if (scan_count == 1) {
        jpeg_part *part = &part_list[scan_slot[0]];
        unit_wide = (part->wide_size + 7) / 8;
        unit_high = (part->high_size + 7) / 8;
      } else {
        unit_wide = mcu_wide;
        unit_high = mcu_high;
      }
      unit_total = unit_wide * unit_high;
      rest_left = rest_span;
      for (slot_index = 0; slot_index < scan_count; ++slot_index)
        part_list[scan_slot[slot_index]].last_dc = 0;
      for (unit_index = 0; unit_index < unit_total; ++unit_index) {
        int unit_x = unit_index % unit_wide, unit_y = unit_index / unit_wide;
        if (rest_span > 0 && rest_left == 0) {
          if (!jpeg_restart(&scan, part_list, part_count)) goto jpeg_done;
          rest_left = rest_span;
        }
        /* Feeding zeros with more than the last unit to go means the entropy
         * stream ended early rather than tidily. */
        if (scan.mark_flag && unit_index + 1 < unit_total) goto jpeg_done;
        for (slot_index = 0; slot_index < scan_count; ++slot_index) {
          jpeg_part *part = &part_list[scan_slot[slot_index]];
          const jpeg_tree *dc_tree = &dc_list[part->dc_slot];
          const jpeg_tree *ac_tree = &ac_list[part->ac_slot];
          int wide_span = scan_count == 1 ? 1 : part->wide_share;
          int high_span = scan_count == 1 ? 1 : part->high_share;
          int block_x, block_y;
          if (!quant_have[part->quant_slot]) goto jpeg_done;
          if (!wave_flag && (!dc_tree->have_flag || !ac_tree->have_flag)) goto jpeg_done;
          if (wave_flag && band_from == 0 && high_bit == 0 && !dc_tree->have_flag) goto jpeg_done;
          if (wave_flag && band_from != 0 && !ac_tree->have_flag) goto jpeg_done;
          for (block_y = 0; block_y < high_span; ++block_y)
            for (block_x = 0; block_x < wide_span; ++block_x) {
              int into_x = (unit_x * wide_span + block_x) * 8;
              int into_y = (unit_y * high_span + block_y) * 8;
              if (into_x + 8 > part->line_step || into_y + 8 > part->high_room) goto jpeg_done;
              if (!wave_flag) {
                if (!jpeg_block_read(&scan, part, dc_tree, ac_tree, quant_list[part->quant_slot],
                                     basis_list,
                                     part->plane_data + (size_t)into_y * (size_t)part->line_step +
                                         (size_t)into_x,
                                     part->line_step, level_mid, peak_value))
                  goto jpeg_done;
                continue;
              }
              {
                int32_t *block_data =
                    part->coef_data +
                    ((size_t)(into_y / 8) * (size_t)part->wide_blocks + (size_t)(into_x / 8)) * 64u;
                if (band_from == 0) {
                  if (high_bit == 0) {
                    if (!jpeg_wave_dc_first(&scan, part, dc_tree, block_data, low_bit))
                      goto jpeg_done;
                  } else {
                    jpeg_wave_dc_next(&scan, block_data, low_bit);
                  }
                } else if (high_bit == 0) {
                  if (!jpeg_wave_ac_first(&scan, ac_tree, block_data, band_from, band_upto,
                                          low_bit))
                    goto jpeg_done;
                } else {
                  if (!jpeg_wave_ac_next(&scan, ac_tree, block_data, band_from, band_upto,
                                         low_bit))
                    goto jpeg_done;
                }
              }
            }
        }
        rest_left -= 1;
      }
      /* The entropy stream is not length prefixed, so the walk resumes at the
       * next marker the reader can find rather than at a recorded offset. */
      scan_flag = 1;
      walk = scan.from_walk;
      while (walk + 1 < file_size &&
             !(file_data[walk] == 0xFF && file_data[walk + 1] != 0x00 &&
               file_data[walk + 1] != 0xFF))
        walk += 1;
      continue;
    }
    walk = body_from + body_size; /* the segment's length counted its own two bytes */
  }

  if (!frame_flag || !scan_flag) goto jpeg_done;
  if (wave_flag)
    for (part_index = 0; part_index < part_count; ++part_index) {
      jpeg_part *part = &part_list[part_index];
      /* A component no scan ever named has no coefficients and no claim on a
       * table; the frame is incomplete rather than grey. */
      if (!part->quant_kept) goto jpeg_done;
      jpeg_part_send(part, basis_list, level_mid, peak_value);
    }

  /* Which colour space the bands are in.  Three bands are YCbCr unless `APP14`
   * or the component ids say otherwise; four are CMYK or YCCK and only `APP14`
   * can say which, so a four band file without one is refused rather than
   * guessed at — and so is a transform value this reader has no rule for. */
  if (part_count == 3) {
    turn_mark = adobe_flag ? adobe_turn
                           : ((part_list[0].id_mark == 'R' && part_list[1].id_mark == 'G' &&
                               part_list[2].id_mark == 'B')
                                  ? 0
                                  : 1);
    if (turn_mark != 0 && turn_mark != 1) { code = APP_FAIL_SUPPORT; goto jpeg_done; }
  } else if (part_count == 4) {
    if (!adobe_flag) { code = APP_FAIL_SUPPORT; goto jpeg_done; }
    turn_mark = adobe_turn;
    if (turn_mark != 0 && turn_mark != 2) { code = APP_FAIL_SUPPORT; goto jpeg_done; }
  }

  band_count = part_count == 1 ? 1 : 3;
  code = grid_open(grid_out, wide_count, high_count, band_count);
  if (code != APP_OKAY) goto jpeg_done;
  for (high_index = 0; high_index < high_count; ++high_index)
    for (wide_index = 0; wide_index < wide_count; ++wide_index) {
      float *out_data = grid_at(grid_out, high_index, wide_index);
      float band_list[4];
      int band_index;
      if (band_count == 1) {
        out_data[0] = jpeg_part_at(&part_list[0], wide_peak, high_peak, wide_index, high_index) /
                      (float)peak_value;
        continue;
      }
      for (part_index = 0; part_index < part_count; ++part_index)
        band_list[part_index] =
            jpeg_part_at(&part_list[part_index], wide_peak, high_peak, wide_index, high_index);
      if (turn_mark != 0) {
        /* The colour transform the format's own conversion clause names, with
         * the chroma pair centred on zero first.  Under YCCK the three it
         * turns are the ink bands rather than the picture's own. */
        float bright = band_list[0];
        float blue_off = band_list[1] - (float)level_mid;
        float red_off = band_list[2] - (float)level_mid;
        band_list[0] = bright + 1.402f * red_off;
        band_list[1] = bright - 0.344136f * blue_off - 0.714136f * red_off;
        band_list[2] = bright + 1.772f * blue_off;
      }
      if (part_count == 4) {
        /* Four bands are ink, and every writer that ships the `APP14` this
         * reader insisted on writes them inverted — a band at its peak is no
         * ink at all — so the picture is the product of the three with the
         * black, and no separate inversion is needed. */
        for (band_index = 0; band_index < 3; ++band_index) {
          float level = band_list[band_index];
          if (level < 0.0f) level = 0.0f;
          if (level > (float)peak_value) level = (float)peak_value;
          band_list[band_index] = level * band_list[3] / (float)peak_value;
        }
      }
      for (band_index = 0; band_index < 3; ++band_index) {
        float level = band_list[band_index] / (float)peak_value;
        out_data[band_index] = level < 0.0f ? 0.0f : (level > 1.0f ? 1.0f : level);
      }
    }
  code = APP_OKAY;

jpeg_done:
  for (part_index = 0; part_index < JPEG_PART_LIMIT; ++part_index) {
    mem_free(part_list[part_index].plane_data);
    mem_free(part_list[part_index].coef_data);
  }
  if (code != APP_OKAY) grid_free(grid_out);
  return code;
}

/* Picks the reader from the leading bytes rather than the file name, because a
 * checkpoint's own sample images are as likely to be misnamed as anything. */
static app_code image_read(const char *path_text, flat_grid *grid_out) {
  file_map map;
  app_code code;
  memset(grid_out, 0, sizeof(*grid_out));
  code = file_open(path_text, &map);
  if (code != APP_OKAY) return code;
  if (map.byte_count >= 8 && map.base_data[0] == 137 && map.base_data[1] == 'P')
    code = png_read(map.base_data, map.byte_count, grid_out);
  else if (map.byte_count >= 2 && map.base_data[0] == 'P' &&
           (map.base_data[1] == '5' || map.base_data[1] == '6'))
    code = pnm_read(map.base_data, map.byte_count, grid_out);
  else if (map.byte_count >= 2 && map.base_data[0] == 'B' && map.base_data[1] == 'M')
    code = bmp_read(map.base_data, map.byte_count, grid_out);
  else if (map.byte_count >= 3 && map.base_data[0] == 0xFF && map.base_data[1] == 0xD8 &&
           map.base_data[2] == 0xFF)
    code = jpeg_read(map.base_data, map.byte_count, grid_out);
  else
    code = APP_FAIL_SUPPORT;
  file_close(&map);
  return code;
}

/* -- bicubic resize ------------------------------------------------------- */

/* The a = -0.5 member of the cubic family, which is what every reference
 * preprocessor means by "bicubic". */
static float grid_curve(float step_value) {
  const float bend_value = -0.5f;
  float step_abs = step_value < 0.0f ? -step_value : step_value;
  if (step_abs < 1.0f)
    return ((bend_value + 2.0f) * step_abs - (bend_value + 3.0f)) * step_abs * step_abs + 1.0f;
  if (step_abs < 2.0f)
    return ((bend_value * step_abs - 5.0f * bend_value) * step_abs + 8.0f * bend_value) * step_abs -
           4.0f * bend_value;
  return 0.0f;
}

/* Weights for one output axis.
 *
 * The support widens with the reduction factor, so shrinking an image averages
 * over everything it passes rather than sampling sixteen pixels out of a
 * thousand and aliasing the rest.  That is what PIL and torchvision do when
 * antialiasing is on, and taking the plain four-tap filter instead shows up as
 * speckle in any downscaled photograph.
 *
 * Centres are half-pixel — `source = (target + 0.5) * ratio - 0.5` — so the two
 * images cover the same area rather than sharing a corner. */
static app_code grid_axis(int from_count, int into_count, int **from_out, int **span_out,
                          float **weight_out, int *tap_out) {
  double ratio_value = (double)from_count / (double)into_count;
  double spread_value = ratio_value > 1.0 ? ratio_value : 1.0;
  double reach_value = 2.0 * spread_value;
  int tap_limit = (int)ceil(reach_value) * 2 + 2;
  int *from_list = (int *)mem_clear(sizeof(int) * (size_t)into_count);
  int *span_list = (int *)mem_clear(sizeof(int) * (size_t)into_count);
  float *weight_list = (float *)mem_clear(sizeof(float) * (size_t)into_count * (size_t)tap_limit);
  int into_index;

  *from_out = from_list;
  *span_out = span_list;
  *weight_out = weight_list;
  *tap_out = tap_limit;
  if (!from_list || !span_list || !weight_list) return APP_FAIL_MEMORY;

  for (into_index = 0; into_index < into_count; ++into_index) {
    double centre_value = ((double)into_index + 0.5) * ratio_value;
    int from_index = (int)floor(centre_value - reach_value + 0.5);
    int upto_index = (int)floor(centre_value + reach_value + 0.5) + 1;
    float *lane_list = weight_list + (size_t)into_index * (size_t)tap_limit;
    float total_value = 0.0f;
    int tap_index, tap_count;
    if (from_index < 0) from_index = 0;
    if (upto_index > from_count) upto_index = from_count;
    if (upto_index <= from_index) upto_index = from_index + 1;
    tap_count = upto_index - from_index;
    if (tap_count > tap_limit) tap_count = tap_limit;
    for (tap_index = 0; tap_index < tap_count; ++tap_index) {
      double offset_value = ((double)(from_index + tap_index) + 0.5 - centre_value) / spread_value;
      lane_list[tap_index] = grid_curve((float)offset_value);
      total_value += lane_list[tap_index];
    }
    if (total_value != 0.0f)
      for (tap_index = 0; tap_index < tap_count; ++tap_index) lane_list[tap_index] /= total_value;
    else
      lane_list[0] = 1.0f;
    from_list[into_index] = from_index;
    span_list[into_index] = tap_count;
  }
  return APP_OKAY;
}

static app_code grid_scale(const flat_grid *from_grid, int wide_want, int high_want,
                           flat_grid *into_grid) {
  flat_grid mid_grid;
  int *wide_from = NULL, *wide_span = NULL, *high_from = NULL, *high_span = NULL;
  float *wide_weight = NULL, *high_weight = NULL;
  int wide_tap = 0, high_tap = 0;
  int band_count = from_grid->band_count;
  int high_index, wide_index, band_index, tap_index;
  app_code code;

  memset(&mid_grid, 0, sizeof(mid_grid));
  memset(into_grid, 0, sizeof(*into_grid));
  if (wide_want < 1 || high_want < 1) return APP_FAIL_ARGUMENT;
  code = grid_axis(from_grid->wide_count, wide_want, &wide_from, &wide_span, &wide_weight, &wide_tap);
  if (code == APP_OKAY)
    code =
        grid_axis(from_grid->high_count, high_want, &high_from, &high_span, &high_weight, &high_tap);
  if (code == APP_OKAY) code = grid_open(&mid_grid, wide_want, from_grid->high_count, band_count);
  if (code == APP_OKAY) code = grid_open(into_grid, wide_want, high_want, band_count);
  if (code != APP_OKAY) goto scale_done;

  for (high_index = 0; high_index < from_grid->high_count; ++high_index)
    for (wide_index = 0; wide_index < wide_want; ++wide_index) {
      const float *lane_list = wide_weight + (size_t)wide_index * (size_t)wide_tap;
      float *out_data = grid_at(&mid_grid, high_index, wide_index);
      for (band_index = 0; band_index < band_count; ++band_index) out_data[band_index] = 0.0f;
      for (tap_index = 0; tap_index < wide_span[wide_index]; ++tap_index) {
        const float *in_data = grid_at(from_grid, high_index, wide_from[wide_index] + tap_index);
        for (band_index = 0; band_index < band_count; ++band_index)
          out_data[band_index] += lane_list[tap_index] * in_data[band_index];
      }
    }

  for (high_index = 0; high_index < high_want; ++high_index)
    for (wide_index = 0; wide_index < wide_want; ++wide_index) {
      const float *lane_list = high_weight + (size_t)high_index * (size_t)high_tap;
      float *out_data = grid_at(into_grid, high_index, wide_index);
      for (band_index = 0; band_index < band_count; ++band_index) out_data[band_index] = 0.0f;
      for (tap_index = 0; tap_index < high_span[high_index]; ++tap_index) {
        const float *in_data = grid_at(&mid_grid, high_from[high_index] + tap_index, wide_index);
        for (band_index = 0; band_index < band_count; ++band_index)
          out_data[band_index] += lane_list[tap_index] * in_data[band_index];
      }
    }

scale_done:
  grid_free(&mid_grid);
  mem_free(wide_from);
  mem_free(wide_span);
  mem_free(wide_weight);
  mem_free(high_from);
  mem_free(high_span);
  mem_free(high_weight);
  if (code != APP_OKAY) grid_free(into_grid);
  return code;
}

/* Spreads a one band raster over three, or folds three onto one, so the reader
 * and the tower need not agree about how many bands the file carried. */
static app_code grid_bands(flat_grid *grid, int band_want) {
  flat_grid made;
  int high_index, wide_index, band_index;
  app_code code;
  if (grid->band_count == band_want) return APP_OKAY;
  code = grid_open(&made, grid->wide_count, grid->high_count, band_want);
  if (code != APP_OKAY) return code;
  for (high_index = 0; high_index < grid->high_count; ++high_index)
    for (wide_index = 0; wide_index < grid->wide_count; ++wide_index) {
      const float *in_data = grid_at(grid, high_index, wide_index);
      float *out_data = grid_at(&made, high_index, wide_index);
      if (grid->band_count == 1) {
        for (band_index = 0; band_index < band_want; ++band_index) out_data[band_index] = in_data[0];
      } else if (band_want == 1) {
        /* The luma weights rather than a plain mean: a one band tower trained
         * on photographs saw luminance. */
        out_data[0] = 0.299f * in_data[0] + 0.587f * in_data[1] + 0.114f * in_data[2];
      } else {
        for (band_index = 0; band_index < band_want; ++band_index)
          out_data[band_index] = band_index < grid->band_count ? in_data[band_index] : 0.0f;
      }
    }
  grid_free(grid);
  *grid = made;
  return APP_OKAY;
}

/* -- wave ----------------------------------------------------------------- */

typedef struct wave_clip {
  float *value_data;
  int    value_count;
  int    rate_value;
} wave_clip;

static void wave_free(wave_clip *clip) {
  if (!clip) return;
  mem_free(clip->value_data);
  memset(clip, 0, sizeof(*clip));
}

static float wave_sample(const uint8_t *cell_data, int deep_count, int real_flag) {
  if (real_flag) {
    union { uint32_t bits; float real; } cast;
    if (deep_count != 32) return 0.0f;
    cast.bits = byte_word(cell_data);
    return cast.real;
  }
  switch (deep_count) {
    case 8: return ((float)cell_data[0] - 128.0f) / 128.0f;
    case 16: {
      int16_t value = (int16_t)((uint16_t)cell_data[0] | ((uint16_t)cell_data[1] << 8));
      return (float)value / 32768.0f;
    }
    case 24: {
      /* Sign extended by landing the three bytes in the top of a word and
       * shifting back down, which needs no branch on the sign bit. */
      int32_t value = (int32_t)(((uint32_t)cell_data[0] << 8) | ((uint32_t)cell_data[1] << 16) |
                                ((uint32_t)cell_data[2] << 24));
      return (float)(value >> 8) / 8388608.0f;
    }
    case 32: {
      int32_t value = (int32_t)byte_word(cell_data);
      return (float)value / 2147483648.0f;
    }
    default: return 0.0f;
  }
}

/* Reads a RIFF wave into one mono track.  Channels are averaged rather than
 * dropped, because a stereo recording with one silent side is common enough
 * that taking the first channel would sometimes hand the tower silence. */
static app_code wave_read(const char *path_text, wave_clip *clip_out) {
  file_map map;
  app_code code;
  size_t walk;
  const uint8_t *body_data = NULL;
  size_t body_size = 0;
  int lane_count = 0, rate_value = 0, deep_count = 0, real_flag = 0, form_kind = 0;
  int value_count, value_index, lane_index;

  memset(clip_out, 0, sizeof(*clip_out));
  code = file_open(path_text, &map);
  if (code != APP_OKAY) return code;
  code = APP_FAIL_FORMAT;
  if (map.byte_count < 44 || memcmp(map.base_data, "RIFF", 4) != 0 ||
      memcmp(map.base_data + 8, "WAVE", 4) != 0)
    goto wave_done;

  walk = 12;
  while (walk + 8 <= map.byte_count) {
    const uint8_t *name_text = map.base_data + walk;
    uint32_t part_size = byte_word(map.base_data + walk + 4);
    const uint8_t *part_data = map.base_data + walk + 8;
    if (walk + 8 + (size_t)part_size > map.byte_count)
      part_size = (uint32_t)(map.byte_count - walk - 8);
    if (memcmp(name_text, "fmt ", 4) == 0 && part_size >= 16) {
      form_kind = (int)((uint32_t)part_data[0] | ((uint32_t)part_data[1] << 8));
      lane_count = (int)((uint32_t)part_data[2] | ((uint32_t)part_data[3] << 8));
      rate_value = (int)byte_word(part_data + 4);
      deep_count = (int)((uint32_t)part_data[14] | ((uint32_t)part_data[15] << 8));
      /* WAVE_FORMAT_EXTENSIBLE keeps the real tag in the first two bytes of its
       * own sub-format record. */
      if (form_kind == 0xFFFE && part_size >= 26)
        form_kind = (int)((uint32_t)part_data[24] | ((uint32_t)part_data[25] << 8));
      real_flag = form_kind == 3;
      if (form_kind != 1 && form_kind != 3) { code = APP_FAIL_SUPPORT; goto wave_done; }
    } else if (memcmp(name_text, "data", 4) == 0) {
      body_data = part_data;
      body_size = part_size;
    }
    walk += 8 + (size_t)part_size + ((size_t)part_size & 1u); /* chunks are word aligned */
  }
  if (!body_data || lane_count < 1 || rate_value < 1) goto wave_done;
  if (deep_count != 8 && deep_count != 16 && deep_count != 24 && deep_count != 32) {
    code = APP_FAIL_SUPPORT;
    goto wave_done;
  }
  value_count = (int)(body_size / ((size_t)lane_count * (size_t)(deep_count / 8)));
  if (value_count < 1) goto wave_done;
  clip_out->value_data = (float *)mem_clear(sizeof(float) * (size_t)value_count);
  if (!clip_out->value_data) { code = APP_FAIL_MEMORY; goto wave_done; }
  clip_out->value_count = value_count;
  clip_out->rate_value = rate_value;
  for (value_index = 0; value_index < value_count; ++value_index) {
    float total_value = 0.0f;
    for (lane_index = 0; lane_index < lane_count; ++lane_index) {
      const uint8_t *cell_data =
          body_data + ((size_t)value_index * (size_t)lane_count + (size_t)lane_index) *
                          (size_t)(deep_count / 8);
      total_value += wave_sample(cell_data, deep_count, real_flag);
    }
    clip_out->value_data[value_index] = total_value / (float)lane_count;
  }
  code = APP_OKAY;

wave_done:
  file_close(&map);
  if (code != APP_OKAY) wave_free(clip_out);
  return code;
}

/* Linear resampling.  A filterbank discards phase and everything above the mel
 * ceiling, so the cost of a windowed sinc would not reach the features the
 * tower reads. */
static app_code wave_rate(wave_clip *clip, int rate_want) {
  double ratio_value;
  int want_count, value_index;
  float *want_data;
  if (rate_want < 1) return APP_OKAY;
  if (clip->rate_value < 1) clip->rate_value = rate_want;
  if (clip->rate_value == rate_want || clip->value_count < 2) return APP_OKAY;
  ratio_value = (double)clip->rate_value / (double)rate_want;
  want_count = (int)((double)clip->value_count / ratio_value);
  if (want_count < 1) want_count = 1;
  want_data = (float *)mem_clear(sizeof(float) * (size_t)want_count);
  if (!want_data) return APP_FAIL_MEMORY;
  for (value_index = 0; value_index < want_count; ++value_index) {
    double place_value = (double)value_index * ratio_value;
    int low_index = (int)place_value;
    double share_value = place_value - (double)low_index;
    int high_index;
    if (low_index >= clip->value_count) low_index = clip->value_count - 1;
    high_index = low_index + 1 < clip->value_count ? low_index + 1 : clip->value_count - 1;
    want_data[value_index] = (float)((1.0 - share_value) * (double)clip->value_data[low_index] +
                                     share_value * (double)clip->value_data[high_index]);
  }
  mem_free(clip->value_data);
  clip->value_data = want_data;
  clip->value_count = want_count;
  clip->rate_value = rate_want;
  return APP_OKAY;
}

/* -- filterbank ----------------------------------------------------------- */

/* In-place radix-2 decimation in time.  The span is always a power of two here
 * because the caller rounds the frame up to one. */
static void wave_spin(float *real_list, float *imag_list, int span_count) {
  int walk_index, pair_index, step_size;
  for (walk_index = 1, pair_index = 0; walk_index < span_count; ++walk_index) {
    int bit_mask = span_count >> 1;
    for (; pair_index & bit_mask; bit_mask >>= 1) pair_index ^= bit_mask;
    pair_index ^= bit_mask;
    if (walk_index < pair_index) {
      float keep_real = real_list[walk_index], keep_imag = imag_list[walk_index];
      real_list[walk_index] = real_list[pair_index];
      imag_list[walk_index] = imag_list[pair_index];
      real_list[pair_index] = keep_real;
      imag_list[pair_index] = keep_imag;
    }
  }
  for (step_size = 1; step_size < span_count; step_size <<= 1) {
    double turn_step = -3.14159265358979323846 / (double)step_size;
    int group_index, slot_index;
    for (group_index = 0; group_index < span_count; group_index += step_size << 1)
      for (slot_index = 0; slot_index < step_size; ++slot_index) {
        double angle_value = turn_step * (double)slot_index;
        float turn_real = (float)cos(angle_value);
        float turn_imag = (float)sin(angle_value);
        int low_slot = group_index + slot_index;
        int high_slot = low_slot + step_size;
        float mix_real = turn_real * real_list[high_slot] - turn_imag * imag_list[high_slot];
        float mix_imag = turn_real * imag_list[high_slot] + turn_imag * real_list[high_slot];
        real_list[high_slot] = real_list[low_slot] - mix_real;
        imag_list[high_slot] = imag_list[low_slot] - mix_imag;
        real_list[low_slot] += mix_real;
        imag_list[low_slot] += mix_imag;
      }
  }
}

static float mel_from_hertz(float hertz_value) {
  return 2595.0f * log10f(1.0f + hertz_value / 700.0f);
}

static float hertz_from_mel(float mel_value) {
  return 700.0f * (powf(10.0f, mel_value / 2595.0f) - 1.0f);
}

/* Triangular filters spaced evenly on the HTK mel scale between zero and the
 * Nyquist rate, one row per filter and one column per spectrum bin.  The
 * triangles are laid out at fractional bin positions rather than whole ones,
 * which is what keeps two neighbouring filters summing to one across the band
 * they share. */
static app_code mel_bank(int mel_count, int bin_count, int rate_value, float **bank_out) {
  float *bank_data = (float *)mem_clear(sizeof(float) * (size_t)mel_count * (size_t)bin_count);
  float *edge_list = (float *)mem_clear(sizeof(float) * (size_t)(mel_count + 2));
  float mel_high, mel_step;
  int edge_index, mel_index, bin_index;
  *bank_out = bank_data;
  if (!bank_data || !edge_list) { mem_free(edge_list); return APP_FAIL_MEMORY; }
  mel_high = mel_from_hertz((float)rate_value * 0.5f);
  mel_step = mel_high / (float)(mel_count + 1);
  for (edge_index = 0; edge_index < mel_count + 2; ++edge_index) {
    float hertz_value = hertz_from_mel(mel_step * (float)edge_index);
    edge_list[edge_index] = hertz_value * (float)(2 * (bin_count - 1)) / (float)rate_value;
  }
  for (mel_index = 0; mel_index < mel_count; ++mel_index) {
    float low_edge = edge_list[mel_index];
    float peak_edge = edge_list[mel_index + 1];
    float high_edge = edge_list[mel_index + 2];
    for (bin_index = 0; bin_index < bin_count; ++bin_index) {
      float place_value = (float)bin_index;
      float weight_value = 0.0f;
      if (place_value > low_edge && place_value < peak_edge && peak_edge > low_edge)
        weight_value = (place_value - low_edge) / (peak_edge - low_edge);
      else if (place_value >= peak_edge && place_value < high_edge && high_edge > peak_edge)
        weight_value = (high_edge - place_value) / (high_edge - peak_edge);
      bank_data[(size_t)mel_index * (size_t)bin_count + (size_t)bin_index] = weight_value;
    }
  }
  mem_free(edge_list);
  return APP_OKAY;
}

/* Log mel spectrogram: one row per frame, one column per filter.
 *
 * The window is a periodic Hann; the transform is taken over `turn_size`, which
 * the checkpoint states rather than the reader deriving it; and the floor is
 * added under the logarithm rather than clamping the argument, which is what
 * the reference feature extractor does and what keeps a silent band finite.
 *
 * The spectrum is taken as a **magnitude**, not a power. Squaring instead is
 * the easy mistake here and it survives every sanity check: the features stay
 * finite, ordered, and roughly the right shape, and only a comparison against
 * the reference catches it.
 *
 * `lead_pad` zeros are prepended so that the first frame is centred on the
 * first sample — the reference calls this semicausal padding — and each frame
 * is cut one sample longer than the window before the last sample is dropped,
 * which is where a preemphasis filter would have consumed it. */
static app_code mel_make(const wave_clip *clip, int mel_count, int frame_size, int frame_step,
                         int turn_size, float floor_value, int lead_pad, flat_grid *grid_out) {
  float *bank_data = NULL;
  float *window_list = NULL;
  float *real_list = NULL;
  float *imag_list = NULL;
  float *power_list = NULL;
  int bin_count, frame_count, reach_count, frame_index, slot_index, mel_index;
  app_code code;

  memset(grid_out, 0, sizeof(*grid_out));
  if (!clip->value_data || mel_count < 1 || frame_size < 2 || frame_step < 1)
    return APP_FAIL_ARGUMENT;
  if (turn_size < 1) {
    turn_size = 1;
    while (turn_size < frame_size) turn_size <<= 1;
  }
  if (turn_size < frame_size) return APP_FAIL_SUPPORT;
  if (lead_pad < 0) lead_pad = 0;
  bin_count = turn_size / 2 + 1;
  /* One sample longer than the window, because the reference cuts the frame
   * that way and then drops the last sample.
   *
   * The frames counted here are the ones the audio actually reaches. The
   * extractor does pad a clip out to a whole block of a hundred and twenty
   * eight samples first, and it does hand back the extra frames, but it marks
   * them in `input_features_mask`; the tower zeros them between its convolution
   * stages and drops their rows from what it returns, so nothing downstream can
   * see them. Counting them here instead would add a soft token the reference
   * never asks for, which the seam catches on about one clip length in seven. */
  reach_count = clip->value_count + lead_pad;
  frame_count = reach_count > frame_size ? 1 + (reach_count - (frame_size + 1)) / frame_step : 1;
  if (frame_count < 1) frame_count = 1;

  code = mel_bank(mel_count, bin_count, clip->rate_value, &bank_data);
  window_list = (float *)mem_clear(sizeof(float) * (size_t)frame_size);
  real_list = (float *)mem_clear(sizeof(float) * (size_t)turn_size);
  imag_list = (float *)mem_clear(sizeof(float) * (size_t)turn_size);
  power_list = (float *)mem_clear(sizeof(float) * (size_t)bin_count);
  if (code == APP_OKAY && (!window_list || !real_list || !imag_list || !power_list))
    code = APP_FAIL_MEMORY;
  if (code == APP_OKAY) code = grid_open(grid_out, mel_count, frame_count, 1);
  if (code != APP_OKAY) goto mel_done;

  for (slot_index = 0; slot_index < frame_size; ++slot_index)
    window_list[slot_index] =
        0.5f - 0.5f * (float)cos(6.283185307179586 * (double)slot_index / (double)frame_size);

  for (frame_index = 0; frame_index < frame_count; ++frame_index) {
    int from_index = frame_index * frame_step;
    float *out_data = grid_at(grid_out, frame_index, 0);
    for (slot_index = 0; slot_index < turn_size; ++slot_index) {
      int read_index = from_index + slot_index - lead_pad;
      real_list[slot_index] =
          (slot_index < frame_size && read_index >= 0 && read_index < clip->value_count)
              ? clip->value_data[read_index] * window_list[slot_index]
              : 0.0f;
      imag_list[slot_index] = 0.0f;
    }
    wave_spin(real_list, imag_list, turn_size);
    for (slot_index = 0; slot_index < bin_count; ++slot_index)
      power_list[slot_index] = (float)sqrt((double)real_list[slot_index] * real_list[slot_index] +
                                           (double)imag_list[slot_index] * imag_list[slot_index]);
    for (mel_index = 0; mel_index < mel_count; ++mel_index) {
      const float *lane_list = bank_data + (size_t)mel_index * (size_t)bin_count;
      float total_value = 0.0f;
      for (slot_index = 0; slot_index < bin_count; ++slot_index)
        total_value += lane_list[slot_index] * power_list[slot_index];
      out_data[mel_index] = logf(total_value + floor_value);
    }
  }

mel_done:
  mem_free(bank_data);
  mem_free(window_list);
  mem_free(real_list);
  mem_free(imag_list);
  mem_free(power_list);
  if (code != APP_OKAY) grid_free(grid_out);
  return code;
}

/* ======================================================================== */
/* 8. model layer                                                           */
/* ======================================================================== */

/* Named activation snapshots, for comparing against the reference one tensor
 * at a time. A disagreement in the logits alone says only that something is
 * wrong; a disagreement in `layer.3.attn` names the function. The whole
 * facility compiles away unless APP_TRACE is defined, so a normal build pays
 * nothing, not even a branch. */
#if defined(APP_TRACE)

static FILE *trace_file = NULL;

static void trace_open(void) {
  const char *path_text;
  if (trace_file) return;
  path_text = getenv("IGLLM_TRACE");
  if (!path_text || !path_text[0]) return;
  trace_file = fopen(path_text, "wb");
  if (trace_file) fwrite("IGTRACE1", 1, 8, trace_file);
}

static void trace_save(const char *name_text, const float *value_list, int value_count) {
  uint32_t name_size, slot_count;
  trace_open();
  if (!trace_file || !value_list || value_count < 1) return;
  name_size = (uint32_t)strlen(name_text);
  slot_count = (uint32_t)value_count;
  fwrite(&name_size, sizeof(name_size), 1, trace_file);
  fwrite(name_text, 1, name_size, trace_file);
  fwrite(&slot_count, sizeof(slot_count), 1, trace_file);
  fwrite(value_list, sizeof(float), (size_t)value_count, trace_file);
}

static void trace_lane(const char *stem_text, int layer_index, int place_index,
                       const float *value_list, int value_count) {
  char name_text[128];
  if (layer_index >= 0)
    snprintf(name_text, sizeof(name_text), "layer.%d.%s.%d", layer_index, stem_text, place_index);
  else
    snprintf(name_text, sizeof(name_text), "%s.%d", stem_text, place_index);
  trace_save(name_text, value_list, value_count);
}

/* A tower's snapshot names the tower rather than a layer of the text stack,
 * so the two never collide in one dump. */
static void trace_tower(const char *tower_text, const char *stem_text, int layer_index,
                        int place_index, const float *value_list, int value_count) {
  char name_text[128];
  if (layer_index >= 0)
    snprintf(name_text, sizeof(name_text), "%s.%d.%s.%d", tower_text, layer_index, stem_text,
             place_index);
  else
    snprintf(name_text, sizeof(name_text), "%s.%s.%d", tower_text, stem_text, place_index);
  trace_save(name_text, value_list, value_count);
}

static void trace_close(void) {
  if (!trace_file) return;
  fclose(trace_file);
  trace_file = NULL;
}

#define TRACE_LANE(stem, layer, place, data, count) \
  trace_lane((stem), (layer), (place), (data), (count))
#define TRACE_TOWER(tower, stem, layer, place, data, count) \
  trace_tower((tower), (stem), (layer), (place), (data), (count))
#define TRACE_CLOSE() trace_close()

#else
#define TRACE_LANE(stem, layer, place, data, count) ((void)0)
#define TRACE_TOWER(tower, stem, layer, place, data, count)   ((void)(tower), (void)(stem), (void)(layer), (void)(place), (void)(data), (void)(count))
#define TRACE_CLOSE() ((void)0)
#endif

#define MODEL_LAYER_LIMIT 128

/* A quantized key and value cache stores the eight bit floats the export
 * calibrates for: one sign bit, four exponent bits, three mantissa bits, and
 * no infinity, so 448 is the largest magnitude the grid reaches and anything
 * over it saturates there.  Below 2^-6 the grid turns subnormal and its step
 * settles at 2^-9.
 *
 * That the denominator is 448 rather than a byte's 127 is not a guess: layer
 * four's value scale is the float32 nearest 128/448 to the bit, a calibration
 * clamped at a round magnitude, and under 127 the ranges would be too small to
 * hold what a plain prompt already puts in the cache. */
#define CACHE_FP8_TOP     448.0f
#define CACHE_FP8_NORM    (1.0f / 64.0f)  /* 2^-6, the smallest normal */
#define CACHE_FP8_STEP    (1.0f / 512.0f) /* 2^-9, the subnormal step */
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
  int expert_count;   /* num_experts */
  int expert_top;     /* top_k_experts */
  int expert_inner;   /* moe_intermediate_size */
  int window_limit;
  float norm_eps;
  float logit_cap;
  int start_id;
  int close_id;
  int pad_id;
  char kind_list[MODEL_LAYER_LIMIT];
  rope_form rope_list[MODEL_KIND_COUNT];
  char rope_kind[MODEL_KIND_COUNT][16];
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

  /* Mixture-of-experts side branch, present only when `enable_moe_block` is set.
   * The dense mlp above stays in place as the always-on shared expert. */
  plane  route_sheet;        /* router.proj, one row per expert */
  float *route_scale;        /* router.scale, one gain per channel */
  float *route_gain;         /* router.per_expert_scale */
  plane *expert_rise_list;   /* experts.gate_up_proj, one view per expert */
  plane *expert_drop_list;   /* experts.down_proj, one view per expert */
  float *after_mlp_norm;     /* post_feedforward_layernorm_1, the dense branch */
  float *before_moe_norm;    /* pre_feedforward_layernorm_2 */
  float *after_moe_norm;     /* post_feedforward_layernorm_2 */

  float *query_norm, *key_norm;
  float *enter_norm, *after_attn_norm, *before_feed_norm, *after_feed_norm, *after_ple_norm;
  float  layer_gain;

  /* Static ranges the export calibrates for a quantized key and value cache,
   * one scalar each per layer.  Zero where the checkpoint ships none, which is
   * the signal to keep that layer's cache in float. */
  float  key_cache_scale, value_cache_scale;
} layer_wing;

/* -- towers -------------------------------------------------------------- */

/* The vision and audio encoders.  Both are the text stack's layer with the
 * causal mask taken off and a different idea of where a token sits: the vision
 * tower turns a head by the patch's row and its column, and the audio tower
 * scores against a projection of the distance between two frames.  Everything
 * either of them does to a residual — the norms, the gated feed-forward, the
 * two residual adds — is the arrangement `session_layer` uses, which is why the
 * two share `tower_feed` rather than each writing it out.
 *
 * A tower ends in a projector: the tower's own norm, a soft-embedding norm, and
 * one linear into the text hidden width.  What comes out is substituted for the
 * embedding of a placeholder token *after* the token path has applied its
 * `sqrt(hidden_size)` scale, so the projector's output is already in the units
 * the residual stream carries and nothing is scaled again. */

#define TOWER_VISION 0
#define TOWER_AUDIO  1
#define TOWER_COUNT  2

typedef struct tower_form {
  int   live_flag;
  int   kind_mark;
  int   state_size;   /* the tower's own hidden width */
  int   inner_size;
  int   layer_count;
  int   head_count;
  int   head_size;
  float norm_eps;
  float head_gain;    /* attention scale, 1/sqrt(head_size) */
  int   token_id;     /* placeholder id in the text vocabulary */
  int   open_id;      /* the id the processor writes before the run */
  int   shut_id;      /* and after it */
  int   soft_limit;   /* soft tokens one image or one clip may produce */

  /* vision */
  int patch_size, band_count, pool_size;
  int grid_wide, grid_high;   /* patches across and down, decided per image */
  int place_size;             /* rows in each axis of the position table */
  int standard_flag;
  rope_form rope; /* built over half a head, because each axis takes a half */

  /* audio */
  int   mel_count, rate_value, frame_size, frame_step, turn_size, lead_pad;
  int   token_ms; /* milliseconds of clip one soft token stands for */
  float mel_floor;
  int   conv_count, conv_step, conv_pad, conv_side;
  int   chunk_size, left_span, right_span; /* the local attention window */
  int   deep_side;                         /* the light convolution's kernel */
  float logit_cap;                         /* tanh softcap on an attention score */
  float share_gain;                        /* what a feed-forward folds back at */

  int  lift_size; /* what the projector reads, which the audio tower widens first */
  char prefix_text[96];
  char layer_text[32]; /* "layers" or "encoder.layers", per the exporter */
  char lift_text[96];
  char lift_leaf[48];
} tower_form;

typedef struct tower_wing {
  plane query_sheet, key_sheet, value_sheet, exit_sheet;
  plane gate_sheet, rise_sheet, drop_sheet;
  float *query_bias, *key_bias, *value_bias, *exit_bias;
  float *gate_bias, *rise_bias, *drop_bias;

  float *query_norm, *key_norm;
  float *enter_norm, *after_attn_norm, *before_feed_norm, *after_feed_norm;
} tower_wing;

typedef struct tower_gear {
  tower_form  form;
  tower_wing *wing_list;

  plane  patch_sheet; /* vision: a row per hidden channel, a column per patch sample */
  float *patch_bias;
  plane  place_sheet; /* vision: the column table stacked above the row table */

  struct sound_wing *sound_list; /* audio: the conformer layers */
  plane  conv_sheet[2];          /* audio: a convolution held as a linear over its window */
  float *conv_norm[2];           /* and the norm across the channels it produced */
  plane  join_sheet;             /* audio: the subsampled map folded into the hidden width */
  float *join_bias;
  plane  out_sheet;              /* audio: the widening the projector reads */
  float *out_bias;

  float *final_norm;
  float *lift_norm;
  plane  lift_sheet;
  float *lift_bias;
} tower_gear;

static const char *tower_name(int kind_mark) {
  return kind_mark == TOWER_VISION ? "vision" : "audio";
}

/* Two dimensional rotary.
 *
 * A head is cut in two: the first half carries the patch's column and the
 * second its row, and each half is rotated **within itself** — channel `j` of a
 * half against channel `j + half/2` of the same half. That is not the text
 * stack's rotate-half, which pairs across the whole head, so the vision tower
 * has its own turn rather than reusing `kern_rope_turn`.
 *
 * The frequencies are the same schedule for both axes, laid out over half a
 * head, which is what `rope_build` over `head_size / 2` produces. */
static void rope_grid_wave(const rope_form *rope, int high_index, int wide_index, float *cos_out,
                           float *sin_out) {
  int angle_index;
  int part_count = rope->half_count; /* head_size / 4 */
  for (angle_index = 0; angle_index < part_count; ++angle_index) {
    double step_value = (double)rope->step_list[angle_index];
    double wide_angle = (double)wide_index * step_value;
    double high_angle = (double)high_index * step_value;
    cos_out[angle_index] = (float)cos(wide_angle);
    sin_out[angle_index] = (float)sin(wide_angle);
    cos_out[part_count + angle_index] = (float)cos(high_angle);
    sin_out[part_count + angle_index] = (float)sin(high_angle);
  }
}

static void rope_grid_turn(float *value_list, int head_size, const float *cos_list,
                           const float *sin_list) {
  int part_size = head_size / 2; /* channels given to one axis */
  int half_size = part_size / 2; /* rotated pairs within that axis */
  int part_index, pair_index;
  for (part_index = 0; part_index < 2; ++part_index) {
    float *part_data = value_list + (size_t)part_index * (size_t)part_size;
    const float *cos_part = cos_list + (size_t)part_index * (size_t)half_size;
    const float *sin_part = sin_list + (size_t)part_index * (size_t)half_size;
    for (pair_index = 0; pair_index < half_size; ++pair_index) {
      float low_value = part_data[pair_index];
      float high_value = part_data[pair_index + half_size];
      part_data[pair_index] = low_value * cos_part[pair_index] - high_value * sin_part[pair_index];
      part_data[pair_index + half_size] =
          high_value * cos_part[pair_index] + low_value * sin_part[pair_index];
    }
  }
}

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
  tower_gear  tower_list[TOWER_COUNT];
  token_book *book_ref;

  pool_group pool;
  back_desk  desk;
  size_t     weight_bytes;
};

/* Rounds the activation onto whatever grid the checkpoint declares, multiplies,
 * and rounds the result onto its own grid.  Every projection in the engine —
 * text stack, mixture branch, and both towers — passes through here, so this is
 * the one place either rule has to be applied.  A lane count above one turns
 * the projection into a matrix product.
 *
 * `quant_room` holds `KERN_LANE_LIMIT` lanes of `quant_stride`, and the batch
 * is chunked to fit it, so a tower that runs two hundred patches at once needs
 * no larger staging buffer than the token loop does. */
static void plane_lift_many(app_model *model, const plane *sheet, const float *act_data,
                            int act_stride, int lane_count, float *out_data, int out_stride,
                            float *quant_room, int quant_stride) {
  int rule_flag = model->book.input_rule.live_flag && sheet->form == PLANE_CODE;
  int lane_index;
  if (!rule_flag && !(sheet->enter_gain > 0.0f)) {
    model->desk.mat_mat(&model->desk, sheet, act_data, act_stride, lane_count, out_data, out_stride);
  } else {
    int done_count = 0;
    while (done_count < lane_count) {
      int chunk_count = lane_count - done_count;
      if (chunk_count > KERN_LANE_LIMIT) chunk_count = KERN_LANE_LIMIT;
      for (lane_index = 0; lane_index < chunk_count; ++lane_index) {
        float *lane_data = quant_room + (size_t)lane_index * (size_t)quant_stride;
        memcpy(lane_data, act_data + (size_t)(done_count + lane_index) * (size_t)act_stride,
               sizeof(float) * (size_t)sheet->col_count);
        if (rule_flag) quant_act(lane_data, sheet->col_count, &model->book.input_rule);
        quant_step(lane_data, sheet->col_count, sheet->enter_gain);
      }
      model->desk.mat_mat(&model->desk, sheet, quant_room, quant_stride, chunk_count,
                          out_data + (size_t)done_count * (size_t)out_stride, out_stride);
      done_count += chunk_count;
    }
  }
  if (sheet->leave_gain > 0.0f)
    for (lane_index = 0; lane_index < lane_count; ++lane_index)
      quant_step(out_data + (size_t)lane_index * (size_t)out_stride, sheet->row_count,
                 sheet->leave_gain);
}

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

/* Locates `stem.weight`, or the bare `stem` when the reference holds the weight
 * as a plain parameter rather than a module, as the stacked expert tensors do. */
static const store_span *plane_find(app_model *model, const char *stem, const char *leaf) {
  char name_text[512];
  const store_span *span;
  snprintf(name_text, sizeof(name_text), "%s.%s", stem, leaf);
  span = store_find(&model->store, name_text);
  if (span) return span;
  if (strcmp(leaf, "weight") != 0) return NULL;
  return store_find(&model->store, stem);
}

/* Reads a plane in the layout the Gemma quantized export uses: the codes are
 * ordinary bytes rather than packed words, the scale carries one column per
 * group, and the eight bit case is stored signed rather than offset.
 *
 * Nothing in the tensor says how many codes share a byte, because the packed
 * width is all the shape records, so `col_hint` is what fixes the bit width. */
static app_code plane_bind_gemma(app_model *model, const char *stem, const store_span *span,
                                 const store_span *gain_span, int part_index, int part_count,
                                 int col_hint, plane *sheet_out) {
  static const int width_list[] = {8, 4, 2, 1};
  int row_count, byte_count, group_count, col_count, bit_count = 0;
  size_t width_index;
  char step_text[512];
  const store_span *step_span;

  if (span->rank_count != (part_count > 1 ? 3 : 2)) return APP_FAIL_FORMAT;
  if (part_count > 1 && (int)span->size_list[0] != part_count) return APP_FAIL_FORMAT;
  row_count = (int)span->size_list[span->rank_count - 2];
  byte_count = (int)span->size_list[span->rank_count - 1];
  group_count =
      gain_span->rank_count >= 2 ? (int)gain_span->size_list[gain_span->rank_count - 1] : 1;
  if (group_count < 1) group_count = 1;

  if (span->type_kind == STORE_I8) {
    bit_count = 8;
    col_count = byte_count;
  } else if (span->type_kind == STORE_U8) {
    col_count = col_hint;
    if (col_count < 1) return APP_FAIL_SUPPORT;
    for (width_index = 0; width_index < sizeof(width_list) / sizeof(width_list[0]); ++width_index)
      if (((int64_t)col_count * width_list[width_index] + 7) / 8 == (int64_t)byte_count) {
        bit_count = width_list[width_index];
        break;
      }
    if (bit_count == 0) return APP_FAIL_FORMAT;
  } else {
    return APP_FAIL_SUPPORT;
  }
  if (col_hint > 0 && col_count != col_hint) return APP_FAIL_FORMAT;
  if (col_count % group_count) return APP_FAIL_FORMAT;

  sheet_out->form = PLANE_CODE;
  sheet_out->row_count = row_count;
  sheet_out->col_count = col_count;
  sheet_out->code_data = (const uint8_t *)span->data_base +
                         (size_t)part_index * (size_t)row_count * (size_t)byte_count;
  sheet_out->row_stride = (size_t)byte_count;
  sheet_out->gain_data = (const uint8_t *)gain_span->data_base +
                         (size_t)part_index * (size_t)row_count * (size_t)group_count *
                             store_type_bytes(gain_span->type_kind);
  sheet_out->gain_type = gain_span->type_kind;
  sheet_out->bit_count = bit_count;
  sheet_out->code_bias = 1 << (bit_count - 1);
  /* A signed byte is the offset code with its top bit flipped, so one xor puts
   * both storage conventions through the same decode. */
  sheet_out->code_flip = span->type_kind == STORE_I8 ? sheet_out->code_bias : 0;
  sheet_out->group_count = group_count;
  sheet_out->group_size = col_count / group_count;
  if (part_index == 0) model->weight_bytes += span->data_bytes + gain_span->data_bytes;

  snprintf(step_text, sizeof(step_text), "%s.input_activation_scale", stem);
  step_span = store_find(&model->store, step_text);
  if (step_span) sheet_out->enter_gain = real_read(step_span->data_base, step_span->type_kind, 0);
  snprintf(step_text, sizeof(step_text), "%s.output_activation_scale", stem);
  step_span = store_find(&model->store, step_text);
  if (step_span) sheet_out->leave_gain = real_read(step_span->data_base, step_span->type_kind, 0);
  return APP_OKAY;
}

/* Resolves `prefix.name` to either a real or a code plane.
 *
 * `part_count` above one selects one slice of a tensor that carries a leading
 * expert axis; the slice is a view, so no payload is copied. `col_hint` is the
 * input width the caller expects, and is used, and checked, only where the
 * stored shape cannot supply it. */
static app_code plane_bind_part(app_model *model, const char *stem, int part_index, int part_count,
                                int col_hint, plane *sheet_out) {
  const store_span *span;
  const store_span *gain_span;
  char other_text[512];
  memset(sheet_out, 0, sizeof(*sheet_out));
  if (part_count < 1 || part_index < 0 || part_index >= part_count) return APP_FAIL_ARGUMENT;

  /* The Gemma export names a quantized embedding table apart, but stores a
   * quantized projection under the plain `weight`. What tells that apart from a
   * real weight is the companion scale, since the codes are ordinary bytes. */
  span = plane_find(model, stem, "embedding_quantized");
  if (span) {
    snprintf(other_text, sizeof(other_text), "%s.embedding_scale", stem);
    gain_span = store_find(&model->store, other_text);
    if (!gain_span) return APP_FAIL_MISSING;
    return plane_bind_gemma(model, stem, span, gain_span, part_index, part_count, col_hint,
                            sheet_out);
  }

  span = plane_find(model, stem, "weight");
  if (span && (span->type_kind == STORE_U8 || span->type_kind == STORE_I8)) {
    snprintf(other_text, sizeof(other_text), "%s.weight_scale", stem);
    gain_span = store_find(&model->store, other_text);
    if (gain_span)
      return plane_bind_gemma(model, stem, span, gain_span, part_index, part_count, col_hint,
                              sheet_out);
  }
  if (span) {
    size_t part_slots;
    if (span->rank_count != (part_count > 1 ? 3 : 2)) return APP_FAIL_FORMAT;
    if (part_count > 1 && (int)span->size_list[0] != part_count) return APP_FAIL_FORMAT;
    sheet_out->form = PLANE_REAL;
    sheet_out->row_count = (int)span->size_list[span->rank_count - 2];
    sheet_out->col_count = (int)span->size_list[span->rank_count - 1];
    part_slots = (size_t)sheet_out->row_count * (size_t)sheet_out->col_count;
    sheet_out->real_data = (const uint8_t *)span->data_base +
                           (size_t)part_index * part_slots * store_type_bytes(span->type_kind);
    sheet_out->real_type = span->type_kind;
    if (part_index == 0) model->weight_bytes += span->data_bytes;
    return APP_OKAY;
  }

  span = plane_find(model, stem, "weight_packed");
  if (!span) return APP_FAIL_MISSING;
  {
    const store_span *zero_span;
    const store_span *shape_span;
    int row_count, col_count, word_count, group_count, bit_count;

    if (span->rank_count != (part_count > 1 ? 3 : 2) || span->type_kind != STORE_I32)
      return APP_FAIL_SUPPORT;
    if (part_count > 1 && (int)span->size_list[0] != part_count) return APP_FAIL_FORMAT;
    row_count = (int)span->size_list[span->rank_count - 2];
    word_count = (int)span->size_list[span->rank_count - 1];

    snprintf(other_text, sizeof(other_text), "%s.weight_scale", stem);
    gain_span = store_find(&model->store, other_text);
    if (!gain_span) return APP_FAIL_MISSING;
    snprintf(other_text, sizeof(other_text), "%s.weight_shape", stem);
    shape_span = store_find(&model->store, other_text);
    if (shape_span && shape_span->rank_count >= 1 &&
        shape_span->size_list[shape_span->rank_count - 1] >= 2)
      col_count = (int)whole_read(shape_span->data_base, shape_span->type_kind,
                                  (size_t)shape_span->size_list[shape_span->rank_count - 1] - 1);
    else
      col_count = (int)((int64_t)word_count * 32 /
                        (model->book.weight_rule.bit_count ? model->book.weight_rule.bit_count : 4));

    group_count = gain_span->rank_count >= 2 ? (int)gain_span->size_list[gain_span->rank_count - 1] : 1;
    if (group_count < 1) group_count = 1;
    bit_count = plane_bits_of(col_count, word_count, model->book.weight_rule.bit_count);
    if (bit_count == 0) return APP_FAIL_FORMAT;

    sheet_out->form = PLANE_CODE;
    sheet_out->row_count = row_count;
    sheet_out->col_count = col_count;
    sheet_out->code_data = (const uint8_t *)span->data_base +
                           (size_t)part_index * (size_t)row_count * (size_t)word_count * 4u;
    sheet_out->row_stride = (size_t)word_count * 4u;
    sheet_out->gain_data =
        (const uint8_t *)gain_span->data_base +
        (size_t)part_index * (size_t)row_count * (size_t)group_count *
            store_type_bytes(gain_span->type_kind);
    sheet_out->gain_type = gain_span->type_kind;
    sheet_out->bit_count = bit_count;
    sheet_out->code_bias = 1 << (bit_count - 1);
    sheet_out->group_count = group_count;
    sheet_out->group_size = (col_count + group_count - 1) / group_count;
    if (part_index == 0) model->weight_bytes += span->data_bytes + gain_span->data_bytes;

    snprintf(other_text, sizeof(other_text), "%s.weight_zero_point", stem);
    zero_span = store_find(&model->store, other_text);
    if (zero_span && zero_span->type_kind == STORE_I32) {
      int8_t *bias_room = (int8_t *)mem_clear((size_t)row_count * (size_t)group_count);
      int word_rows = (int)zero_span->size_list[zero_span->rank_count - 2];
      const uint32_t *zero_data = (const uint32_t *)zero_span->data_base +
                                  (size_t)part_index * (size_t)word_rows * (size_t)group_count;
      int row_index, group_index;
      if (!bias_room) return APP_FAIL_MEMORY;
      for (row_index = 0; row_index < row_count; ++row_index)
        for (group_index = 0; group_index < group_count; ++group_index)
          bias_room[(size_t)row_index * (size_t)group_count + (size_t)group_index] =
              (int8_t)zero_read(zero_data, group_count, group_index, row_index, bit_count);
      sheet_out->bias_data = bias_room;
      sheet_out->own_block = bias_room;
    }
  }
  return APP_OKAY;
}

static app_code plane_bind(app_model *model, const char *stem, int col_hint, plane *sheet_out) {
  return plane_bind_part(model, stem, 0, 1, col_hint, sheet_out);
}

static app_code plane_bind_at(app_model *model, const char *shape_text, int layer_index,
                              const char *leaf, int col_hint, plane *sheet_out) {
  char stem_text[512];
  snprintf(stem_text, sizeof(stem_text), shape_text, model->prefix_text, layer_index, leaf);
  return plane_bind(model, stem_text, col_hint, sheet_out);
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

/* A calibration scalar, read straight out of the checkpoint.  These are rank
 * zero tensors, so `vec_bind` would allocate a one element array to hold what
 * fits in a register; read the single value and hand it back by value. */
static float scale_bind(app_model *model, const char *stem) {
  const store_span *span = store_find(&model->store, stem);
  int value_index, value_count = 1;
  if (!span) return 0.0f;
  for (value_index = 0; value_index < span->rank_count; ++value_index)
    value_count *= (int)span->size_list[value_index];
  if (value_count != 1) return 0.0f;
  model->weight_bytes += span->data_bytes;
  return real_read(span->data_base, span->type_kind, 0);
}

/* -- configuration ------------------------------------------------------- */

static void rope_build(rope_form *rope, int head_size, const char *type_text) {
  int angle_index;
  int half_count = head_size / 2;
  int turn_count = half_count;
  mem_free(rope->step_list);
  rope->step_list = NULL;
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

/* Reads one tower's shape.  What the weights also record is taken from the
 * weights afterwards, in `tower_bind`; what is read here is what only the
 * configuration can say — the patch size, the sample rate, the analysis window,
 * and the pooling factor. */
static void config_tower_read(const json_tree *tree, const char *node_text, int kind_mark,
                              tower_form *form_out) {
  int32_t node_index = json_field(tree, 0, node_text);
  memset(form_out, 0, sizeof(*form_out));
  form_out->kind_mark = kind_mark;
  form_out->token_id = -1;
  form_out->open_id = form_out->shut_id = -1;
  if (node_index < 0) return;
  form_out->live_flag = 1;
  form_out->state_size = (int)json_field_number(tree, node_index, "hidden_size", 0);
  form_out->inner_size = (int)json_field_number(tree, node_index, "intermediate_size", 0);
  form_out->layer_count = (int)json_field_number(tree, node_index, "num_hidden_layers",
                                                 kind_mark == TOWER_VISION ? 16 : 12);
  form_out->head_count = (int)json_field_number(tree, node_index, "num_attention_heads", 0);
  form_out->head_size = (int)json_field_number(tree, node_index, "head_dim", 0);
  form_out->norm_eps = (float)json_field_number(tree, node_index, "rms_norm_eps", 1e-6);
  if (kind_mark == TOWER_VISION) {
    /* The rotary settings sit under `rope_parameters`, as they do for the text
     * stack; a flat `rope_theta` is accepted for an exporter that writes one. */
    int32_t rope_index = json_field(tree, node_index, "rope_parameters");
    form_out->patch_size = (int)json_field_number(tree, node_index, "patch_size", 16);
    form_out->band_count = 3; /* the patch layout the processor writes is fixed */
    form_out->pool_size = (int)json_field_number(
        tree, node_index, "pooling_kernel_size",
        json_field_number(tree, node_index, "pool_size", 3));
    form_out->place_size = (int)json_field_number(tree, node_index, "position_embedding_size", 0);
    form_out->standard_flag = json_field_flag(tree, node_index, "standardize", 0);
    form_out->soft_limit = (int)json_field_number(tree, 0, "vision_soft_tokens_per_image",
                                                  json_field_number(tree, node_index,
                                                                    "default_output_length", 280));
    form_out->rope.theta_value =
        rope_index >= 0 ? json_field_number(tree, rope_index, "rope_theta", 10000.0)
                        : json_field_number(tree, node_index, "rope_theta", 10000.0);
    form_out->rope.part_share = 1.0;
  } else {
    form_out->conv_step = (int)json_field_number(tree, node_index, "conv_stride", 2);
    form_out->conv_pad = (int)json_field_number(tree, node_index, "conv_padding", 1);
    form_out->conv_side = (int)json_field_number(tree, node_index, "conv_side", 3);
    form_out->deep_side = (int)json_field_number(tree, node_index, "conv_kernel_size", 5);
    form_out->chunk_size = (int)json_field_number(tree, node_index, "attention_chunk_size", 0);
    form_out->left_span = (int)json_field_number(tree, node_index, "attention_context_left", 1);
    form_out->right_span = (int)json_field_number(tree, node_index, "attention_context_right", 0);
    form_out->logit_cap = (float)json_field_number(tree, node_index, "attention_logit_cap", 0.0);
    form_out->share_gain = (float)json_field_number(tree, node_index, "residual_weight", 1.0);
    form_out->lift_size = (int)json_field_number(tree, node_index, "output_proj_dims", 0);
    /* The analysis window belongs to the feature extractor rather than the
     * model, and is read from `preprocessor_config.json` beside it; the token
     * budget belongs to the processor, in `processor_config.json`.  Both are
     * defaulted to the shipped export's, which is the only thing a checkpoint
     * that writes neither file can be read as meaning. */
    form_out->mel_count = 128;
    form_out->rate_value = 16000;
    form_out->frame_size = 320;
    form_out->frame_step = 160;
    form_out->turn_size = 0;
    form_out->mel_floor = 1e-3f;
    form_out->soft_limit = 750;
    form_out->token_ms = 40;
  }
}

/* The audio analysis window, which belongs to the feature extractor rather than
 * to the model and is written in its own file beside it.  Everything here has a
 * default, so a checkpoint without the file still runs. */
static void config_sound_read(const char *folder_path, tower_form *form) {
  char path_text[1024];
  size_t text_size = 0;
  char *text_data;
  json_tree *tree;
  if (!form->live_flag) return;
  path_join(path_text, sizeof(path_text), folder_path, "preprocessor_config.json");
  text_data = file_slurp(path_text, &text_size);
  if (!text_data) return;
  tree = json_read(text_data, text_size);
  mem_free(text_data);
  if (!tree) return;
  form->mel_count = (int)json_field_number(tree, 0, "feature_size", form->mel_count);
  form->rate_value = (int)json_field_number(tree, 0, "sampling_rate", form->rate_value);
  form->frame_size = (int)json_field_number(tree, 0, "frame_length", form->frame_size);
  form->frame_step = (int)json_field_number(tree, 0, "hop_length", form->frame_step);
  form->turn_size = (int)json_field_number(tree, 0, "fft_length", 0);
  form->mel_floor = (float)json_field_number(tree, 0, "mel_floor", form->mel_floor);
  /* The first frame is centred on the first sample, which the reference calls
   * semicausal padding: half a window of silence in front of the clip. */
  form->lead_pad = form->frame_size / 2;
  json_free(tree);
}

/* The clip's token budget, which belongs to the processor rather than to the
 * model or to the feature extractor, and is written in a third file beside
 * them.  `audio_seq_length` soft tokens of `audio_ms_per_token` milliseconds
 * each is the most one clip is worth, and the processor pads or trims a clip to
 * fit before it ever reaches the tower.
 *
 * Only the trim is visible here.  The padding is marked in
 * `input_features_mask`, zeroed between the convolution stages and dropped from
 * the rows the tower returns, so a short clip is worth its live frames on both
 * sides and the engine already agrees; a clip past the ceiling is one the
 * reference stops reading and the engine would otherwise keep going through. */
static void config_budget_read(const char *folder_path, tower_form *form) {
  char path_text[1024];
  size_t text_size = 0;
  char *text_data;
  json_tree *tree;
  if (!form->live_flag) return;
  path_join(path_text, sizeof(path_text), folder_path, "processor_config.json");
  text_data = file_slurp(path_text, &text_size);
  if (!text_data) return;
  tree = json_read(text_data, text_size);
  mem_free(text_data);
  if (!tree) return;
  form->soft_limit = (int)json_field_number(tree, 0, "audio_seq_length", form->soft_limit);
  form->token_ms = (int)json_field_number(tree, 0, "audio_ms_per_token", form->token_ms);
  json_free(tree);
}

/* The samples the budget allows, which is where the trim falls: the reference
 * cuts the waveform and then frames what is left, so the engine has to cut in
 * the same place rather than capping the rows afterwards.  A cap would agree on
 * the count and disagree on the last row, whose frames would have been drawn
 * from audio the reference never read.
 *
 * Zero means no ceiling, for a checkpoint that records neither field. */
static long sound_budget_samples(const tower_form *form) {
  double sample_limit;
  if (form->soft_limit < 1 || form->token_ms < 1 || form->rate_value < 1) return 0;
  sample_limit = (double)form->soft_limit * (double)form->token_ms *
                 ((double)form->rate_value / 1000.0);
  if (sample_limit >= (double)INT32_MAX) return 0;
  return (long)sample_limit;
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
  form->whole_head_size = (int)json_field_number(tree, text_index, "global_head_dim", 0);
  form->whole_kv_count =
      (int)json_field_number(tree, text_index, "num_global_key_value_heads", form->kv_count);
  form->slide_span = (int)json_field_number(tree, text_index, "sliding_window", 512);
  form->ple_vocab = (int)json_field_number(tree, text_index, "vocab_size_per_layer_input", 262144);
  form->ple_size = (int)json_field_number(tree, text_index, "hidden_size_per_layer_input", 256);
  form->share_count = (int)json_field_number(tree, text_index, "num_kv_shared_layers", 0);
  form->twin_flag = json_field_flag(tree, text_index, "attention_k_eq_v", 0);
  form->wide_flag = json_field_flag(tree, text_index, "use_double_wide_mlp", 0);
  form->moe_flag = json_field_flag(tree, text_index, "enable_moe_block", 0);
  form->expert_count = (int)json_field_number(tree, text_index, "num_experts", 0);
  form->expert_top = (int)json_field_number(tree, text_index, "top_k_experts", 0);
  form->expert_inner = (int)json_field_number(tree, text_index, "moe_intermediate_size", 0);
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
  if (form->whole_head_size < 1) {
    /* `global_head_dim` is a constructor argument upstream, not a stored field:
     * what `save_pretrained` writes is a `per_layer_config` map of the layers
     * that override it. Read that, and fall back to the plain head size. */
    int32_t over_index = json_field(tree, text_index, "per_layer_config");
    int32_t child_index = over_index >= 0 ? tree->node_list[over_index].head_child : -1;
    for (; child_index >= 0; child_index = tree->node_list[child_index].next_peer) {
      int over_size = (int)json_field_number(tree, child_index, "head_dim", 0);
      if (over_size > form->whole_head_size) form->whole_head_size = over_size;
    }
    if (form->whole_head_size < 1) form->whole_head_size = form->head_size;
  }
  if (form->moe_flag) {
    if (form->expert_count < 1) { json_free(tree); return APP_FAIL_SUPPORT; }
    if (form->expert_top < 1 || form->expert_top > form->expert_count)
      form->expert_top = form->expert_count;
  }

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
    text_fill(form->rope_kind[MODEL_KIND_SLIDE], sizeof(form->rope_kind[0]), type_text);
    rope_build(&form->rope_list[MODEL_KIND_SLIDE], form->head_size, type_text);
    config_rope_read(tree, text_index, "full_attention", &form->rope_list[MODEL_KIND_WHOLE],
                     1000000.0, &type_text);
    text_fill(form->rope_kind[MODEL_KIND_WHOLE], sizeof(form->rope_kind[0]), type_text);
    rope_build(&form->rope_list[MODEL_KIND_WHOLE], form->whole_head_size, type_text);
  }

  {
    int32_t quant_index = json_field(tree, 0, "quantization_config");
    const char *method_text =
        quant_index >= 0 ? json_field_text(tree, quant_index, "quant_method") : NULL;
    memset(&model->book, 0, sizeof(model->book));
    if (method_text && strcmp(method_text, "gemma") == 0) {
      /* The Gemma export declares its widths as a map of module name patterns
       * onto bit counts. Every one of them is recoverable from the shape of the
       * tensor it describes, and the tensor is the side that cannot disagree
       * with the payload, so only the default is taken from here. */
      model->book.pack_flag = 1;
      model->book.weight_rule.live_flag = 1;
      model->book.weight_rule.symmetric_flag = 1;
      model->book.weight_rule.plan_kind = QUANT_CHANNEL;
      model->book.weight_rule.bit_count = (int)json_field_number(tree, quant_index, "num_bits", 4);
    } else if (quant_index >= 0) {
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

  config_tower_read(tree, "vision_config", TOWER_VISION, &model->tower_list[TOWER_VISION].form);
  config_tower_read(tree, "audio_config", TOWER_AUDIO, &model->tower_list[TOWER_AUDIO].form);
  config_sound_read(folder_path, &model->tower_list[TOWER_AUDIO].form);
  config_budget_read(folder_path, &model->tower_list[TOWER_AUDIO].form);
  model->tower_list[TOWER_VISION].form.token_id =
      (int)json_field_number(tree, 0, "image_token_id", -1);
  model->tower_list[TOWER_AUDIO].form.token_id =
      (int)json_field_number(tree, 0, "audio_token_id", -1);
  /* The processor does not lay a bare run of placeholders down: it brackets
   * each one, and the model reads those two ids as ordinary text.  A prompt
   * built without them is a prompt the reference never sees.  The end of an
   * audio run is `eoa_token_id` in one export and `eoa_token_index` in
   * another, so both spellings are accepted. */
  model->tower_list[TOWER_VISION].form.open_id =
      (int)json_field_number(tree, 0, "boi_token_id", -1);
  model->tower_list[TOWER_VISION].form.shut_id =
      (int)json_field_number(tree, 0, "eoi_token_id", -1);
  model->tower_list[TOWER_AUDIO].form.open_id =
      (int)json_field_number(tree, 0, "boa_token_id", -1);
  model->tower_list[TOWER_AUDIO].form.shut_id = (int)json_field_number(
      tree, 0, "eoa_token_id", json_field_number(tree, 0, "eoa_token_index", -1));

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
    snprintf(name_text, sizeof(name_text), "%sembed_tokens.embedding_quantized",
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
  code = plane_bind(model, stem_text, form->state_size, &model->embed_sheet);
  if (code != APP_OKAY) return code;
  if (model->embed_sheet.row_count > 0) form->vocab_count = model->embed_sheet.row_count;
  if (model->embed_sheet.col_count > 0) form->state_size = model->embed_sheet.col_count;

  snprintf(stem_text, sizeof(stem_text), "%sembed_tokens_per_layer", model->prefix_text);
  code = plane_bind(model, stem_text, form->ple_size * form->layer_count,
                    &model->ple_embed_sheet);
  if (code != APP_OKAY) return code;
  form->ple_size = model->ple_embed_sheet.col_count / form->layer_count;

  snprintf(stem_text, sizeof(stem_text), "%sper_layer_model_projection", model->prefix_text);
  code = plane_bind(model, stem_text, form->state_size, &model->ple_lift_sheet);
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
      if (plane_bind(model, head_list[head_index], form->state_size,
                     &model->head_sheet) == APP_OKAY) {
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

    code = plane_bind_at(model, "%slayers.%d.self_attn.%s", layer_index, "q_proj",
                         form->state_size, &wing->query_sheet);
    if (code != APP_OKAY) return code;
    code = plane_bind_at(model, "%slayers.%d.%s", layer_index, "mlp.gate_proj",
                         form->state_size, &wing->gate_sheet);
    if (code != APP_OKAY) return code;
    code = plane_bind_at(model, "%slayers.%d.%s", layer_index, "mlp.up_proj",
                         form->state_size, &wing->rise_sheet);
    if (code != APP_OKAY) return code;
    code = plane_bind_at(model, "%slayers.%d.%s", layer_index, "mlp.down_proj",
                         wing->gate_sheet.row_count, &wing->drop_sheet);
    if (code != APP_OKAY) return code;
    code = plane_bind_at(model, "%slayers.%d.self_attn.%s", layer_index, "o_proj",
                         wing->query_sheet.row_count, &wing->exit_sheet);
    if (code != APP_OKAY) return code;

    wing->head_size = wing->query_sheet.row_count / form->head_count;
    wing->inner_size = wing->gate_sheet.row_count;

    if (!wing->share_flag) {
      code = plane_bind_at(model, "%slayers.%d.self_attn.%s", layer_index, "k_proj",
                           form->state_size, &wing->key_sheet);
      if (code != APP_OKAY) return code;
      wing->kv_count = wing->key_sheet.row_count / wing->head_size;
      if (plane_bind_at(model, "%slayers.%d.self_attn.%s", layer_index, "v_proj",
                        form->state_size, &wing->value_sheet) != APP_OKAY)
        memset(&wing->value_sheet, 0, sizeof(wing->value_sheet)); /* keys double as values */
      snprintf(stem_text, sizeof(stem_text), "%slayers.%d.self_attn.k_norm", model->prefix_text,
               layer_index);
      wing->key_norm = vec_bind(model, stem_text, wing->head_size);
      if (!wing->key_norm) return APP_FAIL_MISSING;
      snprintf(stem_text, sizeof(stem_text), "%slayers.%d.self_attn.k_cache_scale",
               model->prefix_text, layer_index);
      wing->key_cache_scale = scale_bind(model, stem_text);
      snprintf(stem_text, sizeof(stem_text), "%slayers.%d.self_attn.v_cache_scale",
               model->prefix_text, layer_index);
      wing->value_cache_scale = scale_bind(model, stem_text);
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
                           form->state_size, &wing->ple_gate_sheet);
      if (code != APP_OKAY) return code;
      code = plane_bind_at(model, "%slayers.%d.%s", layer_index, "per_layer_projection",
                           form->ple_size, &wing->ple_lift_sheet);
      if (code != APP_OKAY) return code;
      snprintf(stem_text, sizeof(stem_text), "%slayers.%d.post_per_layer_input_norm",
               model->prefix_text, layer_index);
      wing->after_ple_norm = vec_bind(model, stem_text, form->state_size);
      if (!wing->after_ple_norm) return APP_FAIL_MISSING;
    }

    if (form->moe_flag) {
      int expert_index;
      int inner_size;
      snprintf(stem_text, sizeof(stem_text), "%slayers.%d.router.proj", model->prefix_text,
               layer_index);
      code = plane_bind(model, stem_text, form->state_size, &wing->route_sheet);
      if (code != APP_OKAY) return code;
      /* Every layer must agree on the expert count, because the session rooms
       * and the free walk are sized once from the form. */
      if (wing->route_sheet.row_count != form->expert_count) return APP_FAIL_FORMAT;

      snprintf(stem_text, sizeof(stem_text), "%slayers.%d.router.scale", model->prefix_text,
               layer_index);
      wing->route_scale = vec_bind(model, stem_text, form->state_size);
      snprintf(stem_text, sizeof(stem_text), "%slayers.%d.router.per_expert_scale",
               model->prefix_text, layer_index);
      wing->route_gain = vec_bind(model, stem_text, form->expert_count);
      if (!wing->route_scale || !wing->route_gain) return APP_FAIL_MISSING;

      wing->expert_rise_list = (plane *)mem_clear(sizeof(plane) * (size_t)form->expert_count);
      wing->expert_drop_list = (plane *)mem_clear(sizeof(plane) * (size_t)form->expert_count);
      if (!wing->expert_rise_list || !wing->expert_drop_list) return APP_FAIL_MEMORY;
      for (expert_index = 0; expert_index < form->expert_count; ++expert_index) {
        snprintf(stem_text, sizeof(stem_text), "%slayers.%d.experts.gate_up_proj",
                 model->prefix_text, layer_index);
        code = plane_bind_part(model, stem_text, expert_index, form->expert_count,
                               form->state_size, &wing->expert_rise_list[expert_index]);
        if (code != APP_OKAY) return code;
        snprintf(stem_text, sizeof(stem_text), "%slayers.%d.experts.down_proj", model->prefix_text,
                 layer_index);
        code = plane_bind_part(model, stem_text, expert_index, form->expert_count,
                               wing->expert_rise_list[expert_index].row_count / 2,
                               &wing->expert_drop_list[expert_index]);
        if (code != APP_OKAY) return code;
      }
      /* gate_up_proj stacks the gate rows above the rise rows, so the expert
       * width is half its row count whatever the configuration claims. It too
       * has to hold for every layer. */
      inner_size = wing->expert_rise_list[0].row_count / 2;
      if (inner_size < 1) return APP_FAIL_FORMAT;
      if (layer_index == 0) form->expert_inner = inner_size;
      else if (inner_size != form->expert_inner) return APP_FAIL_FORMAT;

      snprintf(stem_text, sizeof(stem_text), "%slayers.%d.post_feedforward_layernorm_1",
               model->prefix_text, layer_index);
      wing->after_mlp_norm = vec_bind(model, stem_text, form->state_size);
      snprintf(stem_text, sizeof(stem_text), "%slayers.%d.pre_feedforward_layernorm_2",
               model->prefix_text, layer_index);
      wing->before_moe_norm = vec_bind(model, stem_text, form->state_size);
      snprintf(stem_text, sizeof(stem_text), "%slayers.%d.post_feedforward_layernorm_2",
               model->prefix_text, layer_index);
      wing->after_moe_norm = vec_bind(model, stem_text, form->state_size);
      if (!wing->after_mlp_norm || !wing->before_moe_norm || !wing->after_moe_norm)
        return APP_FAIL_MISSING;
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

  /* The rope tables are rebuilt from the head size the tensors imply, because
   * that is the only source that cannot disagree with the weights. The
   * configuration is a hint; `q_proj` is the truth. */
  for (layer_index = 0; layer_index < form->layer_count; ++layer_index) {
    layer_wing *wing = &model->wing_list[layer_index];
    rope_form *rope = &form->rope_list[(int)wing->kind_mark];
    if (wing->head_size < 2 || rope->half_count == wing->head_size / 2) continue;
    rope_build(rope, wing->head_size, form->rope_kind[(int)wing->kind_mark]);
    if (!rope->step_list) return APP_FAIL_MEMORY;
  }
  form->head_size = model->wing_list[0].head_size;
  for (layer_index = 0; layer_index < form->layer_count; ++layer_index)
    if (model->wing_list[layer_index].kind_mark == MODEL_KIND_WHOLE)
      form->whole_head_size = model->wing_list[layer_index].head_size;

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

/* -- tower binding ------------------------------------------------------- */

/* Binds a projection that may have been stored either way round.  The reference
 * keeps its multi-modal projection as [in, out] and multiplies on the right,
 * while every other weight in the checkpoint is [out, in].  Rather than pick
 * one and refuse the other, the transpose is materialized when the shape says
 * that is the one that was written. */
static app_code plane_bind_turn(app_model *model, const char *stem, int row_want, int col_want,
                                plane *sheet_out) {
  app_code code = plane_bind(model, stem, 0, sheet_out);
  float *turn_data;
  int row_index, col_index;
  if (code != APP_OKAY) return code;
  if (sheet_out->row_count == row_want && sheet_out->col_count == col_want) return APP_OKAY;
  if (sheet_out->row_count != col_want || sheet_out->col_count != row_want ||
      sheet_out->form != PLANE_REAL)
    return APP_FAIL_FORMAT;
  turn_data = (float *)mem_clear(sizeof(float) * (size_t)row_want * (size_t)col_want);
  if (!turn_data) return APP_FAIL_MEMORY;
  for (row_index = 0; row_index < col_want; ++row_index)
    for (col_index = 0; col_index < row_want; ++col_index)
      turn_data[(size_t)col_index * (size_t)col_want + (size_t)row_index] =
          real_read(sheet_out->real_data, sheet_out->real_type,
                    (size_t)row_index * (size_t)row_want + (size_t)col_index);
  sheet_out->form = PLANE_REAL;
  sheet_out->real_data = turn_data;
  sheet_out->real_type = STORE_F32;
  sheet_out->own_block = turn_data;
  sheet_out->row_count = row_want;
  sheet_out->col_count = col_want;
  return APP_OKAY;
}

/* Binds a weight whose trailing axes are a window rather than a row, which is
 * what a convolution kernel is: torch writes [out, in, high, wide], and the
 * bytes of that already are the [out, in*high*wide] matrix this engine wants. */
static app_code plane_bind_fold(app_model *model, const char *stem, int col_hint,
                                plane *sheet_out) {
  char name_text[512];
  const store_span *span;
  int axis_index;
  size_t col_count = 1;
  memset(sheet_out, 0, sizeof(*sheet_out));
  snprintf(name_text, sizeof(name_text), "%s.weight", stem);
  span = store_find(&model->store, name_text);
  if (!span) span = store_find(&model->store, stem);
  if (!span) return APP_FAIL_MISSING;
  if (span->rank_count < 2) return APP_FAIL_FORMAT;
  if (span->rank_count == 2) return plane_bind(model, stem, col_hint, sheet_out);
  for (axis_index = 1; axis_index < span->rank_count; ++axis_index)
    col_count *= (size_t)span->size_list[axis_index];
  sheet_out->form = PLANE_REAL;
  sheet_out->row_count = (int)span->size_list[0];
  sheet_out->col_count = (int)col_count;
  sheet_out->real_data = span->data_base;
  sheet_out->real_type = span->type_kind;
  model->weight_bytes += span->data_bytes;
  return APP_OKAY;
}

/* Finds which name the exporter gave a tower, by probing for the first weight
 * every arrangement of it has to carry. */
static int tower_prefix_pick(app_model *model, const char *const *candidate_list,
                             int candidate_count, const char *probe_text, char *prefix_out,
                             size_t prefix_limit) {
  static const char *leaf_list[] = {"weight", "weight_packed", "embedding_quantized"};
  char name_text[512];
  int candidate_index;
  for (candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
    size_t leaf_index;
    for (leaf_index = 0; leaf_index < sizeof(leaf_list) / sizeof(leaf_list[0]); ++leaf_index) {
      snprintf(name_text, sizeof(name_text), "%s%s.%s", candidate_list[candidate_index], probe_text,
               leaf_list[leaf_index]);
      if (store_find(&model->store, name_text)) {
        text_fill(prefix_out, prefix_limit, candidate_list[candidate_index]);
        return 1;
      }
    }
    snprintf(name_text, sizeof(name_text), "%s%s", candidate_list[candidate_index], probe_text);
    if (store_find(&model->store, name_text)) {
      text_fill(prefix_out, prefix_limit, candidate_list[candidate_index]);
      return 1;
    }
  }
  return 0;
}

/* The reference wraps every tower projection in a module that can clip its
 * input and its output, so the weight is one level deeper than the name of the
 * projection suggests. Both spellings are accepted. */
static app_code tower_bind_line(app_model *model, const char *stem, int col_hint, plane *sheet_out) {
  char name_text[544];
  app_code code;
  snprintf(name_text, sizeof(name_text), "%s.linear", stem);
  code = plane_bind(model, name_text, col_hint, sheet_out);
  if (code == APP_OKAY) return code;
  return plane_bind(model, stem, col_hint, sheet_out);
}

typedef struct sound_wing {
  plane  rise_sheet[2], drop_sheet[2]; /* the two feed-forwards */
  float *rise_norm[2], *drop_norm[2];

  plane  query_sheet, key_sheet, value_sheet, exit_sheet;
  plane  place_sheet; /* the relative position projection */
  float *dim_gain;    /* one scale per head channel, through a softplus */
  float *enter_norm, *leave_norm, *close_norm;

  plane  conv_start_sheet, conv_end_sheet;
  float *conv_deep; /* the depthwise kernel, one row per tap */
  float *conv_enter_norm, *conv_norm;
} sound_wing;

static void sound_free(tower_gear *gear) {
  int layer_index;
  if (!gear->sound_list) return;
  for (layer_index = 0; layer_index < gear->form.layer_count; ++layer_index) {
    sound_wing *wing = &gear->sound_list[layer_index];
    int side_index;
    for (side_index = 0; side_index < 2; ++side_index) {
      mem_free(wing->rise_sheet[side_index].own_block);
      mem_free(wing->drop_sheet[side_index].own_block);
      mem_free(wing->rise_norm[side_index]);
      mem_free(wing->drop_norm[side_index]);
    }
    mem_free(wing->query_sheet.own_block);
    mem_free(wing->key_sheet.own_block);
    mem_free(wing->value_sheet.own_block);
    mem_free(wing->exit_sheet.own_block);
    mem_free(wing->place_sheet.own_block);
    mem_free(wing->dim_gain);
    mem_free(wing->enter_norm);
    mem_free(wing->leave_norm);
    mem_free(wing->close_norm);
    mem_free(wing->conv_start_sheet.own_block);
    mem_free(wing->conv_end_sheet.own_block);
    mem_free(wing->conv_deep);
    mem_free(wing->conv_enter_norm);
    mem_free(wing->conv_norm);
  }
  mem_free(gear->sound_list);
  gear->sound_list = NULL;
}

static void tower_free(tower_gear *gear) {
  int layer_index;
  if (!gear) return;
  if (gear->wing_list) {
    for (layer_index = 0; layer_index < gear->form.layer_count; ++layer_index) {
      tower_wing *wing = &gear->wing_list[layer_index];
      mem_free(wing->query_sheet.own_block);
      mem_free(wing->key_sheet.own_block);
      mem_free(wing->value_sheet.own_block);
      mem_free(wing->exit_sheet.own_block);
      mem_free(wing->gate_sheet.own_block);
      mem_free(wing->rise_sheet.own_block);
      mem_free(wing->drop_sheet.own_block);
      mem_free(wing->query_bias);
      mem_free(wing->key_bias);
      mem_free(wing->value_bias);
      mem_free(wing->exit_bias);
      mem_free(wing->gate_bias);
      mem_free(wing->rise_bias);
      mem_free(wing->drop_bias);
      mem_free(wing->query_norm);
      mem_free(wing->key_norm);
      mem_free(wing->enter_norm);
      mem_free(wing->after_attn_norm);
      mem_free(wing->before_feed_norm);
      mem_free(wing->after_feed_norm);
    }
    mem_free(gear->wing_list);
    gear->wing_list = NULL;
  }
  mem_free(gear->patch_sheet.own_block);
  mem_free(gear->patch_bias);
  mem_free(gear->place_sheet.own_block);
  mem_free(gear->conv_sheet[0].own_block);
  mem_free(gear->conv_sheet[1].own_block);
  mem_free(gear->conv_norm[0]);
  mem_free(gear->conv_norm[1]);
  mem_free(gear->join_sheet.own_block);
  mem_free(gear->join_bias);
  mem_free(gear->out_sheet.own_block);
  mem_free(gear->out_bias);
  sound_free(gear);
  mem_free(gear->final_norm);
  mem_free(gear->lift_norm);
  mem_free(gear->lift_sheet.own_block);
  mem_free(gear->lift_bias);
  mem_free(gear->form.rope.step_list);
  memset(gear, 0, sizeof(*gear));
}

/* The projector, which both towers end in: one norm and one linear into the
 * text hidden width.  The norm carries no scale in the reference, so a missing
 * weight means "normalize without one" rather than "do not normalize". */
static app_code tower_lift_bind(app_model *model, tower_gear *gear) {
  tower_form *form = &gear->form;
  char stem_text[512];
  app_code code;
  snprintf(stem_text, sizeof(stem_text), "%snorm", form->lift_text);
  gear->lift_norm = vec_bind(model, stem_text, form->lift_size);
  if (!gear->lift_norm) {
    snprintf(stem_text, sizeof(stem_text), "%smm_soft_emb_norm", form->lift_text);
    gear->lift_norm = vec_bind(model, stem_text, form->lift_size);
  }
  snprintf(stem_text, sizeof(stem_text), "%s%s", form->lift_text, form->lift_leaf);
  code = plane_bind_turn(model, stem_text, model->form.state_size, form->lift_size,
                         &gear->lift_sheet);
  if (code != APP_OKAY) return code;
  snprintf(stem_text, sizeof(stem_text), "%s%s.bias", form->lift_text, form->lift_leaf);
  gear->lift_bias = vec_bind(model, stem_text, model->form.state_size);
  return APP_OKAY;
}

/* Turns the depthwise kernel tap-major, once, at bind time.
 *
 * The checkpoint stores it one row per channel, which is the wrong order to
 * convolve with: a tap is then a stride of the whole state through the frames
 * it reads, so the innermost loop walks the kernel per channel per frame and
 * touches one float of a cache line at a time.  Tap-major, a tap is a
 * contiguous run over every channel and the convolution is the element-wise
 * multiply-add every vector path already has.  The taps of one channel are
 * still summed in the order they were, so the arithmetic is unmoved. */
static int sound_conv_flip(float *value_data, int state_size, int side_size) {
  size_t total_count = (size_t)state_size * (size_t)side_size;
  float *copy_data = (float *)mem_alloc(sizeof(float) * total_count);
  int value_index, tap_index;
  if (!copy_data) return 0;
  memcpy(copy_data, value_data, sizeof(float) * total_count);
  for (tap_index = 0; tap_index < side_size; ++tap_index)
    for (value_index = 0; value_index < state_size; ++value_index)
      value_data[(size_t)tap_index * (size_t)state_size + (size_t)value_index] =
          copy_data[(size_t)value_index * (size_t)side_size + (size_t)tap_index];
  mem_free(copy_data);
  return 1;
}

/* The conformer's weights.  The layer shape is different enough from the vision
 * tower's that it gets its own binder rather than sharing `tower_bind`'s. */
static app_code sound_bind(app_model *model, tower_gear *gear) {
  tower_form *form = &gear->form;
  char stem_text[512];
  int layer_index, stage_index;
  app_code code;

  gear->sound_list = (sound_wing *)mem_clear(sizeof(sound_wing) * (size_t)form->layer_count);
  if (!gear->sound_list) return APP_FAIL_MEMORY;

  for (layer_index = 0; layer_index < form->layer_count; ++layer_index) {
    sound_wing *wing = &gear->sound_list[layer_index];
    int side_index;
#define SOUND_STEM(leaf)                                                                     \
  (snprintf(stem_text, sizeof(stem_text), "%s%s.%d.%s", form->prefix_text, form->layer_text,  \
            layer_index, (leaf)),                                                             \
   stem_text)
    code = tower_bind_line(model, SOUND_STEM("self_attn.q_proj"), form->state_size,
                           &wing->query_sheet);
    if (code != APP_OKAY) return code;
    if (layer_index == 0) {
      form->state_size = wing->query_sheet.col_count;
      /* The conformer's configuration leaves the head size out, so it is the
       * hidden width over the head count, which is what the reference does. */
      if (form->head_size < 1) form->head_size = form->state_size / form->head_count;
      if (form->head_size < 2) return APP_FAIL_FORMAT;
      if (wing->query_sheet.row_count != form->head_count * form->head_size)
        return APP_FAIL_FORMAT;
    }
    code = tower_bind_line(model, SOUND_STEM("self_attn.k_proj"), form->state_size,
                           &wing->key_sheet);
    if (code != APP_OKAY) return code;
    code = tower_bind_line(model, SOUND_STEM("self_attn.v_proj"), form->state_size,
                           &wing->value_sheet);
    if (code != APP_OKAY) return code;
    code = tower_bind_line(model, SOUND_STEM("self_attn.post"), wing->query_sheet.row_count,
                           &wing->exit_sheet);
    if (code != APP_OKAY)
      code = tower_bind_line(model, SOUND_STEM("self_attn.o_proj"), wing->query_sheet.row_count,
                             &wing->exit_sheet);
    if (code != APP_OKAY) return code;
    code = tower_bind_line(model, SOUND_STEM("self_attn.relative_k_proj"), form->state_size,
                           &wing->place_sheet);
    if (code != APP_OKAY) return code;
    wing->dim_gain = vec_bind(model, SOUND_STEM("self_attn.per_dim_scale"), form->head_size);

    for (side_index = 0; side_index < 2; ++side_index) {
      char leaf_text[64];
      snprintf(leaf_text, sizeof(leaf_text), "feed_forward%d.ffw_layer_1", side_index + 1);
      code = tower_bind_line(model, SOUND_STEM(leaf_text), form->state_size,
                             &wing->rise_sheet[side_index]);
      if (code != APP_OKAY) return code;
      if (layer_index == 0 && side_index == 0) form->inner_size = wing->rise_sheet[0].row_count;
      snprintf(leaf_text, sizeof(leaf_text), "feed_forward%d.ffw_layer_2", side_index + 1);
      code = tower_bind_line(model, SOUND_STEM(leaf_text), form->inner_size,
                             &wing->drop_sheet[side_index]);
      if (code != APP_OKAY) return code;
      snprintf(leaf_text, sizeof(leaf_text), "feed_forward%d.pre_layer_norm", side_index + 1);
      wing->rise_norm[side_index] = vec_bind(model, SOUND_STEM(leaf_text), form->state_size);
      snprintf(leaf_text, sizeof(leaf_text), "feed_forward%d.post_layer_norm", side_index + 1);
      wing->drop_norm[side_index] = vec_bind(model, SOUND_STEM(leaf_text), form->state_size);
      if (!wing->rise_norm[side_index] || !wing->drop_norm[side_index]) return APP_FAIL_MISSING;
    }

    code = tower_bind_line(model, SOUND_STEM("lconv1d.linear_start"), form->state_size,
                           &wing->conv_start_sheet);
    if (code != APP_OKAY) return code;
    code = tower_bind_line(model, SOUND_STEM("lconv1d.linear_end"), form->state_size,
                           &wing->conv_end_sheet);
    if (code != APP_OKAY) return code;
    if (wing->conv_start_sheet.row_count != 2 * form->state_size) return APP_FAIL_FORMAT;
    wing->conv_enter_norm = vec_bind(model, SOUND_STEM("lconv1d.pre_layer_norm"), form->state_size);
    wing->conv_norm = vec_bind(model, SOUND_STEM("lconv1d.conv_norm"), form->state_size);
    wing->conv_deep = vec_bind(model, SOUND_STEM("lconv1d.depthwise_conv1d"),
                               form->state_size * form->deep_side);
    if (!wing->conv_enter_norm || !wing->conv_norm || !wing->conv_deep) return APP_FAIL_MISSING;
    if (!sound_conv_flip(wing->conv_deep, form->state_size, form->deep_side))
      return APP_FAIL_MEMORY;

    wing->enter_norm = vec_bind(model, SOUND_STEM("norm_pre_attn"), form->state_size);
    wing->leave_norm = vec_bind(model, SOUND_STEM("norm_post_attn"), form->state_size);
    wing->close_norm = vec_bind(model, SOUND_STEM("norm_out"), form->state_size);
    if (!wing->enter_norm || !wing->leave_norm || !wing->close_norm) return APP_FAIL_MISSING;
#undef SOUND_STEM
  }

  for (stage_index = 0; stage_index < 2; ++stage_index) {
    snprintf(stem_text, sizeof(stem_text), "%ssubsample_conv_projection.layer%d.conv",
             form->prefix_text, stage_index);
    if (plane_bind_fold(model, stem_text, 0, &gear->conv_sheet[stage_index]) != APP_OKAY) break;
    snprintf(stem_text, sizeof(stem_text), "%ssubsample_conv_projection.layer%d.norm",
             form->prefix_text, stage_index);
    gear->conv_norm[stage_index] =
        vec_bind(model, stem_text, gear->conv_sheet[stage_index].row_count);
    if (!gear->conv_norm[stage_index]) return APP_FAIL_MISSING;
  }
  form->conv_count = stage_index;
  if (form->conv_count < 1) return APP_FAIL_MISSING;
  /* The kernel is square over (frame, filter), so its side is what the stored
   * column count implies for one input channel. */
  form->conv_side = (int)(sqrt((double)gear->conv_sheet[0].col_count) + 0.5);
  if (gear->conv_sheet[0].col_count != form->conv_side * form->conv_side)
    return APP_FAIL_FORMAT;

  snprintf(stem_text, sizeof(stem_text), "%ssubsample_conv_projection.input_proj_linear",
           form->prefix_text);
  code = plane_bind(model, stem_text, 0, &gear->join_sheet);
  if (code != APP_OKAY) return code;
  if (gear->join_sheet.row_count != form->state_size) return APP_FAIL_FORMAT;

  snprintf(stem_text, sizeof(stem_text), "%soutput_proj", form->prefix_text);
  code = plane_bind(model, stem_text, form->state_size, &gear->out_sheet);
  if (code != APP_OKAY) return code;
  if (form->lift_size < 1) form->lift_size = gear->out_sheet.row_count;
  if (gear->out_sheet.row_count != form->lift_size) return APP_FAIL_FORMAT;
  snprintf(stem_text, sizeof(stem_text), "%soutput_proj.bias", form->prefix_text);
  gear->out_bias = vec_bind(model, stem_text, form->lift_size);

  if (form->conv_step < 1) form->conv_step = 1;
  if (form->conv_pad < 0) form->conv_pad = 0;
  if (form->deep_side < 1) form->deep_side = 1;
  if (form->left_span < 1) form->left_span = 1;
  if (form->chunk_size < 1) form->chunk_size = form->left_span - 1;
  if (form->mel_count < 1 || form->frame_size < 2 || form->frame_step < 1) return APP_FAIL_SUPPORT;
  return APP_OKAY;
}

static app_code tower_bind(app_model *model, tower_gear *gear, int kind_mark) {
  static const char *vision_list[] = {"model.vision_tower.", "vision_tower.", "model.vision_model.",
                                      "vision_model."};
  static const char *audio_list[] = {"model.audio_tower.", "audio_tower.", "model.audio_model.",
                                     "audio_model."};
  static const char *vision_lift[] = {"model.embed_vision.", "embed_vision.",
                                      "model.vision_projector.", "vision_projector.",
                                      "model.multi_modal_projector.", "multi_modal_projector."};
  static const char *audio_lift[] = {"model.embed_audio.", "embed_audio.",
                                     "model.audio_projector.", "audio_projector.",
                                     "model.multi_modal_audio_projector.", "audio_adapter."};
  static const char *lift_leaf[] = {"embedding_projection", "proj", "mm_input_projection_weight"};
  tower_form *form = &gear->form;
  char stem_text[512];
  int layer_index;
  app_code code;

  if (!form->live_flag) return APP_OKAY;

  /* Two shapes are probed for the layers, because the reference keeps its
   * vision layers inside an `encoder` module and its audio layers directly
   * under the tower. */
  {
    static const char *shape_list[] = {"encoder.layers", "layers"};
    const char *const *name_list = kind_mark == TOWER_VISION ? vision_list : audio_list;
    char probe_text[128];
    size_t shape_index;
    int found_flag = 0;
    for (shape_index = 0; !found_flag && shape_index < 2; ++shape_index) {
      snprintf(probe_text, sizeof(probe_text), "%s.0.self_attn.q_proj.linear",
               shape_list[shape_index]);
      found_flag = tower_prefix_pick(model, name_list, 4, probe_text, form->prefix_text,
                                     sizeof(form->prefix_text));
      if (!found_flag) {
        snprintf(probe_text, sizeof(probe_text), "%s.0.self_attn.q_proj", shape_list[shape_index]);
        found_flag = tower_prefix_pick(model, name_list, 4, probe_text, form->prefix_text,
                                       sizeof(form->prefix_text));
      }
      if (found_flag)
        text_fill(form->layer_text, sizeof(form->layer_text), shape_list[shape_index]);
    }
    if (!found_flag) {
      /* The configuration describes a tower the weights do not carry.  That is
       * a text-only export of a multi-modal architecture, not a broken one. */
      form->live_flag = 0;
      return APP_OKAY;
    }
  }
  {
    const char *const *name_list = kind_mark == TOWER_VISION ? vision_lift : audio_lift;
    size_t leaf_index;
    int found_flag = 0;
    for (leaf_index = 0; !found_flag && leaf_index < 3; ++leaf_index)
      if (tower_prefix_pick(model, name_list, 6, lift_leaf[leaf_index], form->lift_text,
                            sizeof(form->lift_text))) {
        text_fill(form->lift_leaf, sizeof(form->lift_leaf), lift_leaf[leaf_index]);
        found_flag = 1;
      }
    if (!found_flag) return APP_FAIL_MISSING;
  }
  if (form->layer_count < 1 || form->layer_count > MODEL_LAYER_LIMIT) return APP_FAIL_SUPPORT;
  if (form->head_count < 1) return APP_FAIL_SUPPORT;
  if (kind_mark == TOWER_AUDIO) {
    code = sound_bind(model, gear);
    if (code != APP_OKAY) return code;
    return tower_lift_bind(model, gear);
  }

  gear->wing_list = (tower_wing *)mem_clear(sizeof(tower_wing) * (size_t)form->layer_count);
  if (!gear->wing_list) return APP_FAIL_MEMORY;

  for (layer_index = 0; layer_index < form->layer_count; ++layer_index) {
    tower_wing *wing = &gear->wing_list[layer_index];
#define TOWER_STEM(leaf)                                                                     \
  (snprintf(stem_text, sizeof(stem_text), "%s%s.%d.%s", form->prefix_text, form->layer_text,  \
            layer_index, (leaf)),                                                             \
   stem_text)
    code = tower_bind_line(model, TOWER_STEM("self_attn.q_proj"), form->state_size,
                           &wing->query_sheet);
    if (code != APP_OKAY) return code;
    if (layer_index == 0) {
      /* The weights decide the width, as they do everywhere else in the loader.
       * A head size the configuration leaves out is the hidden width over the
       * head count, which is what the reference falls back to. */
      form->state_size = wing->query_sheet.col_count;
      if (form->head_size < 1) form->head_size = form->state_size / form->head_count;
      if (wing->query_sheet.row_count != form->head_count * form->head_size)
        return APP_FAIL_FORMAT;
      if (form->head_size < 2) return APP_FAIL_FORMAT;
      /* Attention is scaled by one: the query and key norms absorb the usual
       * `1/sqrt(d)`, exactly as they do in the text stack. */
      form->head_gain = 1.0f;
    }
    code = tower_bind_line(model, TOWER_STEM("self_attn.k_proj"), form->state_size,
                           &wing->key_sheet);
    if (code != APP_OKAY) return code;
    code = tower_bind_line(model, TOWER_STEM("self_attn.v_proj"), form->state_size,
                           &wing->value_sheet);
    if (code != APP_OKAY) return code;
    code = tower_bind_line(model, TOWER_STEM("self_attn.o_proj"), wing->query_sheet.row_count,
                           &wing->exit_sheet);
    if (code != APP_OKAY)
      code = tower_bind_line(model, TOWER_STEM("self_attn.post"), wing->query_sheet.row_count,
                             &wing->exit_sheet);
    if (code != APP_OKAY) return code;
    code = tower_bind_line(model, TOWER_STEM("mlp.gate_proj"), form->state_size, &wing->gate_sheet);
    if (code != APP_OKAY) return code;
    code = tower_bind_line(model, TOWER_STEM("mlp.up_proj"), form->state_size, &wing->rise_sheet);
    if (code != APP_OKAY) return code;
    code = tower_bind_line(model, TOWER_STEM("mlp.down_proj"), wing->gate_sheet.row_count,
                           &wing->drop_sheet);
    if (code != APP_OKAY) return code;
    if (layer_index == 0) form->inner_size = wing->gate_sheet.row_count;
    else if (wing->gate_sheet.row_count != form->inner_size) return APP_FAIL_FORMAT;
    if (wing->key_sheet.row_count != wing->query_sheet.row_count ||
        wing->value_sheet.row_count != wing->query_sheet.row_count)
      return APP_FAIL_SUPPORT; /* a tower keeps one key-value head per query head */

    wing->query_bias =
        vec_bind(model, TOWER_STEM("self_attn.q_proj.bias"), wing->query_sheet.row_count);
    wing->key_bias = vec_bind(model, TOWER_STEM("self_attn.k_proj.bias"), wing->key_sheet.row_count);
    wing->value_bias =
        vec_bind(model, TOWER_STEM("self_attn.v_proj.bias"), wing->value_sheet.row_count);
    wing->exit_bias = vec_bind(model, TOWER_STEM("self_attn.o_proj.bias"), form->state_size);
    wing->gate_bias = vec_bind(model, TOWER_STEM("mlp.gate_proj.bias"), form->inner_size);
    wing->rise_bias = vec_bind(model, TOWER_STEM("mlp.up_proj.bias"), form->inner_size);
    wing->drop_bias = vec_bind(model, TOWER_STEM("mlp.down_proj.bias"), form->state_size);
    wing->query_norm = vec_bind(model, TOWER_STEM("self_attn.q_norm"), form->head_size);
    wing->key_norm = vec_bind(model, TOWER_STEM("self_attn.k_norm"), form->head_size);

    wing->enter_norm = vec_bind(model, TOWER_STEM("input_layernorm"), form->state_size);
    wing->after_attn_norm =
        vec_bind(model, TOWER_STEM("post_attention_layernorm"), form->state_size);
    wing->before_feed_norm =
        vec_bind(model, TOWER_STEM("pre_feedforward_layernorm"), form->state_size);
    wing->after_feed_norm =
        vec_bind(model, TOWER_STEM("post_feedforward_layernorm"), form->state_size);
    if (!wing->enter_norm || !wing->after_attn_norm || !wing->before_feed_norm ||
        !wing->after_feed_norm)
      return APP_FAIL_MISSING;

#undef TOWER_STEM
  }

  /* A trailing norm is not universal: the reference's vision encoder ends at
   * its last layer and leaves the normalizing to the projector. */
  snprintf(stem_text, sizeof(stem_text), "%snorm", form->prefix_text);
  gear->final_norm = vec_bind(model, stem_text, form->state_size);

  if (kind_mark == TOWER_VISION) {
    const store_span *place_span;
    snprintf(stem_text, sizeof(stem_text), "%spatch_embedder.input_proj", form->prefix_text);
    code = plane_bind_fold(model, stem_text,
                           form->band_count * form->patch_size * form->patch_size,
                           &gear->patch_sheet);
    if (code != APP_OKAY) {
      snprintf(stem_text, sizeof(stem_text), "%sembeddings.patch_embedding", form->prefix_text);
      code = plane_bind_fold(model, stem_text,
                             form->band_count * form->patch_size * form->patch_size,
                             &gear->patch_sheet);
    }
    if (code != APP_OKAY) return code;
    if (gear->patch_sheet.row_count != form->state_size) return APP_FAIL_FORMAT;
    if (form->patch_size < 1 || form->band_count < 1 || form->band_count > MEDIA_BAND_LIMIT)
      return APP_FAIL_SUPPORT;
    if (gear->patch_sheet.col_count != form->band_count * form->patch_size * form->patch_size)
      return APP_FAIL_FORMAT;
    snprintf(stem_text, sizeof(stem_text), "%spatch_embedder.input_proj.bias", form->prefix_text);
    gear->patch_bias = vec_bind(model, stem_text, form->state_size);

    /* One table per axis, stacked, looked up by the patch's column and its row
     * and summed.  It is far too large to materialize as f32 — ten thousand
     * rows twice over — so it stays a plane and is read a row at a time. */
    snprintf(stem_text, sizeof(stem_text), "%spatch_embedder.position_embedding_table",
             form->prefix_text);
    place_span = store_find(&model->store, stem_text);
    if (place_span && place_span->rank_count == 3) {
      if ((int)place_span->size_list[0] != 2 ||
          (int)place_span->size_list[2] != form->state_size)
        return APP_FAIL_FORMAT;
      form->place_size = (int)place_span->size_list[1];
      gear->place_sheet.form = PLANE_REAL;
      gear->place_sheet.row_count = 2 * form->place_size;
      gear->place_sheet.col_count = form->state_size;
      gear->place_sheet.real_data = place_span->data_base;
      gear->place_sheet.real_type = place_span->type_kind;
      model->weight_bytes += place_span->data_bytes;
    }
    if (form->pool_size < 1) form->pool_size = 1;
    if (form->soft_limit < 1) form->soft_limit = 280;
    /* Each axis takes half a head and is rotated as a pair, so a head has to
     * divide by four before the two dimensional schedule means anything. */
    if (form->head_size % 4 != 0) return APP_FAIL_SUPPORT;
    rope_build(&form->rope, form->head_size / 2, "default");
    if (!form->rope.step_list) return APP_FAIL_MEMORY;
    form->lift_size = form->state_size;
  }

  return tower_lift_bind(model, gear);
}

/* -- tower forward ------------------------------------------------------- */

/* One tower's scratch.  Nothing here outlives a single image or clip, so it is
 * allocated per call rather than per session: a tower runs once for a prompt
 * and never inside the token loop. */
typedef struct tower_room {
  int    lane_count;
  int    head_wide;
  int    quant_stride;
  float *state_data;
  float *scrap_data;
  float *lift_data;
  float *query_data;
  float *key_data;
  float *value_data;
  float *blend_data;
  float *gate_data;
  float *rise_data;
  float *score_data;
  float *cos_data;
  float *sin_data;
  float *key_pack;  /* one head's keys and values, gathered into a run */
  float *value_pack;
  float *turn_data; /* the projected relative rows, one per reachable offset */
  float *wave_data;
  float *quant_data;
} tower_room;

static void tower_room_free(tower_room *room) {
  if (!room) return;
  mem_free(room->state_data);
  mem_free(room->scrap_data);
  mem_free(room->lift_data);
  mem_free(room->query_data);
  mem_free(room->key_data);
  mem_free(room->value_data);
  mem_free(room->blend_data);
  mem_free(room->gate_data);
  mem_free(room->rise_data);
  mem_free(room->score_data);
  mem_free(room->cos_data);
  mem_free(room->sin_data);
  mem_free(room->key_pack);
  mem_free(room->value_pack);
  mem_free(room->turn_data);
  mem_free(room->wave_data);
  mem_free(room->quant_data);
  memset(room, 0, sizeof(*room));
}

/* The widest column any of a tower's planes reads, which is what the activation
 * staging room has to hold. */
static int tower_wide_peak(const tower_gear *gear) {
  const tower_form *form = &gear->form;
  int wide_peak = form->state_size;
  if (form->inner_size > wide_peak) wide_peak = form->inner_size;
  if (form->head_count * form->head_size > wide_peak) wide_peak = form->head_count * form->head_size;
  if (gear->patch_sheet.col_count > wide_peak) wide_peak = gear->patch_sheet.col_count;
  if (gear->join_sheet.col_count > wide_peak) wide_peak = gear->join_sheet.col_count;
  if (gear->conv_sheet[0].col_count > wide_peak) wide_peak = gear->conv_sheet[0].col_count;
  if (gear->conv_sheet[1].col_count > wide_peak) wide_peak = gear->conv_sheet[1].col_count;
  if (gear->lift_sheet.col_count > wide_peak) wide_peak = gear->lift_sheet.col_count;
  return wide_peak;
}

static app_code tower_room_open(tower_room *room, const tower_gear *gear, int lane_count,
                                int band_count,
                                int turn_flag) {
  const tower_form *form = &gear->form;
  int wide_peak = tower_wide_peak(gear);
  memset(room, 0, sizeof(*room));
  if (lane_count < 1) return APP_FAIL_ARGUMENT;
  room->lane_count = lane_count;
  room->head_wide = form->head_count * form->head_size;
  room->quant_stride = wide_peak;
  room->state_data =
      (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)form->state_size);
  room->scrap_data =
      (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)form->state_size);
  room->lift_data =
      (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)form->state_size);
  room->query_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)room->head_wide);
  room->key_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)room->head_wide);
  room->value_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)room->head_wide);
  room->blend_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)room->head_wide);
  room->gate_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)form->inner_size);
  room->rise_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)form->inner_size);
  room->key_pack =
      (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)form->head_size);
  room->value_pack =
      (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)form->head_size);
  /* One slice of scores per band, because the bands run at the same time. */
  if (band_count < 1) band_count = 1;
  room->score_data =
      (float *)mem_clear(sizeof(float) * (size_t)band_count * (size_t)lane_count);
  room->cos_data = (float *)mem_clear(sizeof(float) * (size_t)(form->head_size / 2 + 1));
  room->sin_data = (float *)mem_clear(sizeof(float) * (size_t)(form->head_size / 2 + 1));
  room->quant_data = (float *)mem_clear(sizeof(float) * (size_t)KERN_LANE_LIMIT * (size_t)wide_peak);
  if (turn_flag) {
    room->turn_data = (float *)mem_clear(sizeof(float) * (size_t)(2 * lane_count - 1) *
                                         (size_t)room->head_wide);
    room->wave_data =
        (float *)mem_clear(sizeof(float) * (size_t)(2 * lane_count - 1) * (size_t)form->state_size);
  }
  if (!room->state_data || !room->scrap_data || !room->lift_data || !room->query_data ||
      !room->key_data || !room->value_data || !room->blend_data || !room->gate_data ||
      !room->rise_data || !room->score_data || !room->cos_data || !room->sin_data ||
      !room->key_pack || !room->value_pack || !room->quant_data || (turn_flag && (!room->turn_data || !room->wave_data))) {
    tower_room_free(room);
    return APP_FAIL_MEMORY;
  }
  return APP_OKAY;
}

/* Adds a bias to every lane of a projection.  The text stack never needs one; a
 * vision encoder almost always does. */
static void tower_bias_add(float *value_data, int value_stride, int lane_count,
                           const float *bias_list, int value_count) {
  int lane_index, value_index;
  if (!bias_list) return;
  for (lane_index = 0; lane_index < lane_count; ++lane_index) {
    float *lane_data = value_data + (size_t)lane_index * (size_t)value_stride;
    for (value_index = 0; value_index < value_count; ++value_index)
      lane_data[value_index] += bias_list[value_index];
  }
}

static void tower_lift(app_model *model, tower_room *room, const plane *sheet, const float *act_data,
                       int act_stride, int lane_count, float *out_data, int out_stride,
                       const float *bias_list) {
  plane_lift_many(model, sheet, act_data, act_stride, lane_count, out_data, out_stride,
                  room->quant_data, room->quant_stride);
  tower_bias_add(out_data, out_stride, lane_count, bias_list, sheet->row_count);
}

/* Softmaxes one head's scores and blends the values behind them.  Both towers
 * end up here; what differs is the score, which the caller has already written
 * into `score_data`. */
/* One band of the patch grid, for one head: every query in the band scored
 * against every key, then blended.
 *
 * That head's keys and values have already been gathered into a run of their
 * own, which is what makes this worth threading. Left where they lie, a head's
 * keys are sixty-four floats in every seven hundred and sixty-eight, so scoring
 * one query walks the whole seven megabyte array and the next query walks it
 * again. Gathered, a head is six hundred kilobytes that stays in cache across
 * every query in the band.
 *
 * A band writes only its own rows of the blend and its own slice of the score
 * scratch, so nothing is shared but the reads. */
typedef struct grid_job {
  app_model   *model;
  tower_room  *room;
  const float *key_pack;
  const float *value_pack;
  int          head_index;
  int          head_size;
  int          lane_count;
  float        head_gain;
} grid_job;

static void tower_attend_band(void *state, int slice_index, int slice_count) {
  grid_job *job = (grid_job *)state;
  tower_room *room = job->room;
  int head_size = job->head_size;
  int head_wide = room->head_wide;
  float *score_data = room->score_data + (size_t)slice_index * (size_t)job->lane_count;
  int from_lane, upto_lane, lane_index, span_index;
  slice_span(job->lane_count, slice_index, slice_count, &from_lane, &upto_lane);
  for (lane_index = from_lane; lane_index < upto_lane; ++lane_index) {
    const float *query_head = room->query_data + (size_t)lane_index * (size_t)head_wide +
                              (size_t)job->head_index * head_size;
    float *blend_head = room->blend_data + (size_t)lane_index * (size_t)head_wide +
                        (size_t)job->head_index * head_size;
    for (span_index = 0; span_index < job->lane_count; ++span_index)
      score_data[span_index] =
          kern_dot_real(job->key_pack + (size_t)span_index * (size_t)head_size, STORE_F32,
                        query_head, head_size) *
          job->head_gain;
    job->model->desk.soft_max(&job->model->desk, score_data, job->lane_count);
    kern_blend_rows(job->value_pack, head_size, score_data, job->lane_count, head_size, blend_head);
  }
}

/* Bidirectional attention over a patch grid, with the row and the column of
 * each patch turned into the head by the two dimensional rotary. */
static void tower_attend_grid(app_model *model, tower_gear *gear, tower_wing *wing,
                              tower_room *room, int lane_count) {
  tower_form *form = &gear->form;
  int head_size = form->head_size;
  int head_wide = room->head_wide;
  int lane_index, head_index;

  tower_lift(model, room, &wing->query_sheet, room->scrap_data, form->state_size, lane_count,
             room->query_data, head_wide, wing->query_bias);
  tower_lift(model, room, &wing->key_sheet, room->scrap_data, form->state_size, lane_count,
             room->key_data, head_wide, wing->key_bias);
  tower_lift(model, room, &wing->value_sheet, room->scrap_data, form->state_size, lane_count,
             room->value_data, head_wide, wing->value_bias);

  for (lane_index = 0; lane_index < lane_count; ++lane_index) {
    rope_grid_wave(&form->rope, lane_index / form->grid_wide, lane_index % form->grid_wide,
                   room->cos_data, room->sin_data);
    for (head_index = 0; head_index < form->head_count; ++head_index) {
      float *query_head = room->query_data + (size_t)lane_index * (size_t)head_wide +
                          (size_t)head_index * head_size;
      float *key_head =
          room->key_data + (size_t)lane_index * (size_t)head_wide + (size_t)head_index * head_size;
      float *value_head =
          room->value_data + (size_t)lane_index * (size_t)head_wide + (size_t)head_index * head_size;
      if (wing->query_norm)
        model->desk.norm_rms(&model->desk, query_head, wing->query_norm, head_size, form->norm_eps,
                             query_head);
      if (wing->key_norm)
        model->desk.norm_rms(&model->desk, key_head, wing->key_norm, head_size, form->norm_eps,
                             key_head);
      /* The values carry a norm without a scale, as they do in the text stack. */
      model->desk.norm_rms(&model->desk, value_head, NULL, head_size, form->norm_eps, value_head);
      rope_grid_turn(query_head, head_size, room->cos_data, room->sin_data);
      rope_grid_turn(key_head, head_size, room->cos_data, room->sin_data);
    }
  }

  /* This is the whole cost of an image: a grid of two thousand patches scored
   * against itself, sixteen times over. One head at a time, so its keys and
   * values can be gathered into a run that stays in cache, and the queries
   * divided across the pool. */
  {
    grid_job job;
    job.model = model;
    job.room = room;
    job.head_size = head_size;
    job.lane_count = lane_count;
    job.head_gain = form->head_gain;
    for (head_index = 0; head_index < form->head_count; ++head_index) {
      for (lane_index = 0; lane_index < lane_count; ++lane_index) {
        memcpy(room->key_pack + (size_t)lane_index * (size_t)head_size,
               room->key_data + (size_t)lane_index * (size_t)head_wide +
                   (size_t)head_index * head_size,
               sizeof(float) * (size_t)head_size);
        memcpy(room->value_pack + (size_t)lane_index * (size_t)head_size,
               room->value_data + (size_t)lane_index * (size_t)head_wide +
                   (size_t)head_index * head_size,
               sizeof(float) * (size_t)head_size);
      }
      job.key_pack = room->key_pack;
      job.value_pack = room->value_pack;
      job.head_index = head_index;
      if (model->pool.worker_count > 0 && lane_count >= 64)
        pool_run(&model->pool, tower_attend_band, &job);
      else
        tower_attend_band(&job, 0, 1);
    }
  }

  tower_lift(model, room, &wing->exit_sheet, room->blend_data, head_wide, lane_count,
             room->lift_data, form->state_size, wing->exit_bias);
}

/* The vision layer: norm, attention, norm, add, then norm, gated feed-forward,
 * norm, add.  It is `session_layer` without the per-layer embedding gate and
 * without the mixture branch, which is the arrangement the reference's vision
 * encoder layer uses exactly. */
static void tower_feed(app_model *model, tower_gear *gear, int layer_index, tower_room *room,
                       int lane_count) {
  tower_form *form = &gear->form;
  tower_wing *wing = &gear->wing_list[layer_index];
  const char *stem_text = tower_name(form->kind_mark);
  int state_size = form->state_size;
  int lane_index;

  for (lane_index = 0; lane_index < lane_count; ++lane_index)
    model->desk.norm_rms(&model->desk, room->state_data + (size_t)lane_index * (size_t)state_size,
                         wing->enter_norm, state_size, form->norm_eps,
                         room->scrap_data + (size_t)lane_index * (size_t)state_size);
  tower_attend_grid(model, gear, wing, room, lane_count);
  for (lane_index = 0; lane_index < lane_count; ++lane_index) {
    float *lift_data = room->lift_data + (size_t)lane_index * (size_t)state_size;
    TRACE_TOWER(stem_text, "attn", layer_index, lane_index, lift_data, state_size);
    model->desk.norm_rms(&model->desk, lift_data, wing->after_attn_norm, state_size, form->norm_eps,
                         lift_data);
    kern_add(room->state_data + (size_t)lane_index * (size_t)state_size, lift_data, state_size);
    model->desk.norm_rms(&model->desk, room->state_data + (size_t)lane_index * (size_t)state_size,
                         wing->before_feed_norm, state_size, form->norm_eps,
                         room->scrap_data + (size_t)lane_index * (size_t)state_size);
  }

  tower_lift(model, room, &wing->gate_sheet, room->scrap_data, state_size, lane_count,
             room->gate_data, form->inner_size, wing->gate_bias);
  tower_lift(model, room, &wing->rise_sheet, room->scrap_data, state_size, lane_count,
             room->rise_data, form->inner_size, wing->rise_bias);
  for (lane_index = 0; lane_index < lane_count; ++lane_index)
    model->desk.gelu_gate(&model->desk,
                          room->gate_data + (size_t)lane_index * (size_t)form->inner_size,
                          room->rise_data + (size_t)lane_index * (size_t)form->inner_size,
                          form->inner_size);
  tower_lift(model, room, &wing->drop_sheet, room->gate_data, form->inner_size, lane_count,
             room->lift_data, state_size, wing->drop_bias);
  for (lane_index = 0; lane_index < lane_count; ++lane_index) {
    float *lift_data = room->lift_data + (size_t)lane_index * (size_t)state_size;
    TRACE_TOWER(stem_text, "mlp", layer_index, lane_index, lift_data, state_size);
    model->desk.norm_rms(&model->desk, lift_data, wing->after_feed_norm, state_size, form->norm_eps,
                         lift_data);
    kern_add(room->state_data + (size_t)lane_index * (size_t)state_size, lift_data, state_size);
    TRACE_TOWER(stem_text, "out", layer_index, lane_index,
                room->state_data + (size_t)lane_index * (size_t)state_size, state_size);
  }
}

/* The projector.  `scrap_data` is safe to use as the staging buffer here
 * because the last layer has already folded everything it held back into the
 * residual. */
static void tower_lift_rows(app_model *model, tower_gear *gear, tower_room *room,
                            const float *from_data, int lane_count, float *out_data) {
  tower_form *form = &gear->form;
  const char *stem_text = tower_name(form->kind_mark);
  int lane_index;
  for (lane_index = 0; lane_index < lane_count; ++lane_index) {
    const float *lane_data = from_data + (size_t)lane_index * (size_t)form->lift_size;
    float *scrap_data = room->scrap_data + (size_t)lane_index * (size_t)form->lift_size;
    if (gear->final_norm)
      model->desk.norm_rms(&model->desk, lane_data, gear->final_norm, form->lift_size,
                           form->norm_eps, scrap_data);
    else if (scrap_data != lane_data)
      memcpy(scrap_data, lane_data, sizeof(float) * (size_t)form->lift_size);
    /* The projector's norm always runs; a null gain means it carries no scale,
     * which is how the reference writes it. */
    model->desk.norm_rms(&model->desk, scrap_data, gear->lift_norm, form->lift_size,
                         form->norm_eps, scrap_data);
  }
  plane_lift_many(model, &gear->lift_sheet, room->scrap_data, form->lift_size, lane_count, out_data,
                  model->form.state_size, room->quant_data, room->quant_stride);
  tower_bias_add(out_data, model->form.state_size, lane_count, gear->lift_bias,
                 model->form.state_size);
  for (lane_index = 0; lane_index < lane_count; ++lane_index)
    TRACE_TOWER(stem_text, "lift", -1, lane_index,
                out_data + (size_t)lane_index * (size_t)model->form.state_size,
                model->form.state_size);
}

/* -- the vision tower ----------------------------------------------------- */

/* The size the picture is resized to.
 *
 * The tower takes a variable resolution rather than a fixed square: the aspect
 * ratio is preserved, the area is capped by the patch budget the soft token
 * limit implies, and both sides are rounded down to a multiple of
 * `pool_size * patch_size` so that the pooler's windows divide exactly. */
static void vision_grid_pick(const tower_form *form, int wide_count, int high_count,
                             int *wide_out, int *high_out) {
  int side_step = form->pool_size * form->patch_size;
  long patch_limit = (long)form->soft_limit * (long)form->pool_size * (long)form->pool_size;
  double target_area = (double)patch_limit * (double)form->patch_size * (double)form->patch_size;
  double shrink_value = sqrt(target_area / ((double)wide_count * (double)high_count));
  int side_limit = (int)(patch_limit / ((long)form->pool_size * (long)form->pool_size)) * side_step;
  int want_wide = (int)floor(shrink_value * (double)wide_count / (double)side_step) * side_step;
  int want_high = (int)floor(shrink_value * (double)high_count / (double)side_step) * side_step;

  /* A picture far from square can round one side to nothing; it keeps one band
   * of patches and gives the other side whatever the ratio allows. */
  if (want_wide < side_step && want_high < side_step) {
    want_wide = side_step;
    want_high = side_step;
  } else if (want_wide < side_step) {
    want_wide = side_step;
    want_high = (int)(wide_count > 0 ? (long)high_count / wide_count : 1) * side_step;
    if (want_high > side_limit) want_high = side_limit;
    if (want_high < side_step) want_high = side_step;
  } else if (want_high < side_step) {
    want_high = side_step;
    want_wide = (int)(high_count > 0 ? (long)wide_count / high_count : 1) * side_step;
    if (want_wide > side_limit) want_wide = side_limit;
    if (want_wide < side_step) want_wide = side_step;
  }
  *wide_out = want_wide;
  *high_out = want_high;
}

/* Cuts the resized image into patches, in the order a convolution whose stride
 * equals its kernel would visit them.  Inside one patch the samples run row,
 * then column, then band — the band is the fastest axis, which is what the
 * reference's flatten produces and what its input projection expects.
 *
 * The pixels arrive in [0,1] and the tower wants [-1,1], a scaling the
 * reference does in the model rather than in the preprocessor. */
static void vision_patch_cut(const flat_grid *grid, const tower_form *form, float *out_data,
                             int out_stride) {
  int high_patch, wide_patch, band_index, high_index, wide_index;
  for (high_patch = 0; high_patch < form->grid_high; ++high_patch)
    for (wide_patch = 0; wide_patch < form->grid_wide; ++wide_patch) {
      float *lane_data =
          out_data + (size_t)(high_patch * form->grid_wide + wide_patch) * (size_t)out_stride;
      for (high_index = 0; high_index < form->patch_size; ++high_index)
        for (wide_index = 0; wide_index < form->patch_size; ++wide_index) {
          const float *cell_data = grid_at(grid, high_patch * form->patch_size + high_index,
                                           wide_patch * form->patch_size + wide_index);
          for (band_index = 0; band_index < form->band_count; ++band_index)
            lane_data[(high_index * form->patch_size + wide_index) * form->band_count +
                      band_index] = 2.0f * (cell_data[band_index] - 0.5f);
        }
    }
}

/* Averages the encoder's output over `pool_size` squared windows of the patch
 * grid and scales by the square root of the hidden width.  Both sides of the
 * grid are multiples of the pool size by construction, so every window is full
 * and the divisor is the same for all of them. */
static void vision_pool(const float *from_data, const tower_form *form, float *into_data) {
  int pool_wide = form->grid_wide / form->pool_size;
  int pool_high = form->grid_high / form->pool_size;
  float root_gain = (float)sqrt((double)form->state_size);
  float share_value = 1.0f / (float)(form->pool_size * form->pool_size);
  int high_pool, wide_pool, high_index, wide_index, value_index;
  for (high_pool = 0; high_pool < pool_high; ++high_pool)
    for (wide_pool = 0; wide_pool < pool_wide; ++wide_pool) {
      float *out_data =
          into_data + (size_t)(high_pool * pool_wide + wide_pool) * (size_t)form->state_size;
      for (value_index = 0; value_index < form->state_size; ++value_index)
        out_data[value_index] = 0.0f;
      for (high_index = high_pool * form->pool_size; high_index < (high_pool + 1) * form->pool_size;
           ++high_index)
        for (wide_index = wide_pool * form->pool_size;
             wide_index < (wide_pool + 1) * form->pool_size; ++wide_index) {
          const float *cell_data = from_data + (size_t)(high_index * form->grid_wide + wide_index) *
                                                   (size_t)form->state_size;
          for (value_index = 0; value_index < form->state_size; ++value_index)
            out_data[value_index] += cell_data[value_index];
        }
      for (value_index = 0; value_index < form->state_size; ++value_index)
        out_data[value_index] *= share_value * root_gain;
    }
}

/* Runs one image.  `grid_in` is consumed: it is resized in place rather than
 * copied, because the caller has no further use for the raw picture. */
static app_code vision_run(app_model *model, flat_grid *grid_in, float **row_out,
                           int *row_count_out) {
  tower_gear *gear = &model->tower_list[TOWER_VISION];
  tower_form *form = &gear->form;
  flat_grid work_grid;
  tower_room room;
  float *patch_data = NULL;
  float *pool_data = NULL;
  float *place_data = NULL;
  float *out_data = NULL;
  int want_wide = 0, want_high = 0, lane_count, pool_count;
  int layer_index, lane_index, value_index;
  app_code code;

  memset(&work_grid, 0, sizeof(work_grid));
  memset(&room, 0, sizeof(room));
  *row_out = NULL;
  *row_count_out = 0;
  if (!form->live_flag) return APP_FAIL_SUPPORT;

  code = grid_bands(grid_in, form->band_count);
  if (code != APP_OKAY) return code;
  vision_grid_pick(form, grid_in->wide_count, grid_in->high_count, &want_wide, &want_high);
  form->grid_wide = want_wide / form->patch_size;
  form->grid_high = want_high / form->patch_size;
  lane_count = form->grid_wide * form->grid_high;
  pool_count = (form->grid_wide / form->pool_size) * (form->grid_high / form->pool_size);
  if (lane_count < 1 || pool_count < 1) return APP_FAIL_SUPPORT;

  code = grid_scale(grid_in, want_wide, want_high, &work_grid);
  if (code == APP_OKAY)
    code = tower_room_open(&room, gear, lane_count, pool_bands(&model->pool), 0);
  if (code != APP_OKAY) goto vision_done;

  patch_data =
      (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)gear->patch_sheet.col_count);
  pool_data = (float *)mem_clear(sizeof(float) * (size_t)pool_count * (size_t)form->state_size);
  out_data = (float *)mem_clear(sizeof(float) * (size_t)pool_count * (size_t)model->form.state_size);
  if (gear->place_sheet.form != PLANE_VOID)
    place_data = (float *)mem_clear(sizeof(float) * (size_t)form->state_size);
  if (!patch_data || !pool_data || !out_data ||
      (gear->place_sheet.form != PLANE_VOID && !place_data)) {
    code = APP_FAIL_MEMORY;
    goto vision_done;
  }

  vision_patch_cut(&work_grid, form, patch_data, gear->patch_sheet.col_count);
  {
    /* The patch grid, so a comparison can rebuild the position of every row. */
    float grid_note[2];
    grid_note[0] = (float)form->grid_wide;
    grid_note[1] = (float)form->grid_high;
    TRACE_TOWER("vision", "grid", -1, 0, grid_note, 2);
  }
  for (lane_index = 0; lane_index < lane_count; ++lane_index)
    TRACE_TOWER("vision", "patch", -1, lane_index,
                patch_data + (size_t)lane_index * (size_t)gear->patch_sheet.col_count,
                gear->patch_sheet.col_count);
  tower_lift(model, &room, &gear->patch_sheet, patch_data, gear->patch_sheet.col_count, lane_count,
             room.state_data, form->state_size, gear->patch_bias);

  /* One row of the table for the patch's column and one for its row, summed
   * into the embedding.  The two tables are stacked, column table first. */
  if (place_data)
    for (lane_index = 0; lane_index < lane_count; ++lane_index) {
      float *state_data = room.state_data + (size_t)lane_index * (size_t)form->state_size;
      int wide_index = lane_index % form->grid_wide;
      int high_index = lane_index / form->grid_wide;
      if (wide_index >= form->place_size || high_index >= form->place_size) {
        code = APP_FAIL_SUPPORT; /* more patches on a side than the table has rows */
        goto vision_done;
      }
      plane_row(&gear->place_sheet, wide_index, place_data);
      for (value_index = 0; value_index < form->state_size; ++value_index)
        state_data[value_index] += place_data[value_index];
      plane_row(&gear->place_sheet, form->place_size + high_index, place_data);
      for (value_index = 0; value_index < form->state_size; ++value_index)
        state_data[value_index] += place_data[value_index];
    }
  for (lane_index = 0; lane_index < lane_count; ++lane_index)
    TRACE_TOWER("vision", "embed", -1, lane_index,
                room.state_data + (size_t)lane_index * (size_t)form->state_size, form->state_size);

  for (layer_index = 0; layer_index < form->layer_count; ++layer_index)
    tower_feed(model, gear, layer_index, &room, lane_count);

  vision_pool(room.state_data, form, pool_data);
  tower_lift_rows(model, gear, &room, pool_data, pool_count, out_data);
  *row_out = out_data;
  *row_count_out = pool_count;
  out_data = NULL;

vision_done:
  mem_free(patch_data);
  mem_free(pool_data);
  mem_free(place_data);
  mem_free(out_data);
  tower_room_free(&room);
  grid_free(&work_grid);
  return code;
}

/* -- the audio tower ------------------------------------------------------ */

/* The audio encoder is a conformer, not a transformer, and it is worth being
 * plain about the difference because the two look alike from a distance. A
 * layer here is
 *
 *     feed-forward (half residual)
 *     norm, chunked local attention, norm, residual
 *     a gated depthwise convolution over time
 *     feed-forward (half residual)
 *     norm
 *
 * so a layer carries two feed-forwards rather than one, each folding back at
 * half weight, and a convolution module the text stack has no equivalent of.
 *
 * The attention is written here as a plain causal window rather than as the
 * reference's blocked form. The reference cuts the sequence into chunks, gives
 * every chunk a context window, and then masks that window down with a sliding
 * rule of `(attention_context_left - 1, attention_context_right)`. What
 * survives both is exactly the sliding window, so the block machinery is an
 * efficiency device rather than part of the arithmetic, and the relative shift
 * it needs collapses into indexing the position row by the lag. */

/* One clip's scratch.  Sized from the frame count, which is known only after
 * the subsampler has run. */
typedef struct sound_room {
  int    lane_count;
  int    quant_stride;
  float *state_data;
  float *scrap_data;
  float *lift_data;
  float *query_data;
  float *key_data;
  float *value_data;
  float *blend_data;
  float *rise_data;
  float *conv_data;
  float *keep_data;  /* the residual the attention folds back into */
  float *wave_data;  /* the sinusoid for every reachable lag */
  float *place_data; /* and its projection */
  float *score_data;
  float *quant_data;
} sound_room;

static void sound_room_free(sound_room *room) {
  if (!room) return;
  mem_free(room->state_data);
  mem_free(room->scrap_data);
  mem_free(room->lift_data);
  mem_free(room->query_data);
  mem_free(room->key_data);
  mem_free(room->value_data);
  mem_free(room->blend_data);
  mem_free(room->rise_data);
  mem_free(room->conv_data);
  mem_free(room->keep_data);
  mem_free(room->wave_data);
  mem_free(room->place_data);
  mem_free(room->score_data);
  mem_free(room->quant_data);
  memset(room, 0, sizeof(*room));
}

static app_code sound_room_open(sound_room *room, const tower_form *form, int lane_count) {
  int state_size = form->state_size;
  int head_wide = form->head_count * form->head_size;
  int win_span = form->left_span > 1 ? form->left_span - 1 : 1;
  int lag_count = (form->chunk_size + win_span + form->right_span) / 2 + form->right_span + 1;
  if (lag_count < win_span + form->right_span) lag_count = win_span + form->right_span;
  int wide_peak = state_size;
  memset(room, 0, sizeof(*room));
  if (lane_count < 1) return APP_FAIL_ARGUMENT;
  if (form->inner_size > wide_peak) wide_peak = form->inner_size;
  if (2 * state_size > wide_peak) wide_peak = 2 * state_size;
  if (head_wide > wide_peak) wide_peak = head_wide;
  room->lane_count = lane_count;
  room->quant_stride = wide_peak;
  room->state_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)state_size);
  room->scrap_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)state_size);
  room->lift_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)state_size);
  room->query_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)head_wide);
  room->key_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)head_wide);
  room->value_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)head_wide);
  room->blend_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)head_wide);
  room->rise_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)form->inner_size);
  room->conv_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)(2 * state_size));
  room->keep_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)state_size);
  room->wave_data = (float *)mem_clear(sizeof(float) * (size_t)lag_count * (size_t)state_size);
  room->place_data = (float *)mem_clear(sizeof(float) * (size_t)lag_count * (size_t)head_wide);
  room->score_data = (float *)mem_clear(sizeof(float) * (size_t)lag_count);
  room->quant_data = (float *)mem_clear(sizeof(float) * (size_t)KERN_LANE_LIMIT * (size_t)wide_peak);
  if (!room->state_data || !room->scrap_data || !room->lift_data || !room->query_data ||
      !room->key_data || !room->value_data || !room->blend_data || !room->rise_data ||
      !room->conv_data || !room->keep_data || !room->wave_data || !room->place_data || !room->score_data ||
      !room->quant_data) {
    sound_room_free(room);
    return APP_FAIL_MEMORY;
  }
  return APP_OKAY;
}

/* One feed-forward: norm, widen, silu, narrow, norm, half weight, residual. */
static void sound_feed_run(app_model *model, tower_gear *gear, sound_wing *wing, sound_room *room,
                           int side_index, int lane_count) {
  tower_form *form = &gear->form;
  int state_size = form->state_size;
  int lane_index, value_index;
  for (lane_index = 0; lane_index < lane_count; ++lane_index)
    model->desk.norm_rms(&model->desk, room->state_data + (size_t)lane_index * (size_t)state_size,
                         wing->rise_norm[side_index], state_size, form->norm_eps,
                         room->scrap_data + (size_t)lane_index * (size_t)state_size);
  plane_lift_many(model, &wing->rise_sheet[side_index], room->scrap_data, state_size, lane_count,
                  room->rise_data, form->inner_size, room->quant_data, room->quant_stride);
  for (lane_index = 0; lane_index < lane_count * form->inner_size; ++lane_index)
    room->rise_data[lane_index] = kern_silu(room->rise_data[lane_index]);
  plane_lift_many(model, &wing->drop_sheet[side_index], room->rise_data, form->inner_size,
                  lane_count, room->lift_data, state_size, room->quant_data, room->quant_stride);
  for (lane_index = 0; lane_index < lane_count; ++lane_index) {
    float *lift_data = room->lift_data + (size_t)lane_index * (size_t)state_size;
    float *state_data = room->state_data + (size_t)lane_index * (size_t)state_size;
    model->desk.norm_rms(&model->desk, lift_data, wing->drop_norm[side_index], state_size,
                         form->norm_eps, lift_data);
    for (value_index = 0; value_index < state_size; ++value_index)
      state_data[value_index] += form->share_gain * lift_data[value_index];
  }
}

/* Chunked local attention, written as the causal window the chunking and the
 * sliding mask agree on.  The position term is a projection of the sinusoid of
 * the lag, shared by every query at that lag. */
static void sound_attend(app_model *model, tower_gear *gear, sound_wing *wing, sound_room *room,
                         int lane_count) {
  tower_form *form = &gear->form;
  int state_size = form->state_size;
  int head_size = form->head_size;
  int head_wide = form->head_count * form->head_size;
  /* `attention_context_left` counts the query's own frame, and the reference's
   * window rule is a strict inequality, so a query reaches `left_span - 1` keys
   * at or before itself.  Getting that bound wrong by one is invisible in every
   * shape and changes every number. */
  int win_span = form->left_span > 1 ? form->left_span - 1 : 1;
  int half_span = (form->chunk_size + win_span + form->right_span) / 2;
  int lag_count = half_span + form->right_span + 1;
  float query_gain = (float)((1.0 / sqrt((double)head_size)) / log(2.0));
  float key_gain = (float)(log(1.0 + 2.718281828459045) / log(2.0));
  int lane_index, head_index, value_index, lag_index;

  plane_lift_many(model, &wing->query_sheet, room->scrap_data, state_size, lane_count,
                  room->query_data, head_wide, room->quant_data, room->quant_stride);
  plane_lift_many(model, &wing->key_sheet, room->scrap_data, state_size, lane_count, room->key_data,
                  head_wide, room->quant_data, room->quant_stride);
  plane_lift_many(model, &wing->value_sheet, room->scrap_data, state_size, lane_count,
                  room->value_data, head_wide, room->quant_data, room->quant_stride);

  /* The query is scaled per head channel through a softplus, and the key by a
   * constant; both are the reference's, and both are folded in once here
   * rather than into every score. */
  for (lane_index = 0; lane_index < lane_count; ++lane_index) {
    float *query_lane = room->query_data + (size_t)lane_index * (size_t)head_wide;
    float *key_lane = room->key_data + (size_t)lane_index * (size_t)head_wide;
    for (head_index = 0; head_index < form->head_count; ++head_index)
      for (value_index = 0; value_index < head_size; ++value_index) {
        float gain_value = wing->dim_gain ? kern_soft_plus(wing->dim_gain[value_index]) : 1.0f;
        query_lane[head_index * head_size + value_index] *= query_gain * gain_value;
        key_lane[head_index * head_size + value_index] *= key_gain;
      }
  }

  /* The sinusoid of each reachable lag, then its projection. */
  for (lag_index = 0; lag_index < lag_count; ++lag_index) {
    float *wave_data = room->wave_data + (size_t)lag_index * (size_t)state_size;
    int pair_count = state_size / 2;
    double step_span = log(10000.0) / (double)(pair_count > 1 ? pair_count - 1 : 1);
    int place_value = half_span - lag_index; /* the lag this row stands for */
    for (value_index = 0; value_index < pair_count; ++value_index) {
      double turn_value = exp((double)value_index * -step_span) * (double)place_value;
      wave_data[value_index] = (float)sin(turn_value);
      wave_data[pair_count + value_index] = (float)cos(turn_value);
    }
  }
  plane_lift_many(model, &wing->place_sheet, room->wave_data, state_size, lag_count,
                  room->place_data, head_wide, room->quant_data, room->quant_stride);

  for (lane_index = 0; lane_index < lane_count; ++lane_index) {
    int from_lane = lane_index - win_span + 1;
    int upto_lane = lane_index + form->right_span + 1;
    if (from_lane < 0) from_lane = 0;
    if (upto_lane > lane_count) upto_lane = lane_count;
    for (head_index = 0; head_index < form->head_count; ++head_index) {
      const float *query_head =
          room->query_data + (size_t)lane_index * (size_t)head_wide + head_index * head_size;
      float *blend_head =
          room->blend_data + (size_t)lane_index * (size_t)head_wide + head_index * head_size;
      int span_index;
      for (span_index = from_lane; span_index < upto_lane; ++span_index) {
        const float *key_head =
            room->key_data + (size_t)span_index * (size_t)head_wide + head_index * head_size;
        /* Row `past_span - lag` is the one built for this lag above. */
        const float *place_head = room->place_data +
                                  (size_t)(half_span - (lane_index - span_index)) *
                                      (size_t)head_wide +
                                  head_index * head_size;
        /* The conformer's window is thirteen keys wide, so this loop is a
         * hundredth of what a picture's attention costs and not worth
         * reassociating: a vector reduction here measured at nothing on the
         * shipped export and moved the tower's numbers by a last bit. */
        float total_value = 0.0f;
        for (value_index = 0; value_index < head_size; ++value_index)
          total_value += query_head[value_index] * (key_head[value_index] + place_head[value_index]);
        /* Softcapping, which keeps a long window from saturating the softmax. */
        if (form->logit_cap > 0.0f)
          total_value = tanhf(total_value / form->logit_cap) * form->logit_cap;
        room->score_data[span_index - from_lane] = total_value;
      }
      model->desk.soft_max(&model->desk, room->score_data, upto_lane - from_lane);
      kern_blend_rows(room->value_data + (size_t)from_lane * (size_t)head_wide +
                          head_index * head_size,
                      head_wide, room->score_data, upto_lane - from_lane, head_size, blend_head);
    }
  }
  plane_lift_many(model, &wing->exit_sheet, room->blend_data, head_wide, lane_count,
                  room->lift_data, state_size, room->quant_data, room->quant_stride);
}

/* The convolution module: a gated widening, a depthwise pass over time, a norm,
 * and a narrowing, folded back whole. */
static void sound_convolve(app_model *model, tower_gear *gear, sound_wing *wing, sound_room *room,
                           int lane_count) {
  tower_form *form = &gear->form;
  int state_size = form->state_size;
  int side_size = form->deep_side;
  int lane_index, value_index, tap_index;

  for (lane_index = 0; lane_index < lane_count; ++lane_index)
    model->desk.norm_rms(&model->desk, room->state_data + (size_t)lane_index * (size_t)state_size,
                         wing->conv_enter_norm, state_size, form->norm_eps,
                         room->scrap_data + (size_t)lane_index * (size_t)state_size);
  plane_lift_many(model, &wing->conv_start_sheet, room->scrap_data, state_size, lane_count,
                  room->conv_data, 2 * state_size, room->quant_data, room->quant_stride);
  /* A gated linear unit: the first half through the sigmoid of the second. */
  for (lane_index = 0; lane_index < lane_count; ++lane_index) {
    float *lane_data = room->conv_data + (size_t)lane_index * (size_t)(2 * state_size);
    float *out_data = room->scrap_data + (size_t)lane_index * (size_t)state_size;
    for (value_index = 0; value_index < state_size; ++value_index)
      out_data[value_index] =
          lane_data[value_index] / (1.0f + expf(-lane_data[state_size + value_index]));
  }
  /* Depthwise over time, left padded so a frame never reads the future.  The
   * kernel is held tap-major, so a tap is one contiguous multiply-add over
   * every channel rather than a strided walk of the kernel per channel; a
   * channel still takes its taps in the order it took them. */
  for (lane_index = lane_count - 1; lane_index >= 0; --lane_index) {
    float *out_data = room->lift_data + (size_t)lane_index * (size_t)state_size;
    for (value_index = 0; value_index < state_size; ++value_index) out_data[value_index] = 0.0f;
    for (tap_index = 0; tap_index < side_size; ++tap_index) {
      int read_index = lane_index + tap_index - (side_size - 1);
      if (read_index < 0) continue;
      kern_fma_row(wing->conv_deep + (size_t)tap_index * (size_t)state_size,
                   room->scrap_data + (size_t)read_index * (size_t)state_size, state_size,
                   out_data);
    }
  }
  for (lane_index = 0; lane_index < lane_count; ++lane_index) {
    float *lift_data = room->lift_data + (size_t)lane_index * (size_t)state_size;
    model->desk.norm_rms(&model->desk, lift_data, wing->conv_norm, state_size, form->norm_eps,
                         lift_data);
    for (value_index = 0; value_index < state_size; ++value_index)
      lift_data[value_index] = kern_silu(lift_data[value_index]);
  }
  plane_lift_many(model, &wing->conv_end_sheet, room->lift_data, state_size, lane_count,
                  room->scrap_data, state_size, room->quant_data, room->quant_stride);
  for (lane_index = 0; lane_index < lane_count * state_size; ++lane_index)
    room->state_data[lane_index] += room->scrap_data[lane_index];
}

static void sound_layer(app_model *model, tower_gear *gear, int layer_index, sound_room *room,
                        int lane_count) {
  tower_form *form = &gear->form;
  sound_wing *wing = &gear->sound_list[layer_index];
  int state_size = form->state_size;
  int lane_index, value_index;

  sound_feed_run(model, gear, wing, room, 0, lane_count);
  for (lane_index = 0; lane_index < lane_count; ++lane_index)
    TRACE_TOWER("audio", "ff1", layer_index, lane_index,
                room->state_data + (size_t)lane_index * (size_t)state_size, state_size);
  /* The residual the attention folds back into is what the first feed-forward
   * produced, not what the layer was handed. */
  memcpy(room->keep_data, room->state_data,
         sizeof(float) * (size_t)lane_count * (size_t)state_size);
  for (lane_index = 0; lane_index < lane_count; ++lane_index)
    model->desk.norm_rms(&model->desk, room->state_data + (size_t)lane_index * (size_t)state_size,
                         wing->enter_norm, state_size, form->norm_eps,
                         room->scrap_data + (size_t)lane_index * (size_t)state_size);
  sound_attend(model, gear, wing, room, lane_count);
  for (lane_index = 0; lane_index < lane_count; ++lane_index)
    TRACE_TOWER("audio", "attn", layer_index, lane_index,
                room->lift_data + (size_t)lane_index * (size_t)state_size, state_size);
  for (lane_index = 0; lane_index < lane_count; ++lane_index) {
    float *lift_data = room->lift_data + (size_t)lane_index * (size_t)state_size;
    float *state_data = room->state_data + (size_t)lane_index * (size_t)state_size;
    const float *keep_lane = room->keep_data + (size_t)lane_index * (size_t)state_size;
    model->desk.norm_rms(&model->desk, lift_data, wing->leave_norm, state_size, form->norm_eps,
                         lift_data);
    for (value_index = 0; value_index < state_size; ++value_index)
      state_data[value_index] = keep_lane[value_index] + lift_data[value_index];
  }

  sound_convolve(model, gear, wing, room, lane_count);
  for (lane_index = 0; lane_index < lane_count; ++lane_index)
    TRACE_TOWER("audio", "conv", layer_index, lane_index,
                room->state_data + (size_t)lane_index * (size_t)state_size, state_size);
  sound_feed_run(model, gear, wing, room, 1, lane_count);
  for (lane_index = 0; lane_index < lane_count; ++lane_index)
    model->desk.norm_rms(&model->desk, room->state_data + (size_t)lane_index * (size_t)state_size,
                         wing->close_norm, state_size, form->norm_eps,
                         room->state_data + (size_t)lane_index * (size_t)state_size);
  for (lane_index = 0; lane_index < lane_count; ++lane_index)
    TRACE_TOWER("audio", "out", layer_index, lane_index,
                room->state_data + (size_t)lane_index * (size_t)state_size, state_size);
}

/* One stage of the subsampler: a two dimensional convolution over (frame,
 * filter), a normalization across the channels it produced, and a rectifier.
 * The map is held as [frame][channel][filter]. */
static app_code sound_stage(app_model *model, tower_gear *gear, int stage_index,
                            const float *from_data, int from_time, int from_band, int from_lane,
                            float **into_out, int *time_out, int *band_out) {
  tower_form *form = &gear->form;
  const plane *sheet = &gear->conv_sheet[stage_index];
  const float *norm_gain = gear->conv_norm[stage_index];
  int side_size = form->conv_side;
  int step_size = form->conv_step;
  int pad_size = form->conv_pad;
  int into_time = (from_time + 2 * pad_size - side_size) / step_size + 1;
  int into_band = (from_band + 2 * pad_size - side_size) / step_size + 1;
  int into_lane = sheet->row_count;
  float *into_data = NULL, *row_room = NULL, *out_room = NULL, *quant_room = NULL, *cell_room = NULL;
  int time_index, band_index, lane_index, tap_time, tap_band, deep_index;

  *into_out = NULL;
  if (into_time < 1 || into_band < 1) return APP_FAIL_SUPPORT;
  if (sheet->col_count != from_lane * side_size * side_size) return APP_FAIL_FORMAT;
  into_data =
      (float *)mem_clear(sizeof(float) * (size_t)into_time * (size_t)into_lane * (size_t)into_band);
  row_room = (float *)mem_clear(sizeof(float) * (size_t)into_band * (size_t)sheet->col_count);
  out_room = (float *)mem_clear(sizeof(float) * (size_t)into_band * (size_t)into_lane);
  cell_room = (float *)mem_clear(sizeof(float) * (size_t)into_lane);
  quant_room = (float *)mem_clear(sizeof(float) * (size_t)KERN_LANE_LIMIT * (size_t)sheet->col_count);
  if (!into_data || !row_room || !out_room || !cell_room || !quant_room) {
    mem_free(into_data);
    mem_free(row_room);
    mem_free(out_room);
    mem_free(cell_room);
    mem_free(quant_room);
    return APP_FAIL_MEMORY;
  }

  for (time_index = 0; time_index < into_time; ++time_index) {
    for (band_index = 0; band_index < into_band; ++band_index) {
      float *lane_data = row_room + (size_t)band_index * (size_t)sheet->col_count;
      for (deep_index = 0; deep_index < from_lane; ++deep_index)
        for (tap_time = 0; tap_time < side_size; ++tap_time)
          for (tap_band = 0; tap_band < side_size; ++tap_band) {
            int read_time = time_index * step_size + tap_time - pad_size;
            int read_band = band_index * step_size + tap_band - pad_size;
            float value_now = 0.0f;
            if (read_time >= 0 && read_time < from_time && read_band >= 0 && read_band < from_band)
              value_now = from_data[((size_t)read_time * (size_t)from_lane + (size_t)deep_index) *
                                        (size_t)from_band +
                                    (size_t)read_band];
            lane_data[((size_t)deep_index * (size_t)side_size + (size_t)tap_time) *
                          (size_t)side_size +
                      (size_t)tap_band] = value_now;
          }
    }
    plane_lift_many(model, sheet, row_room, sheet->col_count, into_band, out_room, into_lane,
                    quant_room, sheet->col_count);
    /* The norm runs across the channels of one cell, so the row has to be
     * gathered before it is normalized and scattered back. */
    for (band_index = 0; band_index < into_band; ++band_index) {
      const float *cell_data = out_room + (size_t)band_index * (size_t)into_lane;
      kern_norm_layer(cell_data, norm_gain, into_lane, form->norm_eps, cell_room);
      for (lane_index = 0; lane_index < into_lane; ++lane_index)
        into_data[((size_t)time_index * (size_t)into_lane + (size_t)lane_index) * (size_t)into_band +
                  (size_t)band_index] = cell_room[lane_index] > 0.0f ? cell_room[lane_index] : 0.0f;
    }
  }
  mem_free(row_room);
  mem_free(out_room);
  mem_free(cell_room);
  mem_free(quant_room);
  *into_out = into_data;
  *time_out = into_time;
  *band_out = into_band;
  return APP_OKAY;
}

static app_code audio_run(app_model *model, const flat_grid *mel_grid, float **row_out,
                          int *row_count_out) {
  tower_gear *gear = &model->tower_list[TOWER_AUDIO];
  tower_form *form = &gear->form;
  sound_room room;
  float *stage_data = NULL;
  float *flat_data = NULL;
  float *wide_data = NULL;
  float *out_data = NULL;
  int stage_time = mel_grid->high_count, stage_band = mel_grid->wide_count, stage_lane = 1;
  int stage_index, lane_count = 0, layer_index, time_index;
  app_code code = APP_OKAY;

  memset(&room, 0, sizeof(room));
  *row_out = NULL;
  *row_count_out = 0;
  if (!form->live_flag) return APP_FAIL_SUPPORT;

  stage_data = (float *)mem_clear(sizeof(float) * (size_t)stage_time * (size_t)stage_band);
  if (!stage_data) return APP_FAIL_MEMORY;
  for (time_index = 0; time_index < stage_time; ++time_index) {
    memcpy(stage_data + (size_t)time_index * (size_t)stage_band, grid_at(mel_grid, time_index, 0),
           sizeof(float) * (size_t)stage_band);
    TRACE_TOWER("audio", "mel", -1, time_index,
                stage_data + (size_t)time_index * (size_t)stage_band, stage_band);
  }

  for (stage_index = 0; stage_index < form->conv_count; ++stage_index) {
    float *next_data = NULL;
    int next_time = 0, next_band = 0;
    code = sound_stage(model, gear, stage_index, stage_data, stage_time, stage_band, stage_lane,
                       &next_data, &next_time, &next_band);
    if (code != APP_OKAY) goto audio_done;
    mem_free(stage_data);
    stage_data = next_data;
    stage_lane = gear->conv_sheet[stage_index].row_count;
    stage_time = next_time;
    stage_band = next_band;
  }
  lane_count = stage_time;
  if (lane_count < 1) { code = APP_FAIL_FORMAT; goto audio_done; }
  if (gear->join_sheet.col_count != stage_lane * stage_band) {
    code = APP_FAIL_FORMAT;
    goto audio_done;
  }

  code = sound_room_open(&room, form, lane_count);
  if (code != APP_OKAY) goto audio_done;
  flat_data =
      (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)gear->join_sheet.col_count);
  wide_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)form->lift_size);
  out_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)model->form.state_size);
  if (!flat_data || !wide_data || !out_data) { code = APP_FAIL_MEMORY; goto audio_done; }

  /* The flatten runs filter before channel, which is the order the reference's
   * permute leaves behind and the order its projection was trained on. */
  for (time_index = 0; time_index < lane_count; ++time_index) {
    float *lane_data = flat_data + (size_t)time_index * (size_t)gear->join_sheet.col_count;
    int band_index, deep_index;
    for (band_index = 0; band_index < stage_band; ++band_index)
      for (deep_index = 0; deep_index < stage_lane; ++deep_index)
        lane_data[(size_t)band_index * (size_t)stage_lane + (size_t)deep_index] =
            stage_data[((size_t)time_index * (size_t)stage_lane + (size_t)deep_index) *
                           (size_t)stage_band +
                       (size_t)band_index];
  }
  plane_lift_many(model, &gear->join_sheet, flat_data, gear->join_sheet.col_count, lane_count,
                  room.state_data, form->state_size, room.quant_data, room.quant_stride);
  for (time_index = 0; time_index < lane_count; ++time_index)
    TRACE_TOWER("audio", "embed", -1, time_index,
                room.state_data + (size_t)time_index * (size_t)form->state_size, form->state_size);

  for (layer_index = 0; layer_index < form->layer_count; ++layer_index)
    sound_layer(model, gear, layer_index, &room, lane_count);

  /* The tower widens into the projector's input before the projector runs. */
  plane_lift_many(model, &gear->out_sheet, room.state_data, form->state_size, lane_count, wide_data,
                  form->lift_size, room.quant_data, room.quant_stride);
  tower_bias_add(wide_data, form->lift_size, lane_count, gear->out_bias, form->lift_size);
  for (time_index = 0; time_index < lane_count; ++time_index) {
    float *lane_data = wide_data + (size_t)time_index * (size_t)form->lift_size;
    float *out_lane = out_data + (size_t)time_index * (size_t)model->form.state_size;
    model->desk.norm_rms(&model->desk, lane_data, gear->lift_norm, form->lift_size, form->norm_eps,
                         lane_data);
    plane_lift_many(model, &gear->lift_sheet, lane_data, 0, 1, out_lane, 0, room.quant_data,
                    room.quant_stride);
    TRACE_TOWER("audio", "lift", -1, time_index, out_lane, model->form.state_size);
  }
  *row_out = out_data;
  *row_count_out = lane_count;
  out_data = NULL;

audio_done:
  mem_free(stage_data);
  mem_free(flat_data);
  mem_free(wide_data);
  mem_free(out_data);
  sound_room_free(&room);
  return code;
}

/* ======================================================================== */
/* 9. token layer                                                           */
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
  uint8_t *mark_list; /* one when the entry is a control token */
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
  mem_free(book->mark_list);
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
  book->mark_list = (uint8_t *)mem_clear((size_t)book->size_count);
  if (!book->text_list || !book->mark_list) { json_free(tree); token_free(book); return APP_FAIL_MEMORY; }
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
    if (json_field_flag(tree, child_index, "special", 0)) book->mark_list[entry_id] = 1;
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
  /* Gemma 4 renamed the turn markers. The older pair is kept behind it so that
   * a Gemma 2 or 3 checkpoint still frames, since nothing else distinguishes
   * the two vocabularies. */
  book->turn_open_id = token_find(book, "<|turn>");
  book->turn_shut_id = token_find(book, "<turn|>");
  if (book->turn_open_id < 0 || book->turn_shut_id < 0) {
    book->turn_open_id = token_find(book, "<start_of_turn>");
    book->turn_shut_id = token_find(book, "<end_of_turn>");
  }
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
  /* The prefix belongs to the start of a chunk of text, not to the start of a
   * call.  The reference splits its input on the special tokens and normalizes
   * each chunk on its own, so the marker follows a special id and nothing else;
   * a caller stitching one turn together out of several pieces says which of
   * them begins a chunk. */
  int want_prefix = lead_marker && book->prefix_mode != 0;

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
  /* A control token decodes to nothing, so the slot is terminated up front:
   * every caller reads the text, and only some of them read the count. */
  if (text_limit > 0) text_out[0] = '\0';
  if (id_value < 0 || id_value >= book->size_count || !book->text_list[id_value]) return 0;
  if (book->mark_list && book->mark_list[id_value]) return 0;
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
/* 10. session layer                                                        */
/* ======================================================================== */


struct app_session {
  app_model *model;
  int        fill_count;
  app_tally  tally;
  uint64_t   draw_state;

  /* One row a slot, held either as floats or as the eight bit floats the
   * export calibrates for.  `key_grid` is the layer's dequantizing table where
   * it holds bytes and null where it holds floats, so it doubles as the flag
   * that says which of the two a store is. */
  void  **key_store;
  void  **value_store;
  float **key_grid;
  float **value_grid;

  /* The largest magnitude this session has put in each layer's cache, tracked
   * so the static ranges the export calibrates can be judged against what the
   * cache is actually asked to hold.  One entry a layer, keys and values apart. */
  float  *key_peak;
  float  *value_peak;

  /* Every room below the stride block holds KERN_LANE_LIMIT lanes, one per
   * token of a batched pass, laid out lane by lane with the stride named. */
  int state_stride;
  int lift_stride;
  int quant_stride;
  int head_stride;
  int score_stride;
  int gate_stride;
  int rise_stride;
  int ple_stride;

  float *state_room;
  float *scrap_room;
  float *lift_room;
  float *quant_room;
  float *query_room;
  float *key_room;
  float *value_room;
  float *blend_room;
  float *score_room;   /* one row of scores a head of the widest group */
  float *cache_room;   /* one block of cached rows, decoded */
  float *gate_room;
  float *rise_room;
  float *cos_room;
  float *sin_room;
  float *ple_seed;
  float *ple_room;
  float *logit_room;
  float *route_room;    /* router probabilities, one per expert */
  float *expert_room;   /* stacked gate and rise activations of one expert */
  float *expert_drop;   /* one expert's contribution, state width */
  float *moe_room;      /* the summed mixture branch, state width */
  float *top_weight;    /* the selected experts' mixing weights */
  int   *top_slot;      /* the selected experts */
  void  *pick_room;
  int32_t *echo_room;
  int      echo_count;
  int      echo_limit;
};

/* The session's view of `plane_lift_many`: the same rule, applied through the
 * staging room the session allocated once at open. */
static void session_lift_many(app_session *session, const plane *sheet, const float *act_data,
                              int act_stride, int lane_count, float *out_data, int out_stride) {
  plane_lift_many(session->model, sheet, act_data, act_stride, lane_count, out_data, out_stride,
                  session->quant_room, session->quant_stride);
}

static void session_lift(app_session *session, const plane *sheet, const float *act_data,
                         float *out_data) {
  session_lift_many(session, sheet, act_data, 0, 1, out_data, 0);
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
  if (session->key_grid) {
    for (layer_index = 0; layer_index < session->model->form.layer_count; ++layer_index) {
      mem_free(session->key_grid[layer_index]);
      mem_free(session->value_grid[layer_index]);
    }
    mem_free(session->key_grid);
    mem_free(session->value_grid);
  }
  mem_free(session->key_peak);
  mem_free(session->value_peak);
  mem_free(session->state_room);
  mem_free(session->scrap_room);
  mem_free(session->lift_room);
  mem_free(session->quant_room);
  mem_free(session->query_room);
  mem_free(session->key_room);
  mem_free(session->value_room);
  mem_free(session->blend_room);
  mem_free(session->score_room);
  mem_free(session->cache_room);
  mem_free(session->gate_room);
  mem_free(session->rise_room);
  mem_free(session->cos_room);
  mem_free(session->sin_room);
  mem_free(session->ple_seed);
  mem_free(session->ple_room);
  mem_free(session->logit_room);
  mem_free(session->route_room);
  mem_free(session->expert_room);
  mem_free(session->expert_drop);
  mem_free(session->moe_room);
  mem_free(session->top_weight);
  mem_free(session->top_slot);
  mem_free(session->pick_room);
  mem_free(session->echo_room);
}

/* Round to the nearest eight bit float the cache grid carries.
 *
 * The value arrives already divided by its layer's scale, so the grid is the
 * plain e4m3 one.  `rintf` rounds half to even under the default mode, which is
 * what the hardware form of this conversion does. */
static float cache_pack8(float value) {
  float sign = value < 0.0f ? -1.0f : 1.0f;
  float size = value < 0.0f ? -value : value;
  float step;
  int power;
  if (!(size < CACHE_FP8_TOP)) return sign * CACHE_FP8_TOP; /* saturates, and catches NaN */
  if (size < CACHE_FP8_NORM) return sign * rintf(size / CACHE_FP8_STEP) * CACHE_FP8_STEP;
  frexpf(size, &power);        /* size is in [2^(power-1), 2^power) */
  step = ldexpf(1.0f, power - 4); /* four bits of significand, one of them implied */
  size = rintf(size / step) * step;
  return sign * (size > CACHE_FP8_TOP ? CACHE_FP8_TOP : size);
}

/* The byte the grid holds a value as, and the value back out of the byte.
 *
 * The layout is the format's own: one sign bit, four exponent bits biased by
 * seven, three of significand.  A zero exponent field is the subnormal range,
 * where the significand counts steps of 2^-9 outright; every other exponent
 * carries the implied leading one, so the value is `(8 + significand) *
 * 2^(exponent - 10)`.  `cache_pack8` has already rounded and saturated by the
 * time the encoder runs, so all it has to do is name the byte the value landed
 * on -- and 0x7F, which the format spends on a NaN, is never reached, because
 * 448 saturates at 0x7E.
 *
 * `cache_real8(cache_code8(x))` is `cache_pack8(x)` for every float.  That is
 * what lets a byte cache be put under a float one without moving a result. */
static uint8_t cache_code8(float value) {
  float size = cache_pack8(value);
  uint8_t sign = 0;
  int power, step;
  if (size < 0.0f) { sign = 0x80u; size = -size; }
  if (size < CACHE_FP8_NORM)
    return (uint8_t)(sign | (uint8_t)(int)(size / CACHE_FP8_STEP + 0.5f));
  frexpf(size, &power);           /* size is in [2^(power-1), 2^power) */
  step = (int)(ldexpf(size, 4 - power) + 0.5f) - 8;
  return (uint8_t)(sign | (uint8_t)((power + 6) << 3) | (uint8_t)step);
}

static float cache_real8(uint8_t code) {
  int power = (code >> 3) & 15;
  int step = code & 7;
  float size = power == 0 ? (float)step * CACHE_FP8_STEP
                          : ldexpf((float)(8 + step), power - 10);
  return (code & 0x80u) ? -size : size;
}

/* A cache row is written once and read on every step that can still see it, so
 * the decoding is a table read rather than a shift and a scale.  The table
 * folds the layer's scale in, which makes an entry the exact float the round
 * trip through the grid stored before anything was held as a byte: a kilobyte a
 * side a layer buys value-for-value agreement with the float cache rather than
 * mere closeness. */
static void cache_grid_fill(float *grid, float scale) {
  int code;
  for (code = 0; code < 256; ++code) grid[code] = cache_real8((uint8_t)code) * scale;
}

#if defined(APP_SIMD_SSE2) && !defined(APP_SIMD_AVX2)
/* Four table reads gathered into one vector, in the lanes a float load would
 * have put them in, so the sum that follows is the same sum in the same order.
 * Neither SSE2 nor NEON has a gather, and four scalar reads of a table that
 * stays in the first level cache are cheaper than the branch to avoid them. */
static __m128 cache_four_wide(const float *grid, const uint8_t *code_list) {
  return _mm_set_ps(grid[code_list[3]], grid[code_list[2]], grid[code_list[1]],
                    grid[code_list[0]]);
}
#define CACHE_SSE_FOUR(grid, code_list) cache_four_wide((grid), (code_list))
#elif defined(APP_SIMD_NEON)
static float32x4_t cache_four_wide(const float *grid, const uint8_t *code_list) {
  float part_list[4];
  part_list[0] = grid[code_list[0]];
  part_list[1] = grid[code_list[1]];
  part_list[2] = grid[code_list[2]];
  part_list[3] = grid[code_list[3]];
  return vld1q_f32(part_list);
}
#define CACHE_NEON_FOUR(grid, code_list) cache_four_wide((grid), (code_list))
#endif

/* What one stored value of a layer's cache costs. */
static size_t cache_slot_bytes(const float *grid) { return grid ? 1u : sizeof(float); }

/* One cached key row of bytes against one query head.
 *
 * The blocking, the pair of accumulators and the horizontal sum are the packed
 * dot kernel's, term for term, and the table hands back exactly the float the
 * float cache held, so a byte cache and a float one reach the same score bit
 * for bit rather than to a tolerance.  A layer holding floats does not come
 * here at all: it keeps the kernel it always used, so `--cache 8` off is the
 * arithmetic it was before any of this. */
static float cache_dot(const uint8_t *code_list, const float *grid, const float *act_data,
                       int span_count) {
  float total = 0.0f;
  int slot = 0;
#if defined(APP_SIMD_AVX2)
  {
    __m256 part_a = _mm256_setzero_ps(), part_b = _mm256_setzero_ps();
    __m256 wide_total;
    for (; slot + 16 <= span_count; slot += 16) {
      __m256i code_a = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(code_list + slot)));
      __m256i code_b =
          _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(code_list + slot + 8)));
      part_a = _mm256_fmadd_ps(_mm256_i32gather_ps(grid, code_a, 4),
                               _mm256_loadu_ps(act_data + slot), part_a);
      part_b = _mm256_fmadd_ps(_mm256_i32gather_ps(grid, code_b, 4),
                               _mm256_loadu_ps(act_data + slot + 8), part_b);
    }
    for (; slot + 8 <= span_count; slot += 8) {
      __m256i code_a = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(code_list + slot)));
      part_a = _mm256_fmadd_ps(_mm256_i32gather_ps(grid, code_a, 4),
                               _mm256_loadu_ps(act_data + slot), part_a);
    }
    wide_total = _mm256_add_ps(part_a, part_b);
    {
      __m128 half_total = _mm_add_ps(_mm256_castps256_ps128(wide_total),
                                     _mm256_extractf128_ps(wide_total, 1));
      half_total = _mm_hadd_ps(half_total, half_total);
      half_total = _mm_hadd_ps(half_total, half_total);
      total = _mm_cvtss_f32(half_total);
    }
  }
#elif defined(APP_SIMD_SSE2)
  {
    __m128 part_a = _mm_setzero_ps(), part_b = _mm_setzero_ps();
    __m128 wide_total;
    for (; slot + 8 <= span_count; slot += 8) {
      part_a = _mm_add_ps(part_a, _mm_mul_ps(CACHE_SSE_FOUR(grid, code_list + slot),
                                             _mm_loadu_ps(act_data + slot)));
      part_b = _mm_add_ps(part_b, _mm_mul_ps(CACHE_SSE_FOUR(grid, code_list + slot + 4),
                                             _mm_loadu_ps(act_data + slot + 4)));
    }
    for (; slot + 4 <= span_count; slot += 4)
      part_a = _mm_add_ps(part_a, _mm_mul_ps(CACHE_SSE_FOUR(grid, code_list + slot),
                                             _mm_loadu_ps(act_data + slot)));
    wide_total = _mm_add_ps(part_a, part_b);
    {
      __m128 half_total = _mm_add_ps(wide_total, _mm_movehl_ps(wide_total, wide_total));
      half_total = _mm_add_ss(half_total, _mm_shuffle_ps(half_total, half_total, 0x55));
      total = _mm_cvtss_f32(half_total);
    }
  }
#elif defined(APP_SIMD_NEON)
  {
    float32x4_t part_a = vdupq_n_f32(0.0f), part_b = vdupq_n_f32(0.0f);
    float32x4_t wide_total;
    for (; slot + 8 <= span_count; slot += 8) {
      part_a = vmlaq_f32(part_a, CACHE_NEON_FOUR(grid, code_list + slot),
                         vld1q_f32(act_data + slot));
      part_b = vmlaq_f32(part_b, CACHE_NEON_FOUR(grid, code_list + slot + 4),
                         vld1q_f32(act_data + slot + 4));
    }
    for (; slot + 4 <= span_count; slot += 4)
      part_a = vmlaq_f32(part_a, CACHE_NEON_FOUR(grid, code_list + slot),
                         vld1q_f32(act_data + slot));
    wide_total = vaddq_f32(part_a, part_b);
    total = vgetq_lane_f32(wide_total, 0) + vgetq_lane_f32(wide_total, 1) +
            vgetq_lane_f32(wide_total, 2) + vgetq_lane_f32(wide_total, 3);
  }
#endif
  for (; slot < span_count; ++slot) total += grid[code_list[slot]] * act_data[slot];
  return total;
}

/* One cached value row of bytes, weighted, into the blend the head is
 * accumulating.  Scalar, which is what the float blend loop is. */
static void cache_add(const uint8_t *code_list, const float *grid, float weight_value,
                      float *blend_data, int span_count) {
  int slot;
  for (slot = 0; slot < span_count; ++slot)
    blend_data[slot] += weight_value * grid[code_list[slot]];
}

/* A run of cached rows for one key-value head, laid out as floats.
 *
 * A side held as bytes is decoded through the layer's table, which hands back
 * exactly the float the float cache held; a side held as floats is copied,
 * which only arises where an export calibrates one of the two and not the
 * other.  Either way what lands here is what a head would have read for
 * itself, so nothing downstream can tell which side it came from.
 *
 * The run is small on purpose: `CACHE_BLOCK_BYTES` of floats, so every head of
 * the group reads it out of the first level cache rather than out of the
 * store. */
#define CACHE_BLOCK_BYTES 8192

static void cache_block(const void *store_data, const float *grid, int cache_span, int place_first,
                        int row_count, int kv_index, int head_size, int kv_width, float *out_data) {
  int row_index;
  for (row_index = 0; row_index < row_count; ++row_index) {
    size_t slot_first = (size_t)((place_first + row_index) % cache_span) * (size_t)kv_width +
                        (size_t)kv_index * (size_t)head_size;
    float *row_out = out_data + (size_t)row_index * (size_t)head_size;
    if (grid) {
      const uint8_t *code_list = (const uint8_t *)store_data + slot_first;
      int slot;
      for (slot = 0; slot < head_size; ++slot) row_out[slot] = grid[code_list[slot]];
    } else {
      memcpy(row_out, (const float *)store_data + slot_first, sizeof(float) * (size_t)head_size);
    }
  }
}

/* How many rows one block holds, given how wide a row is. */
static int cache_block_rows(int head_size) {
  int row_count = (int)(CACHE_BLOCK_BYTES / (sizeof(float) * (size_t)head_size));
  return row_count < 1 ? 1 : row_count;
}

/* Lay one row into the key or value cache.
 *
 * The export calibrates a static range a layer for each of them — a single
 * scalar, the same for every channel and every position — and ships seventy of
 * them the reference never reads.  They describe the eight bit float grid
 * above: a layer whose store is bytes holds `value / scale` rounded to that
 * grid, and reads it back through a table that multiplies the scale in again.
 * With `cache_bits` at zero the row is stored as it arrives and the scales cost
 * nothing.
 *
 * A layer's store is bytes exactly where its dequantizing table exists, which
 * `session_open` decides once: `cache_bits` at eight and a scale the export
 * actually ships.  Nothing is rounded on a layer that has no scale, because
 * there is no grid to round it onto.
 *
 * The peak is tracked either way.  What the range has to cover is the question
 * the scalars answer, and it can only be asked of a real prompt. */
static void cache_lay(app_session *session, int layer_index, size_t slot_first, const float *row,
                      int width, float scale, int value_side) {
  float *peak = (value_side ? session->value_peak : session->key_peak) + layer_index;
  const float *grid = (value_side ? session->value_grid : session->key_grid)[layer_index];
  void *base_data = (value_side ? session->value_store : session->key_store)[layer_index];
  int index;
  for (index = 0; index < width; ++index) {
    float size = row[index] < 0.0f ? -row[index] : row[index];
    if (size > *peak) *peak = size;
  }
  if (!grid) {
    memcpy((float *)base_data + slot_first, row, sizeof(float) * (size_t)width);
    return;
  }
  {
    uint8_t *code_list = (uint8_t *)base_data + slot_first;
    /* A divide rather than a multiply by the reciprocal: this runs once a row
     * where the read runs over the whole span, and the divide is what the float
     * round trip through the grid did, so the two storages agree to the bit. */
    for (index = 0; index < width; ++index) code_list[index] = cache_code8(row[index] / scale);
  }
}

/* Whether a layer reads its cache blocked by row rather than by head.
 *
 * The one thing that has to hold is that more than one head reads the same
 * cached row; where a head has its own row there is nothing to divide and the
 * block would be a copy for its own sake.  It was written for the byte cache,
 * where a read is a table lookup and the block spends one per row instead of
 * one per row per head, but a cache held as floats gains by it too — the block
 * is a run the heads read out of the first level cache where the store is a
 * sweep of hundreds of kilobytes per head — so it is not asked which storage
 * the layer is on. */
static int session_attend_wide(const layer_wing *wing) { return wing->group_share >= 2; }

/* One lane's heads, blocked by cached row rather than by head.
 *
 * This export ships one key-value head against eight attention heads, so
 * `group_share` is eight: all eight score against the same cached key row and
 * blend the same cached value row.  Taken head by head, as the loop below this
 * one takes them, a byte cache decodes every one of those bytes eight times —
 * eight table reads where one would do.  Taken this way round a run of rows is
 * decoded once into a block that stays in the first level cache and every head
 * of the group reads it there, which divides the decode work by the share
 * without giving back any of the storage the byte cache collects, and which
 * serves every backend rather than only the one whose table read is a gather.
 *
 * The block hands each head exactly the floats it would have looked up for
 * itself, and the dot and the blend here are the kernels the float cache always
 * used, so a score and a blend come out bit for bit what they were.  What it
 * costs is a score row per head of the group rather than one, which is what
 * `score_stride` sizes.  A layer whose share is one has nothing to divide and
 * does not come here. */
static void session_attend_group(app_session *session, int layer_index, const float *query_lane,
                                 float *blend_lane, int place_start, int span_count) {
  app_model *model = session->model;
  layer_wing *wing = &model->wing_list[layer_index];
  int owner_slot = wing->share_flag ? wing->source_slot : layer_index;
  layer_wing *owner = &model->wing_list[owner_slot];
  const float *key_grid = session->key_grid[owner_slot];
  const float *value_grid = session->value_grid[owner_slot];
  int head_size = wing->head_size;
  int kv_width = wing->kv_count * head_size;
  int block_rows = cache_block_rows(head_size);
  int kv_index, head_local, row_index, span_from, value_index;

  for (kv_index = 0; kv_index < wing->kv_count; ++kv_index) {
    int head_first = kv_index * wing->group_share;
    for (span_from = 0; span_from < span_count; span_from += block_rows) {
      int block_count = span_count - span_from < block_rows ? span_count - span_from : block_rows;
      cache_block(session->key_store[owner_slot], key_grid, owner->cache_span,
                  place_start + span_from, block_count, kv_index, head_size, kv_width,
                  session->cache_room);
      for (head_local = 0; head_local < wing->group_share; ++head_local) {
        const float *query_head = query_lane + (size_t)(head_first + head_local) * head_size;
        float *score_head = session->score_room + (size_t)head_local * session->score_stride;
        for (row_index = 0; row_index < block_count; ++row_index)
          score_head[span_from + row_index] =
              kern_dot_real(session->cache_room + (size_t)row_index * head_size, STORE_F32,
                            query_head, head_size);
      }
    }
    for (head_local = 0; head_local < wing->group_share; ++head_local) {
      float *blend_head = blend_lane + (size_t)(head_first + head_local) * head_size;
      model->desk.soft_max(&model->desk,
                           session->score_room + (size_t)head_local * session->score_stride,
                           span_count);
      for (value_index = 0; value_index < head_size; ++value_index) blend_head[value_index] = 0.0f;
    }
    for (span_from = 0; span_from < span_count; span_from += block_rows) {
      int block_count = span_count - span_from < block_rows ? span_count - span_from : block_rows;
      cache_block(session->value_store[owner_slot], value_grid, owner->cache_span,
                  place_start + span_from, block_count, kv_index, head_size, kv_width,
                  session->cache_room);
      for (head_local = 0; head_local < wing->group_share; ++head_local) {
        float *blend_head = blend_lane + (size_t)(head_first + head_local) * head_size;
        const float *score_head = session->score_room + (size_t)head_local * session->score_stride;
        for (row_index = 0; row_index < block_count; ++row_index) {
          const float *value_row = session->cache_room + (size_t)row_index * head_size;
          float weight_value = score_head[span_from + row_index];
          for (value_index = 0; value_index < head_size; ++value_index)
            blend_head[value_index] += weight_value * value_row[value_index];
        }
      }
    }
  }
}

static void session_attend(app_session *session, int layer_index, int place_from, int lane_count,
                           const float *enter_data, float *exit_data) {
  app_model *model = session->model;
  model_form *form = &model->form;
  layer_wing *wing = &model->wing_list[layer_index];
  layer_wing *owner = wing->share_flag ? &model->wing_list[wing->source_slot] : wing;
  int owner_slot = wing->share_flag ? wing->source_slot : layer_index;
  int head_size = wing->head_size;
  int kv_width = wing->kv_count * head_size;
  int head_stride = session->head_stride;
  int state_stride = session->state_stride;
  const float *key_grid = session->key_grid[owner_slot];
  const float *value_grid = session->value_grid[owner_slot];
  int head_index, lane_index;

  session_lift_many(session, &wing->query_sheet, enter_data, state_stride, lane_count,
                    session->query_room, head_stride);
  if (!wing->share_flag) {
    session_lift_many(session, &wing->key_sheet, enter_data, state_stride, lane_count,
                      session->key_room, head_stride);
    if (wing->value_sheet.form != PLANE_VOID)
      session_lift_many(session, &wing->value_sheet, enter_data, state_stride, lane_count,
                        session->value_room, head_stride);
  }

  /* Each lane stores its keys and then attends before the next lane runs, so a
   * ring buffer narrower than the batch still holds every slot a lane reads. */
  for (lane_index = 0; lane_index < lane_count; ++lane_index) {
    int place_index = place_from + lane_index;
    float *query_lane = session->query_room + (size_t)lane_index * (size_t)head_stride;
    float *blend_lane = session->blend_room + (size_t)lane_index * (size_t)head_stride;
    int place_start = 0;
    rope_wave(&form->rope_list[wing->kind_mark], place_index, session->cos_room, session->sin_room);
    for (head_index = 0; head_index < form->head_count; ++head_index) {
      float *head_data = query_lane + (size_t)head_index * head_size;
      model->desk.norm_rms(&model->desk, head_data, wing->query_norm, head_size, form->norm_eps,
                           head_data);
      model->desk.rope_turn(&model->desk, head_data, head_size, session->cos_room,
                            session->sin_room);
    }
    if (!wing->share_flag) {
      float *key_lane = session->key_room + (size_t)lane_index * (size_t)head_stride;
      float *value_lane = session->value_room + (size_t)lane_index * (size_t)head_stride;
      size_t slot_first = (size_t)(place_index % wing->cache_span) * (size_t)kv_width;
      if (wing->value_sheet.form == PLANE_VOID)
        memcpy(value_lane, key_lane, sizeof(float) * (size_t)kv_width);
      for (head_index = 0; head_index < wing->kv_count; ++head_index) {
        float *key_head = key_lane + (size_t)head_index * head_size;
        float *value_head = value_lane + (size_t)head_index * head_size;
        model->desk.norm_rms(&model->desk, key_head, wing->key_norm, head_size, form->norm_eps,
                             key_head);
        model->desk.rope_turn(&model->desk, key_head, head_size, session->cos_room,
                              session->sin_room);
        model->desk.norm_rms(&model->desk, value_head, NULL, head_size, form->norm_eps, value_head);
      }
      cache_lay(session, layer_index, slot_first, key_lane, kv_width, wing->key_cache_scale, 0);
      cache_lay(session, layer_index, slot_first, value_lane, kv_width, wing->value_cache_scale, 1);
    }

    if (wing->kind_mark == MODEL_KIND_SLIDE) {
      place_start = place_index - form->slide_span + 1;
      if (place_start < 0) place_start = 0;
    }
    /* Blocked by cached row where a group of heads shares one, and head by
     * head where each head has its own.  The two reach the same numbers to the
     * bit; `session_attend_group` says why the first is the cheaper read. */
    if (session_attend_wide(wing)) {
      session_attend_group(session, layer_index, query_lane, blend_lane, place_start,
                           place_index - place_start + 1);
      continue;
    }
    for (head_index = 0; head_index < form->head_count; ++head_index) {
      const float *query_head = query_lane + (size_t)head_index * head_size;
      int kv_index = head_index / wing->group_share;
      float *blend_head = blend_lane + (size_t)head_index * head_size;
      int span_count = place_index - place_start + 1;
      int span_index, value_index;

      if (key_grid)
        for (span_index = 0; span_index < span_count; ++span_index) {
          int slot_index = (place_start + span_index) % owner->cache_span;
          const uint8_t *key_head = (const uint8_t *)session->key_store[owner_slot] +
                                    (size_t)slot_index * (size_t)kv_width +
                                    (size_t)kv_index * head_size;
          session->score_room[span_index] = cache_dot(key_head, key_grid, query_head, head_size);
        }
      else
        for (span_index = 0; span_index < span_count; ++span_index) {
          int slot_index = (place_start + span_index) % owner->cache_span;
          const float *key_head = (const float *)session->key_store[owner_slot] +
                                (size_t)slot_index * (size_t)kv_width + (size_t)kv_index * head_size;
          session->score_room[span_index] =
              kern_dot_real(key_head, STORE_F32, query_head, head_size);
        }
      model->desk.soft_max(&model->desk, session->score_room, span_count);

      for (value_index = 0; value_index < head_size; ++value_index) blend_head[value_index] = 0.0f;
      if (value_grid)
        for (span_index = 0; span_index < span_count; ++span_index) {
          int slot_index = (place_start + span_index) % owner->cache_span;
          const uint8_t *value_head = (const uint8_t *)session->value_store[owner_slot] +
                                      (size_t)slot_index * (size_t)kv_width +
                                      (size_t)kv_index * head_size;
          cache_add(value_head, value_grid, session->score_room[span_index], blend_head, head_size);
        }
      else
        for (span_index = 0; span_index < span_count; ++span_index) {
          int slot_index = (place_start + span_index) % owner->cache_span;
          const float *value_head = (const float *)session->value_store[owner_slot] +
                                    (size_t)slot_index * (size_t)kv_width +
                                    (size_t)kv_index * head_size;
          float weight_value = session->score_room[span_index];
          for (value_index = 0; value_index < head_size; ++value_index)
            blend_head[value_index] += weight_value * value_head[value_index];
        }
    }
  }
  session_lift_many(session, &wing->exit_sheet, session->blend_room, head_stride, lane_count,
                    exit_data, session->lift_stride);
}

/* Router: normalizes, projects to one score per expert, and keeps the top few.
 * Weights are renormalized to sum to one and then scaled per selected expert. */
static void session_route(app_session *session, layer_wing *wing, const float *state_data) {
  app_model *model = session->model;
  model_form *form = &model->form;
  float root_gain = (float)(1.0 / sqrt((double)form->state_size));
  float total_value = 0.0f;
  int value_index, pick_index;

  model->desk.norm_rms(&model->desk, state_data, NULL, form->state_size, form->norm_eps,
                       session->moe_room);
  for (value_index = 0; value_index < form->state_size; ++value_index)
    session->moe_room[value_index] *= wing->route_scale[value_index] * root_gain;
  session_lift(session, &wing->route_sheet, session->moe_room, session->route_room);
  model->desk.soft_max(&model->desk, session->route_room, form->expert_count);

  for (pick_index = 0; pick_index < form->expert_top; ++pick_index) {
    int best_slot = 0;
    float best_value = -1.0f;
    for (value_index = 0; value_index < form->expert_count; ++value_index)
      if (session->route_room[value_index] > best_value) {
        best_value = session->route_room[value_index];
        best_slot = value_index;
      }
    session->route_room[best_slot] = -1.0f; /* softmax output is never negative */
    session->top_slot[pick_index] = best_slot;
    session->top_weight[pick_index] = best_value;
    total_value += best_value;
  }
  if (total_value <= 0.0f) total_value = 1.0f;
  for (pick_index = 0; pick_index < form->expert_top; ++pick_index)
    session->top_weight[pick_index] =
        session->top_weight[pick_index] / total_value * wing->route_gain[session->top_slot[pick_index]];
}

/* Runs the selected experts and blends their outputs by the router weights. */
static void session_expert(app_session *session, layer_wing *wing, const float *act_data,
                           float *out_data) {
  app_model *model = session->model;
  model_form *form = &model->form;
  int inner_size = form->expert_inner;
  int pick_index, value_index;

  memset(out_data, 0, sizeof(float) * (size_t)form->state_size);
  for (pick_index = 0; pick_index < form->expert_top; ++pick_index) {
    int expert_slot = session->top_slot[pick_index];
    float weight_value = session->top_weight[pick_index];
    session_lift(session, &wing->expert_rise_list[expert_slot], act_data, session->expert_room);
    model->desk.gelu_gate(&model->desk, session->expert_room, session->expert_room + inner_size,
                          inner_size);
    session_lift(session, &wing->expert_drop_list[expert_slot], session->expert_room,
                 session->expert_drop);
    for (value_index = 0; value_index < form->state_size; ++value_index)
      out_data[value_index] += weight_value * session->expert_drop[value_index];
  }
}

static void session_layer(app_session *session, int layer_index, int place_from, int lane_count,
                          const float *ple_data) {
  app_model *model = session->model;
  model_form *form = &model->form;
  layer_wing *wing = &model->wing_list[layer_index];
  int state_size = form->state_size;
  int state_stride = session->state_stride;
  int lift_stride = session->lift_stride;
  int lane_index;

  for (lane_index = 0; lane_index < lane_count; ++lane_index)
    model->desk.norm_rms(&model->desk, session->state_room + (size_t)lane_index * state_stride,
                         wing->enter_norm, state_size, form->norm_eps,
                         session->scrap_room + (size_t)lane_index * state_stride);
  session_attend(session, layer_index, place_from, lane_count, session->scrap_room,
                 session->lift_room);
  for (lane_index = 0; lane_index < lane_count; ++lane_index) {
    float *lift_data = session->lift_room + (size_t)lane_index * lift_stride;
    TRACE_LANE("attn", layer_index, place_from + lane_index, lift_data, state_size);
    model->desk.norm_rms(&model->desk, lift_data, wing->after_attn_norm, state_size, form->norm_eps,
                         lift_data);
    kern_add(session->state_room + (size_t)lane_index * state_stride, lift_data, state_size);
    model->desk.norm_rms(&model->desk, session->state_room + (size_t)lane_index * state_stride,
                         wing->before_feed_norm, state_size, form->norm_eps,
                         session->scrap_room + (size_t)lane_index * state_stride);
  }

  session_lift_many(session, &wing->gate_sheet, session->scrap_room, state_stride, lane_count,
                    session->gate_room, session->gate_stride);
  session_lift_many(session, &wing->rise_sheet, session->scrap_room, state_stride, lane_count,
                    session->rise_room, session->rise_stride);
  for (lane_index = 0; lane_index < lane_count; ++lane_index)
    model->desk.gelu_gate(&model->desk,
                          session->gate_room + (size_t)lane_index * session->gate_stride,
                          session->rise_room + (size_t)lane_index * session->rise_stride,
                          wing->inner_size);
  session_lift_many(session, &wing->drop_sheet, session->gate_room, session->gate_stride, lane_count,
                    session->lift_room, lift_stride);
  for (lane_index = 0; lane_index < lane_count; ++lane_index)
    TRACE_LANE("mlp", layer_index, place_from + lane_index,
               session->lift_room + (size_t)lane_index * lift_stride, state_size);

  if (form->moe_flag) {
    /* The dense mlp above is the shared expert; the mixture reads the residual
     * from before it, and the two branches are summed under a third norm.
     * Each lane picks its own experts, so this branch stays one lane wide. */
    for (lane_index = 0; lane_index < lane_count; ++lane_index) {
      float *state_data = session->state_room + (size_t)lane_index * state_stride;
      float *scrap_data = session->scrap_room + (size_t)lane_index * state_stride;
      float *lift_data = session->lift_room + (size_t)lane_index * lift_stride;
      model->desk.norm_rms(&model->desk, lift_data, wing->after_mlp_norm, state_size,
                           form->norm_eps, lift_data);
      session_route(session, wing, state_data);
      model->desk.norm_rms(&model->desk, state_data, wing->before_moe_norm, state_size,
                           form->norm_eps, scrap_data);
      session_expert(session, wing, scrap_data, session->moe_room);
      TRACE_LANE("moe", layer_index, place_from + lane_index, session->moe_room, state_size);
      model->desk.norm_rms(&model->desk, session->moe_room, wing->after_moe_norm, state_size,
                           form->norm_eps, session->moe_room);
      kern_add(lift_data, session->moe_room, state_size);
    }
  }

  for (lane_index = 0; lane_index < lane_count; ++lane_index) {
    float *lift_data = session->lift_room + (size_t)lane_index * lift_stride;
    model->desk.norm_rms(&model->desk, lift_data, wing->after_feed_norm, state_size, form->norm_eps,
                         lift_data);
    kern_add(session->state_room + (size_t)lane_index * state_stride, lift_data, state_size);
  }

  if (form->ple_size > 0) {
    session_lift_many(session, &wing->ple_gate_sheet, session->state_room, state_stride, lane_count,
                      session->gate_room, session->gate_stride);
    for (lane_index = 0; lane_index < lane_count; ++lane_index) {
      float *gate_data = session->gate_room + (size_t)lane_index * session->gate_stride;
      const float *lane_ple = ple_data + (size_t)lane_index * session->ple_stride;
      int value_index;
      for (value_index = 0; value_index < form->ple_size; ++value_index)
        gate_data[value_index] = kern_gelu_tanh(gate_data[value_index]) * lane_ple[value_index];
    }
    session_lift_many(session, &wing->ple_lift_sheet, session->gate_room, session->gate_stride,
                      lane_count, session->lift_room, lift_stride);
    for (lane_index = 0; lane_index < lane_count; ++lane_index) {
      float *lift_data = session->lift_room + (size_t)lane_index * lift_stride;
      model->desk.norm_rms(&model->desk, lift_data, wing->after_ple_norm, state_size,
                           form->norm_eps, lift_data);
      kern_add(session->state_room + (size_t)lane_index * state_stride, lift_data, state_size);
    }
  }
  if (wing->layer_gain != 1.0f)
    for (lane_index = 0; lane_index < lane_count; ++lane_index)
      kern_scale(session->state_room + (size_t)lane_index * state_stride, wing->layer_gain,
                 state_size);

  for (lane_index = 0; lane_index < lane_count; ++lane_index)
    TRACE_LANE("out", layer_index, place_from + lane_index,
               session->state_room + (size_t)lane_index * state_stride, state_size);
}

/* A batch of tokens through the whole graph.  Logits are produced for the last
 * token only, and skipped entirely when the caller is still filling the prompt. */
static const float *session_pass(app_session *session, const int32_t *id_list,
                                 const float *state_list, const uint8_t *state_flag, int lane_count,
                                 int want_logits) {
  app_model *model = session->model;
  model_form *form = &model->form;
  int place_from = session->fill_count;
  int layer_index, lane_index;

  if (lane_count < 1 || lane_count > KERN_LANE_LIMIT) return NULL;
  if (place_from + lane_count > form->window_limit) return NULL;
  for (lane_index = 0; lane_index < lane_count; ++lane_index)
    if (id_list[lane_index] < 0 || id_list[lane_index] >= model->embed_sheet.row_count) return NULL;

  /* A lane a tower filled carries its embedding rather than looking one up.
   * The substitution happens after the token path's scale, not before it, so
   * what a projector emits is already in the units the residual carries.  What
   * the id is used for afterwards is the per-layer embedding below, and there
   * the reference does not read the placeholder: it rewrites every media
   * position to the pad id first, so the token-identity half of the per-layer
   * input is the pad token's row and carries nothing of the placeholder. */
  for (lane_index = 0; lane_index < lane_count; ++lane_index) {
    float *state_data = session->state_room + (size_t)lane_index * session->state_stride;
    if (state_flag && state_flag[lane_index] && state_list) {
      memcpy(state_data, state_list + (size_t)lane_index * (size_t)form->state_size,
             sizeof(float) * (size_t)form->state_size);
    } else {
      plane_row(&model->embed_sheet, id_list[lane_index], state_data);
      kern_scale(state_data, (float)sqrt((double)form->state_size), form->state_size);
    }
    TRACE_LANE("embed", -1, place_from + lane_index, state_data, form->state_size);
  }

  if (form->ple_size > 0) {
    float blend_gain = (float)(1.0 / sqrt(2.0));
    int chunk_count = form->layer_count * form->ple_size;
    session_lift_many(session, &model->ple_lift_sheet, session->state_room, session->state_stride,
                      lane_count, session->ple_room, session->ple_stride);
    for (lane_index = 0; lane_index < lane_count; ++lane_index) {
      float *seed_data = session->ple_seed + (size_t)lane_index * session->ple_stride;
      float *lane_room = session->ple_room + (size_t)lane_index * session->ple_stride;
      int32_t seed_id = state_flag && state_flag[lane_index] && state_list
                            ? (int32_t)form->pad_id
                            : id_list[lane_index];
      int chunk_index;
      if (seed_id < 0) seed_id = 0;
      plane_row(&model->ple_embed_sheet, seed_id % model->ple_embed_sheet.row_count, seed_data);
      kern_scale(seed_data, (float)sqrt((double)form->ple_size), chunk_count);
      kern_scale(lane_room, (float)(1.0 / sqrt((double)form->state_size)), chunk_count);
      for (chunk_index = 0; chunk_index < form->layer_count; ++chunk_index) {
        float *chunk_data = lane_room + (size_t)chunk_index * form->ple_size;
        const float *chunk_seed = seed_data + (size_t)chunk_index * form->ple_size;
        int value_index;
        model->desk.norm_rms(&model->desk, chunk_data, model->ple_norm, form->ple_size,
                             form->norm_eps, chunk_data);
        for (value_index = 0; value_index < form->ple_size; ++value_index)
          chunk_data[value_index] = (chunk_data[value_index] + chunk_seed[value_index]) * blend_gain;
      }
    }
  }

  for (layer_index = 0; layer_index < form->layer_count; ++layer_index)
    session_layer(session, layer_index, place_from, lane_count,
                  session->ple_room + (size_t)layer_index * form->ple_size);

  session->fill_count += lane_count;
  for (lane_index = 0; lane_index < lane_count; ++lane_index)
    if (session->echo_count < session->echo_limit)
      session->echo_room[session->echo_count++] = id_list[lane_index];

  if (!want_logits) return NULL;
  model->desk.norm_rms(&model->desk,
                       session->state_room + (size_t)(lane_count - 1) * session->state_stride,
                       model->final_norm, form->state_size, form->norm_eps, session->scrap_room);
  TRACE_LANE("final", -1, place_from + lane_count - 1, session->scrap_room, form->state_size);
  session_lift(session, &model->head_sheet, session->scrap_room, session->logit_room);
  if (form->logit_cap > 0.0f) {
    int value_index;
    for (value_index = 0; value_index < model->head_sheet.row_count; ++value_index)
      session->logit_room[value_index] =
          tanhf(session->logit_room[value_index] / form->logit_cap) * form->logit_cap;
  }
  TRACE_LANE("logits", -1, place_from + lane_count - 1, session->logit_room,
             model->head_sheet.row_count);
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
/* 11. public layer                                                         */
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
  setup.cache_bits = 0;
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

  code = tower_bind(model, &model->tower_list[TOWER_VISION], TOWER_VISION);
  if (code != APP_OKAY) { model_free(model); return code; }
  code = tower_bind(model, &model->tower_list[TOWER_AUDIO], TOWER_AUDIO);
  if (code != APP_OKAY) { model_free(model); return code; }

  code = token_load(folder_path, model->form.vocab_count, &model->book_ref);
  if (code != APP_OKAY) { model_free(model); return code; }
  if (model->book_ref->start_id >= 0) model->form.start_id = model->book_ref->start_id;
  if (model->book_ref->close_id >= 0) model->form.close_id = model->book_ref->close_id;
  /* A placeholder id the configuration did not record is looked up in the
   * vocabulary instead, which is where the exporters that omit it put it. */
  {
    static const char *image_name[] = {"<|image|>", "<image_soft_token>", "<image>",
                                       "<start_of_image>"};
    static const char *audio_name[] = {"<|audio|>", "<audio_soft_token>", "<audio>",
                                       "<start_of_audio>"};
    /* The two ids the processor writes either side of a run, in the spellings
     * the exports use: this one's angle-bracket pairs, and the older words. */
    static const char *image_open[] = {"<|image>", "<start_of_image>"};
    static const char *image_shut[] = {"<image|>", "<end_of_image>"};
    static const char *audio_open[] = {"<|audio>", "<start_of_audio>"};
    static const char *audio_shut[] = {"<audio|>", "<end_of_audio>"};
    tower_form *vision_form = &model->tower_list[TOWER_VISION].form;
    tower_form *audio_form = &model->tower_list[TOWER_AUDIO].form;
    size_t name_index;
    for (name_index = 0; name_index < 4; ++name_index) {
      if (vision_form->token_id < 0)
        vision_form->token_id = token_find(model->book_ref, image_name[name_index]);
      if (audio_form->token_id < 0)
        audio_form->token_id = token_find(model->book_ref, audio_name[name_index]);
    }
    for (name_index = 0; name_index < 2; ++name_index) {
      if (vision_form->open_id < 0)
        vision_form->open_id = token_find(model->book_ref, image_open[name_index]);
      if (vision_form->shut_id < 0)
        vision_form->shut_id = token_find(model->book_ref, image_shut[name_index]);
      if (audio_form->open_id < 0)
        audio_form->open_id = token_find(model->book_ref, audio_open[name_index]);
      if (audio_form->shut_id < 0)
        audio_form->shut_id = token_find(model->book_ref, audio_shut[name_index]);
    }
    /* An export that names its placeholder `<start_of_image>` has no bracket at
     * all, whatever the vocabulary is read for, so a bracket that came back as
     * the placeholder itself is not one. */
    if (vision_form->open_id == vision_form->token_id) vision_form->open_id = -1;
    if (audio_form->open_id == audio_form->token_id) audio_form->open_id = -1;
    /* A run is bracketed on both sides or on neither.  Half a bracket is a
     * prompt the reference never lays down, and would be worse than none. */
    if (vision_form->open_id < 0 || vision_form->shut_id < 0)
      vision_form->open_id = vision_form->shut_id = -1;
    if (audio_form->open_id < 0 || audio_form->shut_id < 0)
      audio_form->open_id = audio_form->shut_id = -1;
  }

  thread_count = model->setup.thread_count > 0 ? model->setup.thread_count : host_thread_count();
  if (thread_count > 64) thread_count = 64;
  code = pool_open(&model->pool, thread_count);
  if (code != APP_OKAY) { model_free(model); return code; }

  if (model->embed_sheet.group_count > group_peak) group_peak = model->embed_sheet.group_count;
  if (model->head_sheet.group_count > group_peak) group_peak = model->head_sheet.group_count;
  if (model->ple_lift_sheet.group_count > group_peak) group_peak = model->ple_lift_sheet.group_count;
  for (layer_index = 0; layer_index < model->form.layer_count; ++layer_index) {
    layer_wing *wing = &model->wing_list[layer_index];
    const plane *sheet_list[10];
    int sheet_index;
    sheet_list[0] = &wing->query_sheet; sheet_list[1] = &wing->key_sheet;
    sheet_list[2] = &wing->value_sheet; sheet_list[3] = &wing->exit_sheet;
    sheet_list[4] = &wing->gate_sheet;  sheet_list[5] = &wing->rise_sheet;
    sheet_list[6] = &wing->drop_sheet;  sheet_list[7] = &wing->ple_gate_sheet;
    sheet_list[8] = &wing->ple_lift_sheet; sheet_list[9] = &wing->route_sheet;
    for (sheet_index = 0; sheet_index < 10; ++sheet_index)
      if (sheet_list[sheet_index]->group_count > group_peak)
        group_peak = sheet_list[sheet_index]->group_count;
    if (wing->expert_rise_list && wing->expert_drop_list) {
      int expert_index;
      for (expert_index = 0; expert_index < model->form.expert_count; ++expert_index) {
        if (wing->expert_rise_list[expert_index].group_count > group_peak)
          group_peak = wing->expert_rise_list[expert_index].group_count;
        if (wing->expert_drop_list[expert_index].group_count > group_peak)
          group_peak = wing->expert_drop_list[expert_index].group_count;
      }
    }
  }
  for (layer_index = 0; layer_index < TOWER_COUNT; ++layer_index) {
    tower_gear *gear = &model->tower_list[layer_index];
    const plane *sheet_list[6];
    int wing_index, sheet_index;
    sheet_list[0] = &gear->patch_sheet;   sheet_list[1] = &gear->conv_sheet[0];
    sheet_list[2] = &gear->conv_sheet[1]; sheet_list[3] = &gear->join_sheet;
    sheet_list[4] = &gear->lift_sheet;    sheet_list[5] = &gear->out_sheet;
    for (sheet_index = 0; sheet_index < 6; ++sheet_index)
      if (sheet_list[sheet_index]->group_count > group_peak)
        group_peak = sheet_list[sheet_index]->group_count;
    for (wing_index = 0; gear->wing_list && wing_index < gear->form.layer_count; ++wing_index) {
      tower_wing *wing = &gear->wing_list[wing_index];
      const plane *wing_list[7];
      wing_list[0] = &wing->query_sheet; wing_list[1] = &wing->key_sheet;
      wing_list[2] = &wing->value_sheet; wing_list[3] = &wing->exit_sheet;
      wing_list[4] = &wing->gate_sheet;  wing_list[5] = &wing->rise_sheet;
      wing_list[6] = &wing->drop_sheet;
      for (sheet_index = 0; sheet_index < 7; ++sheet_index)
        if (wing_list[sheet_index]->group_count > group_peak)
          group_peak = wing_list[sheet_index]->group_count;
    }
    for (wing_index = 0; gear->sound_list && wing_index < gear->form.layer_count; ++wing_index) {
      sound_wing *wing = &gear->sound_list[wing_index];
      const plane *wing_list[9];
      wing_list[0] = &wing->query_sheet;      wing_list[1] = &wing->key_sheet;
      wing_list[2] = &wing->value_sheet;      wing_list[3] = &wing->exit_sheet;
      wing_list[4] = &wing->place_sheet;      wing_list[5] = &wing->rise_sheet[0];
      wing_list[6] = &wing->drop_sheet[0];    wing_list[7] = &wing->conv_start_sheet;
      wing_list[8] = &wing->conv_end_sheet;
      for (sheet_index = 0; sheet_index < 9; ++sheet_index)
        if (wing_list[sheet_index]->group_count > group_peak)
          group_peak = wing_list[sheet_index]->group_count;
    }
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
      mem_free(wing->route_sheet.own_block);
      mem_free(wing->route_scale);
      mem_free(wing->route_gain);
      if (wing->expert_rise_list && wing->expert_drop_list) {
        int expert_index;
        for (expert_index = 0; expert_index < model->form.expert_count; ++expert_index) {
          mem_free(wing->expert_rise_list[expert_index].own_block);
          mem_free(wing->expert_drop_list[expert_index].own_block);
        }
      }
      mem_free(wing->expert_rise_list);
      mem_free(wing->expert_drop_list);
      mem_free(wing->after_mlp_norm);
      mem_free(wing->before_moe_norm);
      mem_free(wing->after_moe_norm);
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
  tower_free(&model->tower_list[TOWER_VISION]);
  tower_free(&model->tower_list[TOWER_AUDIO]);
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
int model_state_size(const app_model *model) { return model ? model->form.state_size : 0; }
size_t model_memory_bytes(const app_model *model) { return model ? model->weight_bytes : 0; }

/* Weight bytes one decode step reads.
 *
 * This is deliberately not `model_memory_bytes`.  What is mapped and what is
 * read a token are different quantities on this family of export, and the gap
 * between them is most of the file: the two embedding tables are read one row
 * at a time rather than swept, the vision and audio towers are not in the token
 * loop at all, and a layer that shares another layer's keys binds no projection
 * of its own for them.  A bandwidth figure divided out of the mapped total
 * therefore overstates what the machine is asked to move, which is the whole
 * point of measuring it here rather than assuming it. */
size_t model_decode_bytes(const app_model *model) {
  const model_form *form;
  size_t total = 0;
  int layer_index;
  if (!model) return 0;
  form = &model->form;

  /* Both tables are indexed by the token, so a step touches one row of each. */
  total += plane_row_bytes(&model->embed_sheet);
  total += plane_row_bytes(&model->ple_embed_sheet);
  total += plane_bytes(&model->ple_lift_sheet);
  total += plane_bytes(&model->head_sheet);
  if (model->ple_norm) total += sizeof(float) * (size_t)form->ple_size;
  if (model->final_norm) total += sizeof(float) * (size_t)form->state_size;

  for (layer_index = 0; layer_index < form->layer_count; ++layer_index) {
    const layer_wing *wing = &model->wing_list[layer_index];
    size_t norm_count = 0;
    total += plane_bytes(&wing->query_sheet);
    total += plane_bytes(&wing->key_sheet);   /* void on a sharing layer */
    total += plane_bytes(&wing->value_sheet); /* void where keys double as values */
    total += plane_bytes(&wing->exit_sheet);
    total += plane_bytes(&wing->gate_sheet);
    total += plane_bytes(&wing->rise_sheet);
    total += plane_bytes(&wing->drop_sheet);
    total += plane_bytes(&wing->ple_gate_sheet);
    total += plane_bytes(&wing->ple_lift_sheet);
    if (wing->query_norm) norm_count += (size_t)wing->head_size;
    if (wing->key_norm) norm_count += (size_t)wing->head_size;
    if (wing->enter_norm) norm_count += (size_t)form->state_size;
    if (wing->after_attn_norm) norm_count += (size_t)form->state_size;
    if (wing->before_feed_norm) norm_count += (size_t)form->state_size;
    if (wing->after_feed_norm) norm_count += (size_t)form->state_size;
    if (wing->after_ple_norm) norm_count += (size_t)form->state_size;
    if (form->moe_flag) {
      /* Only the experts the router picks are read, so a mixture block costs
       * the router plus `expert_top` of them rather than the whole bank. */
      int pick_limit = form->expert_top < form->expert_count ? form->expert_top : form->expert_count;
      int pick_index;
      total += plane_bytes(&wing->route_sheet);
      for (pick_index = 0; pick_index < pick_limit; ++pick_index) {
        if (wing->expert_rise_list) total += plane_bytes(&wing->expert_rise_list[pick_index]);
        if (wing->expert_drop_list) total += plane_bytes(&wing->expert_drop_list[pick_index]);
      }
      if (wing->route_scale) norm_count += (size_t)form->state_size;
      if (wing->route_gain) norm_count += (size_t)form->expert_count;
      if (wing->after_mlp_norm) norm_count += (size_t)form->state_size;
      if (wing->before_moe_norm) norm_count += (size_t)form->state_size;
      if (wing->after_moe_norm) norm_count += (size_t)form->state_size;
    }
    total += sizeof(float) * norm_count;
  }
  return total;
}

int model_vision_ready(const app_model *model) {
  return model && model->tower_list[TOWER_VISION].form.live_flag ? 1 : 0;
}

int model_audio_ready(const app_model *model) {
  return model && model->tower_list[TOWER_AUDIO].form.live_flag ? 1 : 0;
}

int model_image_token(const app_model *model) {
  return model ? model->tower_list[TOWER_VISION].form.token_id : -1;
}

int model_audio_token(const app_model *model) {
  return model ? model->tower_list[TOWER_AUDIO].form.token_id : -1;
}

static int model_wrap_of(const app_model *model, int kind_mark, int *open_out, int *shut_out) {
  const tower_form *form = model ? &model->tower_list[kind_mark].form : NULL;
  if (open_out) *open_out = form ? form->open_id : -1;
  if (shut_out) *shut_out = form ? form->shut_id : -1;
  return form && form->open_id >= 0 && form->shut_id >= 0 ? 1 : 0;
}

int model_image_wrap(const app_model *model, int *open_out, int *shut_out) {
  return model_wrap_of(model, TOWER_VISION, open_out, shut_out);
}

int model_audio_wrap(const app_model *model, int *open_out, int *shut_out) {
  return model_wrap_of(model, TOWER_AUDIO, open_out, shut_out);
}

/* The most rows one image can produce.  The tower takes a variable resolution,
 * so what a given picture yields is known only after it is read; this is the
 * cap the checkpoint sets. */
int model_image_rows(const app_model *model) {
  if (!model_vision_ready(model)) return 0;
  return model->tower_list[TOWER_VISION].form.soft_limit;
}

/* The most rows one clip can produce, and what one of them is worth.  A clip
 * shorter than the budget is worth what its own frames come to; a longer one is
 * cut to the budget, so this is a ceiling rather than a count. */
int model_audio_rows(const app_model *model) {
  if (!model_audio_ready(model)) return 0;
  return model->tower_list[TOWER_AUDIO].form.soft_limit;
}

int model_audio_span_ms(const app_model *model) {
  if (!model_audio_ready(model)) return 0;
  return model->tower_list[TOWER_AUDIO].form.token_ms;
}

app_code media_image(app_model *model, const char *path_text, app_media *media_out) {
  flat_grid grid;
  app_code code;
  if (!model || !path_text || !media_out) return APP_FAIL_ARGUMENT;
  memset(media_out, 0, sizeof(*media_out));
  if (!model->tower_list[TOWER_VISION].form.live_flag) return APP_FAIL_SUPPORT;
  code = image_read(path_text, &grid);
  if (code != APP_OKAY) return code;
  code = vision_run(model, &grid, &media_out->state_data, &media_out->row_count);
  grid_free(&grid);
  if (code != APP_OKAY) return code;
  media_out->state_size = model->form.state_size;
  return APP_OKAY;
}

app_code media_audio(app_model *model, const char *path_text, app_media *media_out) {
  tower_form *form;
  wave_clip clip;
  flat_grid mel_grid;
  app_code code;
  if (!model || !path_text || !media_out) return APP_FAIL_ARGUMENT;
  memset(media_out, 0, sizeof(*media_out));
  form = &model->tower_list[TOWER_AUDIO].form;
  if (!form->live_flag) return APP_FAIL_SUPPORT;
  memset(&mel_grid, 0, sizeof(mel_grid));
  code = wave_read(path_text, &clip);
  if (code == APP_OKAY) code = wave_rate(&clip, form->rate_value);
  if (code == APP_OKAY) {
    /* The budget is spent in the processor's units, so it is applied here, on
     * the resampled clip, and before a frame is taken from it. */
    long sample_limit = sound_budget_samples(form);
    if (sample_limit > 0 && (long)clip.value_count > sample_limit) {
      clip.value_count = (int)sample_limit;
      media_out->cut_flag = 1;
    }
  }
  if (code == APP_OKAY)
    code = mel_make(&clip, form->mel_count, form->frame_size, form->frame_step, form->turn_size,
                    form->mel_floor, form->lead_pad, &mel_grid);
  wave_free(&clip);
  if (code != APP_OKAY) return code;
  code = audio_run(model, &mel_grid, &media_out->state_data, &media_out->row_count);
  grid_free(&mel_grid);
  if (code != APP_OKAY) return code;
  media_out->state_size = model->form.state_size;
  return APP_OKAY;
}

void media_free(app_media *media) {
  if (!media) return;
  mem_free(media->state_data);
  memset(media, 0, sizeof(*media));
}

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

/* The ids one image or one clip contributes: the opener, one placeholder per
 * row, and the closer.  That is what the processor substitutes for its marker,
 * and the two brackets are ordinary text to the model — it reads them, and a
 * prompt without them is one the reference never sees.
 *
 * Only the placeholders stand for tower rows, so `place_from` points past the
 * opener; the caller lays its rows down from there. */
int token_media_run(const app_model *model, app_media_span *span_list, int span_count,
                    int place_base, int32_t *id_list, int id_limit) {
  int id_count = 0;
  int span_index, row_index;
  if (!model || (span_count > 0 && !span_list) || !id_list) return -1;
  for (span_index = 0; span_index < span_count; ++span_index) {
    app_media_span *span = &span_list[span_index];
    const tower_form *form = &model->tower_list[span->kind_mark == APP_MEDIA_AUDIO ? TOWER_AUDIO
                                                                                   : TOWER_VISION]
                                  .form;
    span->place_from = -1;
    if (span->row_count < 1) continue;
    if (form->open_id >= 0 && id_count < id_limit) id_list[id_count++] = (int32_t)form->open_id;
    span->place_from = place_base + id_count;
    for (row_index = 0; row_index < span->row_count && id_count < id_limit; ++row_index)
      id_list[id_count++] = (int32_t)form->token_id;
    if (form->shut_id >= 0 && id_count < id_limit) id_list[id_count++] = (int32_t)form->shut_id;
  }
  return id_count;
}

/* The pieces of a turn, in the caller's order, written where the caller has got
 * to.  `part_list` may be null, which is the order the front end had before it
 * could express any other: every span, and then the words.
 *
 * `lead_flag` follows the pieces rather than the loop.  A piece begins a chunk
 * when it follows a special id, which a media run ends in and a stretch of
 * words does not; that is what the reference does, splitting its input on the
 * special tokens and normalizing each chunk on its own, and it is why the first
 * words of a turn continue the line the role marker opened while the words
 * after a picture start a new one. */
static int token_body_parts(const app_model *model, const token_book *book,
                            const app_part *part_list, int part_count, const char *plain_text,
                            app_media_span *span_list, int span_count, int place_base,
                            int32_t *id_list, int id_limit, int lead_flag) {
  int id_count = 0, step_index;
  int step_count = part_list ? part_count : span_count + 1;
  for (step_index = 0; step_index < step_count; ++step_index) {
    app_part step;
    int wrote_count;
    if (part_list) {
      step = part_list[step_index];
    } else if (step_index < span_count) {
      step.kind_mark = APP_PART_MEDIA;
      step.text_ref = NULL;
      step.span_index = step_index;
    } else {
      step.kind_mark = APP_PART_TEXT;
      step.text_ref = plain_text;
      step.span_index = -1;
    }
    if (step.kind_mark == APP_PART_MEDIA) {
      if (step.span_index < 0 || step.span_index >= span_count) return -1;
      wrote_count = token_media_run(model, span_list + step.span_index, 1, place_base + id_count,
                                    id_list + id_count, id_limit - id_count);
      if (wrote_count > 0) {
        id_count += wrote_count;
        lead_flag = 1;
      }
      continue;
    }
    if (!step.text_ref) return -1;
    wrote_count = token_encode_book(book, step.text_ref, lead_flag, id_list + id_count,
                                    id_limit - id_count);
    if (wrote_count > 0) id_count += wrote_count;
    lead_flag = 0;
  }
  return id_count;
}

/* Wraps one user turn in the instruction-tuned chat frame. */
static int token_frame_inner(const app_model *model, const app_part *part_list, int part_count,
                             const char *plain_text, app_media_span *span_list, int span_count,
                             int32_t *id_list, int id_limit, int first_flag) {
  const token_book *book;
  int id_count = 0;
  int wrote_count;
  if (!model || !model->book_ref || !id_list) return -1;
  if (!part_list && !plain_text) return -1;
  book = model->book_ref;
  /* A first turn opens the document; a later one closes the model's turn
   * instead, because the id the sampler stopped on was never fed back. */
  if (first_flag) {
    if (book->start_id >= 0 && id_count < id_limit) id_list[id_count++] = book->start_id;
  } else if (book->turn_open_id >= 0) {
    if (book->turn_shut_id >= 0 && id_count < id_limit) id_list[id_count++] = book->turn_shut_id;
    wrote_count = token_encode_book(book, "\n", 1, id_list + id_count, id_limit - id_count);
    if (wrote_count > 0) id_count += wrote_count;
  }
  if (book->turn_open_id >= 0) {
    if (id_count < id_limit) id_list[id_count++] = book->turn_open_id;
    wrote_count = token_encode_book(book, "user\n", 1, id_list + id_count, id_limit - id_count);
    if (wrote_count > 0) id_count += wrote_count;
    wrote_count = token_body_parts(model, book, part_list, part_count, plain_text, span_list,
                                   span_count, id_count, id_list + id_count,
                                   id_limit - id_count, 0);
    if (wrote_count < 0) return -1;
    id_count += wrote_count;
    if (id_count < id_limit && book->turn_shut_id >= 0) id_list[id_count++] = book->turn_shut_id;
    wrote_count = token_encode_book(book, "\n", 1, id_list + id_count, id_limit - id_count);
    if (wrote_count > 0) id_count += wrote_count;
    if (id_count < id_limit) id_list[id_count++] = book->turn_open_id;
    wrote_count = token_encode_book(book, "model\n", 1, id_list + id_count, id_limit - id_count);
    if (wrote_count > 0) id_count += wrote_count;
    return id_count;
  }
  /* No chat frame: the pieces on their own, and the first of them begins a
   * chunk because nothing precedes it. */
  wrote_count = token_body_parts(model, book, part_list, part_count, plain_text, span_list,
                                 span_count, id_count, id_list + id_count, id_limit - id_count,
                                 1);
  if (wrote_count < 0) return -1;
  return id_count + wrote_count;
}

/* -- keeping a conversation ----------------------------------------------- */

/* A session's cache written out and read back, so a prompt that took thirty
 * seconds to prime need not be primed a second time.
 *
 * What goes in the file is what a session actually holds: the ids it has been
 * fed, and the rows of each layer's key and value cache that have anything in
 * them.  A layer sized for the whole window but holding six hundred rows writes
 * six hundred, so the file is the size of the conversation rather than the size
 * of the window.
 *
 * The file is host native — the same floats and the same bytes the cache holds,
 * in the order the machine holds them, as the mapped checkpoint is — so it is a
 * thing to keep beside a run rather than a thing to send anywhere.  What guards
 * against reading one into the wrong session is `keep_mark`, a mix of every
 * shape the cache's layout depends on and of the checkpoint's own size; a file
 * that disagrees with it is refused rather than restored.
 *
 * The `stamp_value` a caller hands `session_save` is written beside that and
 * handed back by `session_load`.  The engine never reads it: the ids alone
 * cannot say that a picture in the prompt is the same picture, because two
 * pictures lay down the same placeholder ids, and the caller is the one that
 * knows what it fed.
 *
 * `kind_mark` is written beside it and is read, at least far enough to hand it
 * back.  What it distinguishes is not two file layouts but two moments: a
 * prompt saved before its answer is one id short of what it holds and is meant
 * to be primed onto, and a conversation saved after one holds every id and is
 * meant to be framed onto.  The cache is the same either way, which is why
 * nothing but this word tells them apart and why a caller that reads the wrong
 * one gets a turn that drops an id rather than an error.
 *
 * The mark text carries a version because this word was not in the first
 * layout: a file written before it is refused rather than read as a prompt it
 * might not be. */
#define KEEP_MARK_TEXT "igllm cache 2\n\0\0"
#define KEEP_MARK_SIZE 16

static uint64_t keep_mix(uint64_t mark_value, uint64_t value_now) {
  mark_value ^= value_now + 0x9E3779B97F4A7C15ull + (mark_value << 6) + (mark_value >> 2);
  return mark_value;
}

/* Everything about a model and a session that decides how the cache is laid
 * out, and enough about the checkpoint to tell two of the same shape apart. */
static uint64_t keep_mark(const app_session *session) {
  const app_model *model = session->model;
  const model_form *form = &model->form;
  uint64_t mark_value = 0x243F6A8885A308D3ull;
  int layer_index;
  mark_value = keep_mix(mark_value, (uint64_t)form->layer_count);
  mark_value = keep_mix(mark_value, (uint64_t)form->head_count);
  mark_value = keep_mix(mark_value, (uint64_t)form->state_size);
  mark_value = keep_mix(mark_value, (uint64_t)form->window_limit);
  mark_value = keep_mix(mark_value, (uint64_t)form->slide_span);
  mark_value = keep_mix(mark_value, (uint64_t)model->setup.cache_bits);
  mark_value = keep_mix(mark_value, (uint64_t)model->embed_sheet.row_count);
  mark_value = keep_mix(mark_value, (uint64_t)model_memory_bytes(model));
  for (layer_index = 0; layer_index < form->layer_count; ++layer_index) {
    const layer_wing *wing = &model->wing_list[layer_index];
    mark_value = keep_mix(mark_value, (uint64_t)wing->cache_span);
    mark_value = keep_mix(mark_value, (uint64_t)wing->kv_count);
    mark_value = keep_mix(mark_value, (uint64_t)wing->head_size);
    mark_value = keep_mix(mark_value, (uint64_t)wing->share_flag);
    mark_value = keep_mix(mark_value, (uint64_t)cache_slot_bytes(session->key_grid[layer_index]));
    mark_value = keep_mix(mark_value, (uint64_t)cache_slot_bytes(session->value_grid[layer_index]));
  }
  return mark_value;
}

/* How many of a layer's rows carry anything.  A ring that has turned over holds
 * its whole span and every slot of it is live; one that has not holds its rows
 * at the slots it filled, which are the first of them. */
static int keep_row_count(const app_session *session, const layer_wing *wing) {
  int fill_count = session->fill_count;
  return fill_count < wing->cache_span ? fill_count : wing->cache_span;
}

app_code session_save(const app_session *session, const char *path_text, uint64_t stamp_value,
                      int kind_mark) {
  FILE *handle;
  uint64_t head_list[7];
  int layer_index;
  app_code code = APP_OKAY;
  if (!session || !path_text) return APP_FAIL_ARGUMENT;
  if (kind_mark != APP_KEEP_PROMPT && kind_mark != APP_KEEP_TALK) return APP_FAIL_ARGUMENT;
  handle = fopen(path_text, "wb");
  if (!handle) return APP_FAIL_MISSING;
  head_list[0] = keep_mark(session);
  head_list[1] = stamp_value;
  head_list[2] = (uint64_t)session->fill_count;
  head_list[3] = (uint64_t)session->echo_count;
  head_list[4] = (uint64_t)session->model->form.layer_count;
  head_list[5] = (uint64_t)session->model->setup.cache_bits;
  head_list[6] = (uint64_t)kind_mark;
  if (fwrite(KEEP_MARK_TEXT, 1, KEEP_MARK_SIZE, handle) != KEEP_MARK_SIZE ||
      fwrite(head_list, sizeof(uint64_t), 7, handle) != 7)
    code = APP_FAIL_FORMAT;
  if (code == APP_OKAY && session->echo_count > 0 &&
      fwrite(session->echo_room, sizeof(int32_t), (size_t)session->echo_count, handle) !=
          (size_t)session->echo_count)
    code = APP_FAIL_FORMAT;
  for (layer_index = 0; code == APP_OKAY && layer_index < session->model->form.layer_count;
       ++layer_index) {
    const layer_wing *wing = &session->model->wing_list[layer_index];
    size_t row_bytes, keep_bytes;
    int row_count;
    if (wing->share_flag || !session->key_store[layer_index]) continue;
    row_count = keep_row_count(session, wing);
    row_bytes = (size_t)wing->kv_count * (size_t)wing->head_size;
    keep_bytes = row_bytes * (size_t)row_count;
    if (row_count < 1) continue;
    if (fwrite(session->key_store[layer_index], cache_slot_bytes(session->key_grid[layer_index]),
               keep_bytes, handle) != keep_bytes ||
        fwrite(session->value_store[layer_index],
               cache_slot_bytes(session->value_grid[layer_index]), keep_bytes, handle) !=
            keep_bytes)
      code = APP_FAIL_FORMAT;
  }
  if (code == APP_OKAY &&
      (fwrite(session->key_peak, sizeof(float), (size_t)session->model->form.layer_count, handle) !=
           (size_t)session->model->form.layer_count ||
       fwrite(session->value_peak, sizeof(float), (size_t)session->model->form.layer_count,
              handle) != (size_t)session->model->form.layer_count))
    code = APP_FAIL_FORMAT;
  if (fclose(handle) != 0) code = APP_FAIL_FORMAT;
  return code;
}

app_code session_load(app_session *session, const char *path_text, uint64_t *stamp_out,
                      int *kind_out) {
  FILE *handle;
  char mark_room[KEEP_MARK_SIZE];
  uint64_t head_list[7];
  int layer_index, fill_count, echo_count;
  app_code code = APP_OKAY;
  if (!session || !path_text) return APP_FAIL_ARGUMENT;
  if (stamp_out) *stamp_out = 0;
  if (kind_out) *kind_out = APP_KEEP_PROMPT;
  handle = fopen(path_text, "rb");
  if (!handle) return APP_FAIL_MISSING;
  if (fread(mark_room, 1, KEEP_MARK_SIZE, handle) != KEEP_MARK_SIZE ||
      memcmp(mark_room, KEEP_MARK_TEXT, KEEP_MARK_SIZE) != 0 ||
      fread(head_list, sizeof(uint64_t), 7, handle) != 7) {
    fclose(handle);
    return APP_FAIL_FORMAT;
  }
  if (head_list[6] != APP_KEEP_PROMPT && head_list[6] != APP_KEEP_TALK) {
    fclose(handle);
    return APP_FAIL_FORMAT;
  }
  /* A cache belongs to the shapes it was written from.  Reading one into a
   * session laid out differently would be a conversation the model never had,
   * so it is refused rather than made to fit. */
  if (head_list[0] != keep_mark(session) ||
      head_list[4] != (uint64_t)session->model->form.layer_count ||
      head_list[5] != (uint64_t)session->model->setup.cache_bits) {
    fclose(handle);
    return APP_FAIL_STATE;
  }
  fill_count = (int)head_list[2];
  echo_count = (int)head_list[3];
  if (fill_count < 0 || fill_count > session->model->form.window_limit || echo_count < 0 ||
      echo_count > session->echo_limit) {
    fclose(handle);
    return APP_FAIL_FORMAT;
  }

  /* Whatever is read replaces what the session held, and a read that stops
   * half way leaves it cleared rather than half a conversation. */
  session_reset(session);
  if (echo_count > 0 &&
      fread(session->echo_room, sizeof(int32_t), (size_t)echo_count, handle) != (size_t)echo_count)
    code = APP_FAIL_FORMAT;
  session->fill_count = fill_count;
  for (layer_index = 0; code == APP_OKAY && layer_index < session->model->form.layer_count;
       ++layer_index) {
    const layer_wing *wing = &session->model->wing_list[layer_index];
    size_t keep_bytes;
    int row_count;
    if (wing->share_flag || !session->key_store[layer_index]) continue;
    row_count = keep_row_count(session, wing);
    keep_bytes = (size_t)wing->kv_count * (size_t)wing->head_size * (size_t)row_count;
    if (row_count < 1) continue;
    if (fread(session->key_store[layer_index], cache_slot_bytes(session->key_grid[layer_index]),
              keep_bytes, handle) != keep_bytes ||
        fread(session->value_store[layer_index], cache_slot_bytes(session->value_grid[layer_index]),
              keep_bytes, handle) != keep_bytes)
      code = APP_FAIL_FORMAT;
  }
  if (code == APP_OKAY &&
      (fread(session->key_peak, sizeof(float), (size_t)session->model->form.layer_count, handle) !=
           (size_t)session->model->form.layer_count ||
       fread(session->value_peak, sizeof(float), (size_t)session->model->form.layer_count,
             handle) != (size_t)session->model->form.layer_count))
    code = APP_FAIL_FORMAT;
  fclose(handle);
  if (code != APP_OKAY) {
    session_reset(session);
    return code;
  }
  session->echo_count = echo_count;
  if (stamp_out) *stamp_out = head_list[1];
  if (kind_out) *kind_out = (int)head_list[6];
  return APP_OKAY;
}

int session_ids(const app_session *session, int32_t *id_list, int id_limit) {
  if (!session) return -1;
  if (!id_list || id_limit < session->echo_count) return session->echo_count;
  memcpy(id_list, session->echo_room, sizeof(int32_t) * (size_t)session->echo_count);
  return session->echo_count;
}

int token_frame_media(const app_model *model, const char *user_text, app_media_span *span_list,
                      int span_count, int32_t *id_list, int id_limit) {
  if (!user_text) return -1;
  return token_frame_inner(model, NULL, 0, user_text, span_list, span_count, id_list, id_limit, 1);
}

/* The frame for a turn that is not the first.  Everything a session has already
 * been fed stays in its cache, so what this lays down is only what comes after
 * it: the id that closed the model's turn — which the sampler stopped on and
 * never fed back — and then the same user turn and the same opening the model
 * answers into. */
int token_frame_next(const app_model *model, const app_part *part_list, int part_count,
                     app_media_span *span_list, int span_count, int32_t *id_list, int id_limit) {
  if (!part_list || part_count < 0) return -1;
  return token_frame_inner(model, part_list, part_count, NULL, span_list, span_count, id_list,
                           id_limit, 0);
}

int token_frame_parts(const app_model *model, const app_part *part_list, int part_count,
                      app_media_span *span_list, int span_count, int32_t *id_list, int id_limit) {
  if (!part_list || part_count < 0) return -1;
  return token_frame_inner(model, part_list, part_count, NULL, span_list, span_count, id_list,
                           id_limit, 1);
}

int token_frame(const app_model *model, const char *user_text, int32_t *id_list, int id_limit) {
  return token_frame_media(model, user_text, NULL, 0, id_list, id_limit);
}

app_code session_open(app_model *model, app_session **session_out) {
#define LANE_ROOM(stride) \
  ((float *)mem_clear(sizeof(float) * (size_t)(stride) * (size_t)KERN_LANE_LIMIT))
  app_session *session;
  model_form *form;
  int layer_index;
  int head_peak = 0, inner_peak = 0, half_peak = 0, wide_peak;
  int group_peak = 1, block_peak;
  int setup_bits;

  if (!model || !session_out) return APP_FAIL_ARGUMENT;
  *session_out = NULL;
  session = (app_session *)mem_clear(sizeof(app_session));
  if (!session) return APP_FAIL_MEMORY;
  session->model = model;
  form = &model->form;
  setup_bits = model->setup.cache_bits;

  for (layer_index = 0; layer_index < form->layer_count; ++layer_index) {
    layer_wing *wing = &model->wing_list[layer_index];
    if (wing->head_size > head_peak) head_peak = wing->head_size;
    if (wing->inner_size > inner_peak) inner_peak = wing->inner_size;
    if (wing->group_share > group_peak) group_peak = wing->group_share;
  }
  if (form->moe_flag && 2 * form->expert_inner > inner_peak) inner_peak = 2 * form->expert_inner;
  half_peak = head_peak / 2 + 1;
  for (layer_index = 0; layer_index < MODEL_KIND_COUNT; ++layer_index)
    if (form->rope_list[layer_index].half_count >= half_peak)
      half_peak = form->rope_list[layer_index].half_count + 1;
  wide_peak = form->head_count * head_peak;
  if (form->state_size > wide_peak) wide_peak = form->state_size;

  session->key_store = (void **)mem_clear(sizeof(void *) * (size_t)form->layer_count);
  session->value_store = (void **)mem_clear(sizeof(void *) * (size_t)form->layer_count);
  session->key_grid = (float **)mem_clear(sizeof(float *) * (size_t)form->layer_count);
  session->value_grid = (float **)mem_clear(sizeof(float *) * (size_t)form->layer_count);
  session->key_peak = (float *)mem_clear(sizeof(float) * (size_t)form->layer_count);
  session->value_peak = (float *)mem_clear(sizeof(float) * (size_t)form->layer_count);
  if (!session->key_store || !session->value_store || !session->key_grid || !session->value_grid ||
      !session->key_peak || !session->value_peak) {
    session_close(session);
    return APP_FAIL_MEMORY;
  }
  /* A side of a layer is held as bytes where the export ships it a scale and
   * `cache_bits` asks for it, and as floats otherwise, so a checkpoint that
   * calibrates nothing costs nothing for asking.  A zero byte decodes to a zero
   * float, which is what makes a cleared store a cleared cache either way. */
  for (layer_index = 0; layer_index < form->layer_count; ++layer_index) {
    layer_wing *wing = &model->wing_list[layer_index];
    size_t slot_count;
    if (wing->share_flag) continue;
    slot_count = (size_t)wing->cache_span * (size_t)wing->kv_count * (size_t)wing->head_size;
    if (setup_bits == 8 && wing->key_cache_scale > 0.0f) {
      session->key_grid[layer_index] = (float *)mem_clear(sizeof(float) * 256u);
      if (!session->key_grid[layer_index]) { session_close(session); return APP_FAIL_MEMORY; }
      cache_grid_fill(session->key_grid[layer_index], wing->key_cache_scale);
    }
    if (setup_bits == 8 && wing->value_cache_scale > 0.0f) {
      session->value_grid[layer_index] = (float *)mem_clear(sizeof(float) * 256u);
      if (!session->value_grid[layer_index]) { session_close(session); return APP_FAIL_MEMORY; }
      cache_grid_fill(session->value_grid[layer_index], wing->value_cache_scale);
    }
    session->key_store[layer_index] =
        mem_clear(slot_count * cache_slot_bytes(session->key_grid[layer_index]));
    session->value_store[layer_index] =
        mem_clear(slot_count * cache_slot_bytes(session->value_grid[layer_index]));
    if (!session->key_store[layer_index] || !session->value_store[layer_index]) {
      session_close(session);
      return APP_FAIL_MEMORY;
    }
  }

  session->state_stride = form->state_size;
  session->lift_stride = wide_peak;
  session->quant_stride = wide_peak + inner_peak;
  session->head_stride = form->head_count * head_peak;
  /* One row of scores a head of the widest group, because the blocked read
   * scores a whole group against one decoded block before any of them is
   * softmaxed.  A session that never blocks keeps one row and the rest is a
   * few megabytes beside a cache measured in gigabytes. */
  session->score_stride = form->window_limit + 1;
  session->gate_stride = inner_peak + form->ple_size + 1;
  session->rise_stride = inner_peak + 1;
  session->ple_stride = form->layer_count * form->ple_size + 1;

  session->state_room = LANE_ROOM(session->state_stride);
  session->scrap_room = LANE_ROOM(session->state_stride);
  session->lift_room = LANE_ROOM(session->lift_stride);
  session->quant_room = LANE_ROOM(session->quant_stride);
  session->query_room = LANE_ROOM(session->head_stride);
  session->key_room = LANE_ROOM(session->head_stride);
  session->value_room = LANE_ROOM(session->head_stride);
  session->blend_room = LANE_ROOM(session->head_stride);
  session->score_room =
      (float *)mem_clear(sizeof(float) * (size_t)session->score_stride * (size_t)group_peak);
  block_peak = cache_block_rows(head_peak) * head_peak;
  session->cache_room = (float *)mem_clear(sizeof(float) * (size_t)block_peak);
  session->gate_room = LANE_ROOM(session->gate_stride);
  session->rise_room = LANE_ROOM(session->rise_stride);
  session->cos_room = (float *)mem_clear(sizeof(float) * (size_t)half_peak);
  session->sin_room = (float *)mem_clear(sizeof(float) * (size_t)half_peak);
  session->ple_seed = LANE_ROOM(session->ple_stride);
  session->ple_room = LANE_ROOM(session->ple_stride);
  session->logit_room = (float *)mem_clear(sizeof(float) * (size_t)model->head_sheet.row_count);
  if (form->moe_flag) {
    session->route_room = (float *)mem_clear(sizeof(float) * (size_t)form->expert_count);
    session->expert_room = (float *)mem_clear(sizeof(float) * (size_t)(2 * form->expert_inner));
    session->expert_drop = (float *)mem_clear(sizeof(float) * (size_t)form->state_size);
    session->moe_room = (float *)mem_clear(sizeof(float) * (size_t)form->state_size);
    session->top_weight = (float *)mem_clear(sizeof(float) * (size_t)form->expert_top);
    session->top_slot = (int *)mem_clear(sizeof(int) * (size_t)form->expert_top);
    if (!session->route_room || !session->expert_room || !session->expert_drop ||
        !session->moe_room || !session->top_weight || !session->top_slot) {
      session_close(session);
      return APP_FAIL_MEMORY;
    }
  }
  session->pick_room = mem_clear(sizeof(pick_slot) * (size_t)model->head_sheet.row_count);
  session->echo_limit = form->window_limit;
  session->echo_room = (int32_t *)mem_clear(sizeof(int32_t) * (size_t)session->echo_limit);
  session->draw_state = 0x2545F4914F6CDD1DULL;

  if (!session->state_room || !session->scrap_room || !session->lift_room || !session->quant_room ||
      !session->query_room || !session->key_room || !session->value_room || !session->blend_room ||
      !session->score_room || !session->cache_room || !session->gate_room || !session->rise_room || !session->cos_room ||
      !session->sin_room || !session->ple_seed || !session->ple_room || !session->logit_room ||
      !session->pick_room || !session->echo_room) {
    session_close(session);
    return APP_FAIL_MEMORY;
  }
  *session_out = session;
  return APP_OKAY;
#undef LANE_ROOM
}

void session_close(app_session *session) {
  if (!session) return;
  TRACE_CLOSE();
  session_free_rooms(session);
  mem_free(session);
}

void session_reset(app_session *session) {
  int layer_index;
  if (!session) return;
  session->fill_count = 0;
  session->echo_count = 0;
  memset(&session->tally, 0, sizeof(session->tally));
  memset(session->key_peak, 0, sizeof(float) * (size_t)session->model->form.layer_count);
  memset(session->value_peak, 0, sizeof(float) * (size_t)session->model->form.layer_count);
  for (layer_index = 0; layer_index < session->model->form.layer_count; ++layer_index) {
    layer_wing *wing = &session->model->wing_list[layer_index];
    size_t slot_count;
    if (wing->share_flag || !session->key_store[layer_index]) continue;
    slot_count = (size_t)wing->cache_span * (size_t)wing->kv_count * (size_t)wing->head_size;
    memset(session->key_store[layer_index], 0,
           slot_count * cache_slot_bytes(session->key_grid[layer_index]));
    memset(session->value_store[layer_index], 0,
           slot_count * cache_slot_bytes(session->value_grid[layer_index]));
  }
}

/* Consumes every token but the last, which the caller feeds to session_step.
 *
 * `state_list` holds one embedding row per id and `state_flag` says which of
 * them a tower filled; both may be null, which is the text-only case. */
/* Cache bytes one step reads at the fill the session is at.  Every layer scans
 * its keys and then its values over whatever span its kind allows — the whole
 * fill for a full-attention layer, the window for a sliding one — and a sharing
 * layer scans the layer it shares from, so all of them are counted.  What is
 * counted is the distinct bytes of the span: the heads of one group re-read the
 * same key head, but a group's span is tens of kilobytes and stays in cache.
 *
 * A layer that holds its rows as bytes reads a quarter of what it read as
 * floats, which is most of the point of holding them that way, so the width of
 * a stored value is asked of the layer rather than assumed. */
static size_t session_cache_bytes(const app_session *session) {
  const model_form *form = &session->model->form;
  int place_index = session->fill_count;
  size_t total = 0;
  int layer_index;
  if (place_index < 0) return 0;
  for (layer_index = 0; layer_index < form->layer_count; ++layer_index) {
    const layer_wing *wing = &session->model->wing_list[layer_index];
    int owner_slot = wing->share_flag ? wing->source_slot : layer_index;
    int span_count = place_index + 1;
    size_t span_size;
    if (wing->kind_mark == MODEL_KIND_SLIDE && span_count > form->slide_span)
      span_count = form->slide_span;
    span_size = (size_t)span_count * (size_t)wing->kv_count * (size_t)wing->head_size;
    total += span_size * cache_slot_bytes(session->key_grid[owner_slot]);
    total += span_size * cache_slot_bytes(session->value_grid[owner_slot]);
  }
  return total;
}

app_code session_prime_media(app_session *session, const int32_t *id_list, int id_count,
                             const float *state_list, const uint8_t *state_flag) {
  double from_time;
  int id_index;
  int state_size;
  if (!session || !id_list || id_count < 1) return APP_FAIL_ARGUMENT;
  if (session->fill_count + id_count > session->model->form.window_limit) return APP_FAIL_STATE;
  state_size = session->model->form.state_size;
  for (id_index = 0; id_index < id_count; ++id_index)
    if (id_list[id_index] < 0 || id_list[id_index] >= session->model->embed_sheet.row_count)
      return APP_FAIL_ARGUMENT;
  from_time = time_now();
  for (id_index = 0; id_index + 1 < id_count;) {
    int lane_count = id_count - 1 - id_index;
    if (lane_count > KERN_LANE_LIMIT) lane_count = KERN_LANE_LIMIT;
    session_pass(session, id_list + id_index,
                 state_list ? state_list + (size_t)id_index * (size_t)state_size : NULL,
                 state_flag ? state_flag + id_index : NULL, lane_count, 0);
    id_index += lane_count;
  }
  session->tally.prime_seconds += time_now() - from_time;
  session->tally.prime_tokens += (size_t)(id_count > 0 ? id_count - 1 : 0);
  return APP_OKAY;
}

app_code session_prime(app_session *session, const int32_t *id_list, int id_count) {
  return session_prime_media(session, id_list, id_count, NULL, NULL);
}

const float *session_step_state(app_session *session, int32_t id_value, const float *state_data) {
  const float *logit_list;
  uint8_t state_mark = state_data ? 1u : 0u;
  double from_time;
  if (!session) return NULL;
  /* Counted before the clock starts, so the accounting is not in the timing it
   * is there to divide. */
  session->tally.serve_bytes += model_decode_bytes(session->model) + session_cache_bytes(session);
  from_time = time_now();
  logit_list = session_pass(session, &id_value, state_data, &state_mark, 1, 1);
  session->tally.serve_seconds += time_now() - from_time;
  session->tally.serve_tokens += 1;
  session->tally.memory_bytes = app_total_bytes;
  return logit_list;
}

const float *session_step(app_session *session, int32_t id_value) {
  return session_step_state(session, id_value, NULL);
}

int session_fill(const app_session *session) { return session ? session->fill_count : 0; }

float model_cache_scale(const app_model *model, int layer_index, int value_side) {
  const layer_wing *wing;
  if (!model || layer_index < 0 || layer_index >= model->form.layer_count) return 0.0f;
  wing = &model->wing_list[layer_index];
  return value_side ? wing->value_cache_scale : wing->key_cache_scale;
}

size_t session_cache_room_at(const app_session *session, int cache_bits) {
  size_t total = 0;
  int layer_index;
  if (!session) return 0;
  for (layer_index = 0; layer_index < session->model->form.layer_count; ++layer_index) {
    const layer_wing *wing = &session->model->wing_list[layer_index];
    size_t slot_count;
    if (wing->share_flag) continue; /* reads another layer's cache, allocates none */
    slot_count = (size_t)wing->cache_span * (size_t)wing->kv_count * (size_t)wing->head_size;
    total += slot_count * (cache_bits == 8 && wing->key_cache_scale > 0.0f ? 1u : sizeof(float));
    total += slot_count * (cache_bits == 8 && wing->value_cache_scale > 0.0f ? 1u : sizeof(float));
  }
  return total;
}

size_t session_cache_room(const app_session *session) {
  return session ? session_cache_room_at(session, session->model->setup.cache_bits) : 0;
}

float session_cache_peak(const app_session *session, int layer_index, int value_side) {
  if (!session || layer_index < 0 || layer_index >= session->model->form.layer_count) return 0.0f;
  return (value_side ? session->value_peak : session->key_peak)[layer_index];
}

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

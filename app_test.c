/* app_test.c - unit tests for the engine in app_core.c.
 *
 * The suite exercises every layer that can be checked without the pretrained
 * checkpoint: the platform shims, the JSON reader, the safetensors store, the
 * packed-quantization decode, the kernels against plain references, the rotary
 * tables, and the tokenizer. Fixtures are written into a scratch folder and
 * removed on exit.
 *
 * The last section is the exception. When the shipped export is beside the
 * suite it runs three prompts through the whole stack and holds the result to
 * numbers recorded from the build the reference comparison judged, which takes
 * about twenty seconds; without a checkpoint it says so and passes over. */

/* The attention divides its span across the pool only where the span is long
 * enough to pay for the fork, and the synthetic checkpoint's whole window is
 * sixty-four positions.  Left at the shipped figure the divided path would
 * never run here, so the suite would be checking the serial one twice and
 * calling it agreement.  Lowered, `test_wing` runs one session on each path
 * and holds them to the same bits. */
#define ATTEND_BAND_SPAN 8

#include "app_core.c"

#if defined(_WIN32)
#  include <direct.h>  /* _mkdir, _rmdir */
#  include <process.h> /* _getpid */
#  define getpid _getpid
#endif

/* ======================================================================== */
/* test harness                                                             */
/* ======================================================================== */

static int test_pass_count = 0;
static int test_fail_count = 0;
static const char *test_case_name = "";

static void test_open(const char *name) {
  test_case_name = name;
  printf("-- %s\n", name);
}

static void test_true(int truth_flag, const char *claim) {
  if (truth_flag) {
    test_pass_count += 1;
  } else {
    test_fail_count += 1;
    printf("   FAIL %s: %s\n", test_case_name, claim);
  }
}

static void test_near(double value, double want, double slack, const char *claim) {
  double gap = value - want;
  if (gap < 0.0) gap = -gap;
  if (gap <= slack) {
    test_pass_count += 1;
  } else {
    test_fail_count += 1;
    printf("   FAIL %s: %s (got %.9g want %.9g)\n", test_case_name, claim, value, want);
  }
}

/* ======================================================================== */
/* scratch fixtures                                                         */
/* ======================================================================== */

static char test_yard_path[512];

static void test_yard_open(void) {
  const char *root_text = getenv("TMPDIR");
#if defined(_WIN32)
  if (!root_text) root_text = getenv("TEMP");
#endif
  if (!root_text) root_text = "/tmp";
  snprintf(test_yard_path, sizeof(test_yard_path), "%s/igllm_test_%d", root_text, (int)getpid());
#if defined(_WIN32)
  _mkdir(test_yard_path);
#else
  mkdir(test_yard_path, 0700);
#endif
}

static void test_yard_close(void) {
  /* Fixture names are known, so the folder is emptied by hand. */
  static const char *leaf_list[] = {"config.json", "tokenizer.json", "model.safetensors",
                                    "shape.json", "image.png",  "image.pnm",
                                    "image.bmp",  "image.bad",  "clip.wav",
                                    "preprocessor_config.json",
                                    "processor_config.json", NULL};
  int leaf_index;
  char path_text[1024];
  for (leaf_index = 0; leaf_list[leaf_index]; ++leaf_index) {
    path_join(path_text, sizeof(path_text), test_yard_path, leaf_list[leaf_index]);
    remove(path_text);
  }
#if defined(_WIN32)
  _rmdir(test_yard_path);
#else
  rmdir(test_yard_path);
#endif
}

static int test_file_write(const char *leaf, const void *data, size_t byte_count) {
  char path_text[1024];
  FILE *handle;
  path_join(path_text, sizeof(path_text), test_yard_path, leaf);
  handle = fopen(path_text, "wb");
  if (!handle) return 0;
  if (byte_count) fwrite(data, 1, byte_count, handle);
  fclose(handle);
  return 1;
}

/* ======================================================================== */
/* 1. platform layer                                                        */
/* ======================================================================== */

static void test_pool_band(void *state, int slice_index, int slice_count) {
  int *tally_list = (int *)state;
  int from_index, upto_index, slot;
  slice_span(1000, slice_index, slice_count, &from_index, &upto_index);
  for (slot = from_index; slot < upto_index; ++slot) tally_list[slot] += 1;
}

static void test_platform(void) {
  char path_text[64];
  test_open("platform");

  path_join(path_text, sizeof(path_text), "/base", "leaf.txt");
  test_true(strcmp(path_text, "/base/leaf.txt") == 0 || strcmp(path_text, "/base\\leaf.txt") == 0,
            "path_join joins with a separator");
  path_join(path_text, sizeof(path_text), "/base/", "leaf.txt");
  test_true(strstr(path_text, "leaf.txt") != NULL, "path_join keeps the leaf");
  test_true(strstr(path_text, "//") == NULL, "path_join avoids a doubled separator");

  {
    int from_index = 0, upto_index = 0;
    slice_span(10, 0, 3, &from_index, &upto_index);
    test_true(from_index == 0 && upto_index == 4, "slice_span spreads the remainder first");
    slice_span(10, 2, 3, &from_index, &upto_index);
    test_true(upto_index == 10, "slice_span reaches the end");
    slice_span(2, 2, 3, &from_index, &upto_index);
    test_true(from_index == upto_index, "slice_span yields empty bands when short");
  }

  {
    double first_time = time_now();
    double later_time = time_now();
    test_true(later_time >= first_time, "time_now does not run backwards");
  }

  test_true(host_thread_count() >= 1, "host_thread_count is positive");

  {
    pool_group pool;
    int *tally_list = (int *)mem_clear(sizeof(int) * 1000);
    int slot, sum_value = 0;
    test_true(pool_open(&pool, 4) == APP_OKAY, "pool_open succeeds");
    pool_run(&pool, test_pool_band, tally_list);
    pool_run(&pool, test_pool_band, tally_list);
    for (slot = 0; slot < 1000; ++slot) sum_value += tally_list[slot];
    test_true(sum_value == 2000, "pool_run covers every element exactly once per pass");
    test_true(pool.spin_limit == (4 <= host_thread_count() ? POOL_SPIN_LIMIT : 0),
              "a pool spins only where every thread has a core");
    pool_close(&pool);
    mem_free(tally_list);
  }

  {
    /* Oversubscribed: the waiters sleep on the first look rather than spin,
     * because the core a spinner holds is one another worker is waiting for.
     * The bands are the same bands either way, which is what this checks. */
    pool_group pool;
    int wide_count = host_thread_count() * 2 + 1;
    int *tally_list = (int *)mem_clear(sizeof(int) * 1000);
    int slot, sum_value = 0;
    test_true(pool_open(&pool, wide_count) == APP_OKAY, "pool_open succeeds oversubscribed");
    test_true(pool.spin_limit == 0, "an oversubscribed pool does not spin");
    pool_run(&pool, test_pool_band, tally_list);
    for (slot = 0; slot < 1000; ++slot) sum_value += tally_list[slot];
    test_true(sum_value == 1000, "pool_run covers every element once without the spin");
    pool_close(&pool);
    mem_free(tally_list);
  }
}

/* ======================================================================== */
/* 2. json layer                                                            */
/* ======================================================================== */

static void test_json(void) {
  static const char source_text[] =
      "{\"name\":\"gemma\\u00e94\\n\",\"size\":2304,\"scale\":-1.5e-3,"
      "\"flag\":true,\"void\":null,\"list\":[1,2,3],"
      "\"deep\":{\"leaf\":[{\"a\":1},{\"a\":2}]},\"pair\":\"\\ud83d\\ude00\"}";
  json_tree *tree = json_read(source_text, sizeof(source_text) - 1);
  test_open("json");
  test_true(tree != NULL, "json_read accepts a nested document");
  if (!tree) return;

  test_true(strcmp(json_field_text(tree, 0, "name"), "gemma\xc3\xa9" "4\n") == 0,
            "json decodes escapes and utf-8");
  test_near(json_field_number(tree, 0, "size", 0), 2304, 0, "json reads an integer");
  test_near(json_field_number(tree, 0, "scale", 0), -1.5e-3, 1e-12, "json reads an exponent");
  test_true(json_field_flag(tree, 0, "flag", 0) == 1, "json reads true");
  test_true(json_field_flag(tree, 0, "missing", 7) == 7, "json falls back for a missing field");
  test_true(json_number(tree, json_field(tree, 0, "void"), 42.0) == 42.0, "json null uses the spare");
  test_true(json_count(tree, json_field(tree, 0, "list")) == 3, "json counts array items");
  test_near(json_number(tree, json_item(tree, json_field(tree, 0, "list"), 2), 0), 3, 0,
            "json indexes array items");
  {
    int32_t deep_index = json_field(tree, 0, "deep");
    int32_t leaf_index = json_field(tree, deep_index, "leaf");
    int32_t slot_index = json_item(tree, leaf_index, 1);
    test_near(json_field_number(tree, slot_index, "a", 0), 2, 0, "json walks nested containers");
  }
  test_true(strcmp(json_field_text(tree, 0, "pair"), "\xf0\x9f\x98\x80") == 0,
            "json joins a surrogate pair");
  json_free(tree);

  test_true(json_read("{\"a\":", 5) == NULL, "json rejects a truncated document");
  test_true(json_read("[1,2", 4) == NULL, "json rejects an unclosed array");
}

/* ======================================================================== */
/* 3. safetensors store                                                     */
/* ======================================================================== */

static void test_store(void) {
  /* Two tensors: an f32 vector and a bf16 matrix, in one hand-built file. */
  const char header_text[] =
      "{\"alpha\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[0,12]},"
      "\"beta\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[12,20]}}";
  size_t header_size = sizeof(header_text) - 1;
  size_t pad_count = (8 - (header_size % 8)) % 8;
  size_t body_size = 20;
  size_t file_size = 8 + header_size + pad_count + body_size;
  uint8_t *file_data = (uint8_t *)mem_clear(file_size);
  float alpha_list[3] = {1.0f, -2.0f, 0.5f};
  uint16_t beta_list[4] = {0x3F80u, 0xBF00u, 0x4000u, 0x0000u}; /* 1, -0.5, 2, 0 */
  store_set set;
  uint64_t header_count = (uint64_t)(header_size + pad_count);
  int byte_index;

  test_open("store");
  for (byte_index = 0; byte_index < 8; ++byte_index)
    file_data[byte_index] = (uint8_t)((header_count >> (8 * byte_index)) & 0xFFu);
  memcpy(file_data + 8, header_text, header_size);
  for (byte_index = 0; byte_index < (int)pad_count; ++byte_index)
    file_data[8 + header_size + byte_index] = ' ';
  memcpy(file_data + 8 + header_count, alpha_list, sizeof(alpha_list));
  memcpy(file_data + 8 + header_count + 12, beta_list, sizeof(beta_list));

  test_true(test_file_write("model.safetensors", file_data, file_size), "fixture written");
  mem_free(file_data);

  test_true(store_open(&set, test_yard_path) == APP_OKAY, "store_open reads a single shard");
  {
    const store_span *alpha = store_find(&set, "alpha");
    const store_span *beta = store_find(&set, "beta");
    test_true(alpha != NULL, "store_find locates alpha");
    test_true(beta != NULL, "store_find locates beta");
    test_true(store_find(&set, "gamma") == NULL, "store_find rejects an absent name");
    if (alpha) {
      test_true(alpha->type_kind == STORE_F32 && alpha->rank_count == 1 && alpha->size_list[0] == 3,
                "alpha keeps its dtype and shape");
      test_near(real_read(alpha->data_base, alpha->type_kind, 1), -2.0, 0, "alpha payload maps");
    }
    if (beta) {
      test_true(beta->rank_count == 2 && beta->size_list[1] == 2, "beta keeps its shape");
      test_near(real_read(beta->data_base, beta->type_kind, 0), 1.0, 0, "bf16 1.0 decodes");
      test_near(real_read(beta->data_base, beta->type_kind, 1), -0.5, 0, "bf16 -0.5 decodes");
      test_near(real_read(beta->data_base, beta->type_kind, 2), 2.0, 0, "bf16 2.0 decodes");
    }
  }
  store_close(&set);
}

/* ======================================================================== */
/* 4. number and bit decoding                                               */
/* ======================================================================== */

/* Plain reference packer: element i occupies bits [i*bits, i*bits+bits). */
static void test_pack_write(uint8_t *code_data, size_t element_index, int bit_count,
                            uint32_t code_value) {
  int bit_index;
  for (bit_index = 0; bit_index < bit_count; ++bit_index) {
    size_t bit_place = element_index * (size_t)bit_count + (size_t)bit_index;
    if ((code_value >> bit_index) & 1u) code_data[bit_place >> 3] |= (uint8_t)(1u << (bit_place & 7u));
  }
}

static void test_number(void) {
  test_open("number");
  test_near(real_from_bf16(0x0000u), 0.0, 0, "bf16 zero");
  test_near(real_from_bf16(0x3F80u), 1.0, 0, "bf16 one");
  test_near(real_from_bf16(0xC080u), -4.0, 0, "bf16 minus four");
  test_near(real_from_f16(0x3C00u), 1.0, 0, "f16 one");
  test_near(real_from_f16(0xC000u), -2.0, 0, "f16 minus two");
  test_near(real_from_f16(0x0000u), 0.0, 0, "f16 zero");
  test_near(real_from_f16(0x3555u), 0.333251953125, 1e-9, "f16 a third");
  test_near(real_from_f16(0x0200u), 3.0517578125e-05, 1e-12, "f16 subnormal");

  {
    int8_t byte_list[2] = {-3, 5};
    test_near(real_read(byte_list, STORE_I8, 0), -3.0, 0, "i8 reads signed");
    test_true(whole_read(byte_list, STORE_I8, 1) == 5, "whole_read reads i8");
  }
  {
    int64_t wide_list[1] = {123456789};
    test_true(whole_read(wide_list, STORE_I64, 0) == 123456789, "whole_read reads i64");
  }
}

static void test_pack(void) {
  int bit_list[4] = {2, 3, 4, 8};
  int trial_index;
  test_open("pack");
  for (trial_index = 0; trial_index < 4; ++trial_index) {
    int bit_count = bit_list[trial_index];
    uint32_t mask_value = (1u << bit_count) - 1u;
    int element_count = 97;
    uint8_t *code_data = (uint8_t *)mem_clear((size_t)(element_count * bit_count + 64) / 8 + 8);
    int element_index;
    int okay_flag = 1;
    for (element_index = 0; element_index < element_count; ++element_index)
      test_pack_write(code_data, (size_t)element_index, bit_count,
                      (uint32_t)(element_index * 7 + trial_index) & mask_value);
    for (element_index = 0; element_index < element_count; ++element_index) {
      uint32_t want_value = (uint32_t)(element_index * 7 + trial_index) & mask_value;
      if (pack_read(code_data, (size_t)element_index, bit_count) != want_value) okay_flag = 0;
    }
    test_true(okay_flag, "pack_read matches the reference bit stream");
    mem_free(code_data);
  }

  {
    /* Zero points are packed along the row axis, one word column per group. */
    int bit_count = 4, row_count = 9, group_count = 3;
    int word_count = (row_count * bit_count + 31) / 32;
    uint32_t *word_list = (uint32_t *)mem_clear(sizeof(uint32_t) * (size_t)(word_count * group_count));
    int row_index, group_index, okay_flag = 1;
    for (group_index = 0; group_index < group_count; ++group_index)
      for (row_index = 0; row_index < row_count; ++row_index) {
        int bit_place = row_index * bit_count;
        word_list[(bit_place / 32) * group_count + group_index] |=
            (uint32_t)((row_index + group_index) & 0xF) << (bit_place % 32);
      }
    for (group_index = 0; group_index < group_count; ++group_index)
      for (row_index = 0; row_index < row_count; ++row_index)
        if (zero_read(word_list, group_count, group_index, row_index, bit_count) !=
            (((row_index + group_index) & 0xF) - (1 << (bit_count - 1))))
          okay_flag = 0;
    test_true(okay_flag, "zero_read walks the column-major zero point block");
    mem_free(word_list);
  }
}

/* ======================================================================== */
/* 5. planes and kernels                                                    */
/* ======================================================================== */

typedef struct test_plane_kit {
  plane   sheet;
  float  *dense_list; /* the same matrix as plain f32 */
  uint8_t *code_data;
  float   *gain_data;
  int8_t  *bias_data;
} test_plane_kit;

/* Builds a code plane whose exact dense value is known. */
static void test_plane_open(test_plane_kit *kit, int row_count, int col_count, int bit_count,
                            int group_size, int with_bias) {
  int group_count = (col_count + group_size - 1) / group_size;
  size_t row_stride = (size_t)(((col_count * bit_count + 31) / 32) * 4);
  int row_index, col_index;
  int code_bias = 1 << (bit_count - 1);
  uint32_t mask_value = (1u << bit_count) - 1u;

  memset(kit, 0, sizeof(*kit));
  kit->code_data = (uint8_t *)mem_clear(row_stride * (size_t)row_count);
  kit->gain_data = (float *)mem_clear(sizeof(float) * (size_t)(row_count * group_count));
  kit->bias_data = with_bias ? (int8_t *)mem_clear((size_t)(row_count * group_count)) : NULL;
  kit->dense_list = (float *)mem_clear(sizeof(float) * (size_t)(row_count * col_count));

  for (row_index = 0; row_index < row_count; ++row_index) {
    int group_index;
    for (group_index = 0; group_index < group_count; ++group_index) {
      int slot = row_index * group_count + group_index;
      kit->gain_data[slot] = 0.01f * (float)((slot % 7) + 1);
      if (kit->bias_data) kit->bias_data[slot] = (int8_t)((slot % 5) - 2);
    }
    for (col_index = 0; col_index < col_count; ++col_index) {
      int group_index = col_index / group_size;
      int slot = row_index * group_count + group_index;
      uint32_t code_value = (uint32_t)(row_index * 13 + col_index * 5) & mask_value;
      int bias_value = kit->bias_data ? kit->bias_data[slot] : 0;
      test_pack_write(kit->code_data + (size_t)row_index * row_stride, (size_t)col_index, bit_count,
                      code_value);
      kit->dense_list[row_index * col_count + col_index] =
          (float)((int)code_value - code_bias - bias_value) * kit->gain_data[slot];
    }
  }
  kit->sheet.form = PLANE_CODE;
  kit->sheet.row_count = row_count;
  kit->sheet.col_count = col_count;
  kit->sheet.code_data = kit->code_data;
  kit->sheet.row_stride = row_stride;
  kit->sheet.gain_data = kit->gain_data;
  kit->sheet.gain_type = STORE_F32;
  kit->sheet.bias_data = kit->bias_data;
  kit->sheet.bit_count = bit_count;
  kit->sheet.code_bias = code_bias;
  kit->sheet.group_size = group_size;
  kit->sheet.group_count = group_count;
}

static void test_plane_close(test_plane_kit *kit) {
  mem_free(kit->code_data);
  mem_free(kit->gain_data);
  mem_free(kit->bias_data);
  mem_free(kit->dense_list);
}

static void test_plane(void) {
  int bit_list[3] = {2, 4, 8};
  int trial_index;
  test_open("plane");
  for (trial_index = 0; trial_index < 3; ++trial_index) {
    test_plane_kit kit;
    int row_count = 5, col_count = 96;
    float *row_list = (float *)mem_clear(sizeof(float) * (size_t)col_count);
    int row_index, col_index, okay_flag = 1;
    test_plane_open(&kit, row_count, col_count, bit_list[trial_index], 32, trial_index != 0);
    for (row_index = 0; row_index < row_count; ++row_index) {
      plane_row(&kit.sheet, row_index, row_list);
      for (col_index = 0; col_index < col_count; ++col_index) {
        float want = kit.dense_list[row_index * col_count + col_index];
        float gap = row_list[col_index] - want;
        if (gap < 0.0f) gap = -gap;
        if (gap > 1e-5f) okay_flag = 0;
      }
    }
    test_true(okay_flag, "plane_row reproduces the dense matrix");
    test_true(plane_bits_of(col_count, (col_count * bit_list[trial_index]) / 32, 0) ==
                  bit_list[trial_index],
              "plane_bits_of recovers the bit width");
    /* What a product against this plane reads, counted from what the fixture
     * allocated rather than from the function under test. */
    {
      int group_count = (col_count + 31) / 32;
      size_t row_stride = (size_t)(((col_count * bit_list[trial_index] + 31) / 32) * 4);
      size_t want_bytes = row_stride * (size_t)row_count +
                          sizeof(float) * (size_t)row_count * (size_t)group_count +
                          (kit.sheet.bias_data ? (size_t)row_count * (size_t)group_count : 0u);
      test_true(plane_bytes(&kit.sheet) == want_bytes,
                "plane_bytes counts the codes, the gains and the zero points");
      test_true(plane_row_bytes(&kit.sheet) * (size_t)row_count == want_bytes,
                "plane_row_bytes is one row of that");
    }
    mem_free(row_list);
    test_plane_close(&kit);
  }
  {
    /* A real plane is its own payload, and a plane that was never bound reads
     * nothing at all — which is what a layer that shares another's keys has. */
    plane real_sheet;
    plane void_sheet;
    memset(&real_sheet, 0, sizeof(real_sheet));
    memset(&void_sheet, 0, sizeof(void_sheet));
    real_sheet.form = PLANE_REAL;
    real_sheet.row_count = 7;
    real_sheet.col_count = 11;
    real_sheet.real_type = STORE_BF16;
    test_true(plane_bytes(&real_sheet) == 7u * 11u * 2u, "plane_bytes counts a real payload");
    test_true(plane_row_bytes(&real_sheet) == 11u * 2u, "plane_row_bytes counts one real row");
    test_true(plane_bytes(&void_sheet) == 0 && plane_row_bytes(&void_sheet) == 0,
              "an unbound plane reads nothing");
  }
}

/* The layout the Gemma quantized export uses: codes as plain bytes under the
 * name a real weight would have, a scale with one column per group, and the
 * eight bit case stored signed rather than offset. */
static void test_gemma(void) {
  const char header_text[] =
      "{\"q_proj.weight\":{\"dtype\":\"U8\",\"shape\":[3,4],\"data_offsets\":[0,12]},"
      "\"q_proj.weight_scale\":{\"dtype\":\"F32\",\"shape\":[3,1],\"data_offsets\":[12,24]},"
      "\"q_proj.input_activation_scale\":{\"dtype\":\"F32\",\"shape\":[],"
      "\"data_offsets\":[24,28]},"
      "\"q_proj.output_activation_scale\":{\"dtype\":\"F32\",\"shape\":[],"
      "\"data_offsets\":[28,32]},"
      "\"gate.weight\":{\"dtype\":\"I8\",\"shape\":[2,6],\"data_offsets\":[32,44]},"
      "\"gate.weight_scale\":{\"dtype\":\"F32\",\"shape\":[2,1],\"data_offsets\":[44,52]},"
      "\"embed_tokens.embedding_quantized\":{\"dtype\":\"U8\",\"shape\":[2,2],"
      "\"data_offsets\":[52,56]},"
      "\"embed_tokens.embedding_scale\":{\"dtype\":\"F32\",\"shape\":[2,2],"
      "\"data_offsets\":[56,72]}}";
  /* Four bit codes, low nibble first: 0, 1, 2, 3, -7, -6, 7, -8 in every row. */
  static const uint8_t four_byte[4] = {0x98u, 0xBAu, 0x21u, 0x0Fu};
  /* Signed bytes: 0, 1, 127, -1, -128, -56. */
  static const uint8_t sign_byte[6] = {0x00u, 0x01u, 0x7Fu, 0xFFu, 0x80u, 0xC8u};
  /* Two bit codes, lowest pair first: 1, 0, -1, -2 then -2, -1, 0, 1. */
  static const uint8_t pair_byte[2] = {0x1Bu, 0xE4u};
  static const float four_gain[3] = {0.5f, 1.0f, 2.0f};
  static const float sign_gain[2] = {0.25f, 0.5f};
  static const float pair_gain[4] = {1.0f, 4.0f, 0.5f, 2.0f};
  static const float step_pair[2] = {0.125f, 0.0625f};

  size_t header_size = sizeof(header_text) - 1;
  size_t pad_count = (8 - (header_size % 8)) % 8;
  size_t body_size = 72;
  size_t file_size = 8 + header_size + pad_count + body_size;
  uint8_t *file_data = (uint8_t *)mem_clear(file_size);
  uint64_t header_count = (uint64_t)(header_size + pad_count);
  uint8_t *body_data;
  app_model *model = (app_model *)mem_clear(sizeof(app_model));
  int byte_index, row_index;

  test_open("gemma");
  for (byte_index = 0; byte_index < 8; ++byte_index)
    file_data[byte_index] = (uint8_t)((header_count >> (8 * byte_index)) & 0xFFu);
  memcpy(file_data + 8, header_text, header_size);
  for (byte_index = 0; byte_index < (int)pad_count; ++byte_index)
    file_data[8 + header_size + byte_index] = ' ';
  body_data = file_data + 8 + header_count;
  for (row_index = 0; row_index < 3; ++row_index)
    memcpy(body_data + row_index * 4, four_byte, sizeof(four_byte));
  memcpy(body_data + 12, four_gain, sizeof(four_gain));
  memcpy(body_data + 24, &step_pair[0], sizeof(float));
  memcpy(body_data + 28, &step_pair[1], sizeof(float));
  for (row_index = 0; row_index < 2; ++row_index)
    memcpy(body_data + 32 + row_index * 6, sign_byte, sizeof(sign_byte));
  memcpy(body_data + 44, sign_gain, sizeof(sign_gain));
  for (row_index = 0; row_index < 2; ++row_index)
    memcpy(body_data + 52 + row_index * 2, pair_byte, sizeof(pair_byte));
  memcpy(body_data + 56, pair_gain, sizeof(pair_gain));
  test_true(test_file_write("model.safetensors", file_data, file_size), "gemma fixture written");
  mem_free(file_data);

  test_true(store_open(&model->store, test_yard_path) == APP_OKAY, "gemma fixture opens");
  {
    const char *prefix_text = model_prefix_pick(model);
    test_true(prefix_text && prefix_text[0] == '\0',
              "the prefix probe finds a quantized embedding table");
  }
  {
    plane sheet;
    float row_list[8] = {0};
    test_true(plane_bind(model, "q_proj", 8, &sheet) == APP_OKAY,
              "a byte packed weight binds as a code plane");
    test_true(sheet.form == PLANE_CODE && sheet.bit_count == 4 && sheet.col_count == 8,
              "the column hint fixes the bit width");
    test_true(sheet.code_flip == 0 && sheet.code_bias == 8, "four bit codes carry an offset");
    test_true(sheet.group_count == 1 && sheet.group_size == 8, "one scale spans the row");
    test_near(sheet.enter_gain, 0.125, 0, "the input step is read");
    test_near(sheet.leave_gain, 0.0625, 0, "the output step is read");
    plane_row(&sheet, 0, row_list);
    test_near(row_list[0], 0.0, 0, "four bit decode, first code");
    test_near(row_list[3], 1.5, 0, "four bit decode, low nibble first");
    test_near(row_list[4], -3.5, 0, "four bit decode, negative code");
    test_near(row_list[7], -4.0, 0, "four bit decode, last code");
    plane_row(&sheet, 2, row_list);
    test_near(row_list[6], 14.0, 0, "the scale is per row");

    test_true(plane_bind(model, "q_proj", 9, &sheet) == APP_FAIL_FORMAT,
              "a column hint the packing cannot explain is refused");
  }
  {
    plane sheet;
    float row_list[6] = {0};
    float act_list[6];
    float sum_list[1];
    int slot;
    test_true(plane_bind(model, "gate", 6, &sheet) == APP_OKAY, "a signed byte weight binds");
    test_true(sheet.bit_count == 8 && sheet.code_flip == 128 && sheet.code_bias == 128,
              "eight bit codes are flipped rather than offset");
    plane_row(&sheet, 0, row_list);
    test_near(row_list[2], 31.75, 0, "signed decode, largest positive");
    test_near(row_list[3], -0.25, 0, "signed decode, minus one");
    test_near(row_list[4], -32.0, 0, "signed decode, most negative");
    test_near(row_list[5], -14.0, 0, "signed decode, interior");
    for (slot = 0, sum_list[0] = 0.0f; slot < 6; ++slot) {
      act_list[slot] = (float)(slot + 1);
      sum_list[0] += act_list[slot];
    }
    /* The eight bit dot product walks the bytes directly, so it has to take the
     * flip the row decode takes. */
    test_near(kern_row_code(&sheet, 0, act_list, sum_list), -149.25, 1e-4,
              "kern_row_code reads a signed plane");
    test_near(kern_row_code(&sheet, 1, act_list, sum_list), -298.5, 1e-4,
              "kern_row_code scales a signed plane per row");
  }
  {
    plane sheet;
    float row_list[8] = {0};
    test_true(plane_bind(model, "embed_tokens", 8, &sheet) == APP_OKAY,
              "a quantized embedding table binds");
    test_true(sheet.bit_count == 2 && sheet.group_count == 2 && sheet.group_size == 4,
              "the scale shape gives the group size");
    plane_row(&sheet, 0, row_list);
    test_near(row_list[0], 1.0, 0, "two bit decode, lowest pair first");
    test_near(row_list[3], -2.0, 0, "two bit decode, last of the group");
    test_near(row_list[4], -8.0, 0, "the second group takes the second scale");
    test_near(row_list[7], 4.0, 0, "two bit decode, last code");
    plane_row(&sheet, 1, row_list);
    test_near(row_list[4], -4.0, 0, "the group scales are per row");
  }
  store_close(&model->store);
  mem_free(model);
}

/* The calibrated integer grid: the staging, the dot it feeds, and the product
 * the two make.
 *
 * Every check here holds the integer path against a reference computed in
 * integers, so it is exact rather than near: `kern_dot_level` returns the sum a
 * plain loop over the same codes and levels returns, or it is wrong.  The
 * comparison against the float path is the other way round and is deliberately
 * loose — the two are not the same arithmetic, and which is closer to the truth
 * is the point of the change rather than something to assert to the last bit.
 *
 * The tail past the last whole block is what most of the column counts below
 * are for: a block is sixty-four columns and the staging orders them by the bit
 * position they sit at, so a span that ends inside one has to be read the way
 * it lies.  Off by one either way and the sums stop matching. */
static void test_level(void) {
  int bit_list[3] = {2, 4, 8};
  int col_list[6] = {64, 128, 192, 130, 100, 67};
  float step_value = 0.017384f;
  int bit_slot, col_slot;
  test_open("level");

  { /* A value on the step comes back as its level; one between two steps says
     * so rather than rounding to the nearer. */
    float back_step = 1.0f / step_value;
    int okay_flag = 1, level_index;
    for (level_index = -128; level_index <= 127; ++level_index)
      if (kern_level_of((float)level_index * step_value, step_value, back_step) != level_index)
        okay_flag = 0;
    test_true(okay_flag, "every level of the grid is recovered exactly");
    test_true(kern_level_of(0.5f * step_value, step_value, back_step) == KERN_LEVEL_OFF,
              "a value between two levels is refused");
    test_true(kern_level_of(129.0f * step_value, step_value, back_step) == KERN_LEVEL_OFF,
              "a value past the top of the grid is refused");
    test_true(kern_level_of(-129.0f * step_value, step_value, back_step) == KERN_LEVEL_OFF,
              "and past the bottom of it");
  }

  for (bit_slot = 0; bit_slot < 3; ++bit_slot) {
    for (col_slot = 0; col_slot < 6; ++col_slot) {
      int bit_count = bit_list[bit_slot];
      int col_count = col_list[col_slot];
      int row_count = 70;
      test_plane_kit kit;
      float *act_list = (float *)mem_clear(sizeof(float) * (size_t)col_count);
      int8_t *level_list = (int8_t *)mem_clear((size_t)col_count);
      int32_t *isum_list = (int32_t *)mem_clear(sizeof(int32_t));
      int32_t *want_level = (int32_t *)mem_clear(sizeof(int32_t) * (size_t)col_count);
      int col_index, row_index, stage_flag, okay_flag = 1;

      test_plane_open(&kit, row_count, col_count, bit_count, col_count, 1);
      kit.sheet.enter_gain = step_value;
      for (col_index = 0; col_index < col_count; ++col_index) {
        want_level[col_index] = ((col_index * 37 + bit_count * 11) % 255) - 127;
        act_list[col_index] = (float)want_level[col_index] * step_value;
      }
      stage_flag = kern_level_stage(&kit.sheet, act_list, level_list, isum_list);
      test_true(stage_flag, "activations on the step stage as levels");

      { /* The staging is a permutation of the levels, and its sum is theirs. */
        int32_t sum_want = 0;
        int part_count = 8 / bit_count, run_wide = 64 / part_count;
        int base_index, part_index, run_index;
        for (col_index = 0; col_index < col_count; ++col_index) sum_want += want_level[col_index];
        for (base_index = 0; base_index + 64 <= col_count; base_index += 64)
          for (part_index = 0; part_index < part_count; ++part_index)
            for (run_index = 0; run_index < run_wide; ++run_index)
              if (level_list[base_index + run_wide * part_index + run_index] !=
                  (int8_t)want_level[base_index + part_count * run_index + part_index])
                okay_flag = 0;
        for (col_index = base_index; col_index < col_count; ++col_index)
          if (level_list[col_index] != (int8_t)want_level[col_index]) okay_flag = 0;
        test_true(okay_flag, "each level lands where its code will");
        test_true(isum_list[0] == sum_want, "and the group's integer sum is the levels' own");
      }

      { /* The dot against a plain integer loop over the same codes. */
        okay_flag = 1;
        for (row_index = 0; row_index < row_count; ++row_index) {
          const uint8_t *code_row = kit.sheet.code_data + (size_t)row_index * kit.sheet.row_stride;
          int32_t want_value = 0;
          for (col_index = 0; col_index < col_count; ++col_index)
            want_value += (int32_t)pack_read(code_row, (size_t)col_index, bit_count) *
                          want_level[col_index];
          if (kern_dot_level(code_row, 0, col_count, level_list, bit_count, 0) != want_value)
            okay_flag = 0;
        }
        test_true(okay_flag, "kern_dot_level is the integer sum, exactly");
      }

      { /* And the row it makes agrees with the dense matrix it encodes. */
        okay_flag = 1;
        for (row_index = 0; row_index < row_count; ++row_index) {
          double want_value = 0.0;
          float have_value = kern_row_code_level(&kit.sheet, row_index, level_list, isum_list);
          float gap;
          for (col_index = 0; col_index < col_count; ++col_index)
            want_value += (double)kit.dense_list[row_index * col_count + col_index] *
                          (double)act_list[col_index];
          gap = have_value - (float)want_value;
          if (gap < 0.0f) gap = -gap;
          if (gap > 1e-3f) okay_flag = 0;
        }
        test_true(okay_flag, "kern_row_code_level matches the dense product");
      }

      { /* And the block of rows is the one row path, bit for bit.
         *
         * The block shares one staged load between four rows and closes four
         * epilogues beside each other, which is a schedule and not a different
         * sum: an integer dot does not care in what order it is taken, and the
         * two multiplies that put it back in the activation's units are the
         * same two in the same order.  So the claim is equality of floats and
         * not nearness of them.  `row_count` is seventy, which is seventeen
         * whole blocks and two rows past them, so the tail is reached too. */
        float *block_list = (float *)mem_clear(sizeof(float) * (size_t)row_count);
        okay_flag = 1;
        for (row_index = 0; row_index + KERN_ROW_BLOCK <= row_count;
             row_index += KERN_ROW_BLOCK)
          kern_row_code_level_rows(&kit.sheet, row_index, level_list, isum_list, block_list);
        for (; row_index < row_count; ++row_index)
          block_list[row_index] = kern_row_code_level(&kit.sheet, row_index, level_list, isum_list);
        for (row_index = 0; row_index < row_count; ++row_index)
          if (block_list[row_index] !=
              kern_row_code_level(&kit.sheet, row_index, level_list, isum_list))
            okay_flag = 0;
        test_true(okay_flag, "a block of rows is the one row path bit for bit");
        mem_free(block_list);
      }

      { /* One activation off the grid puts the whole product back on floats. */
        act_list[col_count / 2] += 0.5f * step_value;
        test_true(!kern_level_stage(&kit.sheet, act_list, level_list, isum_list),
                  "one activation off the step refuses the whole staging");
        act_list[col_count / 2] -= 0.5f * step_value;
      }

      mem_free(act_list);
      mem_free(level_list);
      mem_free(isum_list);
      mem_free(want_level);
      test_plane_close(&kit);
    }
  }

  { /* The whole product, through the backend, on a plane that carries a step:
     * every lane count from one up, so the batch's blocks of four and the lanes
     * past them are both reached, and against the same rows fed one at a time. */
    int lane_list[4] = {1, 4, 7, 16};
    int lane_slot;
    for (bit_slot = 0; bit_slot < 3; ++bit_slot) {
      for (lane_slot = 0; lane_slot < 4; ++lane_slot) {
        int bit_count = bit_list[bit_slot];
        int lane_count = lane_list[lane_slot];
        int row_count = 70, col_count = 194;
        test_plane_kit kit;
        float *act_list = (float *)mem_clear(sizeof(float) * (size_t)(col_count * lane_count));
        float *many_list = (float *)mem_clear(sizeof(float) * (size_t)(row_count * lane_count));
        pool_group pool;
        back_desk desk;
        int lane_index, row_index, col_index, okay_flag = 1;

        test_plane_open(&kit, row_count, col_count, bit_count, col_count, 1);
        kit.sheet.enter_gain = step_value;
        for (lane_index = 0; lane_index < lane_count; ++lane_index)
          for (col_index = 0; col_index < col_count; ++col_index)
            act_list[lane_index * col_count + col_index] =
                (float)(((lane_index * 53 + col_index * 29) % 255) - 127) * step_value;
        pool_open(&pool, 3);
        back_open(&desk, &pool, kit.sheet.group_count, kit.sheet.col_count);
        desk.mat_mat(&desk, &kit.sheet, act_list, col_count, lane_count, many_list, row_count);
        for (lane_index = 0; lane_index < lane_count; ++lane_index)
          for (row_index = 0; row_index < row_count; ++row_index) {
            double want_value = 0.0;
            float gap;
            for (col_index = 0; col_index < col_count; ++col_index)
              want_value += (double)kit.dense_list[row_index * col_count + col_index] *
                            (double)act_list[lane_index * col_count + col_index];
            gap = many_list[lane_index * row_count + row_index] - (float)want_value;
            if (gap < 0.0f) gap = -gap;
            if (gap > 1e-3f) okay_flag = 0;
          }
        test_true(okay_flag, "a product on the grid matches the dense one at every lane count");
        back_close(&desk);
        pool_close(&pool);
        mem_free(act_list);
        mem_free(many_list);
        test_plane_close(&kit);
      }
    }
  }

  { /* A plane whose activations are not on its step is answered by the float
     * path, and answers the same thing. */
    test_plane_kit kit;
    int row_count = 70, col_count = 194;
    float *act_list = (float *)mem_clear(sizeof(float) * (size_t)col_count);
    float *grid_list = (float *)mem_clear(sizeof(float) * (size_t)row_count);
    float *free_list = (float *)mem_clear(sizeof(float) * (size_t)row_count);
    pool_group pool;
    back_desk desk;
    int col_index, row_index, okay_flag = 1;
    test_plane_open(&kit, row_count, col_count, 4, col_count, 1);
    for (col_index = 0; col_index < col_count; ++col_index)
      act_list[col_index] = (float)sin((double)col_index * 0.19);
    pool_open(&pool, 3);
    back_open(&desk, &pool, kit.sheet.group_count, kit.sheet.col_count);
    desk.mat_vec(&desk, &kit.sheet, act_list, free_list);
    kit.sheet.enter_gain = step_value; /* claimed, but the activations are not on it */
    desk.mat_vec(&desk, &kit.sheet, act_list, grid_list);
    for (row_index = 0; row_index < row_count; ++row_index)
      if (grid_list[row_index] != free_list[row_index]) okay_flag = 0;
    test_true(okay_flag, "a step the activations are not on changes nothing");
    back_close(&desk);
    pool_close(&pool);
    mem_free(act_list);
    mem_free(grid_list);
    mem_free(free_list);
    test_plane_close(&kit);
  }
}

/* Binds one slice of a stacked expert tensor and checks the view lands right. */
static void test_expert(void) {
  const char header_text[] =
      "{\"experts.gate_up_proj\":{\"dtype\":\"F32\",\"shape\":[3,2,4],\"data_offsets\":[0,96]}}";
  size_t header_size = sizeof(header_text) - 1;
  size_t pad_count = (8 - (header_size % 8)) % 8;
  size_t body_size = 96;
  size_t file_size = 8 + header_size + pad_count + body_size;
  uint8_t *file_data = (uint8_t *)mem_clear(file_size);
  uint64_t header_count = (uint64_t)(header_size + pad_count);
  float value_list[24];
  app_model *model = (app_model *)mem_clear(sizeof(app_model));
  int byte_index, slot;

  test_open("expert");
  for (slot = 0; slot < 24; ++slot) value_list[slot] = (float)slot;
  for (byte_index = 0; byte_index < 8; ++byte_index)
    file_data[byte_index] = (uint8_t)((header_count >> (8 * byte_index)) & 0xFFu);
  memcpy(file_data + 8, header_text, header_size);
  for (byte_index = 0; byte_index < (int)pad_count; ++byte_index)
    file_data[8 + header_size + byte_index] = ' ';
  memcpy(file_data + 8 + header_count, value_list, sizeof(value_list));
  test_true(test_file_write("model.safetensors", file_data, file_size), "expert fixture written");
  mem_free(file_data);

  test_true(store_open(&model->store, test_yard_path) == APP_OKAY, "expert fixture opens");
  {
    plane sheet;
    float row_list[4];
    test_true(plane_bind_part(model, "experts.gate_up_proj", 2, 3, 0, &sheet) == APP_OKAY,
              "plane_bind_part accepts a stacked tensor");
    test_true(sheet.row_count == 2 && sheet.col_count == 4, "the slice drops the expert axis");
    plane_row(&sheet, 1, row_list);
    test_near(row_list[0], 20.0, 0, "the slice lands on the right expert");
    test_near(row_list[3], 23.0, 0, "the slice keeps its row stride");
    test_true(plane_bind_part(model, "experts.gate_up_proj", 0, 1, 0, &sheet) == APP_FAIL_FORMAT,
              "plane_bind_part rejects a rank mismatch");
  }
  store_close(&model->store);
  mem_free(model);
}

static void test_kernel(void) {
  test_open("kernel");

  { /* Dense dot against a plain loop, in every stored dtype. */
    int value_count = 133;
    float *act_list = (float *)mem_clear(sizeof(float) * (size_t)value_count);
    float *row_list = (float *)mem_clear(sizeof(float) * (size_t)value_count);
    uint16_t *half_list = (uint16_t *)mem_clear(sizeof(uint16_t) * (size_t)value_count);
    double want_value = 0.0;
    int slot;
    for (slot = 0; slot < value_count; ++slot) {
      act_list[slot] = (float)sin((double)slot * 0.37);
      row_list[slot] = (float)cos((double)slot * 0.11);
      want_value += (double)act_list[slot] * (double)row_list[slot];
    }
    test_near(kern_dot_real(row_list, STORE_F32, act_list, value_count), want_value, 1e-4,
              "kern_dot_real matches a plain f32 loop");
    for (slot = 0; slot < value_count; ++slot) {
      uint32_t raw_bits;
      memcpy(&raw_bits, &row_list[slot], 4);
      half_list[slot] = (uint16_t)(raw_bits >> 16);
    }
    want_value = 0.0;
    for (slot = 0; slot < value_count; ++slot)
      want_value += (double)act_list[slot] * (double)real_from_bf16(half_list[slot]);
    test_near(kern_dot_real(half_list, STORE_BF16, act_list, value_count), want_value, 1e-3,
              "kern_dot_real matches a plain bf16 loop");
    mem_free(act_list);
    mem_free(row_list);
    mem_free(half_list);
  }

  { /* The soft cap against the call it replaces.
     *
     * `kern_logit_cap` is a series rather than `tanhf`, so what is claimed of
     * it is a bound and not equality — and the bound is absolute, because the
     * subtraction from one is where the low bits of a small result go.  Two and
     * a half parts in ten million of the cap is what the sweep measures; the
     * test asks for five, so a host whose series lands a bit differently is
     * still inside it and a path that is actually wrong is not.
     *
     * The three claims after it are the ones a sampler rests on: the order of
     * two logits is never swapped, the ends are exact rather than near, and an
     * argument far past the cap comes back the cap rather than a nan. */
    int value_count = 4001;
    float cap_value = 30.0f;
    float *have_list = (float *)mem_clear(sizeof(float) * (size_t)value_count);
    float *want_list = (float *)mem_clear(sizeof(float) * (size_t)value_count);
    float worst_gap = 0.0f;
    int slot, order_flag = 1;
    for (slot = 0; slot < value_count; ++slot) {
      have_list[slot] = -200.0f + 400.0f * (float)slot / (float)(value_count - 1);
      want_list[slot] = tanhf(have_list[slot] / cap_value) * cap_value;
    }
    kern_logit_cap(have_list, value_count, cap_value);
    for (slot = 0; slot < value_count; ++slot) {
      float gap = have_list[slot] - want_list[slot];
      if (gap < 0.0f) gap = -gap;
      if (gap > worst_gap) worst_gap = gap;
      if (slot && have_list[slot] < have_list[slot - 1]) order_flag = 0;
    }
    test_true(worst_gap <= 5e-7f * cap_value,
              "kern_logit_cap is within five parts in ten million of the cap of tanhf");
    test_true(order_flag, "and never swaps the order of two logits");
    {
      float edge_list[5] = {0.0f, 300.0f, -300.0f, 1e30f, -1e30f};
      kern_logit_cap(edge_list, 5, cap_value);
      test_true(edge_list[0] == 0.0f, "zero caps to zero exactly");
      test_true(edge_list[1] == cap_value && edge_list[2] == -cap_value,
                "and an argument past the cap to the cap itself");
      test_true(edge_list[3] == cap_value && edge_list[4] == -cap_value,
                "however far past it the argument is");
    }
    mem_free(have_list);
    mem_free(want_list);
  }

  { /* The batch's dot against the one it blocks.  Four lanes at a time share
     * the row's load, and the claim the loop rests on is that this is the same
     * arithmetic and not merely close: every lane's sum has to be the float
     * `kern_dot_real` returns, bit for bit, at every span and lane count where
     * the block, its vector tail and its scalar tail all land differently.
     *
     * The lanes are strided as the batch strides them, wider than the span, so
     * a path that read the wrong lane would read a different vector rather
     * than a neighbouring one. */
    int lane_limit = 9, span_limit = 40;
    int lane_stride = 71;
    float *act_list = (float *)mem_clear(sizeof(float) * (size_t)lane_limit * (size_t)lane_stride);
    float *row_list = (float *)mem_clear(sizeof(float) * (size_t)lane_stride);
    int slot, lane_count, span_count, okay_flag = 1;
    for (slot = 0; slot < lane_limit * lane_stride; ++slot)
      act_list[slot] = (float)sin((double)slot * 0.29);
    for (slot = 0; slot < lane_stride; ++slot) row_list[slot] = (float)cos((double)slot * 0.13);
    for (lane_count = 1; lane_count <= lane_limit && okay_flag; ++lane_count) {
      for (span_count = 1; span_count <= span_limit && okay_flag; ++span_count) {
        float many_room[16], one_room[16];
        int lane_index;
        for (lane_index = 0; lane_index < lane_count; ++lane_index) {
          many_room[lane_index] = (float)lane_index * 0.5f; /* a live accumulator */
          one_room[lane_index] = many_room[lane_index];
        }
        kern_dot_real_many(row_list, act_list, lane_stride, lane_count, span_count, many_room);
        for (lane_index = 0; lane_index < lane_count; ++lane_index)
          one_room[lane_index] += kern_dot_real(
              row_list, STORE_F32, act_list + (size_t)lane_index * (size_t)lane_stride, span_count);
        for (lane_index = 0; lane_index < lane_count; ++lane_index)
          if (many_room[lane_index] != one_room[lane_index]) okay_flag = 0;
      }
    }
    test_true(okay_flag, "kern_dot_real_many reaches kern_dot_real's own float, every lane");
    mem_free(act_list);
    mem_free(row_list);
  }

  { /* The two bit table against the bit stream it stands in for.  It is built
     * by a nest of macros at compile time, which is the sort of thing that is
     * either right or catastrophically wrong, and every kernel that reads it
     * would agree with itself either way.  `pack_read` is the independent
     * account of what a byte of two bit codes holds. */
    int byte_index, code_index, okay_flag = 1;
    for (byte_index = 0; byte_index < 256; ++byte_index) {
      uint8_t byte_value = (uint8_t)byte_index;
      for (code_index = 0; code_index < 4; ++code_index)
        if (kern_code_two[byte_index][code_index] !=
            (float)pack_read(&byte_value, (size_t)code_index, 2))
          okay_flag = 0;
    }
    test_true(okay_flag, "the two bit table is the bit stream it stands in for");
  }

  { /* Packed dot and packed spread against a plain bit-stream loop, over every
     * width the format allows, at spans that end mid-block and at leads that
     * are and are not where a block of eight begins — which is what decides
     * whether a width's own path is taken or the walk it falls back to. */
    int bit_list[7] = {2, 3, 4, 5, 6, 7, 8};
    int span_list[8] = {1, 3, 7, 8, 15, 16, 31, 60};
    int lead_list[4] = {0, 8, 24, 5};
    int bit_slot, span_slot, lead_slot, flip_slot;
    for (bit_slot = 0; bit_slot < 7; ++bit_slot) {
      int bit_count = bit_list[bit_slot];
      int element_count = 128;
      uint32_t mask_value = (uint32_t)((1u << bit_count) - 1u);
      uint8_t *code_data = (uint8_t *)mem_clear((size_t)(element_count * bit_count + 7) / 8 + 8);
      float *act_list = (float *)mem_clear(sizeof(float) * (size_t)element_count);
      float *out_list = (float *)mem_clear(sizeof(float) * (size_t)element_count);
      int element_index, okay_flag = 1;
      for (element_index = 0; element_index < element_count; ++element_index) {
        test_pack_write(code_data, (size_t)element_index, bit_count,
                        (uint32_t)(element_index * 11 + bit_slot * 5) & mask_value);
        act_list[element_index] = (float)sin((double)element_index * 0.19);
      }
      for (span_slot = 0; span_slot < 8; ++span_slot)
        for (lead_slot = 0; lead_slot < 4; ++lead_slot)
          for (flip_slot = 0; flip_slot < 2; ++flip_slot) {
            int span_count = span_list[span_slot];
            int from_index = lead_list[lead_slot];
            int code_flip = flip_slot ? (int)mask_value : 0;
            double want_value = 0.0;
            int slot;
            if (from_index + span_count > element_count) continue;
            for (slot = 0; slot < span_count; ++slot)
              want_value +=
                  (double)(pack_read(code_data, (size_t)(from_index + slot), bit_count) ^
                           (uint32_t)code_flip) *
                  (double)act_list[from_index + slot];
            if (fabs((double)kern_dot_code(code_data, from_index, span_count,
                                           act_list + from_index, bit_count, code_flip) -
                     want_value) > 1e-3)
              okay_flag = 0;
            /* The spread has to lay down the same codes the dot summed, value
             * for value: it is the same decode with the multiply left out. */
            kern_code_spread(code_data, from_index, span_count, bit_count, code_flip, out_list);
            for (slot = 0; slot < span_count; ++slot)
              if (out_list[slot] !=
                  (float)(pack_read(code_data, (size_t)(from_index + slot), bit_count) ^
                          (uint32_t)code_flip))
                okay_flag = 0;
          }
      test_true(okay_flag,
                "the packed dot and the packed spread are the bit stream, at every width");
      mem_free(code_data);
      mem_free(act_list);
      mem_free(out_list);
    }
  }

  { /* The same, over a row allocated to exactly the bytes it packs into.
     *
     * The odd widths read their block of eight codes as one eight byte word
     * where the run has eight bytes left, which is more bytes than the block
     * itself is: at three bits a block is three of them.  A row with nothing
     * allocated behind it is what says whether the bound on that read is real,
     * because under the sanitizers a read past it is a failure rather than a
     * value nobody looks at.  The block above pads its row and cannot say.
     *
     * It is `malloc` rather than `mem_clear` for the same reason: the engine's
     * allocator rounds every block up to the alignment its kernels want, so a
     * row asked for at twenty-four bytes is sixty-four and the reads past it
     * land inside the allocation.  Here the row is exactly its own bytes and
     * the sanitizer's guard is the next thing after it.  A weight row is a
     * mapped one rather than either, and a mapping ends at a page: the same
     * read is a fault there rather than a value, which is what the bound is
     * for. */
    int bit_list[7] = {2, 3, 4, 5, 6, 7, 8};
    int bit_slot, element_index, span_count, okay_flag = 1;
    for (bit_slot = 0; bit_slot < 7; ++bit_slot) {
      int bit_count = bit_list[bit_slot];
      uint32_t mask_value = (uint32_t)((1u << bit_count) - 1u);
      int element_count = 64;
      size_t byte_count = (size_t)(element_count * bit_count + 7) / 8;
      uint8_t *code_data = (uint8_t *)malloc(byte_count);
      float *act_list = (float *)mem_clear(sizeof(float) * (size_t)element_count);
      float *out_list = (float *)mem_clear(sizeof(float) * (size_t)element_count);
      if (!code_data || !act_list || !out_list) { okay_flag = 0; }
      if (code_data) memset(code_data, 0, byte_count);
      for (element_index = 0; code_data && element_index < element_count; ++element_index) {
        test_pack_write(code_data, (size_t)element_index, bit_count,
                        (uint32_t)(element_index * 7 + bit_slot) & mask_value);
        act_list[element_index] = (float)cos((double)element_index * 0.31);
      }
      for (span_count = 8; code_data && span_count <= element_count; span_count += 8) {
        double want_value = 0.0;
        int slot;
        for (slot = 0; slot < span_count; ++slot)
          want_value += (double)pack_read(code_data, (size_t)slot, bit_count) *
                        (double)act_list[slot];
        if (fabs((double)kern_dot_code(code_data, 0, span_count, act_list, bit_count, 0) -
                 want_value) > 1e-3)
          okay_flag = 0;
        kern_code_spread(code_data, 0, span_count, bit_count, 0, out_list);
        for (slot = 0; slot < span_count; ++slot)
          if (out_list[slot] != (float)pack_read(code_data, (size_t)slot, bit_count))
            okay_flag = 0;
      }
      free(code_data);
      mem_free(act_list);
      mem_free(out_list);
    }
    test_true(okay_flag, "and stay inside a row that has nothing allocated behind it");
  }

  { /* The same again, over every row length rather than one.
     *
     * A block is read one of two ways now: a shuffle over a sixteen byte load
     * where the row has sixteen bytes behind the block, and the narrower
     * broadcast where it has not.  Which way a block goes is decided once for
     * the run by `kern_code_wide_span`, so the split lands at a different block
     * for every width and every row length — and a row of under sixteen bytes
     * has no wide block at all.  One row length exercises one split.  Sweeping
     * them puts the boundary at every block of every width, which is what says
     * the bound is the right one rather than merely a bound: off by one either
     * way, this fails, and under the sanitizers the row that has nothing behind
     * it says which way. */
    int bit_list[7] = {2, 3, 4, 5, 6, 7, 8};
    int bit_slot, element_count, okay_flag = 1;
    for (bit_slot = 0; bit_slot < 7; ++bit_slot) {
      int bit_count = bit_list[bit_slot];
      for (element_count = 8; element_count <= 200; element_count += 8) {
        size_t byte_count = (size_t)(element_count * bit_count + 7) / 8;
        uint8_t *code_data = (uint8_t *)malloc(byte_count);
        float *act_list = (float *)mem_clear(sizeof(float) * (size_t)element_count);
        float *out_list = (float *)mem_clear(sizeof(float) * (size_t)element_count);
        uint32_t mask_value = (uint32_t)((1u << bit_count) - 1u);
        double want_value = 0.0;
        int slot;
        if (!code_data || !act_list || !out_list) { okay_flag = 0; }
        if (code_data) memset(code_data, 0, byte_count);
        for (slot = 0; code_data && slot < element_count; ++slot) {
          test_pack_write(code_data, (size_t)slot, bit_count,
                          (uint32_t)(slot * 13 + bit_slot * 3) & mask_value);
          act_list[slot] = (float)sin((double)slot * 0.11 + (double)bit_slot);
        }
        for (slot = 0; code_data && slot < element_count; ++slot)
          want_value += (double)pack_read(code_data, (size_t)slot, bit_count) *
                        (double)act_list[slot];
        if (code_data &&
            fabs((double)kern_dot_code(code_data, 0, element_count, act_list, bit_count, 0) -
                 want_value) > 1e-3)
          okay_flag = 0;
        if (code_data) {
          kern_code_spread(code_data, 0, element_count, bit_count, 0, out_list);
          for (slot = 0; slot < element_count; ++slot)
            if (out_list[slot] != (float)pack_read(code_data, (size_t)slot, bit_count))
              okay_flag = 0;
        }
        free(code_data);
        mem_free(act_list);
        mem_free(out_list);
      }
    }
    test_true(okay_flag, "at every row length, so the split between the two lands everywhere");
  }

  { /* Quantized matvec against the dense matrix it encodes. */
    int bit_list[3] = {2, 4, 8};
    int trial_index;
    for (trial_index = 0; trial_index < 3; ++trial_index) {
      test_plane_kit kit;
      int row_count = 70, col_count = 128;
      float *act_list = (float *)mem_clear(sizeof(float) * (size_t)col_count);
      float *out_list = (float *)mem_clear(sizeof(float) * (size_t)row_count);
      pool_group pool;
      back_desk desk;
      int row_index, col_index, okay_flag = 1;
      test_plane_open(&kit, row_count, col_count, bit_list[trial_index], 32, 1);
      for (col_index = 0; col_index < col_count; ++col_index)
        act_list[col_index] = (float)sin((double)col_index * 0.21);
      pool_open(&pool, 3);
      back_open(&desk, &pool, kit.sheet.group_count, kit.sheet.col_count);
      desk.mat_vec(&desk, &kit.sheet, act_list, out_list);
      for (row_index = 0; row_index < row_count; ++row_index) {
        double want_value = 0.0;
        float gap;
        for (col_index = 0; col_index < col_count; ++col_index)
          want_value += (double)kit.dense_list[row_index * col_count + col_index] *
                        (double)act_list[col_index];
        gap = out_list[row_index] - (float)want_value;
        if (gap < 0.0f) gap = -gap;
        if (gap > 1e-3f) okay_flag = 0;
      }
      test_true(okay_flag, "back_mat_vec matches the dense product");
      test_true(strcmp(back_flavor(), "") != 0, "back_flavor names the backend");
      back_close(&desk);
      pool_close(&pool);
      mem_free(act_list);
      mem_free(out_list);
      test_plane_close(&kit);
    }
  }

  { /* A batched product agrees with the same rows fed one at a time. */
    int bit_list[4] = {2, 3, 4, 8};
    int trial_index;
    for (trial_index = 0; trial_index < 4; ++trial_index) {
      test_plane_kit kit;
      int row_count = 70, col_count = 130, lane_count = 19;
      float *act_list = (float *)mem_clear(sizeof(float) * (size_t)(col_count * lane_count));
      float *many_list = (float *)mem_clear(sizeof(float) * (size_t)(row_count * lane_count));
      float *one_list = (float *)mem_clear(sizeof(float) * (size_t)row_count);
      pool_group pool;
      back_desk desk;
      int lane_index, row_index, col_index, okay_flag = 1;
      test_plane_open(&kit, row_count, col_count, bit_list[trial_index], 48, 1);
      for (lane_index = 0; lane_index < lane_count * col_count; ++lane_index)
        act_list[lane_index] = (float)sin((double)lane_index * 0.13);
      pool_open(&pool, 3);
      back_open(&desk, &pool, kit.sheet.group_count, kit.sheet.col_count);
      desk.mat_mat(&desk, &kit.sheet, act_list, col_count, lane_count, many_list, row_count);
      for (lane_index = 0; lane_index < lane_count; ++lane_index) {
        desk.mat_vec(&desk, &kit.sheet, act_list + lane_index * col_count, one_list);
        for (row_index = 0; row_index < row_count; ++row_index) {
          float gap = many_list[lane_index * row_count + row_index] - one_list[row_index];
          if (gap < 0.0f) gap = -gap;
          if (gap > 1e-3f) okay_flag = 0;
        }
      }
      test_true(okay_flag, "back_mat_mat matches a lane at a time");
      (void)col_index;
      back_close(&desk);
      pool_close(&pool);
      mem_free(act_list);
      mem_free(many_list);
      mem_free(one_list);
      test_plane_close(&kit);
    }
  }

  { /* RMS norm, plain weight, no one-plus term. */
    float value_list[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float gain_list[4] = {0.5f, 1.0f, 1.5f, 2.0f};
    float out_list[4];
    double mean_value = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
    double turn_value = 1.0 / sqrt(mean_value + 1e-6);
    int slot;
    int okay_flag = 1;
    kern_norm_rms(value_list, gain_list, 4, 1e-6f, out_list);
    for (slot = 0; slot < 4; ++slot) {
      double want = value_list[slot] * turn_value * gain_list[slot];
      if (fabs((double)out_list[slot] - want) > 1e-5) okay_flag = 0;
    }
    test_true(okay_flag, "kern_norm_rms scales by weight without a one-plus term");
    kern_norm_rms(value_list, NULL, 4, 1e-6f, out_list);
    test_near(out_list[0], 1.0 * turn_value, 1e-5, "kern_norm_rms allows an absent weight");
  }

  { /* GELU tanh against the closed form. */
    double probe_list[5] = {-3.0, -0.5, 0.0, 0.7, 2.5};
    int slot;
    for (slot = 0; slot < 5; ++slot) {
      double value = probe_list[slot];
      double inner = 0.7978845608028654 * (value + 0.044715 * value * value * value);
      double want = 0.5 * value * (1.0 + tanh(inner));
      test_near(kern_gelu_tanh((float)value), want, 1e-5, "kern_gelu_tanh matches the closed form");
    }
  }

  { /* Gated MLP activation. */
    float gate_list[3] = {1.0f, -1.0f, 0.25f};
    float rise_list[3] = {2.0f, 3.0f, -4.0f};
    float want_list[3];
    int slot;
    for (slot = 0; slot < 3; ++slot) want_list[slot] = kern_gelu_tanh(gate_list[slot]) * rise_list[slot];
    kern_gelu_gate(gate_list, rise_list, 3);
    for (slot = 0; slot < 3; ++slot)
      test_near(gate_list[slot], want_list[slot], 1e-6, "kern_gelu_gate gates in place");
  }

  { /* Softmax: normalized, shift invariant, stable for large inputs. */
    float value_list[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float shift_list[5] = {101.0f, 102.0f, 103.0f, 104.0f, 105.0f};
    double sum_value = 0.0;
    int slot;
    kern_soft_max(value_list, 5);
    kern_soft_max(shift_list, 5);
    for (slot = 0; slot < 5; ++slot) sum_value += value_list[slot];
    test_near(sum_value, 1.0, 1e-6, "kern_soft_max sums to one");
    for (slot = 0; slot < 5; ++slot)
      test_near(shift_list[slot], value_list[slot], 1e-6, "kern_soft_max is shift invariant");
    {
      float wide_list[3] = {1000.0f, 1000.0f, 1000.0f};
      kern_soft_max(wide_list, 3);
      test_near(wide_list[0], 1.0 / 3.0, 1e-6, "kern_soft_max survives large inputs");
    }
  }

  { /* Rotate-half rotary transform, and its inverse. */
    int head_size = 8, half_count = 4;
    float value_list[8], keep_list[8];
    float cos_list[4], sin_list[4], back_sin[4];
    int slot, okay_flag = 1;
    for (slot = 0; slot < head_size; ++slot) value_list[slot] = keep_list[slot] = (float)(slot + 1);
    for (slot = 0; slot < half_count; ++slot) {
      cos_list[slot] = (float)cos(0.3 * (slot + 1));
      sin_list[slot] = (float)sin(0.3 * (slot + 1));
      back_sin[slot] = -sin_list[slot];
    }
    kern_rope_turn(value_list, head_size, cos_list, sin_list);
    for (slot = 0; slot < half_count; ++slot) {
      double want_low = keep_list[slot] * cos_list[slot] - keep_list[slot + half_count] * sin_list[slot];
      double want_high = keep_list[slot + half_count] * cos_list[slot] + keep_list[slot] * sin_list[slot];
      if (fabs(value_list[slot] - want_low) > 1e-5) okay_flag = 0;
      if (fabs(value_list[slot + half_count] - want_high) > 1e-5) okay_flag = 0;
    }
    test_true(okay_flag, "kern_rope_turn follows the rotate-half convention");
    kern_rope_turn(value_list, head_size, cos_list, back_sin);
    okay_flag = 1;
    for (slot = 0; slot < head_size; ++slot)
      if (fabs(value_list[slot] - keep_list[slot]) > 1e-4) okay_flag = 0;
    test_true(okay_flag, "kern_rope_turn is inverted by the negated angle");
  }

  { /* Activation fake quantization stays inside the declared grid. */
    quant_rule rule;
    float value_list[6] = {-1.0f, -0.25f, 0.0f, 0.4f, 0.9f, 1.0f};
    float keep_list[6];
    int slot, okay_flag = 1;
    memset(&rule, 0, sizeof(rule));
    rule.live_flag = 1;
    rule.bit_count = 8;
    rule.symmetric_flag = 1;
    rule.plan_kind = QUANT_TENSOR;
    memcpy(keep_list, value_list, sizeof(keep_list));
    quant_act(value_list, 6, &rule);
    for (slot = 0; slot < 6; ++slot)
      if (fabs(value_list[slot] - keep_list[slot]) > 0.02) okay_flag = 0;
    test_true(okay_flag, "quant_act stays close to the input at eight bits");
    rule.live_flag = 0;
    memcpy(value_list, keep_list, sizeof(keep_list));
    quant_act(value_list, 6, &rule);
    test_near(value_list[3], keep_list[3], 0.0, "quant_act is a no-op when the rule is idle");
  }

  { /* The blocked blend against the row-at-a-time loop it replaces.
     *
     * The blocking is only about where the running sum lives, so the two are
     * held to each other rather than to a tolerance on a third account: every
     * value takes the same spans in the same order either way.  A stride wider
     * than the values covers the case the audio tower brings, where the rows
     * are heads of a wider array rather than a run of their own, and the value
     * counts straddle the block so the tail is walked as well as the body. */
    int value_list[5] = {1, 31, 32, 64, 72};
    int span_list[4] = {1, 2, 13, 96};
    int value_slot, span_slot, okay_flag = 1;
    for (value_slot = 0; value_slot < 5; ++value_slot) {
      int value_count = value_list[value_slot];
      int wide_stride = value_count + 7; /* rows further apart than they are long */
      for (span_slot = 0; span_slot < 4; ++span_slot) {
        int span_count = span_list[span_slot];
        int stride_slot;
        for (stride_slot = 0; stride_slot < 2; ++stride_slot) {
          int from_stride = stride_slot ? wide_stride : value_count;
          float *from_data =
              (float *)mem_clear(sizeof(float) * (size_t)span_count * (size_t)from_stride);
          float *weight_list = (float *)mem_clear(sizeof(float) * (size_t)span_count);
          float *into_data = (float *)mem_clear(sizeof(float) * (size_t)value_count);
          float *want_data = (float *)mem_clear(sizeof(float) * (size_t)value_count);
          int span_index, value_index;
          for (span_index = 0; span_index < span_count; ++span_index) {
            weight_list[span_index] = (float)sin((double)span_index * 0.41) * 0.5f;
            for (value_index = 0; value_index < value_count; ++value_index)
              from_data[(size_t)span_index * (size_t)from_stride + (size_t)value_index] =
                  (float)cos((double)(span_index * 13 + value_index) * 0.07);
          }
          for (value_index = 0; value_index < value_count; ++value_index) want_data[value_index] = 0.0f;
          for (span_index = 0; span_index < span_count; ++span_index)
            for (value_index = 0; value_index < value_count; ++value_index)
              want_data[value_index] +=
                  weight_list[span_index] *
                  from_data[(size_t)span_index * (size_t)from_stride + (size_t)value_index];
          kern_blend_rows(from_data, from_stride, weight_list, span_count, value_count, into_data);
          for (value_index = 0; value_index < value_count; ++value_index)
            if (fabs((double)into_data[value_index] - (double)want_data[value_index]) > 1e-5)
              okay_flag = 0;
          mem_free(from_data);
          mem_free(weight_list);
          mem_free(into_data);
          mem_free(want_data);
        }
      }
    }
    test_true(okay_flag, "kern_blend_rows matches the row-at-a-time blend it replaces");
  }

}

/* ======================================================================== */
/* 6. rotary tables                                                         */
/* ======================================================================== */

static void test_rope(void) {
  rope_form rope;
  int head_size = 16;
  test_open("rope");

  memset(&rope, 0, sizeof(rope));
  rope.theta_value = 10000.0;
  rope.part_share = 1.0;
  rope_build(&rope, head_size, "default");
  test_true(rope.half_count == head_size / 2, "default rope keeps every pair");
  test_true(rope.turn_count == head_size / 2, "default rope rotates every pair");
  test_near(rope.step_list[0], 1.0, 1e-9, "the first inverse frequency is one");
  test_near(rope.step_list[1], 1.0 / pow(10000.0, 2.0 / 16.0), 1e-6,
            "inverse frequencies follow the base");
  {
    float cos_list[8], sin_list[8];
    rope_wave(&rope, 0, cos_list, sin_list);
    test_near(cos_list[3], 1.0, 1e-9, "position zero is the identity rotation");
    test_near(sin_list[3], 0.0, 1e-9, "position zero has no sine term");
    rope_wave(&rope, 2, cos_list, sin_list);
    test_near(cos_list[0], cos(2.0), 1e-6, "rope_wave scales by the position");
  }
  mem_free(rope.step_list);

  memset(&rope, 0, sizeof(rope));
  rope.theta_value = 1000000.0;
  rope.part_share = 0.25;
  rope_build(&rope, head_size, "proportional");
  test_true(rope.half_count == head_size / 2, "proportional rope keeps the full head");
  test_true(rope.turn_count == 2, "proportional rope rotates a quarter of the head");
  test_near(rope.step_list[2], 0.0, 0.0, "unrotated pairs have a zero frequency");
  {
    float cos_list[8], sin_list[8];
    rope_wave(&rope, 5, cos_list, sin_list);
    test_near(cos_list[5], 1.0, 1e-9, "unrotated pairs stay identity");
    test_near(sin_list[5], 0.0, 1e-9, "unrotated pairs have no sine term");
  }
  mem_free(rope.step_list);
}

/* ======================================================================== */
/* 7. tokenizer                                                             */
/* ======================================================================== */

static const char test_token_json[] =
    "{\"normalizer\":{\"type\":\"Prepend\",\"prepend\":\"\\u2581\"},"
    "\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"\\u2581\"},"
    "\"added_tokens\":["
    "{\"id\":0,\"content\":\"<pad>\",\"special\":true},"
    "{\"id\":1,\"content\":\"<bos>\",\"special\":true},"
    "{\"id\":2,\"content\":\"<eos>\",\"special\":true},"
    "{\"id\":3,\"content\":\"<start_of_turn>\",\"special\":true},"
    "{\"id\":4,\"content\":\"<end_of_turn>\",\"special\":true}],"
    "\"model\":{\"type\":\"BPE\",\"vocab\":{"
    "\"<pad>\":0,\"<bos>\":1,\"<eos>\":2,\"<start_of_turn>\":3,\"<end_of_turn>\":4,"
    "\"<0x0A>\":5,"
    "\"\\u2581\":6,\"h\":7,\"e\":8,\"l\":9,\"o\":10,\"w\":11,\"r\":12,\"d\":13,"
    "\"he\":14,\"ll\":15,\"hell\":16,\"hello\":17,\"\\u2581hello\":18,"
    "\"\\u2581w\":19,\"\\u2581wo\":20,\"\\u2581wor\":21,\"\\u2581world\":22,\"orld\":23,"
    "\"model\":24,\"user\":25,\"ld\":26},"
    "\"merges\":[\"h e\",\"l l\",\"he ll\",\"hell o\",\"\\u2581 hello\","
    "\"\\u2581 w\",\"o r\",\"l d\",\"or ld\",\"\\u2581w orld\"]}}";

static void test_token(void) {
  token_book *book = NULL;
  test_open("token");
  test_true(test_file_write("tokenizer.json", test_token_json, sizeof(test_token_json) - 1),
            "tokenizer fixture written");
  test_true(token_load(test_yard_path, 0, &book) == APP_OKAY, "token_load reads tokenizer.json");
  if (!book) return;

  test_true(book->start_id == 1, "the start id is recovered");
  test_true(book->close_id == 2, "the close id is recovered");
  test_true(book->turn_open_id == 3, "the turn opener is recovered");
  test_true(book->turn_shut_id == 4, "the turn closer is recovered");

  {
    int32_t id_list[32];
    int id_count = token_encode_book(book, "hello world", 1, id_list, 32);
    test_true(id_count == 2, "the greedy merges reach two tokens");
    if (id_count == 2) {
      test_true(id_list[0] == 18, "the leading word takes the metaspace form");
      test_true(id_list[1] == 22, "the second word merges fully");
    }
  }
  {
    /* The prefix marks the start of a chunk of text, not the start of a call.
     * The reference splits its input on the special tokens and normalizes each
     * chunk on its own, so a piece that continues the chunk before it — the
     * user's words after the role marker's newline — is not marked, and one
     * that follows a special id is. */
    int32_t id_list[32];
    int id_count = token_encode_book(book, "hello", 0, id_list, 32);
    test_true(id_count == 1 && id_list[0] == 17,
              "a piece continuing a chunk is not marked, whatever the normalizer says");
    id_count = token_encode_book(book, "hello", 1, id_list, 32);
    test_true(id_count == 1 && id_list[0] == 18, "a piece beginning a chunk is marked");
  }
  {
    char text_room[64];
    int text_count = token_decode_book(book, 18, text_room, (int)sizeof(text_room));
    test_true(text_count == 6 && strcmp(text_room, " hello") == 0,
              "decode turns the metaspace back into a space");
    text_count = token_decode_book(book, 3, text_room, (int)sizeof(text_room));
    test_true(text_count == 0, "special tokens decode to nothing");
    text_count = token_decode_book(book, 5, text_room, (int)sizeof(text_room));
    test_true(text_count == 1 && text_room[0] == '\n', "byte fallback tokens decode to their byte");
  }
  {
    /* Round trip: every encodable word comes back unchanged. */
    static const char *word_list[] = {"hello world", "world", "hello", NULL};
    int word_index;
    for (word_index = 0; word_list[word_index]; ++word_index) {
      int32_t id_list[32];
      char text_room[128];
      int id_count = token_encode_book(book, word_list[word_index], 1, id_list, 32);
      int id_index, fill_count = 0;
      char want_room[128];
      snprintf(want_room, sizeof(want_room), " %s", word_list[word_index]);
      for (id_index = 0; id_index < id_count; ++id_index)
        fill_count += token_decode_book(book, id_list[id_index], text_room + fill_count,
                                        (int)sizeof(text_room) - fill_count);
      text_room[fill_count] = 0;
      test_true(strcmp(text_room, want_room) == 0, "encode and decode round trip");
    }
  }
  token_free(book);
}

/* ======================================================================== */
/* 7. media layer                                                           */
/* ======================================================================== */

/* The content of the two fixtures below is produced by a formula the test can
 * evaluate for itself, and the bytes were written by an encoder that shares no
 * code with the reader.  That is the point: a recorded output only shows that
 * nothing has changed, while a formula plus a foreign encoder shows that the
 * reader is right. */

/* 400 bytes of `((i*i*7 + i*13) >> 3) & 0x1F`, deflated by an independent
 * compressor into a dynamic Huffman block. */
static const unsigned char test_puff_data[] = {
    0x78, 0xDA, 0xCD, 0x8F, 0xC7, 0x11, 0xC4, 0x20, 0x00, 0x03, 0x4D, 0x36, 0xD9, 0x64, 0x93,
    0xFB, 0xEF, 0xF2, 0x4C, 0x17, 0xF7, 0xDD, 0x19, 0x69, 0xA5, 0x0B, 0x52, 0x15, 0x17, 0xCF,
    0x24, 0xB1, 0xEE, 0x38, 0x9C, 0xED, 0x7D, 0xDB, 0xC2, 0x32, 0x6E, 0x59, 0x79, 0xB7, 0xB8,
    0x47, 0x23, 0xEE, 0x5B, 0x68, 0x5F, 0x81, 0x2C, 0x24, 0xD1, 0x6A, 0xF0, 0xC8, 0xFE, 0xB1,
    0x4F, 0x28, 0x93, 0xD8, 0x7E, 0x97, 0xBB, 0x1D, 0xE2, 0x8C, 0xD6, 0x36, 0x54, 0x20, 0x12,
    0x7C, 0x2E, 0x4F, 0x47, 0x50, 0x0C, 0x01, 0x80, 0x98, 0x0A, 0x83, 0xFA, 0xEB, 0x81, 0x49,
    0x80, 0x1A, 0xAC, 0xD6, 0xC6, 0xE5, 0x81, 0x4D, 0xFB, 0x62, 0xDD, 0x92, 0x59, 0xC2, 0x57,
    0xE4, 0x0F, 0xA9, 0x34, 0x91, 0x22, 0x41, 0xF5, 0xFA, 0xD8, 0x4C, 0xEC, 0xD8, 0x76, 0x5E,
    0xE5, 0x8E, 0x12, 0xAF, 0xB3, 0x68, 0x42, 0xEE, 0x3A, 0x4B, 0x24, 0xF3, 0x15, 0x15, 0x85,
    0xD7, 0xBE, 0x10, 0xD3, 0x69, 0xCB, 0x97, 0xBD, 0xE2, 0x10, 0xB0, 0xE6, 0xDC, 0x88, 0xBB,
    0x46, 0x8E, 0x2D, 0x2B, 0x34, 0x4A, 0x70, 0x2E, 0x94, 0x81, 0x54, 0x3E, 0x8B, 0x48, 0x73,
    0x1C, 0xED, 0x39, 0x17, 0x38, 0x6F, 0xC5, 0x17, 0x93, 0x3B, 0x69, 0x86, 0xBE, 0xA2, 0x3F,
    0xFB, 0xFF, 0x03, 0xE2, 0xD2, 0x18, 0x0B,
};

/* A 12x8 eight bit RGB png whose pixel (x,y) is
 * (x*17 + y*5, x*3 + y*29, x*x + y*y) modulo 256, written by an independent
 * encoder with the five line filters used in turn so that undoing each of
 * them is exercised. */
static const unsigned char test_png_data[] = {
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
    0x52, 0x00, 0x00, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x08, 0x08, 0x02, 0x00, 0x00, 0x00, 0x42,
    0x86, 0x89, 0xA6, 0x00, 0x00, 0x00, 0xBD, 0x49, 0x44, 0x41, 0x54, 0x78, 0xDA, 0x63, 0x60,
    0x60, 0x60, 0x10, 0x64, 0x66, 0x54, 0x62, 0x63, 0x31, 0xE6, 0xE4, 0x74, 0xE1, 0x11, 0x08,
    0xE5, 0x97, 0x4C, 0x13, 0x52, 0x29, 0x17, 0x35, 0xEC, 0x90, 0x70, 0x98, 0x29, 0x1D, 0xB8,
    0x4A, 0x2E, 0x65, 0xB7, 0x62, 0x25, 0x23, 0xAB, 0x2C, 0x23, 0x50, 0x91, 0x20, 0x33, 0xB3,
    0x20, 0x33, 0xAB, 0x20, 0x33, 0xBB, 0x20, 0x33, 0xA7, 0x20, 0x33, 0xB7, 0x20, 0x33, 0xAF,
    0x20, 0x33, 0xBF, 0x20, 0xB3, 0xA0, 0x20, 0xB3, 0xB0, 0x20, 0xB3, 0x28, 0x13, 0xAB, 0x2C,
    0x33, 0x41, 0xC4, 0xCC, 0x65, 0xC5, 0xCE, 0x2D, 0xC0, 0xCC, 0x2D, 0xC0, 0xC2, 0x2D, 0xC0,
    0xCA, 0x2D, 0xC0, 0xC6, 0x2D, 0x00, 0xE4, 0x72, 0x70, 0x0B, 0x70, 0x72, 0x0B, 0x70, 0x71,
    0x0B, 0x70, 0x73, 0x0B, 0xF0, 0x70, 0x0B, 0xF0, 0xB2, 0xB0, 0xCA, 0xB2, 0xB3, 0x32, 0x33,
    0xB2, 0x32, 0x33, 0xB3, 0x32, 0xB3, 0xB2, 0x32, 0xB3, 0x63, 0x45, 0x0C, 0x92, 0x13, 0x25,
    0xB5, 0xA6, 0x48, 0x59, 0x4F, 0x97, 0xF5, 0x99, 0xA5, 0x14, 0x3B, 0x57, 0x33, 0x6F, 0x81,
    0x51, 0xFD, 0x62, 0xDB, 0x09, 0xCB, 0xBC, 0x16, 0xAE, 0x8C, 0xDC, 0xB4, 0x26, 0xEB, 0xF0,
    0xFA, 0xDA, 0x2B, 0x9B, 0x26, 0x31, 0xCA, 0xAD, 0x53, 0x21, 0xC6, 0xE1, 0xBC, 0x04, 0x11,
    0x00, 0x45, 0x1C, 0x21, 0xB1, 0x62, 0xA3, 0x76, 0xD4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
    0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
};

static void test_puff(void) {
  uint8_t *plain_room;
  uint8_t stored_data[64];
  uint8_t stored_out[32];
  long wrote_count;
  int slot_index, okay_flag = 1;
  test_open("puff");

  /* A stored block, assembled here rather than by a compressor: the zlib
   * header, one final uncompressed block, its length and the complement. */
  stored_data[0] = 0x78;
  stored_data[1] = 0x01;
  stored_data[2] = 0x01; /* final, stored */
  stored_data[3] = 5;
  stored_data[4] = 0;
  stored_data[5] = (uint8_t)~5;
  stored_data[6] = 0xFF;
  for (slot_index = 0; slot_index < 5; ++slot_index)
    stored_data[7 + slot_index] = (uint8_t)('a' + slot_index);
  wrote_count = puff_run(stored_data, 12, stored_out, sizeof(stored_out));
  test_true(wrote_count == 5, "a stored block inflates to its own length");
  test_true(wrote_count == 5 && memcmp(stored_out, "abcde", 5) == 0,
            "a stored block inflates to its own bytes");

  plain_room = (uint8_t *)mem_clear(512);
  if (!plain_room) { test_true(0, "the inflate room is allocated"); return; }
  wrote_count = puff_run(test_puff_data, sizeof(test_puff_data), plain_room, 512);
  test_true(wrote_count == 400, "a dynamic Huffman block inflates to the right length");
  for (slot_index = 0; slot_index < 400 && wrote_count == 400; ++slot_index)
    if (plain_room[slot_index] !=
        (uint8_t)((((slot_index * slot_index * 7) + slot_index * 13) >> 3) & 0x1F))
      okay_flag = 0;
  test_true(okay_flag, "a dynamic Huffman block inflates to the right bytes");

  /* A stream that stops in the middle has to be refused rather than returning
   * whatever it managed, because the caller sizes its buffer from the header
   * and would read the rest as image data. */
  test_true(puff_run(test_puff_data, sizeof(test_puff_data) / 2, plain_room, 512) < 0,
            "a truncated stream is refused");
  test_true(puff_run(test_puff_data, sizeof(test_puff_data), plain_room, 16) < 0,
            "a stream that overruns its room is refused");
  mem_free(plain_room);
}

static int test_image_png_want(int wide_index, int high_index, int band_index) {
  switch (band_index) {
    case 0: return (wide_index * 17 + high_index * 5) & 0xFF;
    case 1: return (wide_index * 3 + high_index * 29) & 0xFF;
    default: return (wide_index * wide_index + high_index * high_index) & 0xFF;
  }
}

/* A binary portable pixmap and a bottom-up 24 bit bitmap of the same picture,
 * so that the three readers can be held to one answer. */
static int test_image_pnm_write(int wide_count, int high_count) {
  uint8_t *file_data;
  size_t head_size, file_size;
  char head_text[64];
  int high_index, wide_index, band_index, okay_flag;
  snprintf(head_text, sizeof(head_text), "P6\n# a comment\n%d %d\n255\n", wide_count, high_count);
  head_size = strlen(head_text);
  file_size = head_size + (size_t)wide_count * (size_t)high_count * 3u;
  file_data = (uint8_t *)mem_clear(file_size);
  if (!file_data) return 0;
  memcpy(file_data, head_text, head_size);
  for (high_index = 0; high_index < high_count; ++high_index)
    for (wide_index = 0; wide_index < wide_count; ++wide_index)
      for (band_index = 0; band_index < 3; ++band_index)
        file_data[head_size +
                  ((size_t)high_index * (size_t)wide_count + (size_t)wide_index) * 3u +
                  (size_t)band_index] =
            (uint8_t)test_image_png_want(wide_index, high_index, band_index);
  okay_flag = test_file_write("image.pnm", file_data, file_size);
  mem_free(file_data);
  return okay_flag;
}

static int test_image_bmp_write(int wide_count, int high_count) {
  size_t line_bytes = (((size_t)wide_count * 3u + 3u) / 4u) * 4u;
  size_t file_size = 54u + line_bytes * (size_t)high_count;
  uint8_t *file_data = (uint8_t *)mem_clear(file_size);
  int high_index, wide_index, okay_flag;
  if (!file_data) return 0;
  file_data[0] = 'B';
  file_data[1] = 'M';
  file_data[10] = 54;
  file_data[14] = 40;
  file_data[18] = (uint8_t)(wide_count & 0xFF);
  file_data[19] = (uint8_t)((wide_count >> 8) & 0xFF);
  file_data[22] = (uint8_t)(high_count & 0xFF);
  file_data[23] = (uint8_t)((high_count >> 8) & 0xFF);
  file_data[26] = 1;
  file_data[28] = 24;
  for (high_index = 0; high_index < high_count; ++high_index) {
    /* The first row of the file is the last row of the picture. */
    uint8_t *line_data = file_data + 54u + (size_t)(high_count - 1 - high_index) * line_bytes;
    for (wide_index = 0; wide_index < wide_count; ++wide_index) {
      line_data[wide_index * 3 + 2] = (uint8_t)test_image_png_want(wide_index, high_index, 0);
      line_data[wide_index * 3 + 1] = (uint8_t)test_image_png_want(wide_index, high_index, 1);
      line_data[wide_index * 3 + 0] = (uint8_t)test_image_png_want(wide_index, high_index, 2);
    }
  }
  okay_flag = test_file_write("image.bmp", file_data, file_size);
  mem_free(file_data);
  return okay_flag;
}

static void test_image_check(const char *leaf_text, const char *claim_text, int wide_count,
                             int high_count) {
  char path_text[1024];
  flat_grid grid;
  int high_index, wide_index, band_index, okay_flag = 1;
  path_join(path_text, sizeof(path_text), test_yard_path, leaf_text);
  if (image_read(path_text, &grid) != APP_OKAY) {
    test_true(0, claim_text);
    return;
  }
  if (grid.wide_count != wide_count || grid.high_count != high_count || grid.band_count != 3)
    okay_flag = 0;
  for (high_index = 0; okay_flag && high_index < high_count; ++high_index)
    for (wide_index = 0; wide_index < wide_count; ++wide_index) {
      const float *cell_data = grid_at(&grid, high_index, wide_index);
      for (band_index = 0; band_index < 3; ++band_index) {
        float want_value = (float)test_image_png_want(wide_index, high_index, band_index) / 255.0f;
        float gap_value = cell_data[band_index] - want_value;
        if (gap_value < 0.0f) gap_value = -gap_value;
        if (gap_value > 1e-6f) okay_flag = 0;
      }
    }
  test_true(okay_flag, claim_text);
  grid_free(&grid);
}

static void test_image(void) {
  static const int wide_count = 12, high_count = 8;
  flat_grid grid;
  char path_text[1024];
  test_open("image");
  test_true(test_file_write("image.png", test_png_data, sizeof(test_png_data)),
            "the png fixture is written");
  test_true(test_image_pnm_write(wide_count, high_count), "the pnm fixture is written");
  test_true(test_image_bmp_write(wide_count, high_count), "the bmp fixture is written");
  test_image_check("image.png", "a png decodes to the pixels it was built from", wide_count,
                   high_count);
  test_image_check("image.pnm", "a pnm decodes to the same pixels", wide_count, high_count);
  test_image_check("image.bmp", "a bmp decodes to the same pixels", wide_count, high_count);

  /* A file the reader does not recognise is refused rather than read as noise. */
  test_true(test_file_write("image.bad", "not a picture at all", 20), "a decoy file is written");
  path_join(path_text, sizeof(path_text), test_yard_path, "image.bad");
  test_true(image_read(path_text, &grid) != APP_OKAY, "an unknown container is refused");
  path_join(path_text, sizeof(path_text), test_yard_path, "no_such_image.png");
  test_true(image_read(path_text, &grid) != APP_OKAY, "a missing file is refused");
}

/* An independent png writer, so the reader's wider range is checked against the
 * format rather than against files it was developed on.  It shares nothing with
 * the reader: its own chunk framing and check values, its own line filters
 * applied forward from the definitions, its own bit packing, and a deflate
 * stream of stored blocks — which is a compressor the test does not need to
 * have. */
typedef struct test_png_room {
  uint8_t *file_data;
  size_t   file_fill;
  size_t   file_room;
  int      fault_flag;
} test_png_room;

static void test_png_byte(test_png_room *room, int byte_value) {
  if (room->file_fill >= room->file_room) { room->fault_flag = 1; return; }
  room->file_data[room->file_fill++] = (uint8_t)byte_value;
}

static void test_png_word(test_png_room *room, uint32_t word_value) {
  test_png_byte(room, (int)((word_value >> 24) & 0xFF));
  test_png_byte(room, (int)((word_value >> 16) & 0xFF));
  test_png_byte(room, (int)((word_value >> 8) & 0xFF));
  test_png_byte(room, (int)(word_value & 0xFF));
}

static uint32_t test_png_crc(const uint8_t *data, size_t byte_count) {
  uint32_t value_now = 0xFFFFFFFFu;
  size_t slot_index;
  int bit_index;
  for (slot_index = 0; slot_index < byte_count; ++slot_index) {
    value_now ^= data[slot_index];
    for (bit_index = 0; bit_index < 8; ++bit_index)
      value_now = (value_now >> 1) ^ (0xEDB88320u & (uint32_t)(0u - (value_now & 1u)));
  }
  return value_now ^ 0xFFFFFFFFu;
}

static uint32_t test_png_adler(const uint8_t *data, size_t byte_count) {
  uint32_t low_sum = 1, high_sum = 0;
  size_t slot_index;
  for (slot_index = 0; slot_index < byte_count; ++slot_index) {
    low_sum = (low_sum + data[slot_index]) % 65521u;
    high_sum = (high_sum + low_sum) % 65521u;
  }
  return (high_sum << 16) | low_sum;
}

static void test_png_chunk(test_png_room *room, const char *name_text, const uint8_t *body_data,
                           size_t body_size) {
  uint8_t *name_from;
  size_t slot_index;
  test_png_word(room, (uint32_t)body_size);
  name_from = room->file_data + room->file_fill;
  for (slot_index = 0; slot_index < 4; ++slot_index) test_png_byte(room, name_text[slot_index]);
  for (slot_index = 0; slot_index < body_size; ++slot_index) test_png_byte(room, body_data[slot_index]);
  if (room->fault_flag) return;
  test_png_word(room, test_png_crc(name_from, body_size + 4u));
}

/* The sample a pixel carries, and the palette a slot names.  Both are formulas
 * so the check can restate them without holding the picture. */
static int test_png_level(int wide_index, int high_index, int band_index, int deep_count) {
  int span_value = deep_count >= 16 ? 65535 : (1 << deep_count) - 1;
  int seed_value = wide_index * 7 + high_index * 13 + band_index * 29;
  return (seed_value * 2654435761u) % (unsigned)(span_value + 1);
}

static int test_png_shade(int slot_index, int band_index) {
  return (slot_index * (11 + band_index * 9) + band_index * 37) & 0xFF;
}

static void test_png_pack(uint8_t *row_data, int slot_index, int deep_count, int value_now) {
  if (deep_count == 16) {
    row_data[(size_t)slot_index * 2] = (uint8_t)((value_now >> 8) & 0xFF);
    row_data[(size_t)slot_index * 2 + 1] = (uint8_t)(value_now & 0xFF);
    return;
  }
  if (deep_count == 8) {
    row_data[slot_index] = (uint8_t)value_now;
    return;
  }
  {
    int per_byte = 8 / deep_count;
    int shift_count = 8 - deep_count * (slot_index % per_byte + 1);
    row_data[slot_index / per_byte] |= (uint8_t)((value_now & ((1 << deep_count) - 1)) << shift_count);
  }
}

/* The forward filters, from the definitions rather than from the reader's
 * inverse: each byte less a prediction from the byte to its left, the one above
 * it, and the one above that one. */
static int test_png_guess(int rule_mark, int left_value, int over_value, int corner_value) {
  int near_value, gap_left, gap_over, gap_corner;
  switch (rule_mark) {
    case 1: return left_value;
    case 2: return over_value;
    case 3: return (left_value + over_value) / 2;
    case 4:
      near_value = left_value + over_value - corner_value;
      gap_left = near_value > left_value ? near_value - left_value : left_value - near_value;
      gap_over = near_value > over_value ? near_value - over_value : over_value - near_value;
      gap_corner = near_value > corner_value ? near_value - corner_value : corner_value - near_value;
      if (gap_left <= gap_over && gap_left <= gap_corner) return left_value;
      return gap_over <= gap_corner ? over_value : corner_value;
    default: return 0;
  }
}

static int test_png_write(const char *leaf_text, int wide_count, int high_count, int deep_count,
                          int kind_mark, int weave_mark) {
  static const uint8_t mark_list[8] = {137, 'P', 'N', 'G', 13, 10, 26, 10};
  static const uint8_t from_x_list[7] = {0, 4, 0, 2, 0, 1, 0};
  static const uint8_t from_y_list[7] = {0, 0, 4, 0, 2, 0, 1};
  static const uint8_t step_x_list[7] = {8, 8, 4, 4, 2, 2, 1};
  static const uint8_t step_y_list[7] = {8, 8, 8, 4, 4, 2, 2};
  test_png_room room;
  uint8_t head_list[13], *raw_data, *plain_data, *last_data, *this_data;
  size_t raw_room, raw_fill = 0, line_room, walk;
  int lane_count = kind_mark == 2 ? 3 : (kind_mark == 6 ? 4 : (kind_mark == 4 ? 2 : 1));
  int step_bytes = lane_count * deep_count / 8;
  int pass_count = weave_mark ? 7 : 1, pass_index, okay_flag, slot_index;

  if (step_bytes < 1) step_bytes = 1;
  memset(&room, 0, sizeof(room));
  room.file_room = 4096u + (size_t)wide_count * (size_t)high_count * (size_t)lane_count * 8u;
  room.file_data = (uint8_t *)mem_clear(room.file_room);
  line_room = ((size_t)wide_count * (size_t)lane_count * (size_t)deep_count + 7u) / 8u + 1u;
  raw_room = (line_room + 1u) * (size_t)high_count + 64u;
  raw_data = (uint8_t *)mem_clear(raw_room);
  last_data = (uint8_t *)mem_clear(line_room);
  this_data = (uint8_t *)mem_clear(line_room);
  plain_data = (uint8_t *)mem_clear(line_room);
  if (!room.file_data || !raw_data || !last_data || !this_data || !plain_data) {
    mem_free(room.file_data);
    mem_free(raw_data);
    mem_free(last_data);
    mem_free(this_data);
    mem_free(plain_data);
    return 0;
  }

  /* The raw stream: every lattice in turn, every row of it filtered against the
   * row above it inside that lattice alone. */
  for (pass_index = 0; pass_index < pass_count; ++pass_index) {
    int from_x = weave_mark ? from_x_list[pass_index] : 0;
    int from_y = weave_mark ? from_y_list[pass_index] : 0;
    int step_x = weave_mark ? step_x_list[pass_index] : 1;
    int step_y = weave_mark ? step_y_list[pass_index] : 1;
    int pass_wide = (wide_count - from_x + step_x - 1) / step_x;
    int pass_high = (high_count - from_y + step_y - 1) / step_y;
    size_t pass_bytes = ((size_t)pass_wide * (size_t)lane_count * (size_t)deep_count + 7u) / 8u;
    int high_index, wide_index, band_index;
    if (pass_wide < 1 || pass_high < 1) continue;
    memset(last_data, 0, line_room);
    for (high_index = 0; high_index < pass_high; ++high_index) {
      int rule_mark = (high_index + pass_index) % 5;
      memset(plain_data, 0, line_room);
      for (wide_index = 0; wide_index < pass_wide; ++wide_index)
        for (band_index = 0; band_index < lane_count; ++band_index) {
          int level_value =
              kind_mark == 3
                  ? test_png_level(from_x + wide_index * step_x, from_y + high_index * step_y, 0,
                                   deep_count)
                  : test_png_level(from_x + wide_index * step_x, from_y + high_index * step_y,
                                   band_index, deep_count);
          test_png_pack(plain_data, wide_index * lane_count + band_index, deep_count, level_value);
        }
      for (slot_index = 0; slot_index < (int)pass_bytes; ++slot_index) {
        int left_value = slot_index >= step_bytes ? plain_data[slot_index - step_bytes] : 0;
        int corner_value = slot_index >= step_bytes ? last_data[slot_index - step_bytes] : 0;
        this_data[slot_index] =
            (uint8_t)((plain_data[slot_index] -
                       test_png_guess(rule_mark, left_value, last_data[slot_index], corner_value)) &
                      0xFF);
      }
      raw_data[raw_fill++] = (uint8_t)rule_mark;
      memcpy(raw_data + raw_fill, this_data, pass_bytes);
      raw_fill += pass_bytes;
      memcpy(last_data, plain_data, pass_bytes);
    }
  }

  for (slot_index = 0; slot_index < 8; ++slot_index) test_png_byte(&room, mark_list[slot_index]);
  head_list[0] = (uint8_t)((wide_count >> 24) & 0xFF);
  head_list[1] = (uint8_t)((wide_count >> 16) & 0xFF);
  head_list[2] = (uint8_t)((wide_count >> 8) & 0xFF);
  head_list[3] = (uint8_t)(wide_count & 0xFF);
  head_list[4] = (uint8_t)((high_count >> 24) & 0xFF);
  head_list[5] = (uint8_t)((high_count >> 16) & 0xFF);
  head_list[6] = (uint8_t)((high_count >> 8) & 0xFF);
  head_list[7] = (uint8_t)(high_count & 0xFF);
  head_list[8] = (uint8_t)deep_count;
  head_list[9] = (uint8_t)kind_mark;
  head_list[10] = 0;
  head_list[11] = 0;
  head_list[12] = (uint8_t)weave_mark;
  test_png_chunk(&room, "IHDR", head_list, sizeof(head_list));
  if (kind_mark == 3) {
    uint8_t shade_list[768];
    int slot_count = 1 << deep_count;
    for (slot_index = 0; slot_index < slot_count; ++slot_index) {
      shade_list[slot_index * 3] = (uint8_t)test_png_shade(slot_index, 0);
      shade_list[slot_index * 3 + 1] = (uint8_t)test_png_shade(slot_index, 1);
      shade_list[slot_index * 3 + 2] = (uint8_t)test_png_shade(slot_index, 2);
    }
    test_png_chunk(&room, "PLTE", shade_list, (size_t)slot_count * 3u);
  }
  {
    /* A zlib stream of stored blocks: the two header bytes, then the raw
     * stream in runs of at most a block, then the adler sum. */
    uint8_t *pack_data = (uint8_t *)mem_clear(raw_fill + raw_fill / 65535u * 5u + 64u);
    size_t pack_fill = 0;
    if (!pack_data) {
      mem_free(room.file_data);
      mem_free(raw_data);
      mem_free(last_data);
      mem_free(this_data);
      mem_free(plain_data);
      return 0;
    }
    pack_data[pack_fill++] = 0x78;
    pack_data[pack_fill++] = 0x01;
    walk = 0;
    do {
      size_t span = raw_fill - walk > 65535u ? 65535u : raw_fill - walk;
      pack_data[pack_fill++] = (uint8_t)(walk + span >= raw_fill ? 1 : 0);
      pack_data[pack_fill++] = (uint8_t)(span & 0xFF);
      pack_data[pack_fill++] = (uint8_t)((span >> 8) & 0xFF);
      pack_data[pack_fill++] = (uint8_t)(~span & 0xFF);
      pack_data[pack_fill++] = (uint8_t)((~span >> 8) & 0xFF);
      memcpy(pack_data + pack_fill, raw_data + walk, span);
      pack_fill += span;
      walk += span;
    } while (walk < raw_fill);
    {
      uint32_t sum_value = test_png_adler(raw_data, raw_fill);
      pack_data[pack_fill++] = (uint8_t)((sum_value >> 24) & 0xFF);
      pack_data[pack_fill++] = (uint8_t)((sum_value >> 16) & 0xFF);
      pack_data[pack_fill++] = (uint8_t)((sum_value >> 8) & 0xFF);
      pack_data[pack_fill++] = (uint8_t)(sum_value & 0xFF);
    }
    test_png_chunk(&room, "IDAT", pack_data, pack_fill);
    mem_free(pack_data);
  }
  test_png_chunk(&room, "IEND", NULL, 0);

  okay_flag = !room.fault_flag && test_file_write(leaf_text, room.file_data, room.file_fill);
  mem_free(room.file_data);
  mem_free(raw_data);
  mem_free(last_data);
  mem_free(this_data);
  mem_free(plain_data);
  return okay_flag;
}

static void test_png_check(const char *leaf_text, const char *claim_text, int wide_count,
                           int high_count, int deep_count, int kind_mark) {
  char path_text[1024];
  flat_grid grid;
  int band_want = (kind_mark == 2 || kind_mark == 6 || kind_mark == 3) ? 3 : 1;
  int high_index, wide_index, band_index, okay_flag = 1;
  float level_span = deep_count >= 16 ? 65535.0f : (float)((1 << deep_count) - 1);
  path_join(path_text, sizeof(path_text), test_yard_path, leaf_text);
  if (image_read(path_text, &grid) != APP_OKAY) {
    test_true(0, claim_text);
    return;
  }
  if (grid.wide_count != wide_count || grid.high_count != high_count ||
      grid.band_count != band_want)
    okay_flag = 0;
  for (high_index = 0; okay_flag && high_index < high_count; ++high_index)
    for (wide_index = 0; wide_index < wide_count; ++wide_index) {
      const float *cell_data = grid_at(&grid, high_index, wide_index);
      for (band_index = 0; band_index < band_want; ++band_index) {
        float want_value;
        if (kind_mark == 3)
          want_value =
              (float)test_png_shade(test_png_level(wide_index, high_index, 0, deep_count),
                                    band_index) /
              255.0f;
        else
          want_value =
              (float)test_png_level(wide_index, high_index, band_index, deep_count) / level_span;
        if (cell_data[band_index] - want_value > 1e-6f ||
            want_value - cell_data[band_index] > 1e-6f)
          okay_flag = 0;
      }
    }
  test_true(okay_flag, claim_text);
  grid_free(&grid);
}

/* The depths and the two lattices, over sizes chosen so that a lattice's last
 * row is a partial byte and the smaller passes are empty. */
static void test_png_wide(void) {
  static const int deep_list[5] = {1, 2, 4, 8, 16};
  char leaf_text[64], claim_text[128];
  int deep_index, weave_mark;
  test_open("png");

  for (deep_index = 0; deep_index < 5; ++deep_index)
    for (weave_mark = 0; weave_mark < 2; ++weave_mark) {
      int deep_count = deep_list[deep_index];
      snprintf(leaf_text, sizeof(leaf_text), "grey%d_%d.png", deep_count, weave_mark);
      snprintf(claim_text, sizeof(claim_text), "%d bit grey%s decodes to its samples", deep_count,
               weave_mark ? ", interlaced" : "");
      if (!test_png_write(leaf_text, 13, 11, deep_count, 0, weave_mark)) {
        test_true(0, claim_text);
        continue;
      }
      test_png_check(leaf_text, claim_text, 13, 11, deep_count, 0);
    }

  for (deep_index = 0; deep_index < 4; ++deep_index)
    for (weave_mark = 0; weave_mark < 2; ++weave_mark) {
      int deep_count = deep_list[deep_index];
      snprintf(leaf_text, sizeof(leaf_text), "slot%d_%d.png", deep_count, weave_mark);
      snprintf(claim_text, sizeof(claim_text), "%d bit palette%s decodes through its table",
               deep_count, weave_mark ? ", interlaced" : "");
      if (!test_png_write(leaf_text, 13, 11, deep_count, 3, weave_mark)) {
        test_true(0, claim_text);
        continue;
      }
      test_png_check(leaf_text, claim_text, 13, 11, deep_count, 3);
    }

  /* The wider kinds interlaced, and a size where four of the seven lattices
   * carry nothing at all. */
  test_true(test_png_write("rgb8i.png", 13, 11, 8, 2, 1), "an interlaced rgb fixture is written");
  test_png_check("rgb8i.png", "an interlaced rgb file decodes to its pixels", 13, 11, 8, 2);
  test_true(test_png_write("rgba16i.png", 13, 11, 16, 6, 1),
            "an interlaced sixteen bit rgba fixture is written");
  test_png_check("rgba16i.png", "an interlaced sixteen bit rgba file drops its alpha", 13, 11, 16,
                 6);
  test_true(test_png_write("tiny.png", 3, 2, 4, 3, 1), "a fixture smaller than the lattice is written");
  test_png_check("tiny.png", "a picture smaller than the lattice decodes from the passes that "
                             "carry anything",
                 3, 2, 4, 3);
  test_true(test_png_write("one.png", 1, 1, 1, 0, 1), "a one pixel interlaced fixture is written");
  test_png_check("one.png", "a one pixel interlaced file is one pass of one sample", 1, 1, 1, 0);
}

/* An independent jpeg encoder, written here so that the reader is checked
 * against the format rather than against a recorded file.  It shares nothing
 * with the reader: its own forward transform in double precision, its own
 * canonical code assignment, its own bit writer.  It writes sequential frames
 * and progressive ones, at eight or twelve bits, over one to four components.
 *
 * Its quantization tables are all ones, so a round trip through it loses only
 * what the two transforms round, and the pixels that come back can be held to
 * within a level or two of the pixels that went in.  Its Huffman table is
 * deliberately not the specification's: lengths of eight to twelve bits over
 * all 256 symbols, which is an incomplete code — legal, and a shape no encoder
 * in the wild produces — so the reader's walk is exercised rather than a table
 * it might have been written around. */
typedef struct test_jpeg_room {
  uint8_t *file_data;
  size_t   file_fill;
  size_t   file_room;
  uint32_t bit_room;
  int      bit_count;
  int      fault_flag;
} test_jpeg_room;

static void test_jpeg_byte(test_jpeg_room *room, int byte_value) {
  if (room->file_fill >= room->file_room) { room->fault_flag = 1; return; }
  room->file_data[room->file_fill++] = (uint8_t)byte_value;
}

static void test_jpeg_word(test_jpeg_room *room, int word_value) {
  test_jpeg_byte(room, (word_value >> 8) & 0xFF);
  test_jpeg_byte(room, word_value & 0xFF);
}

/* Bits land most significant first, and a byte that comes out `FF` is followed
 * by a zero so that no marker can appear inside the entropy stream. */
static void test_jpeg_bits(test_jpeg_room *room, int code_value, int bit_count) {
  int bit_index;
  for (bit_index = bit_count - 1; bit_index >= 0; --bit_index) {
    room->bit_room = (room->bit_room << 1) | (uint32_t)((code_value >> bit_index) & 1);
    room->bit_count += 1;
    if (room->bit_count == 8) {
      int byte_value = (int)(room->bit_room & 0xFFu);
      test_jpeg_byte(room, byte_value);
      if (byte_value == 0xFF) test_jpeg_byte(room, 0x00);
      room->bit_count = 0;
      room->bit_room = 0;
    }
  }
}

static void test_jpeg_flush(test_jpeg_room *room) {
  while (room->bit_count != 0) test_jpeg_bits(room, 1, 1); /* the padding is ones */
}

/* The table the encoder uses, in the two forms it needs it: the lengths and
 * symbol order a `DHT` segment carries, and the code per symbol. */
typedef struct test_jpeg_code {
  int length_list[256];
  int code_list[256];
  int count_list[17];
} test_jpeg_code;

static void test_jpeg_code_build(test_jpeg_code *table) {
  int sign_index, length_index, code_value = 0;
  for (sign_index = 0; sign_index < 256; ++sign_index)
    table->length_list[sign_index] = 8 + (sign_index % 5);
  for (length_index = 0; length_index <= 16; ++length_index) table->count_list[length_index] = 0;
  for (sign_index = 0; sign_index < 256; ++sign_index)
    table->count_list[table->length_list[sign_index]] += 1;
  for (length_index = 1; length_index <= 16; ++length_index) {
    for (sign_index = 0; sign_index < 256; ++sign_index)
      if (table->length_list[sign_index] == length_index) table->code_list[sign_index] = code_value++;
    code_value <<= 1;
  }
}

static void test_jpeg_sign(test_jpeg_room *room, const test_jpeg_code *table, int sign_value) {
  test_jpeg_bits(room, table->code_list[sign_value], table->length_list[sign_value]);
}

/* How many bits a coefficient needs, and the bits themselves — a negative
 * value written as the ones complement of its magnitude, which is what the
 * format's extension rule means. */
static int test_jpeg_size(int value_now) {
  int size_value = 0, size_abs = value_now < 0 ? -value_now : value_now;
  while (size_abs) { size_value += 1; size_abs >>= 1; }
  return size_value;
}

static void test_jpeg_value(test_jpeg_room *room, int value_now, int size_value) {
  if (size_value == 0) return;
  if (value_now < 0) value_now += (1 << size_value) - 1;
  test_jpeg_bits(room, value_now, size_value);
}

/* The two point transforms a progressive scan sends a coefficient through: the
 * dc one is a division that floors, the ac one a division that truncates
 * towards zero.  Written as divisions rather than as shifts because a shift of
 * a negative is the host's business and these two have to differ. */
static int test_jpeg_floor(int value_now, int low_bit) {
  int step_value = 1 << low_bit;
  return value_now >= 0 ? value_now / step_value
                        : -((-value_now + step_value - 1) / step_value);
}

static int test_jpeg_trim(int value_now, int low_bit) {
  int step_value = 1 << low_bit;
  return value_now >= 0 ? value_now / step_value : -((-value_now) / step_value);
}

/* The forward transform, gathered directly from the definition rather than
 * through any factorization. */
static void test_jpeg_turn(const int *cell_list, double *coef_out) {
  int freq_wide, freq_high, slot_wide, slot_high;
  for (freq_high = 0; freq_high < 8; ++freq_high)
    for (freq_wide = 0; freq_wide < 8; ++freq_wide) {
      double total = 0.0;
      double gain_high = freq_high == 0 ? sqrt(0.125) : 0.5;
      double gain_wide = freq_wide == 0 ? sqrt(0.125) : 0.5;
      for (slot_high = 0; slot_high < 8; ++slot_high)
        for (slot_wide = 0; slot_wide < 8; ++slot_wide)
          total += (double)cell_list[slot_high * 8 + slot_wide] *
                   cos((2.0 * slot_wide + 1.0) * freq_wide * 3.14159265358979323846 / 16.0) *
                   cos((2.0 * slot_high + 1.0) * freq_high * 3.14159265358979323846 / 16.0);
      coef_out[freq_high * 8 + freq_wide] = total * gain_wide * gain_high;
    }
}

/* The order the coefficients are sent in, derived rather than copied: the
 * anti-diagonals of the block in turn, each walked the way the one before it
 * was not.  The reader carries the same order as a literal table, and the two
 * are held against each other below. */
static void test_jpeg_zig_fill(int *zig_out) {
  int slot_index = 0, sum_index, step_index;
  for (sum_index = 0; sum_index <= 14; ++sum_index) {
    int from_index = sum_index < 7 ? sum_index : 7;
    int upto_index = sum_index - 7 > 0 ? sum_index - 7 : 0;
    for (step_index = from_index; step_index >= upto_index; --step_index) {
      int high_index = sum_index % 2 == 0 ? step_index : sum_index - step_index;
      int wide_index = sum_index - high_index;
      zig_out[slot_index++] = high_index * 8 + wide_index;
    }
  }
}

/* One cell of one block, as the sample the component carries there: the picture
 * pixels it covers averaged where the component is sampled below the peak, and
 * clamped to the picture at the padded edge.  `lift_value` is what a twelve bit
 * frame scales an eight bit fixture by. */
static int test_jpeg_cell(const uint8_t *band_data, int wide_count, int high_count, int band_count,
                          int part_index, int wide_share, int high_share, int wide_peak,
                          int high_peak, int block_x, int block_y, int slot_wide, int slot_high,
                          int lift_value) {
  int part_x = block_x * 8 + slot_wide;
  int part_y = block_y * 8 + slot_high;
  int step_wide = wide_peak / wide_share, step_high = high_peak / high_share;
  int walk_x, walk_y, total = 0, seen = 0;
  for (walk_y = 0; walk_y < step_high; ++walk_y)
    for (walk_x = 0; walk_x < step_wide; ++walk_x) {
      int pick_x = part_x * step_wide + walk_x, pick_y = part_y * step_high + walk_y;
      if (pick_x >= wide_count) pick_x = wide_count - 1;
      if (pick_y >= high_count) pick_y = high_count - 1;
      total += band_data[((size_t)pick_y * (size_t)wide_count + (size_t)pick_x) *
                             (size_t)band_count + (size_t)part_index];
      seen += 1;
    }
  return (total / seen) * lift_value;
}

/* The header a frame of either kind opens with: the quantization table, the
 * frame itself, one Huffman table serving both classes, an `APP14` where the
 * caller asked for one, and a restart interval where it asked for that. */
static void test_jpeg_head(test_jpeg_room *room, const test_jpeg_code *table, int wide_count,
                           int high_count, int band_count, const int *wide_share_list,
                           const int *high_share_list, int rest_span, int frame_mark,
                           int deep_count, int adobe_turn) {
  int part_index, length_index, sign_index;
  test_jpeg_word(room, 0xFFD8);
  /* Every step a one, at whatever precision the frame's own does not forbid:
   * twelve bit frames are only allowed sixteen bit tables. */
  if (deep_count == 12) {
    test_jpeg_word(room, 0xFFDB);
    test_jpeg_word(room, 131);
    test_jpeg_byte(room, 0x10);
    for (part_index = 0; part_index < 64; ++part_index) test_jpeg_word(room, 1);
  } else {
    test_jpeg_word(room, 0xFFDB);
    test_jpeg_word(room, 67);
    test_jpeg_byte(room, 0x00);
    for (part_index = 0; part_index < 64; ++part_index) test_jpeg_byte(room, 1);
  }
  if (adobe_turn >= 0) {
    test_jpeg_word(room, 0xFFEE);
    test_jpeg_word(room, 14);
    test_jpeg_byte(room, 'A');
    test_jpeg_byte(room, 'd');
    test_jpeg_byte(room, 'o');
    test_jpeg_byte(room, 'b');
    test_jpeg_byte(room, 'e');
    test_jpeg_word(room, 100); /* version */
    test_jpeg_word(room, 0);   /* the two flag words nothing here reads */
    test_jpeg_word(room, 0);
    test_jpeg_byte(room, adobe_turn);
  }
  test_jpeg_word(room, 0xFF00 | frame_mark);
  test_jpeg_word(room, 8 + 3 * band_count);
  test_jpeg_byte(room, deep_count);
  test_jpeg_word(room, high_count);
  test_jpeg_word(room, wide_count);
  test_jpeg_byte(room, band_count);
  for (part_index = 0; part_index < band_count; ++part_index) {
    test_jpeg_byte(room, part_index + 1);
    test_jpeg_byte(room, (wide_share_list[part_index] << 4) | high_share_list[part_index]);
    test_jpeg_byte(room, 0);
  }
  for (part_index = 0; part_index < 2; ++part_index) { /* the same table as dc and as ac */
    test_jpeg_word(room, 0xFFC4);
    test_jpeg_word(room, 19 + 256);
    test_jpeg_byte(room, part_index << 4);
    for (length_index = 1; length_index <= 16; ++length_index)
      test_jpeg_byte(room, table->count_list[length_index]);
    for (length_index = 1; length_index <= 16; ++length_index)
      for (sign_index = 0; sign_index < 256; ++sign_index)
        if (table->length_list[sign_index] == length_index) test_jpeg_byte(room, sign_index);
  }
  if (rest_span > 0) {
    test_jpeg_word(room, 0xFFDD);
    test_jpeg_word(room, 4);
    test_jpeg_word(room, rest_span);
  }
}

/* Writes one sequential file.  `band_count` bands at the picture's own size are
 * handed in as bytes; a component whose sampling factor is below the peak is
 * built by averaging the samples it covers. */
static int test_jpeg_write(const char *leaf_text, const uint8_t *band_data, int wide_count,
                           int high_count, int band_count, const int *wide_share_list,
                           const int *high_share_list, int rest_span, int frame_mark,
                           int deep_count, int adobe_turn) {
  test_jpeg_code table;
  test_jpeg_room room;
  int zig_list[64];
  int wide_peak = 1, high_peak = 1, mcu_wide, mcu_high, mcu_index, part_index;
  int lift_value = deep_count == 12 ? 16 : 1, level_mid = 1 << (deep_count - 1);
  int last_dc[4];
  int okay_flag;
  size_t file_room = 2048u + (size_t)wide_count * (size_t)high_count * (size_t)band_count * 16u;

  memset(&room, 0, sizeof(room));
  memset(last_dc, 0, sizeof(last_dc));
  test_jpeg_code_build(&table);
  test_jpeg_zig_fill(zig_list);
  room.file_room = file_room;
  room.file_data = (uint8_t *)mem_clear(file_room);
  if (!room.file_data) return 0;
  for (part_index = 0; part_index < band_count; ++part_index) {
    if (wide_share_list[part_index] > wide_peak) wide_peak = wide_share_list[part_index];
    if (high_share_list[part_index] > high_peak) high_peak = high_share_list[part_index];
  }
  mcu_wide = (wide_count + wide_peak * 8 - 1) / (wide_peak * 8);
  mcu_high = (high_count + high_peak * 8 - 1) / (high_peak * 8);

  test_jpeg_head(&room, &table, wide_count, high_count, band_count, wide_share_list,
                 high_share_list, rest_span, frame_mark, deep_count, adobe_turn);
  test_jpeg_word(&room, 0xFFDA);
  test_jpeg_word(&room, 6 + 2 * band_count);
  test_jpeg_byte(&room, band_count);
  for (part_index = 0; part_index < band_count; ++part_index) {
    test_jpeg_byte(&room, part_index + 1);
    test_jpeg_byte(&room, 0x00);
  }
  test_jpeg_byte(&room, 0);
  test_jpeg_byte(&room, 63);
  test_jpeg_byte(&room, 0);

  for (mcu_index = 0; mcu_index < mcu_wide * mcu_high; ++mcu_index) {
    int mcu_x = mcu_index % mcu_wide, mcu_y = mcu_index / mcu_wide;
    if (rest_span > 0 && mcu_index > 0 && mcu_index % rest_span == 0) {
      test_jpeg_flush(&room);
      test_jpeg_word(&room, 0xFFD0 | ((mcu_index / rest_span - 1) & 7));
      for (part_index = 0; part_index < band_count; ++part_index) last_dc[part_index] = 0;
    }
    for (part_index = 0; part_index < band_count; ++part_index) {
      int wide_share = wide_share_list[part_index], high_share = high_share_list[part_index];
      int block_x, block_y;
      for (block_y = 0; block_y < high_share; ++block_y)
        for (block_x = 0; block_x < wide_share; ++block_x) {
          int cell_list[64];
          double coef_list[64];
          int slot_wide, slot_high, slot_index, run_value, size_value;
          for (slot_high = 0; slot_high < 8; ++slot_high)
            for (slot_wide = 0; slot_wide < 8; ++slot_wide)
              cell_list[slot_high * 8 + slot_wide] =
                  test_jpeg_cell(band_data, wide_count, high_count, band_count, part_index,
                                 wide_share, high_share, wide_peak, high_peak,
                                 mcu_x * wide_share + block_x, mcu_y * high_share + block_y,
                                 slot_wide, slot_high, lift_value) -
                  level_mid;
          test_jpeg_turn(cell_list, coef_list);
          {
            int coef_whole[64];
            for (slot_index = 0; slot_index < 64; ++slot_index)
              coef_whole[slot_index] = (int)(coef_list[slot_index] < 0.0
                                                 ? coef_list[slot_index] - 0.5
                                                 : coef_list[slot_index] + 0.5);
            size_value = test_jpeg_size(coef_whole[0] - last_dc[part_index]);
            test_jpeg_sign(&room, &table, size_value);
            test_jpeg_value(&room, coef_whole[0] - last_dc[part_index], size_value);
            last_dc[part_index] = coef_whole[0];
            run_value = 0;
            for (slot_index = 1; slot_index < 64; ++slot_index) {
              int coef_value = coef_whole[zig_list[slot_index]];
              if (coef_value == 0) { run_value += 1; continue; }
              while (run_value > 15) {
                test_jpeg_sign(&room, &table, 0xF0);
                run_value -= 16;
              }
              size_value = test_jpeg_size(coef_value);
              test_jpeg_sign(&room, &table, (run_value << 4) | size_value);
              test_jpeg_value(&room, coef_value, size_value);
              run_value = 0;
            }
            if (run_value > 0) test_jpeg_sign(&room, &table, 0x00); /* end of block */
          }
        }
    }
  }
  test_jpeg_flush(&room);
  test_jpeg_word(&room, 0xFFD9);
  okay_flag = !room.fault_flag && test_file_write(leaf_text, room.file_data, room.file_fill);
  mem_free(room.file_data);
  return okay_flag;
}

/* The progressive encoder.  A progressive file sends the whole frame's
 * coefficients several times over — a band of them at a time, and a bit plane
 * at a time within a band — so the encoder holds them all before it writes
 * anything, which is the mirror of the store the reader has to hold. */
typedef struct test_jpeg_hoard {
  int *coef_data[JPEG_PART_LIMIT];
  int  wide_blocks[JPEG_PART_LIMIT], high_blocks[JPEG_PART_LIMIT];
  int  wide_units[JPEG_PART_LIMIT], high_units[JPEG_PART_LIMIT];
} test_jpeg_hoard;

static void test_jpeg_hoard_drop(test_jpeg_hoard *hoard) {
  int part_index;
  for (part_index = 0; part_index < JPEG_PART_LIMIT; ++part_index)
    mem_free(hoard->coef_data[part_index]);
  memset(hoard, 0, sizeof(*hoard));
}

static int test_jpeg_hoard_fill(test_jpeg_hoard *hoard, const uint8_t *band_data, int wide_count,
                            int high_count, int band_count, const int *wide_share_list,
                            const int *high_share_list, int lift_value, int level_mid) {
  int zig_list[64];
  int wide_peak = 1, high_peak = 1, mcu_wide, mcu_high, part_index;
  memset(hoard, 0, sizeof(*hoard));
  test_jpeg_zig_fill(zig_list);
  for (part_index = 0; part_index < band_count; ++part_index) {
    if (wide_share_list[part_index] > wide_peak) wide_peak = wide_share_list[part_index];
    if (high_share_list[part_index] > high_peak) high_peak = high_share_list[part_index];
  }
  mcu_wide = (wide_count + wide_peak * 8 - 1) / (wide_peak * 8);
  mcu_high = (high_count + high_peak * 8 - 1) / (high_peak * 8);
  for (part_index = 0; part_index < band_count; ++part_index) {
    int wide_share = wide_share_list[part_index], high_share = high_share_list[part_index];
    int wide_size = (wide_count * wide_share + wide_peak - 1) / wide_peak;
    int high_size = (high_count * high_share + high_peak - 1) / high_peak;
    int block_x, block_y;
    hoard->wide_blocks[part_index] = mcu_wide * wide_share;
    hoard->high_blocks[part_index] = mcu_high * high_share;
    hoard->wide_units[part_index] = (wide_size + 7) / 8;
    hoard->high_units[part_index] = (high_size + 7) / 8;
    hoard->coef_data[part_index] = (int *)mem_clear(
        sizeof(int) * (size_t)hoard->wide_blocks[part_index] *
        (size_t)hoard->high_blocks[part_index] * 64u);
    if (!hoard->coef_data[part_index]) return 0;
    for (block_y = 0; block_y < hoard->high_blocks[part_index]; ++block_y)
      for (block_x = 0; block_x < hoard->wide_blocks[part_index]; ++block_x) {
        int cell_list[64], slot_wide, slot_high, slot_index;
        double coef_list[64];
        int *into_data = hoard->coef_data[part_index] +
                         ((size_t)block_y * (size_t)hoard->wide_blocks[part_index] +
                          (size_t)block_x) * 64u;
        for (slot_high = 0; slot_high < 8; ++slot_high)
          for (slot_wide = 0; slot_wide < 8; ++slot_wide)
            cell_list[slot_high * 8 + slot_wide] =
                test_jpeg_cell(band_data, wide_count, high_count, band_count, part_index,
                               wide_share, high_share, wide_peak, high_peak, block_x, block_y,
                               slot_wide, slot_high, lift_value) -
                level_mid;
        test_jpeg_turn(cell_list, coef_list);
        /* Held in the order the stream sends them, which is the order the
         * reader's store holds them in too. */
        for (slot_index = 0; slot_index < 64; ++slot_index) {
          double value_now = coef_list[zig_list[slot_index]];
          into_data[slot_index] = (int)(value_now < 0.0 ? value_now - 0.5 : value_now + 0.5);
        }
      }
  }
  return 1;
}

/* The scan writer's own state.  A progressive scan defers two things: the run
 * of blocks whose band is finished — the end-of-band run, which is only worth
 * writing once it is known how long it is — and, in a refining scan, the
 * correction bits of the blocks inside such a run, which have to follow the
 * code the run is written as. */
#define TEST_JPEG_HOLD 1024

typedef struct test_jpeg_wave {
  test_jpeg_room       *room;
  const test_jpeg_code *table;
  int      band_left;
  uint8_t  hold_list[TEST_JPEG_HOLD]; /* corrections waiting on the run */
  int      hold_count;
  uint8_t  fix_list[64];              /* corrections waiting on the next code */
  int      fix_count;
} test_jpeg_wave;

static void test_jpeg_wave_bits(test_jpeg_wave *wave, const uint8_t *bit_list, int bit_count) {
  int bit_index;
  for (bit_index = 0; bit_index < bit_count; ++bit_index)
    test_jpeg_bits(wave->room, bit_list[bit_index], 1);
}

/* Writes the pending end-of-band run and the corrections that belong with it.
 * The run's code carries the position of its top bit and the rest follow it. */
static void test_jpeg_wave_shut(test_jpeg_wave *wave) {
  if (wave->band_left <= 0) return;
  {
    int size_value = 0, step_value = wave->band_left;
    while ((step_value >>= 1) != 0) size_value += 1;
    test_jpeg_sign(wave->room, wave->table, size_value << 4);
    if (size_value) test_jpeg_bits(wave->room, wave->band_left, size_value);
  }
  wave->band_left = 0;
  test_jpeg_wave_bits(wave, wave->hold_list, wave->hold_count);
  wave->hold_count = 0;
}

static void test_jpeg_wave_fix(test_jpeg_wave *wave) {
  test_jpeg_wave_bits(wave, wave->fix_list, wave->fix_count);
  wave->fix_count = 0;
}

static void test_jpeg_wave_dc_first(test_jpeg_wave *wave, const int *coef_data, int *last_dc,
                               int low_bit) {
  int value_now = test_jpeg_floor(coef_data[0], low_bit);
  int diff_value = value_now - *last_dc;
  int size_value = test_jpeg_size(diff_value);
  test_jpeg_sign(wave->room, wave->table, size_value);
  test_jpeg_value(wave->room, diff_value, size_value);
  *last_dc = value_now;
}

static void test_jpeg_wave_dc_next(test_jpeg_wave *wave, const int *coef_data, int low_bit) {
  int value_now = test_jpeg_floor(coef_data[0], low_bit);
  test_jpeg_bits(wave->room, value_now - 2 * test_jpeg_floor(value_now, 1), 1);
}

/* A band's first scan: runs of zeros and a coefficient, as a sequential block
 * sends them, except that a block whose band ends in zeros is counted into the
 * end-of-band run instead of being closed with a code of its own. */
static void test_jpeg_wave_ac_first(test_jpeg_wave *wave, const int *coef_data, int from_slot,
                               int upto_slot, int low_bit) {
  int slot_index, run_value = 0;
  for (slot_index = from_slot; slot_index <= upto_slot; ++slot_index) {
    int value_now = test_jpeg_trim(coef_data[slot_index], low_bit);
    int size_value;
    if (value_now == 0) { run_value += 1; continue; }
    test_jpeg_wave_shut(wave);
    while (run_value > 15) {
      test_jpeg_sign(wave->room, wave->table, 0xF0);
      run_value -= 16;
    }
    size_value = test_jpeg_size(value_now);
    test_jpeg_sign(wave->room, wave->table, (run_value << 4) | size_value);
    test_jpeg_value(wave->room, value_now, size_value);
    run_value = 0;
  }
  if (run_value > 0) {
    wave->band_left += 1;
    if (wave->band_left == 0x7FFF) test_jpeg_wave_shut(wave);
  }
}

/* A band's refinement.  Every coefficient an earlier scan left nonzero carries
 * one correction bit wherever the walk passes it, so the run field of a code
 * counts only the ones they left zero, and the correction bits gathered on the
 * way follow the code rather than leading it.  A coefficient this scan makes
 * nonzero has a magnitude of exactly one, which is why the size field is
 * always one and the bit after the code is only its sign. */
static void test_jpeg_wave_ac_next(test_jpeg_wave *wave, const int *coef_data, int from_slot,
                              int upto_slot, int low_bit) {
  int size_list[64];
  int slot_index, last_new = 0, run_value = 0;
  for (slot_index = from_slot; slot_index <= upto_slot; ++slot_index) {
    int value_now = test_jpeg_trim(coef_data[slot_index], low_bit);
    size_list[slot_index] = value_now < 0 ? -value_now : value_now;
    if (size_list[slot_index] == 1) last_new = slot_index;
  }
  for (slot_index = from_slot; slot_index <= upto_slot; ++slot_index) {
    int value_now = size_list[slot_index];
    if (value_now == 0) { run_value += 1; continue; }
    /* A run of sixteen zeros is only worth its own code while a newly nonzero
     * coefficient is still to come; past the last of them the rest of the band
     * is what the end-of-band run stands for. */
    while (run_value > 15 && slot_index <= last_new) {
      test_jpeg_wave_shut(wave);
      test_jpeg_sign(wave->room, wave->table, 0xF0);
      run_value -= 16;
      test_jpeg_wave_fix(wave);
    }
    if (value_now > 1) { /* already nonzero: the next bit of its magnitude */
      wave->fix_list[wave->fix_count++] = (uint8_t)(value_now & 1);
      continue;
    }
    test_jpeg_wave_shut(wave);
    test_jpeg_sign(wave->room, wave->table, (run_value << 4) | 1);
    test_jpeg_bits(wave->room, test_jpeg_trim(coef_data[slot_index], low_bit) < 0 ? 0 : 1, 1);
    test_jpeg_wave_fix(wave);
    run_value = 0;
  }
  if (run_value > 0 || wave->fix_count > 0) {
    int fix_index;
    wave->band_left += 1;
    for (fix_index = 0; fix_index < wave->fix_count; ++fix_index)
      wave->hold_list[wave->hold_count++] = wave->fix_list[fix_index];
    wave->fix_count = 0;
    /* Forced out before either the run counter or the correction buffer could
     * overflow on the next block. */
    if (wave->band_left == 0x7FFF || wave->hold_count > TEST_JPEG_HOLD - 64) test_jpeg_wave_shut(wave);
  }
}

static void test_jpeg_wave_scan(test_jpeg_wave *wave, const test_jpeg_hoard *hoard, const int *slot_list,
                           int slot_count, int band_from, int band_upto, int high_bit, int low_bit,
                           int rest_span, const int *wide_share_list, const int *high_share_list,
                           int mcu_wide, int mcu_high) {
  test_jpeg_room *room = wave->room;
  int last_dc[JPEG_PART_LIMIT];
  int unit_wide, unit_high, unit_index, unit_total, rest_left, slot_index, part_index;

  test_jpeg_word(room, 0xFFDA);
  test_jpeg_word(room, 6 + 2 * slot_count);
  test_jpeg_byte(room, slot_count);
  for (slot_index = 0; slot_index < slot_count; ++slot_index) {
    test_jpeg_byte(room, slot_list[slot_index] + 1);
    test_jpeg_byte(room, 0x00); /* the one table serves as both classes */
  }
  test_jpeg_byte(room, band_from);
  test_jpeg_byte(room, band_upto);
  test_jpeg_byte(room, (high_bit << 4) | low_bit);

  for (part_index = 0; part_index < JPEG_PART_LIMIT; ++part_index) last_dc[part_index] = 0;
  wave->band_left = 0;
  wave->hold_count = 0;
  wave->fix_count = 0;
  if (slot_count == 1) {
    unit_wide = hoard->wide_units[slot_list[0]];
    unit_high = hoard->high_units[slot_list[0]];
  } else {
    unit_wide = mcu_wide;
    unit_high = mcu_high;
  }
  unit_total = unit_wide * unit_high;
  rest_left = rest_span;
  for (unit_index = 0; unit_index < unit_total; ++unit_index) {
    int unit_x = unit_index % unit_wide, unit_y = unit_index / unit_wide;
    if (rest_span > 0 && unit_index > 0 && rest_left == 0) {
      test_jpeg_wave_shut(wave);
      test_jpeg_flush(room);
      test_jpeg_word(room, 0xFFD0 | ((unit_index / rest_span - 1) & 7));
      for (part_index = 0; part_index < JPEG_PART_LIMIT; ++part_index) last_dc[part_index] = 0;
      rest_left = rest_span;
    }
    for (slot_index = 0; slot_index < slot_count; ++slot_index) {
      int part = slot_list[slot_index];
      int wide_span = slot_count == 1 ? 1 : wide_share_list[part];
      int high_span = slot_count == 1 ? 1 : high_share_list[part];
      int block_x, block_y;
      for (block_y = 0; block_y < high_span; ++block_y)
        for (block_x = 0; block_x < wide_span; ++block_x) {
          int into_x = unit_x * wide_span + block_x, into_y = unit_y * high_span + block_y;
          const int *coef_data =
              hoard->coef_data[part] +
              ((size_t)into_y * (size_t)hoard->wide_blocks[part] + (size_t)into_x) * 64u;
          if (band_from == 0) {
            if (high_bit == 0)
              test_jpeg_wave_dc_first(wave, coef_data, &last_dc[part], low_bit);
            else
              test_jpeg_wave_dc_next(wave, coef_data, low_bit);
          } else if (high_bit == 0) {
            test_jpeg_wave_ac_first(wave, coef_data, band_from, band_upto, low_bit);
          } else {
            test_jpeg_wave_ac_next(wave, coef_data, band_from, band_upto, low_bit);
          }
        }
    }
    rest_left -= 1;
  }
  test_jpeg_wave_shut(wave);
  test_jpeg_flush(room);
}

/* Writes one progressive file, over a scan script that reaches every one of the
 * four scan kinds and splits a band across two scans besides: the dc plane sent
 * a bit short and then refined, the luma's low frequencies and its high ones as
 * separate first scans two bits short, then a refinement of each band in turn
 * down to the bit the coefficients actually carry. */
static int test_jpeg_wave_write(const char *leaf_text, const uint8_t *band_data, int wide_count,
                           int high_count, int band_count, const int *wide_share_list,
                           const int *high_share_list, int rest_span, int deep_count,
                           int adobe_turn) {
  test_jpeg_code table;
  test_jpeg_room room;
  test_jpeg_wave wave;
  test_jpeg_hoard hoard;
  int wide_peak = 1, high_peak = 1, mcu_wide, mcu_high, part_index, okay_flag;
  int lift_value = deep_count == 12 ? 16 : 1, level_mid = 1 << (deep_count - 1);
  int all_list[JPEG_PART_LIMIT], one_list[1];
  size_t file_room = 8192u + (size_t)wide_count * (size_t)high_count * (size_t)band_count * 48u;

  memset(&room, 0, sizeof(room));
  memset(&wave, 0, sizeof(wave));
  test_jpeg_code_build(&table);
  room.file_room = file_room;
  room.file_data = (uint8_t *)mem_clear(file_room);
  if (!room.file_data) return 0;
  wave.room = &room;
  wave.table = &table;
  for (part_index = 0; part_index < band_count; ++part_index) {
    all_list[part_index] = part_index;
    if (wide_share_list[part_index] > wide_peak) wide_peak = wide_share_list[part_index];
    if (high_share_list[part_index] > high_peak) high_peak = high_share_list[part_index];
  }
  mcu_wide = (wide_count + wide_peak * 8 - 1) / (wide_peak * 8);
  mcu_high = (high_count + high_peak * 8 - 1) / (high_peak * 8);
  if (!test_jpeg_hoard_fill(&hoard, band_data, wide_count, high_count, band_count, wide_share_list,
                        high_share_list, lift_value, level_mid)) {
    test_jpeg_hoard_drop(&hoard);
    mem_free(room.file_data);
    return 0;
  }

  test_jpeg_head(&room, &table, wide_count, high_count, band_count, wide_share_list,
                 high_share_list, rest_span, 0xC2, deep_count, adobe_turn);

  test_jpeg_wave_scan(&wave, &hoard, all_list, band_count, 0, 0, 0, 1, rest_span, wide_share_list,
                 high_share_list, mcu_wide, mcu_high);
  test_jpeg_wave_scan(&wave, &hoard, all_list, band_count, 0, 0, 1, 0, rest_span, wide_share_list,
                 high_share_list, mcu_wide, mcu_high);
  one_list[0] = 0;
  test_jpeg_wave_scan(&wave, &hoard, one_list, 1, 1, 5, 0, 2, rest_span, wide_share_list,
                 high_share_list, mcu_wide, mcu_high);
  test_jpeg_wave_scan(&wave, &hoard, one_list, 1, 6, 63, 0, 2, rest_span, wide_share_list,
                 high_share_list, mcu_wide, mcu_high);
  for (part_index = 1; part_index < band_count; ++part_index) {
    one_list[0] = part_index;
    test_jpeg_wave_scan(&wave, &hoard, one_list, 1, 1, 63, 0, 1, rest_span, wide_share_list,
                   high_share_list, mcu_wide, mcu_high);
  }
  one_list[0] = 0;
  test_jpeg_wave_scan(&wave, &hoard, one_list, 1, 1, 63, 2, 1, rest_span, wide_share_list,
                 high_share_list, mcu_wide, mcu_high);
  test_jpeg_wave_scan(&wave, &hoard, one_list, 1, 1, 63, 1, 0, rest_span, wide_share_list,
                 high_share_list, mcu_wide, mcu_high);
  for (part_index = 1; part_index < band_count; ++part_index) {
    one_list[0] = part_index;
    test_jpeg_wave_scan(&wave, &hoard, one_list, 1, 1, 63, 1, 0, rest_span, wide_share_list,
                   high_share_list, mcu_wide, mcu_high);
  }

  test_jpeg_word(&room, 0xFFD9);
  okay_flag = !room.fault_flag && test_file_write(leaf_text, room.file_data, room.file_fill);
  test_jpeg_hoard_drop(&hoard);
  mem_free(room.file_data);
  return okay_flag;
}

/* The bands a fixture is built from, in four shapes.  The luma is always
 * busy.  The chroma pair is busy too at shape zero, flat along a row at shape
 * one — so that halving the horizontal sampling loses nothing to the averaging
 * — and a plane at shape two.
 *
 * A plane is the shape that pins the upsampler down.  Averaging a plane over a
 * block gives its value at the block's centre, and interpolating linearly
 * between those centres gives the plane back exactly, so a chroma pair at half
 * the sampling in both directions has to come back to the level it went in at.
 * A reader that repeated the nearest sample instead would come back a step
 * away, which is the difference these coefficients are chosen large enough to
 * show. */
/* Which eight by eight tile a pixel is in, for the shape that holds a tile
 * constant so that the transform round trip is exact and the colour transform
 * is the only thing between the samples and the pixels. */
static int test_jpeg_tile(int wide_index, int high_index) {
  return (wide_index / 8) + (high_index / 8) * 3;
}

static int test_jpeg_band(int wide_index, int high_index, int band_index, int shape_mark) {
  switch (band_index) {
    case 0:
      if (shape_mark == 3) return 100 + (test_jpeg_tile(wide_index, high_index) % 5) * 20;
      return (wide_index * 17 + high_index * 5) & 0xFF;
    case 1:
      if (shape_mark == 1) return 100 + (high_index % 3) * 10;
      if (shape_mark == 2) return 60 + wide_index * 3 + high_index * 5;
      if (shape_mark == 3) return 40 + (test_jpeg_tile(wide_index, high_index) % 6) * 35;
      return (wide_index * 3 + high_index * 29) & 0xFF;
    case 2:
      if (shape_mark == 1) return 150 - (high_index % 4) * 7;
      if (shape_mark == 2) return 200 - wide_index * 4 - high_index * 2;
      if (shape_mark == 3) return 30 + (test_jpeg_tile(wide_index, high_index) % 5) * 40;
      return (wide_index * wide_index + high_index * high_index) & 0xFF;
    default: /* the black an ink file carries beside its three */
      if (shape_mark == 3) return 90 + (test_jpeg_tile(wide_index, high_index) % 4) * 45;
      return 200 - ((wide_index * 7 + high_index * 11) & 0x7F);
  }
}

static uint8_t *test_jpeg_bands(int wide_count, int high_count, int band_count, int shape_mark) {
  uint8_t *band_data =
      (uint8_t *)mem_clear((size_t)wide_count * (size_t)high_count * (size_t)band_count);
  int high_index, wide_index, band_index;
  if (!band_data) return NULL;
  for (high_index = 0; high_index < high_count; ++high_index)
    for (wide_index = 0; wide_index < wide_count; ++wide_index)
      for (band_index = 0; band_index < band_count; ++band_index)
        band_data[((size_t)high_index * (size_t)wide_count + (size_t)wide_index) *
                      (size_t)band_count + (size_t)band_index] =
            (uint8_t)test_jpeg_band(wide_index, high_index, band_index, shape_mark);
  return band_data;
}

/* What the reader should hand back for one pixel of a fixture: the bands it was
 * built from, through whichever colour transform the file said it carried.
 * `turn_mark` is the `APP14` transform value the reader will have read, and the
 * fourth band of an ink file multiplies the other three, exactly as a file
 * written inverted means. */
static void test_jpeg_want(int wide_index, int high_index, int band_count, int shape_mark,
                           int turn_mark, float lift_value, float peak_value, float *want_list) {
  float band_list[4];
  int band_index;
  for (band_index = 0; band_index < band_count; ++band_index)
    band_list[band_index] =
        (float)test_jpeg_band(wide_index, high_index, band_index, shape_mark) * lift_value;
  if (band_count == 1) {
    want_list[0] = band_list[0];
    return;
  }
  if (turn_mark != 0) {
    float bright = band_list[0];
    float blue_off = band_list[1] - (peak_value + 1.0f) / 2.0f;
    float red_off = band_list[2] - (peak_value + 1.0f) / 2.0f;
    band_list[0] = bright + 1.402f * red_off;
    band_list[1] = bright - 0.344136f * blue_off - 0.714136f * red_off;
    band_list[2] = bright + 1.772f * blue_off;
  }
  for (band_index = 0; band_index < 3; ++band_index) {
    float level = band_list[band_index];
    if (band_count == 4) {
      if (level < 0.0f) level = 0.0f;
      if (level > peak_value) level = peak_value;
      level = level * band_list[3] / peak_value;
    }
    want_list[band_index] = level;
  }
}

/* Reads a fixture back and holds every pixel to the bands it was built from. */
static void test_jpeg_check(const char *leaf_text, const char *claim_text, int wide_count,
                            int high_count, int band_count, int shape_mark, int edge_skip,
                            float slack_value, int deep_count, int turn_mark) {
  char path_text[1024];
  flat_grid grid;
  float lift_value = deep_count == 12 ? 16.0f : 1.0f;
  float peak_value = (float)((1 << deep_count) - 1);
  int high_index, wide_index, band_index, okay_flag = 1;
  path_join(path_text, sizeof(path_text), test_yard_path, leaf_text);
  if (image_read(path_text, &grid) != APP_OKAY) {
    test_true(0, claim_text);
    return;
  }
  if (grid.wide_count != wide_count || grid.high_count != high_count ||
      grid.band_count != (band_count == 1 ? 1 : 3))
    okay_flag = 0;
  /* A plane read at the picture's edge is the one place the blend cannot
   * reproduce it: there is no sample beyond the border to blend with, so both
   * this reader and libjpeg hold the edge sample flat.  The margin is skipped
   * rather than modelled. */
  for (high_index = edge_skip; okay_flag && high_index < high_count - edge_skip; ++high_index)
    for (wide_index = edge_skip; wide_index < wide_count - edge_skip; ++wide_index) {
      const float *cell_data = grid_at(&grid, high_index, wide_index);
      float want_list[3];
      test_jpeg_want(wide_index, high_index, band_count, shape_mark, turn_mark, lift_value,
                     peak_value, want_list);
      for (band_index = 0; band_index < grid.band_count; ++band_index) {
        float want_value = want_list[band_index] / peak_value;
        float gap_value;
        if (want_value < 0.0f) want_value = 0.0f;
        if (want_value > 1.0f) want_value = 1.0f;
        gap_value = cell_data[band_index] - want_value;
        if (gap_value < 0.0f) gap_value = -gap_value;
        if (gap_value > slack_value) okay_flag = 0;
      }
    }
  test_true(okay_flag, claim_text);
  grid_free(&grid);
}

static void test_jpeg(void) {
  static const int wide_count = 24, high_count = 16;
  static const int share_one[4] = {1, 1, 1, 1};
  static const int share_two[3] = {2, 1, 1};
  uint8_t *gray_data, *band_data, *flat_data, *ramp_data, *tile_data, *ink_data;
  char path_text[1024];
  flat_grid grid;
  test_open("jpeg");

  {
    int zig_list[64], slot_index, okay_flag = 1;
    test_jpeg_zig_fill(zig_list);
    for (slot_index = 0; slot_index < 64; ++slot_index)
      if (zig_list[slot_index] != (int)jpeg_zig_list[slot_index]) okay_flag = 0;
    test_true(okay_flag, "the coefficient order is the diagonals of the block in turn");
  }

  gray_data = test_jpeg_bands(wide_count, high_count, 1, 0);
  band_data = test_jpeg_bands(wide_count, high_count, 3, 0);
  flat_data = test_jpeg_bands(wide_count, high_count, 3, 1);
  ramp_data = test_jpeg_bands(wide_count, high_count, 3, 2);
  tile_data = test_jpeg_bands(wide_count, high_count, 3, 3);
  ink_data = test_jpeg_bands(wide_count, high_count, 4, 3);
  if (!gray_data || !band_data || !flat_data || !ramp_data || !tile_data || !ink_data) {
    test_true(0, "the jpeg fixtures are allocated");
    mem_free(gray_data);
    mem_free(band_data);
    mem_free(flat_data);
    mem_free(ramp_data);
    mem_free(tile_data);
    mem_free(ink_data);
    return;
  }

  /* One component, every component at the picture's own sampling: what comes
   * back is the picture the encoder was given, to within what two transforms
   * round. */
  test_true(test_jpeg_write("gray.jpg", gray_data, wide_count, high_count, 1, share_one, share_one,
                            0, 0xC0, 8, -1),
            "a grey jpeg fixture is written");
  test_jpeg_check("gray.jpg", "a grey jpeg decodes to the samples it was built from", wide_count,
                  high_count, 1, 0, 0, 1.5f / 255.0f, 8, 1);

  test_true(test_jpeg_write("colour.jpg", band_data, wide_count, high_count, 3, share_one,
                            share_one, 0, 0xC0, 8, -1),
            "a three component jpeg fixture is written");
  test_jpeg_check("colour.jpg", "a three component jpeg decodes through the colour transform",
                  wide_count, high_count, 3, 0, 0, 2.0f / 255.0f, 8, 1);

  /* Chroma at half the horizontal sampling, over a picture whose chroma does
   * not vary along a row: the averaging the encoder does is exact and so is
   * the blend the reader undoes it with, so this is held to the same floor as
   * the unsampled one rather than a looser one. */
  test_true(test_jpeg_write("share.jpg", flat_data, wide_count, high_count, 3, share_two, share_one,
                            0, 0xC0, 8, -1),
            "a chroma subsampled fixture is written");
  test_jpeg_check("share.jpg", "chroma at half the sampling is lifted back to the picture's grid",
                  wide_count, high_count, 3, 1, 0, 1.5f / 255.0f, 8, 1);

  /* A picture that is constant over each eight by eight tile has nothing in it
   * but dc coefficients, so the transform round trip is exact to the level and
   * what is left between the samples and the pixels is the colour transform
   * alone.  Half a level is a tight enough floor to catch a coefficient that is
   * a percent wrong, or a pair the wrong way round. */
  test_true(test_jpeg_write("tile.jpg", tile_data, wide_count, high_count, 3, share_one, share_one,
                            0, 0xC0, 8, -1),
            "a tile constant fixture is written");
  test_jpeg_check("tile.jpg", "the colour transform is exact where the transform loses nothing",
                  wide_count, high_count, 3, 3, 0, 0.51f / 255.0f, 8, 1);

  /* Chroma at half the sampling in both directions, over a picture whose
   * chroma is a plane: the blend that undoes it has to put every interior
   * pixel back where it started, which repeating the nearest sample would
   * not. */
  test_true(test_jpeg_write("plane.jpg", ramp_data, wide_count, high_count, 3, share_two,
                            share_two, 0, 0xC0, 8, -1),
            "a fixture subsampled in both directions is written");
  test_jpeg_check("plane.jpg", "chroma at half the sampling both ways is blended, not repeated",
                  wide_count, high_count, 3, 2, 2, 2.5f / 255.0f, 8, 1);

  /* The same picture with a restart marker after every unit, which resets both
   * the bit reader and the dc predictors. */
  test_true(test_jpeg_write("rest.jpg", band_data, wide_count, high_count, 3, share_one, share_one,
                            1, 0xC0, 8, -1),
            "a restarting fixture is written");
  test_jpeg_check("rest.jpg", "a file broken by restart markers decodes to the same picture",
                  wide_count, high_count, 3, 0, 0, 2.0f / 255.0f, 8, 1);

  /* Twelve bits a sample, which an extended sequential frame header allows: the
   * tables and the level shift widen with it, and the fixture is the eight bit
   * one scaled up so that the picture that comes back is the same picture. */
  test_true(test_jpeg_write("deep.jpg", band_data, wide_count, high_count, 3, share_one, share_one,
                            0, 0xC1, 12, -1),
            "a twelve bit fixture is written");
  test_jpeg_check("deep.jpg", "a twelve bit frame decodes on its own wider grid", wide_count,
                  high_count, 3, 0, 0, 2.0f / 4095.0f, 12, 1);

  /* Three bands the file says are the picture's own rather than a luma and a
   * chroma pair.  Nothing but `APP14` distinguishes the two, which is why the
   * reader reads it. */
  test_true(test_jpeg_write("plain.jpg", band_data, wide_count, high_count, 3, share_one, share_one,
                            0, 0xC0, 8, 0),
            "an untransformed three band fixture is written");
  test_jpeg_check("plain.jpg", "three bands marked untransformed are the picture's own",
                  wide_count, high_count, 3, 0, 0, 1.5f / 255.0f, 8, 0);

  /* Four bands are ink, written inverted the way every writer that ships an
   * `APP14` writes them, so the picture is the product of the three with the
   * black.  The tile constant shape leaves the transform nothing to round, so
   * the product itself is what is being held. */
  test_true(test_jpeg_write("ink.jpg", ink_data, wide_count, high_count, 4, share_one, share_one,
                            0, 0xC0, 8, 0),
            "a four band ink fixture is written");
  test_jpeg_check("ink.jpg", "four bands are read as ink and multiplied by the black", wide_count,
                  high_count, 4, 3, 0, 1.5f / 255.0f, 8, 0);

  test_true(test_jpeg_write("inky.jpg", ink_data, wide_count, high_count, 4, share_one, share_one,
                            0, 0xC0, 8, 2),
            "a four band ink fixture under the colour transform is written");
  test_jpeg_check("inky.jpg", "ink under the colour transform is turned before it is multiplied",
                  wide_count, high_count, 4, 3, 0, 2.0f / 255.0f, 8, 2);

  /* Four bands with nothing to say which four they are: refused rather than
   * guessed at. */
  test_true(test_jpeg_write("mute.jpg", ink_data, wide_count, high_count, 4, share_one, share_one,
                            0, 0xC0, 8, -1),
            "a four band fixture without an APP14 is written");
  path_join(path_text, sizeof(path_text), test_yard_path, "mute.jpg");
  test_true(image_read(path_text, &grid) == APP_FAIL_SUPPORT,
            "four bands with no APP14 to read are refused");

  /* The progressive frames.  Each is the same picture as the sequential
   * fixture above it, sent as a dc plane and then refined, its bands split
   * across scans and each of them refined in turn, so all four scan kinds and
   * the end-of-band run are walked before the picture comes back. */
  test_true(test_jpeg_wave_write("wave.jpg", band_data, wide_count, high_count, 3, share_one, share_one,
                            0, 8, -1),
            "a progressive fixture is written");
  test_jpeg_check("wave.jpg", "a progressive file decodes to the picture its scans carry",
                  wide_count, high_count, 3, 0, 0, 2.0f / 255.0f, 8, 1);

  test_true(test_jpeg_wave_write("wavegray.jpg", gray_data, wide_count, high_count, 1, share_one,
                            share_one, 0, 8, -1),
            "a one component progressive fixture is written");
  test_jpeg_check("wavegray.jpg", "a one component progressive file decodes the same",
                  wide_count, high_count, 1, 0, 0, 1.5f / 255.0f, 8, 1);

  /* A progressive file whose chroma is sampled below the luma walks its own
   * blocks in the band scans and the shared units in the dc one, which is the
   * pairing a subsampled progressive file is. */
  test_true(test_jpeg_wave_write("waveshare.jpg", ramp_data, wide_count, high_count, 3, share_two,
                            share_two, 0, 8, -1),
            "a subsampled progressive fixture is written");
  test_jpeg_check("waveshare.jpg", "a subsampled progressive file lifts its chroma back",
                  wide_count, high_count, 3, 2, 2, 2.5f / 255.0f, 8, 1);

  /* The tile constant picture again, progressively: with nothing above dc in
   * it, the successive approximation of the dc plane is the whole of what is
   * being held, at the same half level floor as the sequential one. */
  test_true(test_jpeg_wave_write("wavetile.jpg", tile_data, wide_count, high_count, 3, share_one,
                            share_one, 0, 8, -1),
            "a tile constant progressive fixture is written");
  test_jpeg_check("wavetile.jpg", "a progressive dc plane is refined back to the level",
                  wide_count, high_count, 3, 3, 0, 0.51f / 255.0f, 8, 1);

  /* Restart markers inside a progressive scan end the end-of-band run as well
   * as the predictors, in the band scans as much as in the dc ones. */
  test_true(test_jpeg_wave_write("waverest.jpg", band_data, wide_count, high_count, 3, share_one,
                            share_one, 2, 8, -1),
            "a restarting progressive fixture is written");
  test_jpeg_check("waverest.jpg", "a progressive file broken by restarts decodes the same",
                  wide_count, high_count, 3, 0, 0, 2.0f / 255.0f, 8, 1);

  test_true(test_jpeg_wave_write("wavedeep.jpg", band_data, wide_count, high_count, 3, share_one,
                            share_one, 0, 12, -1),
            "a twelve bit progressive fixture is written");
  test_jpeg_check("wavedeep.jpg", "a twelve bit progressive frame decodes on its wider grid",
                  wide_count, high_count, 3, 0, 0, 2.0f / 4095.0f, 12, 1);

  /* A frame header this reader still has no decoder for — arithmetic coding —
   * is refused the way progressive used to be. */
  test_true(test_jpeg_write("guess.jpg", band_data, wide_count, high_count, 3, share_one, share_one,
                            0, 0xC9, 8, -1),
            "an arithmetic coded fixture is written");
  path_join(path_text, sizeof(path_text), test_yard_path, "guess.jpg");
  test_true(image_read(path_text, &grid) == APP_FAIL_SUPPORT, "an arithmetic coded jpeg is refused");

  /* An entropy stream that stops before its last unit is a truncated file, not
   * a picture with a grey corner. */
  {
    char full_text[1024];
    uint8_t *file_data;
    size_t file_size = 0;
    path_join(full_text, sizeof(full_text), test_yard_path, "colour.jpg");
    file_data = (uint8_t *)file_slurp(full_text, &file_size);
    test_true(file_data != NULL && file_size > 400, "the fixture is read back for truncation");
    if (file_data) {
      test_true(test_file_write("short.jpg", file_data, file_size - file_size / 4),
                "a truncated fixture is written");
      path_join(path_text, sizeof(path_text), test_yard_path, "short.jpg");
      test_true(image_read(path_text, &grid) != APP_OKAY, "a truncated jpeg is refused");
      mem_free(file_data);
    }
  }

  mem_free(gray_data);
  mem_free(band_data);
  mem_free(flat_data);
  mem_free(ramp_data);
  mem_free(tile_data);
  mem_free(ink_data);
}

/* An independent bicubic, gathered in two dimensions at once rather than as two
 * separable passes, so agreement is between two different arrangements of the
 * same definition. */
static float test_scale_curve(float step_value) {
  float step_abs = step_value < 0.0f ? -step_value : step_value;
  if (step_abs < 1.0f) return (1.5f * step_abs - 2.5f) * step_abs * step_abs + 1.0f;
  if (step_abs < 2.0f)
    return ((-0.5f * step_abs + 2.5f) * step_abs - 4.0f) * step_abs + 2.0f;
  return 0.0f;
}

static float test_scale_axis(const float *from_list, int from_count, int into_index, int into_count) {
  double ratio_value = (double)from_count / (double)into_count;
  double spread_value = ratio_value > 1.0 ? ratio_value : 1.0;
  double reach_value = 2.0 * spread_value;
  double centre_value = ((double)into_index + 0.5) * ratio_value;
  int from_index = (int)floor(centre_value - reach_value + 0.5);
  int upto_index = (int)floor(centre_value + reach_value + 0.5) + 1;
  float total_value = 0.0f, sum_value = 0.0f;
  int tap_index;
  if (from_index < 0) from_index = 0;
  if (upto_index > from_count) upto_index = from_count;
  for (tap_index = from_index; tap_index < upto_index; ++tap_index) {
    float weight_value =
        test_scale_curve((float)(((double)tap_index + 0.5 - centre_value) / spread_value));
    total_value += weight_value;
    sum_value += weight_value * from_list[tap_index];
  }
  return total_value != 0.0f ? sum_value / total_value : 0.0f;
}

/* The eight bit float grid a quantized key and value cache stores on.
 *
 * Every value the grid carries has to come back unchanged, everything between
 * two of them has to land on the nearer, and a magnitude over the top has to
 * saturate rather than run off into an infinity the format has no room for. */
static void test_cache(void) {
  int okay_flag = 1;
  int power_index, step_index;
  test_open("cache");

  /* The normal range: significands of four bits, one of them implied. */
  for (power_index = -6; power_index <= 8; ++power_index)
    for (step_index = 0; step_index < 8; ++step_index) {
      float exact = ldexpf(1.0f + (float)step_index / 8.0f, power_index);
      if (exact > 448.0f) continue;
      if (cache_pack8(exact) != exact || cache_pack8(-exact) != -exact) okay_flag = 0;
    }
  test_true(okay_flag, "every normal the grid carries survives the round trip");

  okay_flag = 1;
  for (step_index = 0; step_index < 8; ++step_index) {
    float exact = (float)step_index / 512.0f;
    if (cache_pack8(exact) != exact) okay_flag = 0;
  }
  test_true(okay_flag, "the subnormal steps survive the round trip");

  test_true(cache_pack8(448.0f) == 448.0f, "the largest magnitude is held");
  test_true(cache_pack8(1e9f) == 448.0f && cache_pack8(-1e9f) == -448.0f,
            "a magnitude over the top saturates rather than overflowing");
  test_true(cache_pack8(0.0f) == 0.0f, "zero stays zero");

  /* Between 1.0 and 1.125 the step is 0.125, so the halfway point is 1.0625
   * and either side of it has a nearer neighbour. */
  test_true(cache_pack8(1.03f) == 1.0f, "a value below the midpoint rounds down");
  test_true(cache_pack8(1.10f) == 1.125f, "a value above the midpoint rounds up");
  test_true(cache_pack8(1.0625f) == 1.0f, "a midpoint rounds to the even significand");

  /* Anything under half the subnormal step has no neighbour but zero. */
  test_true(cache_pack8(1.0f / 4096.0f) == 0.0f, "a magnitude under the smallest step falls to zero");

  /* The byte the cache actually holds.  Everything above is the grid as an
   * arithmetic; what a stored cache costs is the grid as a byte, and the two
   * have to name the same value or a byte cache would not be the float cache
   * it replaced.
   *
   * Two codes stand outside the round trip and are meant to.  0x7F and 0xFF are
   * where the format keeps its NaN, and this grid has no NaN: the encoder
   * saturates at 448 and stops one code short of them, so decoding either and
   * encoding it back lands on 0x7E.  A negative zero encodes to the positive
   * one for the same reason a float cache never held one. */
  okay_flag = 1;
  for (step_index = 0; step_index < 256; ++step_index) {
    float exact = cache_real8((uint8_t)step_index);
    if ((step_index & 0x7F) == 0x7F) continue;
    if (cache_code8(exact) != (uint8_t)step_index && exact != 0.0f) okay_flag = 0;
    if (cache_real8(cache_code8(exact)) != cache_pack8(exact)) okay_flag = 0;
  }
  test_true(okay_flag, "every code the grid carries encodes back to itself");
  test_true(cache_code8(cache_real8(0x7Fu)) == 0x7Eu && cache_code8(cache_real8(0xFFu)) == 0xFEu,
            "the two codes the format spends on a NaN saturate rather than round-tripping");

  okay_flag = 1;
  for (step_index = -20000; step_index <= 20000; ++step_index) {
    float value = (float)step_index / 37.0f;
    if (cache_real8(cache_code8(value)) != cache_pack8(value)) okay_flag = 0;
  }
  for (power_index = -12; power_index <= 10; ++power_index) {
    float value = ldexpf(1.3f, power_index);
    if (cache_real8(cache_code8(value)) != cache_pack8(value)) okay_flag = 0;
    if (cache_real8(cache_code8(-value)) != cache_pack8(-value)) okay_flag = 0;
  }
  test_true(okay_flag, "the byte and the rounding agree on every value tried");

  test_true(cache_code8(0.0f) == 0x00u, "zero is the zero byte");
  test_true(cache_code8(-1e9f) == 0xFEu && cache_real8(0xFEu) == -448.0f,
            "the saturating magnitude is the last code before the format's NaN");
  test_true(cache_code8(1.0f) == 0x38u, "one is exponent seven with an empty significand");
  test_true(cache_real8(0x08u) == CACHE_FP8_NORM, "the smallest normal is the first normal code");
  test_true(cache_real8(0x01u) == CACHE_FP8_STEP, "the smallest subnormal is one step");

  /* The table the session reads a cached row through.  Its whole purpose is
   * that an entry equals the round trip a float cache made, scale and all, so
   * a layer holding bytes reaches the same score as one holding floats. */
  {
    float grid_room[256];
    float scale_list[4];
    int scale_index;
    scale_list[0] = 2.686925f;   /* layer 0's keys on the shipped export */
    scale_list[1] = 21.165359f;  /* and its values */
    scale_list[2] = 128.0f / 448.0f;
    scale_list[3] = 1.0f;
    okay_flag = 1;
    for (scale_index = 0; scale_index < 4; ++scale_index) {
      float scale = scale_list[scale_index];
      cache_grid_fill(grid_room, scale);
      for (step_index = -3000; step_index <= 3000; ++step_index) {
        float value = (float)step_index * scale / 211.0f;
        if (grid_room[cache_code8(value / scale)] != cache_pack8(value / scale) * scale)
          okay_flag = 0;
      }
    }
    test_true(okay_flag, "the table hands back exactly what the float round trip stored");
  }
}

static void test_scale(void) {
  flat_grid from_grid, into_grid;
  int wide_index, high_index, okay_flag = 1;
  test_open("scale");

  /* A constant picture has to survive any resize, in either direction: the
   * weights of one output sample sum to one. */
  test_true(grid_open(&from_grid, 9, 7, 3) == APP_OKAY, "a source raster opens");
  for (high_index = 0; high_index < 7; ++high_index)
    for (wide_index = 0; wide_index < 9; ++wide_index) {
      float *cell_data = grid_at(&from_grid, high_index, wide_index);
      cell_data[0] = 0.25f;
      cell_data[1] = 0.5f;
      cell_data[2] = 0.75f;
    }
  test_true(grid_scale(&from_grid, 20, 16, &into_grid) == APP_OKAY, "an upscale runs");
  for (high_index = 0; high_index < 16; ++high_index)
    for (wide_index = 0; wide_index < 20; ++wide_index) {
      const float *cell_data = grid_at(&into_grid, high_index, wide_index);
      if (fabs((double)cell_data[0] - 0.25) > 1e-5 || fabs((double)cell_data[2] - 0.75) > 1e-5)
        okay_flag = 0;
    }
  test_true(okay_flag, "an upscaled constant stays constant");
  grid_free(&into_grid);
  okay_flag = 1;
  test_true(grid_scale(&from_grid, 4, 3, &into_grid) == APP_OKAY, "a downscale runs");
  for (high_index = 0; high_index < 3; ++high_index)
    for (wide_index = 0; wide_index < 4; ++wide_index) {
      const float *cell_data = grid_at(&into_grid, high_index, wide_index);
      if (fabs((double)cell_data[1] - 0.5) > 1e-5) okay_flag = 0;
    }
  test_true(okay_flag, "a downscaled constant stays constant");
  grid_free(&into_grid);
  grid_free(&from_grid);

  /* A ramp along one axis is reproduced exactly wherever the four taps fit,
   * because a cubic interpolant reproduces a straight line. */
  test_true(grid_open(&from_grid, 8, 1, 1) == APP_OKAY, "a ramp opens");
  for (wide_index = 0; wide_index < 8; ++wide_index)
    grid_at(&from_grid, 0, wide_index)[0] = (float)wide_index;
  test_true(grid_scale(&from_grid, 16, 1, &into_grid) == APP_OKAY, "the ramp upscales");
  okay_flag = 1;
  for (wide_index = 4; wide_index < 12; ++wide_index) {
    double want_value = ((double)wide_index + 0.5) * 0.5 - 0.5;
    if (fabs((double)grid_at(&into_grid, 0, wide_index)[0] - want_value) > 2e-5) okay_flag = 0;
  }
  test_true(okay_flag, "an upscaled ramp stays a straight line away from the edges");
  grid_free(&into_grid);
  grid_free(&from_grid);

  /* And the whole thing against the two dimensional gather. */
  test_true(grid_open(&from_grid, 11, 9, 2) == APP_OKAY, "a textured raster opens");
  for (high_index = 0; high_index < 9; ++high_index)
    for (wide_index = 0; wide_index < 11; ++wide_index) {
      float *cell_data = grid_at(&from_grid, high_index, wide_index);
      cell_data[0] = (float)sin((double)(wide_index * 3 + high_index) * 0.4);
      cell_data[1] = (float)cos((double)(wide_index + high_index * 5) * 0.2);
    }
  test_true(grid_scale(&from_grid, 7, 5, &into_grid) == APP_OKAY, "the textured raster resizes");
  okay_flag = 1;
  {
    float lane_list[16];
    float mid_list[16];
    int band_index, tap_index;
    for (high_index = 0; high_index < 5; ++high_index)
      for (wide_index = 0; wide_index < 7; ++wide_index)
        for (band_index = 0; band_index < 2; ++band_index) {
          double want_value;
          for (tap_index = 0; tap_index < 9; ++tap_index) {
            int lane_index;
            for (lane_index = 0; lane_index < 11; ++lane_index)
              lane_list[lane_index] = grid_at(&from_grid, tap_index, lane_index)[band_index];
            mid_list[tap_index] = test_scale_axis(lane_list, 11, wide_index, 7);
          }
          want_value = (double)test_scale_axis(mid_list, 9, high_index, 5);
          if (fabs((double)grid_at(&into_grid, high_index, wide_index)[band_index] - want_value) >
              1e-5)
            okay_flag = 0;
        }
  }
  test_true(okay_flag, "the separable resize matches a direct gather");
  grid_free(&into_grid);
  grid_free(&from_grid);

  /* Folding three bands onto one takes the luma weights, not a plain mean. */
  test_true(grid_open(&from_grid, 2, 2, 3) == APP_OKAY, "a colour raster opens");
  for (high_index = 0; high_index < 2; ++high_index)
    for (wide_index = 0; wide_index < 2; ++wide_index) {
      float *cell_data = grid_at(&from_grid, high_index, wide_index);
      cell_data[0] = 1.0f;
      cell_data[1] = 0.0f;
      cell_data[2] = 0.0f;
    }
  test_true(grid_bands(&from_grid, 1) == APP_OKAY, "three bands fold onto one");
  test_near((double)grid_at(&from_grid, 1, 1)[0], 0.299, 1e-6, "the fold uses the luma weights");
  test_true(grid_bands(&from_grid, 3) == APP_OKAY, "one band spreads over three");
  test_near((double)grid_at(&from_grid, 0, 1)[2], 0.299, 1e-6, "the spread copies the one band");
  grid_free(&from_grid);
}

/* -- wave ----------------------------------------------------------------- */

static void test_wave_put(uint8_t *data, size_t at, uint32_t value, int byte_count) {
  int byte_index;
  for (byte_index = 0; byte_index < byte_count; ++byte_index)
    data[at + (size_t)byte_index] = (uint8_t)((value >> (8 * byte_index)) & 0xFFu);
}

/* Writes a two channel sixteen bit wave with an unknown chunk in front of the
 * data, which is what a real recorder leaves behind. */
static int test_wave_write_at(const char *leaf_text, int rate_value, int value_count,
                              double turn_step) {
  size_t body_size = (size_t)value_count * 4u;
  size_t file_size = 12u + 24u + 12u + 8u + body_size;
  uint8_t *file_data = (uint8_t *)mem_clear(file_size);
  int value_index, okay_flag;
  if (!file_data) return 0;
  memcpy(file_data, "RIFF", 4);
  test_wave_put(file_data, 4, (uint32_t)(file_size - 8), 4);
  memcpy(file_data + 8, "WAVE", 4);
  memcpy(file_data + 12, "fmt ", 4);
  test_wave_put(file_data, 16, 16, 4);
  test_wave_put(file_data, 20, 1, 2);  /* pcm */
  test_wave_put(file_data, 22, 2, 2);  /* channels */
  test_wave_put(file_data, 24, (uint32_t)rate_value, 4);
  test_wave_put(file_data, 28, (uint32_t)(rate_value * 4), 4);
  test_wave_put(file_data, 32, 4, 2);
  test_wave_put(file_data, 34, 16, 2); /* bits */
  memcpy(file_data + 36, "LIST", 4);
  test_wave_put(file_data, 40, 4, 4);
  memcpy(file_data + 44, "INFO", 4);
  memcpy(file_data + 48, "data", 4);
  test_wave_put(file_data, 52, (uint32_t)body_size, 4);
  for (value_index = 0; value_index < value_count; ++value_index) {
    int left_value = (int)(8000.0 * sin((double)value_index * turn_step));
    int right_value = (int)(4000.0 * sin((double)value_index * turn_step));
    test_wave_put(file_data, 56 + (size_t)value_index * 4u, (uint32_t)(left_value & 0xFFFF), 2);
    test_wave_put(file_data, 58 + (size_t)value_index * 4u, (uint32_t)(right_value & 0xFFFF), 2);
  }
  okay_flag = test_file_write(leaf_text, file_data, file_size);
  mem_free(file_data);
  return okay_flag;
}

static int test_wave_write(const char *leaf_text, int rate_value, int value_count) {
  return test_wave_write_at(leaf_text, rate_value, value_count, 0.31);
}

static void test_wave(void) {
  wave_clip clip;
  char path_text[1024];
  int value_index, okay_flag = 1;
  test_open("wave");
  test_true(test_wave_write("clip.wav", 8000, 200), "the wave fixture is written");
  path_join(path_text, sizeof(path_text), test_yard_path, "clip.wav");
  test_true(wave_read(path_text, &clip) == APP_OKAY, "a riff wave reads");
  test_true(clip.rate_value == 8000, "the sample rate comes from the format chunk");
  test_true(clip.value_count == 200, "an unknown chunk before the data is stepped over");
  for (value_index = 0; value_index < clip.value_count; ++value_index) {
    /* Two channels at 8000 and 4000 average to 6000, and the reader scales a
     * signed sixteen bit sample by 32768. */
    double want_value = 6000.0 * sin((double)value_index * 0.31) / 32768.0;
    if (fabs((double)clip.value_data[value_index] - want_value) > 2e-4) okay_flag = 0;
  }
  test_true(okay_flag, "the channels are averaged rather than dropped");
  test_true(wave_rate(&clip, 4000) == APP_OKAY, "the clip resamples");
  test_true(clip.rate_value == 4000 && clip.value_count == 100,
            "halving the rate halves the sample count");
  wave_free(&clip);
}

/* -- filterbank ----------------------------------------------------------- */

static void test_mel(void) {
  static const int turn_size = 64;
  float real_list[64], imag_list[64];
  float *bank_data = NULL;
  wave_clip clip;
  flat_grid mel_grid;
  int slot_index, bin_index, mel_index, okay_flag = 1;
  test_open("mel");

  /* The transform against the sum it is a fast way of computing. */
  for (slot_index = 0; slot_index < turn_size; ++slot_index) {
    real_list[slot_index] = (float)sin((double)slot_index * 0.7) + 0.3f * (float)slot_index;
    imag_list[slot_index] = 0.0f;
  }
  {
    float want_real[64], want_imag[64];
    for (bin_index = 0; bin_index < turn_size; ++bin_index) {
      double sum_real = 0.0, sum_imag = 0.0;
      for (slot_index = 0; slot_index < turn_size; ++slot_index) {
        double angle_value =
            -6.283185307179586 * (double)bin_index * (double)slot_index / (double)turn_size;
        sum_real += (double)real_list[slot_index] * cos(angle_value);
        sum_imag += (double)real_list[slot_index] * sin(angle_value);
      }
      want_real[bin_index] = (float)sum_real;
      want_imag[bin_index] = (float)sum_imag;
    }
    wave_spin(real_list, imag_list, turn_size);
    for (bin_index = 0; bin_index < turn_size; ++bin_index)
      if (fabs((double)real_list[bin_index] - want_real[bin_index]) > 1e-2 ||
          fabs((double)imag_list[bin_index] - want_imag[bin_index]) > 1e-2)
        okay_flag = 0;
    test_true(okay_flag, "the fast transform matches the discrete one");
  }

  /* Every filter has to be a triangle that starts and ends at zero, and two
   * neighbours have to hand over between them: at the peak of one, the other
   * two are already or still at nothing. */
  test_true(mel_bank(8, 33, 8000, &bank_data) == APP_OKAY, "a filterbank builds");
  okay_flag = 1;
  for (mel_index = 0; mel_index < 8 && bank_data; ++mel_index) {
    float peak_value = 0.0f;
    float total_value = 0.0f;
    for (bin_index = 0; bin_index < 33; ++bin_index) {
      float weight_value = bank_data[mel_index * 33 + bin_index];
      if (weight_value < 0.0f) okay_flag = 0;
      if (weight_value > peak_value) peak_value = weight_value;
      total_value += weight_value;
    }
    if (peak_value <= 0.0f || peak_value > 1.0f + 1e-6f) okay_flag = 0;
    if (total_value <= 0.0f) okay_flag = 0;
  }
  test_true(okay_flag, "every filter is a positive triangle no taller than one");
  /* The mel scale is not linear, so the filters have to widen with frequency. */
  if (bank_data) {
    int low_span = 0, high_span = 0;
    for (bin_index = 0; bin_index < 33; ++bin_index) {
      if (bank_data[0 * 33 + bin_index] > 0.0f) low_span += 1;
      if (bank_data[7 * 33 + bin_index] > 0.0f) high_span += 1;
    }
    test_true(high_span > low_span, "the filters widen with frequency");
  }
  mem_free(bank_data);

  /* A pure tone lands in the filter that covers it. */
  memset(&clip, 0, sizeof(clip));
  clip.rate_value = 8000;
  clip.value_count = 512;
  clip.value_data = (float *)mem_clear(sizeof(float) * 512);
  if (!clip.value_data) { test_true(0, "the tone is allocated"); return; }
  for (slot_index = 0; slot_index < 512; ++slot_index)
    clip.value_data[slot_index] =
        (float)sin(6.283185307179586 * 1000.0 * (double)slot_index / 8000.0);
  test_true(mel_make(&clip, 16, 128, 64, 128, 1e-10f, 0, &mel_grid) == APP_OKAY,
            "a spectrogram is taken");
  test_true(mel_grid.high_count == 1 + (512 - (128 + 1)) / 64,
            "the frame count follows the hop, over a frame cut one sample long");
  test_true(mel_grid.wide_count == 16, "there is one column per filter");
  {
    const float *row_data = grid_at(&mel_grid, 2, 0);
    int best_index = 0;
    float want_edge;
    for (mel_index = 1; mel_index < 16; ++mel_index)
      if (row_data[mel_index] > row_data[best_index]) best_index = mel_index;
    /* Where the peak belongs, from the mel scale alone. */
    want_edge = mel_from_hertz(1000.0f) / (mel_from_hertz(4000.0f) / 17.0f);
    test_true(best_index >= (int)want_edge - 2 && best_index <= (int)want_edge + 1,
              "a pure tone peaks in the filter that covers it");
  }
  grid_free(&mel_grid);
  wave_free(&clip);

  /* How many frames a clip makes is how many soft tokens it becomes, so the
   * count is held against the reference's own at the shipped export's framing:
   * a 320 sample window, a 160 sample hop, and half a window of lead. The
   * lengths below are ones `Gemma4AudioFeatureExtractor` was measured on, and
   * the frame counts are the ones it marks live in `input_features_mask`.
   *
   * The trap here is that the extractor also pads a clip out to a whole block
   * of a hundred and twenty eight samples and hands back the extra frames. They
   * are masked, zeroed between the convolution stages and dropped from the rows
   * the tower returns, so counting them would add a soft token the reference
   * never asks for: 4000 samples make 24 live frames and 25 padded, and 24
   * frames are six tokens where 25 would be seven. */
  {
    static const struct {
      int sample_count;
      int frame_count;
    } reach_list[] = {{800, 4}, {2000, 12}, {4000, 24}, {24000, 149}, {24065, 150}, {40000, 249}};
    size_t reach_index;
    for (reach_index = 0; reach_index < sizeof(reach_list) / sizeof(reach_list[0]); ++reach_index) {
      wave_clip reach_clip;
      flat_grid reach_grid;
      int slot;
      memset(&reach_clip, 0, sizeof(reach_clip));
      reach_clip.rate_value = 16000;
      reach_clip.value_count = reach_list[reach_index].sample_count;
      reach_clip.value_data =
          (float *)mem_clear(sizeof(float) * (size_t)reach_clip.value_count);
      if (!reach_clip.value_data) { test_true(0, "the clip is allocated"); return; }
      for (slot = 0; slot < reach_clip.value_count; ++slot)
        reach_clip.value_data[slot] = (float)sin((double)slot * 0.05);
      test_true(mel_make(&reach_clip, 8, 320, 160, 512, 1e-3f, 160, &reach_grid) == APP_OKAY,
                "a spectrogram is taken at the shipped framing");
      test_true(reach_grid.high_count == reach_list[reach_index].frame_count,
                "the frame count is the one the extractor marks live");
      grid_free(&reach_grid);
      wave_free(&reach_clip);
    }
  }
}

/* ======================================================================== */
/* 8. public surface                                                        */
/* ======================================================================== */

/* ======================================================================== */
/* 9. whole graph on a synthetic checkpoint                                 */
/* ======================================================================== */

/* A safetensors file under construction.  The values stay reachable by name
 * after the file is written, which is what lets a reference forward read the
 * same weights the engine mapped without parsing the file a second time. */
#define TEST_KIT_LIMIT 280

typedef struct test_kit {
  char   name_list[TEST_KIT_LIMIT][160];
  int    size_list[TEST_KIT_LIMIT][4];
  int    rank_list[TEST_KIT_LIMIT];
  size_t from_list[TEST_KIT_LIMIT];
  size_t span_list[TEST_KIT_LIMIT];
  int    item_count;
  size_t body_size;
  float *body_data;
} test_kit;

static void test_kit_add(test_kit *pack, const char *name_text, int size_a, int size_b,
                          int size_c, int size_d) {
  int slot = pack->item_count;
  int axis_list[4];
  int axis_index, rank_count = 0;
  size_t value_count = 1;
  if (slot >= TEST_KIT_LIMIT) return;
  axis_list[0] = size_a;
  axis_list[1] = size_b;
  axis_list[2] = size_c;
  axis_list[3] = size_d;
  for (axis_index = 0; axis_index < 4 && axis_list[axis_index] > 0; ++axis_index) {
    value_count *= (size_t)axis_list[axis_index];
    rank_count += 1;
  }
  pack->item_count += 1;
  snprintf(pack->name_list[slot], sizeof(pack->name_list[slot]), "%s", name_text);
  for (axis_index = 0; axis_index < 4; ++axis_index)
    pack->size_list[slot][axis_index] = axis_list[axis_index];
  pack->rank_list[slot] = rank_count;
  pack->from_list[slot] = pack->body_size;
  pack->span_list[slot] = value_count * sizeof(float);
  pack->body_size += pack->span_list[slot];
}

/* Allocates the payload and gives every tensor its own values.  Leaving whole
 * families equal would hide any mistake about which one is picked. */
static int test_kit_open(test_kit *pack) {
  size_t value_index, value_count;
  int slot;
  pack->body_data = (float *)mem_clear(pack->body_size);
  if (!pack->body_data) return 0;
  for (slot = 0; slot < pack->item_count; ++slot) {
    size_t from_slot = pack->from_list[slot] / sizeof(float);
    value_count = pack->span_list[slot] / sizeof(float);
    for (value_index = 0; value_index < value_count; ++value_index)
      pack->body_data[from_slot + value_index] =
          (float)(0.6 * sin((double)(value_index * 13 + (size_t)slot * 7 + 1) * 0.37));
  }
  return 1;
}

static float *test_kit_find(test_kit *pack, const char *name_text) {
  int slot;
  for (slot = 0; slot < pack->item_count; ++slot)
    if (strcmp(pack->name_list[slot], name_text) == 0)
      return pack->body_data + pack->from_list[slot] / sizeof(float);
  return NULL;
}

static int test_kit_save(const test_kit *pack, const char *leaf_text) {
  char *header_text;
  size_t header_size, pad_count, file_size, header_fill = 0;
  uint8_t *file_data;
  uint64_t header_count;
  int slot, byte_index, okay_flag;

  if (!pack->body_data) return 0;
  header_size = 256 * (size_t)pack->item_count + 64;
  header_text = (char *)mem_clear(header_size);
  if (!header_text) return 0;
  header_fill += (size_t)snprintf(header_text, header_size, "{");
  for (slot = 0; slot < pack->item_count; ++slot) {
    char shape_text[64];
    int wrote, axis_index, shape_fill = 1;
    shape_text[0] = '[';
    for (axis_index = 0; axis_index < pack->rank_list[slot]; ++axis_index)
      shape_fill += snprintf(shape_text + shape_fill, sizeof(shape_text) - (size_t)shape_fill,
                             "%s%d", axis_index ? "," : "", pack->size_list[slot][axis_index]);
    snprintf(shape_text + shape_fill, sizeof(shape_text) - (size_t)shape_fill, "]");
    wrote = snprintf(header_text + header_fill, header_size - header_fill,
                     "%s\"%s\":{\"dtype\":\"F32\",\"shape\":%s,"
                     "\"data_offsets\":[%lu,%lu]}",
                     slot ? "," : "", pack->name_list[slot], shape_text,
                     (unsigned long)pack->from_list[slot],
                     (unsigned long)(pack->from_list[slot] + pack->span_list[slot]));
    /* The header room is generous, but a truncated entry would silently
     * produce a broken checkpoint, so refuse instead. */
    if (wrote < 0 || (size_t)wrote >= header_size - header_fill) {
      mem_free(header_text);
      return 0;
    }
    header_fill += (size_t)wrote;
  }
  if (header_fill + 1 >= header_size) { mem_free(header_text); return 0; }
  header_text[header_fill++] = '}';
  header_text[header_fill] = '\0';

  pad_count = (8 - (header_fill % 8)) % 8;
  header_count = (uint64_t)(header_fill + pad_count);
  file_size = 8 + (size_t)header_count + pack->body_size;
  file_data = (uint8_t *)mem_clear(file_size);
  if (!file_data) { mem_free(header_text); return 0; }
  for (byte_index = 0; byte_index < 8; ++byte_index)
    file_data[byte_index] = (uint8_t)((header_count >> (8 * byte_index)) & 0xFFu);
  memcpy(file_data + 8, header_text, header_fill);
  for (byte_index = 0; byte_index < (int)pad_count; ++byte_index)
    file_data[8 + header_fill + byte_index] = ' ';
  memcpy(file_data + 8 + (size_t)header_count, pack->body_data, pack->body_size);
  okay_flag = test_file_write(leaf_text, file_data, file_size);
  mem_free(file_data);
  mem_free(header_text);
  return okay_flag;
}

/* The text stack of the miniature checkpoint. */
#define TEST_WING_STATE  16
#define TEST_WING_INNER  24
#define TEST_WING_LAYERS 2
#define TEST_WING_HEADS  2
#define TEST_WING_HEAD   8
#define TEST_WING_KV     1
#define TEST_WING_PLE    4
#define TEST_WING_VOCAB  27
#define TEST_WING_EXPERT 4
#define TEST_WING_WIDTH  6

static void test_wing_pack(test_kit *pack, int moe_flag) {
  char name_text[96];
  int layer_index;
  test_kit_add(pack, "model.embed_tokens.weight", TEST_WING_VOCAB, TEST_WING_STATE, 0, 0);
  test_kit_add(pack, "model.embed_tokens_per_layer.weight", 32,
                TEST_WING_LAYERS * TEST_WING_PLE, 0, 0);
  test_kit_add(pack, "model.per_layer_model_projection.weight", TEST_WING_LAYERS * TEST_WING_PLE,
                TEST_WING_STATE, 0, 0);
  test_kit_add(pack, "model.per_layer_projection_norm.weight", TEST_WING_PLE, 0, 0, 0);
  test_kit_add(pack, "model.norm.weight", TEST_WING_STATE, 0, 0, 0);
  for (layer_index = 0; layer_index < TEST_WING_LAYERS; ++layer_index) {
#define TEST_WING_NAME(leaf) \
  (snprintf(name_text, sizeof(name_text), "model.layers.%d.%s", layer_index, leaf), name_text)
    test_kit_add(pack, TEST_WING_NAME("self_attn.q_proj.weight"), TEST_WING_HEADS * TEST_WING_HEAD,
                  TEST_WING_STATE, 0, 0);
    test_kit_add(pack, TEST_WING_NAME("self_attn.k_proj.weight"), TEST_WING_KV * TEST_WING_HEAD,
                  TEST_WING_STATE, 0, 0);
    test_kit_add(pack, TEST_WING_NAME("self_attn.v_proj.weight"), TEST_WING_KV * TEST_WING_HEAD,
                  TEST_WING_STATE, 0, 0);
    test_kit_add(pack, TEST_WING_NAME("self_attn.o_proj.weight"), TEST_WING_STATE,
                  TEST_WING_HEADS * TEST_WING_HEAD, 0, 0);
    test_kit_add(pack, TEST_WING_NAME("self_attn.q_norm.weight"), TEST_WING_HEAD, 0, 0, 0);
    test_kit_add(pack, TEST_WING_NAME("self_attn.k_norm.weight"), TEST_WING_HEAD, 0, 0, 0);
    test_kit_add(pack, TEST_WING_NAME("mlp.gate_proj.weight"), TEST_WING_INNER, TEST_WING_STATE, 0, 0);
    test_kit_add(pack, TEST_WING_NAME("mlp.up_proj.weight"), TEST_WING_INNER, TEST_WING_STATE, 0, 0);
    test_kit_add(pack, TEST_WING_NAME("mlp.down_proj.weight"), TEST_WING_STATE, TEST_WING_INNER, 0, 0);
    test_kit_add(pack, TEST_WING_NAME("input_layernorm.weight"), TEST_WING_STATE, 0, 0, 0);
    test_kit_add(pack, TEST_WING_NAME("post_attention_layernorm.weight"), TEST_WING_STATE, 0, 0, 0);
    test_kit_add(pack, TEST_WING_NAME("pre_feedforward_layernorm.weight"), TEST_WING_STATE, 0, 0, 0);
    test_kit_add(pack, TEST_WING_NAME("post_feedforward_layernorm.weight"), TEST_WING_STATE, 0, 0, 0);
    test_kit_add(pack, TEST_WING_NAME("per_layer_input_gate.weight"), TEST_WING_PLE,
                  TEST_WING_STATE, 0, 0);
    test_kit_add(pack, TEST_WING_NAME("per_layer_projection.weight"), TEST_WING_STATE,
                  TEST_WING_PLE, 0, 0);
    test_kit_add(pack, TEST_WING_NAME("post_per_layer_input_norm.weight"), TEST_WING_STATE, 0, 0, 0);
    if (!moe_flag) continue;
    test_kit_add(pack, TEST_WING_NAME("router.proj.weight"), TEST_WING_EXPERT, TEST_WING_STATE, 0, 0);
    test_kit_add(pack, TEST_WING_NAME("router.scale"), TEST_WING_STATE, 0, 0, 0);
    test_kit_add(pack, TEST_WING_NAME("router.per_expert_scale"), TEST_WING_EXPERT, 0, 0, 0);
    test_kit_add(pack, TEST_WING_NAME("experts.gate_up_proj"), TEST_WING_EXPERT,
                  2 * TEST_WING_WIDTH, TEST_WING_STATE, 0);
    test_kit_add(pack, TEST_WING_NAME("experts.down_proj"), TEST_WING_EXPERT, TEST_WING_STATE,
                  TEST_WING_WIDTH, 0);
    test_kit_add(pack, TEST_WING_NAME("post_feedforward_layernorm_1.weight"), TEST_WING_STATE, 0, 0, 0);
    test_kit_add(pack, TEST_WING_NAME("pre_feedforward_layernorm_2.weight"), TEST_WING_STATE, 0, 0, 0);
    test_kit_add(pack, TEST_WING_NAME("post_feedforward_layernorm_2.weight"), TEST_WING_STATE, 0, 0, 0);
#undef TEST_WING_NAME
  }
}

/* The configuration beside it.  `media_flag` adds the two tower blocks, which
 * are the only thing that tells the loader to look for them at all. */
static int test_wing_config(int moe_flag, int media_flag) {
  char config_text[2048];
  int fill_count;
  fill_count = snprintf(config_text, sizeof(config_text),
           "{\"vocab_size\":%d,\"hidden_size\":%d,\"intermediate_size\":%d,"
           "\"num_hidden_layers\":%d,\"num_attention_heads\":%d,\"num_key_value_heads\":%d,"
           "\"head_dim\":%d,\"global_head_dim\":%d,\"sliding_window\":4,"
           "\"vocab_size_per_layer_input\":32,\"hidden_size_per_layer_input\":%d,"
           "\"num_kv_shared_layers\":0,\"rms_norm_eps\":1e-6,\"max_position_embeddings\":64,"
           "\"bos_token_id\":1,\"eos_token_id\":2,\"pad_token_id\":0,"
           "\"layer_types\":[\"sliding_attention\",\"full_attention\"],"
           "\"enable_moe_block\":%s,\"num_experts\":%d,\"top_k_experts\":2,"
           "\"moe_intermediate_size\":%d",
           TEST_WING_VOCAB, TEST_WING_STATE, TEST_WING_INNER, TEST_WING_LAYERS, TEST_WING_HEADS,
           TEST_WING_KV, TEST_WING_HEAD, TEST_WING_HEAD, TEST_WING_PLE,
           moe_flag ? "true" : "false", TEST_WING_EXPERT, TEST_WING_WIDTH);
  if (media_flag && fill_count > 0)
    fill_count += snprintf(config_text + fill_count, sizeof(config_text) - (size_t)fill_count,
             ",\"image_token_id\":5,\"audio_token_id\":6,"
             "\"boi_token_id\":20,\"eoi_token_id\":21,"
             "\"boa_token_id\":22,\"eoa_token_index\":23,"
             "\"vision_soft_tokens_per_image\":9,"
             "\"vision_config\":{\"hidden_size\":12,\"intermediate_size\":16,"
             "\"num_hidden_layers\":1,\"num_attention_heads\":2,\"head_dim\":8,"
             "\"patch_size\":2,\"pooling_kernel_size\":2,\"position_embedding_size\":16,"
             "\"standardize\":false,\"rms_norm_eps\":1e-6,"
             "\"rope_parameters\":{\"rope_theta\":100.0,\"rope_type\":\"default\"}},"
             "\"audio_config\":{\"hidden_size\":12,\"num_hidden_layers\":1,"
             "\"num_attention_heads\":2,\"rms_norm_eps\":1e-6,\"conv_kernel_size\":3,"
             "\"attention_chunk_size\":4,\"attention_context_left\":3,"
             "\"attention_context_right\":0,\"attention_logit_cap\":5.0,"
             "\"residual_weight\":0.5,\"output_proj_dims\":12}");
  if (fill_count < 0 || (size_t)fill_count + 2 >= sizeof(config_text)) return 0;
  config_text[fill_count++] = '}';
  config_text[fill_count] = '\0';
  if (media_flag) {
    /* The analysis window lives beside the model, as the export writes it, and
     * the clip's token budget beside that, in the processor's own file.  Four
     * frames of four samples at eight kilohertz is one soft token, so a token
     * here is two milliseconds and sixty-four of them are a budget a fixture
     * can be written past without the tower costing anything. */
    static const char sound_text[] =
        "{\"feature_size\":8,\"sampling_rate\":8000,\"frame_length\":8,"
        "\"hop_length\":4,\"fft_length\":16,\"mel_floor\":0.001}";
    static const char budget_text[] = "{\"audio_seq_length\":64,\"audio_ms_per_token\":2}";
    if (!test_file_write("preprocessor_config.json", sound_text, sizeof(sound_text) - 1)) return 0;
    if (!test_file_write("processor_config.json", budget_text, sizeof(budget_text) - 1)) return 0;
  }
  return test_file_write("config.json", config_text, (size_t)fill_count);
}

/* Writes a complete miniature checkpoint so the whole graph can be exercised. */
static int test_wing_write(int moe_flag) {
  test_kit *pack = (test_kit *)mem_clear(sizeof(test_kit));
  int okay_flag;
  if (!pack) return 0;
  test_wing_pack(pack, moe_flag);
  okay_flag = test_kit_open(pack) && test_kit_save(pack, "model.safetensors");
  mem_free(pack->body_data);
  mem_free(pack);
  if (!test_wing_config(moe_flag, 0)) okay_flag = 0;
  if (!test_file_write("tokenizer.json", test_token_json, sizeof(test_token_json) - 1)) okay_flag = 0;
  return okay_flag;
}

/* A batched prefill has to land on exactly the same state as a token at a time. */
static void test_wing(void) {
  static const int32_t id_list[21] = {1, 7, 8, 9, 10, 11, 12, 13, 7,  8, 9,
                                      10, 11, 12, 13, 7, 8, 9, 10, 11, 12};
  int moe_flag;
  test_open("wing");
  for (moe_flag = 0; moe_flag < 2; ++moe_flag) {
    app_setup setup = app_setup_plain();
    app_model *model = NULL;
    app_session *wide_session = NULL;
    app_session *thin_session = NULL;
    float *want_list = NULL;
    const float *have_list;
    int okay_flag = 1, slot;
    setup.thread_count = 2;
    if (!test_wing_write(moe_flag)) { test_true(0, "the synthetic checkpoint is written"); return; }
    test_true(model_load(test_yard_path, &setup, &model) == APP_OKAY,
              moe_flag ? "a mixture checkpoint loads" : "a dense checkpoint loads");
    if (!model) return;
    /* What one step reads is a part of what the model holds, and for the
     * mixture branch the part is nameable exactly: the router picks
     * `expert_top` of the bank, so raising that to the whole bank has to move
     * the figure by precisely the experts that were being left out.  Counting
     * the bank whole, or counting none of it, both fail this. */
    {
      size_t read_bytes = model_decode_bytes(model);
      test_true(read_bytes > 0, "a decode step reads weights");
      test_true(read_bytes < model_memory_bytes(model),
                "a decode step reads less than the model holds");
      if (moe_flag) {
        size_t rest_bytes = 0, wide_bytes;
        int keep_top = model->form.expert_top;
        int layer_index, expert_index;
        test_true(keep_top < model->form.expert_count,
                  "the fixture holds more experts than a step picks");
        for (layer_index = 0; layer_index < model->form.layer_count; ++layer_index) {
          const layer_wing *wing = &model->wing_list[layer_index];
          for (expert_index = keep_top; expert_index < model->form.expert_count; ++expert_index)
            rest_bytes += plane_bytes(&wing->expert_rise_list[expert_index]) +
                          plane_bytes(&wing->expert_drop_list[expert_index]);
        }
        model->form.expert_top = model->form.expert_count;
        wide_bytes = model_decode_bytes(model);
        model->form.expert_top = keep_top; /* the session below routes with it */
        test_true(rest_bytes > 0, "the bank holds experts a step does not read");
        test_true(wide_bytes - read_bytes == rest_bytes,
                  "only the experts the router picks are in the read");
      }
    }

    /* The phase split is the same sweep counted a second way, so the two
     * totals have to agree to the byte.  A sheet added to one and forgotten in
     * the other is what this catches, and it is the only thing that keeps the
     * rate a phase reports from being quietly wrong. */
    {
      size_t phase_list[APP_PHASE_COUNT];
      size_t phase_total = 0;
      int phase_slot;
      model_phase_bytes(model, phase_list);
      for (phase_slot = 0; phase_slot < APP_PHASE_COUNT; ++phase_slot)
        phase_total += phase_list[phase_slot];
      test_true(phase_total == model_decode_bytes(model),
                "the phase split counts the same bytes as the sweep");
      test_true(phase_list[APP_PHASE_MLP] > 0 && phase_list[APP_PHASE_HEAD] > 0,
                "the sheets a step sweeps land in a named phase");
      test_true(phase_list[APP_PHASE_PICK] == 0,
                "the sampler reads no weights");
      test_true((phase_list[APP_PHASE_MOE] > 0) == (moe_flag != 0),
                "the mixture's bytes are the mixture's");
    }

    test_true(session_open(model, &thin_session) == APP_OKAY, "a session opens");
    test_true(session_open(model, &wide_session) == APP_OKAY, "a second session opens");
    if (!thin_session || !wide_session) { model_free(model); return; }

    for (slot = 0; slot < 21; ++slot) have_list = session_step(thin_session, id_list[slot]);
    want_list = (float *)mem_clear(sizeof(float) * 27);
    memcpy(want_list, have_list, sizeof(float) * 27);

    test_true(session_prime(wide_session, id_list, 21) == APP_OKAY, "a batched prefill runs");
    have_list = session_step(wide_session, id_list[20]);
    test_true(have_list != NULL, "the batched pass reaches the head");
    if (have_list) {
      for (slot = 0; slot < 27; ++slot) {
        float gap = have_list[slot] - want_list[slot];
        if (gap < 0.0f) gap = -gap;
        if (gap > 1e-3f) okay_flag = 0;
      }
    }
    test_true(okay_flag, "a batched prefill matches one token at a time");
    test_true(session_fill(wide_session) == session_fill(thin_session),
              "both routes land on the same position");

    /* The attention's two jobs divide along axes chosen so that neither can
     * move a number: the scores by position, which are independent, and the
     * blends by head, so a running sum is never regrouped.  The claim is that
     * a logit row is bit for bit the same however many threads carried it, and
     * bits are what this compares — not a tolerance, which would pass on a
     * regrouped sum too. */
    {
      app_setup lone_setup = setup;
      app_model *lone_model = NULL;
      lone_setup.thread_count = 1;
      if (model_load(test_yard_path, &lone_setup, &lone_model) == APP_OKAY && lone_model) {
        app_session *lone_session = NULL;
        if (session_open(lone_model, &lone_session) == APP_OKAY && lone_session) {
          const float *lone_list;
          int same_flag = 1;
          test_true(session_prime(lone_session, id_list, 21) == APP_OKAY,
                    "a one thread prefill runs");
          lone_list = session_step(lone_session, id_list[20]);
          if (!lone_list) same_flag = 0;
          else
            for (slot = 0; slot < 27; ++slot)
              if (memcmp(&lone_list[slot], &have_list[slot], sizeof(float)) != 0) same_flag = 0;
          test_true(same_flag, "one thread and two agree to the bit");
          session_close(lone_session);
        }
        model_free(lone_model);
      }
    }

    /* The timer is a partition, and the property that makes it worth reading is
     * that the parts sum to the step rather than sample it.  A session that did
     * not ask for it records nothing at all, which is the other half of the
     * claim: a run that does not want the timer does not pay for it. */
    {
      app_phase_book quiet_book = session_phases(thin_session);
      app_setup loud_setup = setup;
      app_model *loud_model = NULL;
      test_true(quiet_book.step_count == 0 && quiet_book.read_count == 0,
                "the phase timer stays shut unless it is asked for");

      loud_setup.verbose_level = 1;
      if (model_load(test_yard_path, &loud_setup, &loud_model) == APP_OKAY && loud_model) {
        app_session *loud_session = NULL;
        if (session_open(loud_model, &loud_session) == APP_OKAY && loud_session) {
          app_phase_book book;
          double part_total = 0.0;
          int phase_slot, name_count = 0;
          for (slot = 0; slot < 4; ++slot) session_step(loud_session, id_list[slot]);
          book = session_phases(loud_session);
          for (phase_slot = 0; phase_slot < APP_PHASE_COUNT; ++phase_slot) {
            part_total += book.seconds[phase_slot];
            if (book.counts[phase_slot] > 0) ++name_count;
            /* Every part is entered a whole number of times per step, and the
             * count is the same every step, so this divides exactly. */
            test_true(book.seconds[phase_slot] >= 0.0, "no part of a step takes negative time");
          }
          test_true(book.step_count == 4, "the timer divides every step and no other pass");
          test_true(name_count >= 6, "the step is divided into parts, not left in one");
          test_true(book.read_count > 0, "the timer reads the clock");
          /* The sampler is outside the step's own span, so the parts of the
           * pass cannot exceed it, and cannot fall far short of it either. */
          test_true(part_total <= book.step_seconds * 1.001 + 1e-6,
                    "the parts of a pass do not outrun the pass");
          test_true(part_total >= book.step_seconds * 0.95,
                    "the parts of a pass account for nearly all of it");
          {
            /* A reset clears the counts and leaves the timer armed, exactly as
             * it does the tally beside it. */
            app_phase_book after_book;
            session_reset(loud_session);
            after_book = session_phases(loud_session);
            test_true(after_book.step_count == 0 && after_book.read_count == 0,
                      "a reset clears the phase counts");
            session_step(loud_session, id_list[0]);
            after_book = session_phases(loud_session);
            test_true(after_book.step_count == 1, "a reset leaves the timer armed");
          }
          session_close(loud_session);
        }
        model_free(loud_model);
      }
    }
    mem_free(want_list);
    session_close(wide_session);
    session_close(thin_session);
    model_free(model);
  }
}

/* ======================================================================== */
/* 10. the vision and audio towers                                          */
/* ======================================================================== */

/* A miniature multi-modal checkpoint in the shipped export's arrangement — the
 * reference's module names, its wrapper around every projection, its two axis
 * position tables, its conformer — and a definition of the vision tower written
 * out separately here to hold the engine against.
 *
 * The audio tower is not reproduced a second time in C.  It is a conformer with
 * a dozen interacting parts, and a second transcription of it here would mostly
 * be a copy of the first; what it gets instead is a check on each piece that is
 * peculiar to it — the mean-subtracting norm, the depthwise causal convolution,
 * the gated unit, and the window the local attention actually spans — plus the
 * end-to-end run.  The tower as a whole is held against the real reference by
 * `app_diff.py --media`, which is a stronger test than anything written here
 * could be. */

#define TOWER_TEXT   16 /* the text stack's hidden width, from test_wing */
#define TOWER_HIDDEN 12
#define TOWER_INNER  16
#define TOWER_LAYERS 1
#define TOWER_HEADS  2
#define TOWER_HEAD   8  /* divisible by four: each axis takes half a head */
#define TOWER_PATCH  2
#define TOWER_POOL   2
#define TOWER_PLACE  16
#define TOWER_SOFT   9
#define SOUND_HEADS  2
#define SOUND_HEAD   6
#define SOUND_INNER  16
#define SOUND_MEL    8
#define SOUND_CONV   4
#define SOUND_DEEP   3
#define SOUND_CHUNK  4
#define SOUND_LEFT   3
#define SOUND_BUDGET 64 /* soft tokens one clip may spend, from the processor */
#define SOUND_SPAN   2  /* milliseconds one of them stands for */

static void test_tower_add(test_kit *pack) {
  char name_text[160];
  int layer_index;
#define VISION_AT(leaf)                                                                     \
  (snprintf(name_text, sizeof(name_text), "model.vision_tower.encoder.layers.%d.%s",         \
            layer_index, (leaf)),                                                             \
   name_text)
  for (layer_index = 0; layer_index < TOWER_LAYERS; ++layer_index) {
    test_kit_add(pack, VISION_AT("self_attn.q_proj.linear.weight"), TOWER_HEADS * TOWER_HEAD,
                 TOWER_HIDDEN, 0, 0);
    test_kit_add(pack, VISION_AT("self_attn.k_proj.linear.weight"), TOWER_HEADS * TOWER_HEAD,
                 TOWER_HIDDEN, 0, 0);
    test_kit_add(pack, VISION_AT("self_attn.v_proj.linear.weight"), TOWER_HEADS * TOWER_HEAD,
                 TOWER_HIDDEN, 0, 0);
    test_kit_add(pack, VISION_AT("self_attn.o_proj.linear.weight"), TOWER_HIDDEN,
                 TOWER_HEADS * TOWER_HEAD, 0, 0);
    test_kit_add(pack, VISION_AT("self_attn.q_norm.weight"), TOWER_HEAD, 0, 0, 0);
    test_kit_add(pack, VISION_AT("self_attn.k_norm.weight"), TOWER_HEAD, 0, 0, 0);
    test_kit_add(pack, VISION_AT("input_layernorm.weight"), TOWER_HIDDEN, 0, 0, 0);
    test_kit_add(pack, VISION_AT("post_attention_layernorm.weight"), TOWER_HIDDEN, 0, 0, 0);
    test_kit_add(pack, VISION_AT("pre_feedforward_layernorm.weight"), TOWER_HIDDEN, 0, 0, 0);
    test_kit_add(pack, VISION_AT("post_feedforward_layernorm.weight"), TOWER_HIDDEN, 0, 0, 0);
    test_kit_add(pack, VISION_AT("mlp.gate_proj.linear.weight"), TOWER_INNER, TOWER_HIDDEN, 0, 0);
    test_kit_add(pack, VISION_AT("mlp.up_proj.linear.weight"), TOWER_INNER, TOWER_HIDDEN, 0, 0);
    test_kit_add(pack, VISION_AT("mlp.down_proj.linear.weight"), TOWER_HIDDEN, TOWER_INNER, 0, 0);
  }
#undef VISION_AT
  test_kit_add(pack, "model.vision_tower.patch_embedder.input_proj.weight", TOWER_HIDDEN,
               3 * TOWER_PATCH * TOWER_PATCH, 0, 0);
  test_kit_add(pack, "model.vision_tower.patch_embedder.position_embedding_table", 2, TOWER_PLACE,
               TOWER_HIDDEN, 0);
  test_kit_add(pack, "model.embed_vision.embedding_projection.weight", TOWER_TEXT, TOWER_HIDDEN, 0, 0);

#define SOUND_AT(leaf)                                                                      \
  (snprintf(name_text, sizeof(name_text), "model.audio_tower.layers.%d.%s", layer_index,     \
            (leaf)),                                                                          \
   name_text)
  for (layer_index = 0; layer_index < TOWER_LAYERS; ++layer_index) {
    int side_index;
    for (side_index = 1; side_index <= 2; ++side_index) {
      char leaf_text[64];
      snprintf(leaf_text, sizeof(leaf_text), "feed_forward%d.ffw_layer_1.linear.weight", side_index);
      test_kit_add(pack, SOUND_AT(leaf_text), SOUND_INNER, TOWER_HIDDEN, 0, 0);
      snprintf(leaf_text, sizeof(leaf_text), "feed_forward%d.ffw_layer_2.linear.weight", side_index);
      test_kit_add(pack, SOUND_AT(leaf_text), TOWER_HIDDEN, SOUND_INNER, 0, 0);
      snprintf(leaf_text, sizeof(leaf_text), "feed_forward%d.pre_layer_norm.weight", side_index);
      test_kit_add(pack, SOUND_AT(leaf_text), TOWER_HIDDEN, 0, 0, 0);
      snprintf(leaf_text, sizeof(leaf_text), "feed_forward%d.post_layer_norm.weight", side_index);
      test_kit_add(pack, SOUND_AT(leaf_text), TOWER_HIDDEN, 0, 0, 0);
    }
    test_kit_add(pack, SOUND_AT("self_attn.q_proj.linear.weight"), SOUND_HEADS * SOUND_HEAD,
                 TOWER_HIDDEN, 0, 0);
    test_kit_add(pack, SOUND_AT("self_attn.k_proj.linear.weight"), SOUND_HEADS * SOUND_HEAD,
                 TOWER_HIDDEN, 0, 0);
    test_kit_add(pack, SOUND_AT("self_attn.v_proj.linear.weight"), SOUND_HEADS * SOUND_HEAD,
                 TOWER_HIDDEN, 0, 0);
    test_kit_add(pack, SOUND_AT("self_attn.post.linear.weight"), TOWER_HIDDEN,
                 SOUND_HEADS * SOUND_HEAD, 0, 0);
    test_kit_add(pack, SOUND_AT("self_attn.relative_k_proj.weight"), SOUND_HEADS * SOUND_HEAD,
                 TOWER_HIDDEN, 0, 0);
    test_kit_add(pack, SOUND_AT("self_attn.per_dim_scale"), SOUND_HEAD, 0, 0, 0);
    test_kit_add(pack, SOUND_AT("lconv1d.linear_start.linear.weight"), 2 * TOWER_HIDDEN,
                 TOWER_HIDDEN, 0, 0);
    test_kit_add(pack, SOUND_AT("lconv1d.linear_end.linear.weight"), TOWER_HIDDEN, TOWER_HIDDEN, 0, 0);
    test_kit_add(pack, SOUND_AT("lconv1d.depthwise_conv1d.weight"), TOWER_HIDDEN, 1, SOUND_DEEP, 0);
    test_kit_add(pack, SOUND_AT("lconv1d.pre_layer_norm.weight"), TOWER_HIDDEN, 0, 0, 0);
    test_kit_add(pack, SOUND_AT("lconv1d.conv_norm.weight"), TOWER_HIDDEN, 0, 0, 0);
    test_kit_add(pack, SOUND_AT("norm_pre_attn.weight"), TOWER_HIDDEN, 0, 0, 0);
    test_kit_add(pack, SOUND_AT("norm_post_attn.weight"), TOWER_HIDDEN, 0, 0, 0);
    test_kit_add(pack, SOUND_AT("norm_out.weight"), TOWER_HIDDEN, 0, 0, 0);
  }
#undef SOUND_AT
  test_kit_add(pack, "model.audio_tower.subsample_conv_projection.layer0.conv.weight", SOUND_CONV,
               1, 3, 3);
  test_kit_add(pack, "model.audio_tower.subsample_conv_projection.layer0.norm.weight", SOUND_CONV,
               0, 0, 0);
  test_kit_add(pack, "model.audio_tower.subsample_conv_projection.layer1.conv.weight", SOUND_CONV,
               SOUND_CONV, 3, 3);
  test_kit_add(pack, "model.audio_tower.subsample_conv_projection.layer1.norm.weight", SOUND_CONV,
               0, 0, 0);
  test_kit_add(pack, "model.audio_tower.subsample_conv_projection.input_proj_linear.weight",
               TOWER_HIDDEN, SOUND_CONV * 2, 0, 0);
  test_kit_add(pack, "model.audio_tower.output_proj.weight", TOWER_HIDDEN, TOWER_HIDDEN, 0, 0);
  test_kit_add(pack, "model.audio_tower.output_proj.bias", TOWER_HIDDEN, 0, 0, 0);
  test_kit_add(pack, "model.embed_audio.embedding_projection.weight", TOWER_TEXT, TOWER_HIDDEN, 0, 0);
}

/* -- an independent definition of the vision tower ------------------------ */

static void test_tower_norm(const float *value_list, const float *gain_list, int value_count,
                            float eps_value, float *value_out) {
  double square_total = 0.0, shrink_value;
  int value_index;
  for (value_index = 0; value_index < value_count; ++value_index)
    square_total += (double)value_list[value_index] * (double)value_list[value_index];
  shrink_value = pow(square_total / (double)value_count + (double)eps_value, -0.5);
  for (value_index = 0; value_index < value_count; ++value_index)
    value_out[value_index] = (float)((double)value_list[value_index] * shrink_value *
                                     (gain_list ? (double)gain_list[value_index] : 1.0));
}

static void test_tower_lift(const float *sheet_data, int row_count, int col_count,
                            const float *act_data, float *out_data) {
  int row_index, col_index;
  for (row_index = 0; row_index < row_count; ++row_index) {
    double total_value = 0.0;
    for (col_index = 0; col_index < col_count; ++col_index)
      total_value += (double)sheet_data[(size_t)row_index * (size_t)col_count + (size_t)col_index] *
                     (double)act_data[col_index];
    out_data[row_index] = (float)total_value;
  }
}

static void test_tower_soft(float *value_list, int value_count) {
  float peak_value = value_list[0], total_value = 0.0f;
  int value_index;
  for (value_index = 1; value_index < value_count; ++value_index)
    if (value_list[value_index] > peak_value) peak_value = value_list[value_index];
  for (value_index = 0; value_index < value_count; ++value_index) {
    value_list[value_index] = expf(value_list[value_index] - peak_value);
    total_value += value_list[value_index];
  }
  for (value_index = 0; value_index < value_count; ++value_index)
    value_list[value_index] /= total_value;
}

static float test_tower_gelu(float value) {
  double cube_value = (double)value * value * value;
  return (float)(0.5 * value *
                 (1.0 + tanh(0.7978845608028654 * ((double)value + 0.044715 * cube_value))));
}

/* The vision tower, written from the reference's description rather than from
 * the engine's code: patch, position tables, two dimensional rotary paired
 * inside each half of a head, attention scaled by one, pooling with the square
 * root of the hidden width, and a projector whose norm carries no scale. */
static void test_tower_vision(test_kit *pack, const flat_grid *grid, int wide_grid, int high_grid,
                              float *out_data) {
  static const float eps_value = 1e-6f;
  int lane_count = wide_grid * high_grid;
  int pool_wide = wide_grid / TOWER_POOL, pool_high = high_grid / TOWER_POOL;
  int head_wide = TOWER_HEADS * TOWER_HEAD;
  float *state_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * TOWER_HIDDEN);
  float *query_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)head_wide);
  float *key_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)head_wide);
  float *value_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)head_wide);
  float *blend_data = (float *)mem_clear(sizeof(float) * (size_t)lane_count * (size_t)head_wide);
  const float *patch_sheet =
      test_kit_find(pack, "model.vision_tower.patch_embedder.input_proj.weight");
  const float *place_table =
      test_kit_find(pack, "model.vision_tower.patch_embedder.position_embedding_table");
  const float *lift_sheet = test_kit_find(pack, "model.embed_vision.embedding_projection.weight");
  float patch_list[3 * TOWER_PATCH * TOWER_PATCH];
  float scrap_list[64], score_list[256], lift_list[64];
  int lane_index, layer_index, head_index, span_index, value_index;

  if (!state_data || !query_data || !key_data || !value_data || !blend_data) goto vision_done;

  for (lane_index = 0; lane_index < lane_count; ++lane_index) {
    int wide_patch = lane_index % wide_grid, high_patch = lane_index / wide_grid;
    int band_index, high_index, wide_index;
    float *lane_data = state_data + (size_t)lane_index * TOWER_HIDDEN;
    for (high_index = 0; high_index < TOWER_PATCH; ++high_index)
      for (wide_index = 0; wide_index < TOWER_PATCH; ++wide_index) {
        const float *cell_data = grid_at(grid, high_patch * TOWER_PATCH + high_index,
                                         wide_patch * TOWER_PATCH + wide_index);
        for (band_index = 0; band_index < 3; ++band_index)
          patch_list[(high_index * TOWER_PATCH + wide_index) * 3 + band_index] =
              2.0f * (cell_data[band_index] - 0.5f);
      }
    test_tower_lift(patch_sheet, TOWER_HIDDEN, 3 * TOWER_PATCH * TOWER_PATCH, patch_list, lane_data);
    for (value_index = 0; value_index < TOWER_HIDDEN; ++value_index)
      lane_data[value_index] += place_table[wide_patch * TOWER_HIDDEN + value_index] +
                                place_table[(TOWER_PLACE + high_patch) * TOWER_HIDDEN + value_index];
  }

  for (layer_index = 0; layer_index < TOWER_LAYERS; ++layer_index) {
    char name_text[160];
#define VISION_FIND(leaf)                                                                  \
  (snprintf(name_text, sizeof(name_text), "model.vision_tower.encoder.layers.%d.%s",        \
            layer_index, (leaf)),                                                            \
   test_kit_find(pack, name_text))
    const float *enter_norm = VISION_FIND("input_layernorm.weight");
    const float *after_norm = VISION_FIND("post_attention_layernorm.weight");
    const float *query_sheet = VISION_FIND("self_attn.q_proj.linear.weight");
    const float *key_sheet = VISION_FIND("self_attn.k_proj.linear.weight");
    const float *value_sheet = VISION_FIND("self_attn.v_proj.linear.weight");
    const float *exit_sheet = VISION_FIND("self_attn.o_proj.linear.weight");
    const float *query_norm = VISION_FIND("self_attn.q_norm.weight");
    const float *key_norm = VISION_FIND("self_attn.k_norm.weight");
    const float *before_feed = VISION_FIND("pre_feedforward_layernorm.weight");
    const float *after_feed = VISION_FIND("post_feedforward_layernorm.weight");
    const float *gate_sheet = VISION_FIND("mlp.gate_proj.linear.weight");
    const float *rise_sheet = VISION_FIND("mlp.up_proj.linear.weight");
    const float *drop_sheet = VISION_FIND("mlp.down_proj.linear.weight");
#undef VISION_FIND

    for (lane_index = 0; lane_index < lane_count; ++lane_index) {
      float *query_lane = query_data + (size_t)lane_index * (size_t)head_wide;
      float *key_lane = key_data + (size_t)lane_index * (size_t)head_wide;
      float *value_lane = value_data + (size_t)lane_index * (size_t)head_wide;
      int wide_place = lane_index % wide_grid, high_place = lane_index / wide_grid;
      test_tower_norm(state_data + (size_t)lane_index * TOWER_HIDDEN, enter_norm, TOWER_HIDDEN,
                      eps_value, scrap_list);
      test_tower_lift(query_sheet, head_wide, TOWER_HIDDEN, scrap_list, query_lane);
      test_tower_lift(key_sheet, head_wide, TOWER_HIDDEN, scrap_list, key_lane);
      test_tower_lift(value_sheet, head_wide, TOWER_HIDDEN, scrap_list, value_lane);
      for (head_index = 0; head_index < TOWER_HEADS; ++head_index) {
        float *query_head = query_lane + head_index * TOWER_HEAD;
        float *key_head = key_lane + head_index * TOWER_HEAD;
        float *value_head = value_lane + head_index * TOWER_HEAD;
        int part_index, pair_index;
        test_tower_norm(query_head, query_norm, TOWER_HEAD, eps_value, query_head);
        test_tower_norm(key_head, key_norm, TOWER_HEAD, eps_value, key_head);
        test_tower_norm(value_head, NULL, TOWER_HEAD, eps_value, value_head);
        /* Half a head to each axis, rotated as pairs inside that half. */
        for (part_index = 0; part_index < 2; ++part_index) {
          int place_value = part_index == 0 ? wide_place : high_place;
          int half_size = TOWER_HEAD / 4;
          for (pair_index = 0; pair_index < half_size; ++pair_index) {
            double step_value =
                1.0 / pow(100.0, (double)(2 * pair_index) / (double)(TOWER_HEAD / 2));
            double angle_value = (double)place_value * step_value;
            float cos_value = (float)cos(angle_value), sin_value = (float)sin(angle_value);
            int slot = part_index * (TOWER_HEAD / 2) + pair_index;
            float low_query = query_head[slot], high_query = query_head[slot + half_size];
            float low_key = key_head[slot], high_key = key_head[slot + half_size];
            query_head[slot] = low_query * cos_value - high_query * sin_value;
            query_head[slot + half_size] = high_query * cos_value + low_query * sin_value;
            key_head[slot] = low_key * cos_value - high_key * sin_value;
            key_head[slot + half_size] = high_key * cos_value + low_key * sin_value;
          }
        }
      }
    }

    for (lane_index = 0; lane_index < lane_count; ++lane_index)
      for (head_index = 0; head_index < TOWER_HEADS; ++head_index) {
        const float *query_head =
            query_data + (size_t)lane_index * (size_t)head_wide + head_index * TOWER_HEAD;
        float *blend_head =
            blend_data + (size_t)lane_index * (size_t)head_wide + head_index * TOWER_HEAD;
        for (span_index = 0; span_index < lane_count; ++span_index) {
          const float *key_head =
              key_data + (size_t)span_index * (size_t)head_wide + head_index * TOWER_HEAD;
          double total_value = 0.0;
          for (value_index = 0; value_index < TOWER_HEAD; ++value_index)
            total_value += (double)query_head[value_index] * (double)key_head[value_index];
          score_list[span_index] = (float)total_value; /* the norms absorb the scale */
        }
        test_tower_soft(score_list, lane_count);
        for (value_index = 0; value_index < TOWER_HEAD; ++value_index) blend_head[value_index] = 0.0f;
        for (span_index = 0; span_index < lane_count; ++span_index) {
          const float *value_head =
              value_data + (size_t)span_index * (size_t)head_wide + head_index * TOWER_HEAD;
          for (value_index = 0; value_index < TOWER_HEAD; ++value_index)
            blend_head[value_index] += score_list[span_index] * value_head[value_index];
        }
      }

    for (lane_index = 0; lane_index < lane_count; ++lane_index) {
      float *lane_data = state_data + (size_t)lane_index * TOWER_HIDDEN;
      float gate_list[TOWER_INNER], rise_list[TOWER_INNER];
      test_tower_lift(exit_sheet, TOWER_HIDDEN, head_wide,
                      blend_data + (size_t)lane_index * (size_t)head_wide, lift_list);
      test_tower_norm(lift_list, after_norm, TOWER_HIDDEN, eps_value, lift_list);
      for (value_index = 0; value_index < TOWER_HIDDEN; ++value_index)
        lane_data[value_index] += lift_list[value_index];
      test_tower_norm(lane_data, before_feed, TOWER_HIDDEN, eps_value, scrap_list);
      test_tower_lift(gate_sheet, TOWER_INNER, TOWER_HIDDEN, scrap_list, gate_list);
      test_tower_lift(rise_sheet, TOWER_INNER, TOWER_HIDDEN, scrap_list, rise_list);
      for (value_index = 0; value_index < TOWER_INNER; ++value_index)
        gate_list[value_index] = test_tower_gelu(gate_list[value_index]) * rise_list[value_index];
      test_tower_lift(drop_sheet, TOWER_HIDDEN, TOWER_INNER, gate_list, lift_list);
      test_tower_norm(lift_list, after_feed, TOWER_HIDDEN, eps_value, lift_list);
      for (value_index = 0; value_index < TOWER_HIDDEN; ++value_index)
        lane_data[value_index] += lift_list[value_index];
    }
  }

  {
    int high_pool, wide_pool, high_index, wide_index;
    float root_gain = (float)sqrt((double)TOWER_HIDDEN);
    for (high_pool = 0; high_pool < pool_high; ++high_pool)
      for (wide_pool = 0; wide_pool < pool_wide; ++wide_pool) {
        float pool_list[TOWER_HIDDEN];
        for (value_index = 0; value_index < TOWER_HIDDEN; ++value_index) pool_list[value_index] = 0.0f;
        for (high_index = high_pool * TOWER_POOL; high_index < (high_pool + 1) * TOWER_POOL;
             ++high_index)
          for (wide_index = wide_pool * TOWER_POOL; wide_index < (wide_pool + 1) * TOWER_POOL;
               ++wide_index) {
            const float *cell_data =
                state_data + (size_t)(high_index * wide_grid + wide_index) * TOWER_HIDDEN;
            for (value_index = 0; value_index < TOWER_HIDDEN; ++value_index)
              pool_list[value_index] += cell_data[value_index];
          }
        for (value_index = 0; value_index < TOWER_HIDDEN; ++value_index)
          pool_list[value_index] *= root_gain / (float)(TOWER_POOL * TOWER_POOL);
        /* The projector's norm carries no scale. */
        test_tower_norm(pool_list, NULL, TOWER_HIDDEN, eps_value, pool_list);
        test_tower_lift(lift_sheet, TOWER_TEXT, TOWER_HIDDEN, pool_list,
                        out_data + (size_t)(high_pool * pool_wide + wide_pool) * TOWER_TEXT);
      }
  }

vision_done:
  mem_free(state_data);
  mem_free(query_data);
  mem_free(key_data);
  mem_free(value_data);
  mem_free(blend_data);
}

/* -- the pieces peculiar to the conformer --------------------------------- */

static void test_sound_parts(void) {
  /* The mean-subtracting norm the subsampler uses, against its definition. */
  {
    float value_list[5] = {1.0f, -2.0f, 3.0f, 0.5f, -1.5f};
    float gain_list[5] = {2.0f, 1.0f, 0.5f, 1.5f, 1.0f};
    float out_list[5];
    double mean_value = 0.0, spread_value = 0.0;
    int slot;
    for (slot = 0; slot < 5; ++slot) mean_value += value_list[slot];
    mean_value /= 5.0;
    for (slot = 0; slot < 5; ++slot)
      spread_value += (value_list[slot] - mean_value) * (value_list[slot] - mean_value);
    spread_value = pow(spread_value / 5.0 + 1e-6, -0.5);
    kern_norm_layer(value_list, gain_list, 5, 1e-6f, out_list);
    for (slot = 0; slot < 5; ++slot)
      test_near((double)out_list[slot],
                ((double)value_list[slot] - mean_value) * spread_value * gain_list[slot], 1e-6,
                "the layer norm subtracts the mean before it scales");
  }
  /* And that it is not the root-mean-square norm wearing its name. */
  {
    float value_list[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    float out_list[4];
    kern_norm_layer(value_list, NULL, 4, 1e-6f, out_list);
    test_true(fabs((double)out_list[0]) < 1e-2, "a constant row normalizes to nothing");
  }
  {
    float value_list[3] = {0.0f, 1.0f, -1.0f};
    test_near((double)kern_silu(0.0f), 0.0, 1e-9, "silu is zero at zero");
    test_near((double)kern_silu(value_list[1]), 1.0 / (1.0 + exp(-1.0)), 1e-6,
              "silu is x over one plus e to the minus x");
    test_near((double)kern_silu(value_list[2]), -1.0 / (1.0 + exp(1.0)), 1e-6,
              "silu is negative below zero");
    test_near((double)kern_soft_plus(0.0f), log(2.0), 1e-6, "softplus of zero is log two");
    test_near((double)kern_soft_plus(30.0f), 30.0, 1e-3, "softplus is linear when large");
  }
}

static void test_tower(void) {
  test_kit *pack;
  app_setup setup = app_setup_plain();
  app_model *model = NULL;
  app_session *session = NULL;
  app_media media;
  char path_text[1024];
  flat_grid grid, work_grid;
  float *want_list = NULL;
  int row_index, value_index, okay_flag = 1;
  int wide_grid = 0, high_grid = 0;

  test_open("tower");
  test_sound_parts();

  pack = (test_kit *)mem_clear(sizeof(test_kit));
  if (!pack) { test_true(0, "the fixture is allocated"); return; }
  test_wing_pack(pack, 0);
  test_tower_add(pack);
  if (!test_kit_open(pack) || !test_kit_save(pack, "model.safetensors")) {
    test_true(0, "the multi-modal checkpoint is written");
    mem_free(pack->body_data);
    mem_free(pack);
    return;
  }
  test_true(test_wing_config(0, 1), "the multi-modal configuration is written");
  test_true(test_file_write("tokenizer.json", test_token_json, sizeof(test_token_json) - 1),
            "the tokenizer is written");
  test_true(test_file_write("image.png", test_png_data, sizeof(test_png_data)),
            "the image is written");
  test_true(test_wave_write("clip.wav", 8000, 600), "the clip is written");

  setup.thread_count = 2;
  test_true(model_load(test_yard_path, &setup, &model) == APP_OKAY,
            "a checkpoint with towers loads");
  if (!model) { mem_free(pack->body_data); mem_free(pack); return; }
  test_true(model_vision_ready(model), "the vision tower is bound");
  test_true(model_audio_ready(model), "the audio tower is bound");
  test_true(model_image_token(model) == 5, "the image placeholder id comes from the configuration");
  test_true(model_image_rows(model) == TOWER_SOFT, "the soft token cap comes from the configuration");
  test_true(model_audio_rows(model) == SOUND_BUDGET && model_audio_span_ms(model) == SOUND_SPAN,
            "the clip's token budget comes from the processor configuration");

  /* -- vision -------------------------------------------------------- */
  path_join(path_text, sizeof(path_text), test_yard_path, "image.png");
  test_true(media_image(model, path_text, &media) == APP_OKAY, "an image runs the vision tower");
  /* The picture is twelve by eight and the budget allows nine soft tokens, so
   * the resize lands on twelve by eight pixels, a six by four patch grid, and a
   * three by two pooled grid. */
  test_true(media.row_count == 6, "the tower pools the grid the budget allows");
  wide_grid = 6;
  high_grid = 4;
  want_list = (float *)mem_clear(sizeof(float) * 6u * TOWER_TEXT);
  if (image_read(path_text, &grid) == APP_OKAY && want_list) {
    test_true(grid_bands(&grid, 3) == APP_OKAY, "the picture is read for the reference");
    test_true(grid_scale(&grid, wide_grid * TOWER_PATCH, high_grid * TOWER_PATCH, &work_grid) ==
                  APP_OKAY,
              "the picture is resized for the reference");
    test_tower_vision(pack, &work_grid, wide_grid, high_grid, want_list);
    for (row_index = 0; row_index < media.row_count && row_index < 6; ++row_index)
      for (value_index = 0; value_index < TOWER_TEXT; ++value_index)
        if (fabs((double)media.state_data[row_index * TOWER_TEXT + value_index] -
                 (double)want_list[row_index * TOWER_TEXT + value_index]) > 1e-4)
          okay_flag = 0;
    test_true(okay_flag, "the vision tower matches an independent definition of it");
    grid_free(&work_grid);
    grid_free(&grid);
  } else {
    test_true(0, "the reference reads the picture");
  }
  mem_free(want_list);

  /* -- audio --------------------------------------------------------- */
  {
    app_media sound;
    path_join(path_text, sizeof(path_text), test_yard_path, "clip.wav");
    test_true(media_audio(model, path_text, &sound) == APP_OKAY, "a clip runs the audio tower");
    /* Six hundred samples at eight kilohertz become twelve hundred at sixteen,
     * then frames of eight stepping four, then two halvings in the subsampler. */
    test_true(sound.row_count > 0, "the conformer emits a row per subsampled frame");
    okay_flag = 1;
    for (row_index = 0; row_index < sound.row_count; ++row_index)
      for (value_index = 0; value_index < TOWER_TEXT; ++value_index) {
        float value_now = sound.state_data[row_index * TOWER_TEXT + value_index];
        if (!(value_now == value_now) || fabs((double)value_now) > 1e6) okay_flag = 0;
      }
    test_true(okay_flag, "every value the conformer emits is finite");
    /* The attention reaches backwards only, so a longer clip has to reproduce
     * the rows of a shorter one that starts the same way; and a clip that
     * sounds different has to reach a different answer, or the tower is not
     * reading its input at all. */
    {
      app_media longer, other;
      int same_flag = 1, keep_flag = 1;
      test_true(test_wave_write("clip.wav", 8000, 900), "a longer clip is written");
      test_true(media_audio(model, path_text, &longer) == APP_OKAY, "the longer clip runs");
      test_true(longer.row_count > sound.row_count, "a longer clip makes more rows");
      for (value_index = 0; value_index < TOWER_TEXT; ++value_index)
        if (fabs((double)longer.state_data[value_index] - (double)sound.state_data[value_index]) >
            1e-4)
          keep_flag = 0;
      test_true(keep_flag, "a longer clip repeats the first row of a shorter one");
      media_free(&longer);

      test_true(test_wave_write_at("clip.wav", 8000, 600, 0.77), "a different clip is written");
      test_true(media_audio(model, path_text, &other) == APP_OKAY, "the different clip runs");
      for (value_index = 0; value_index < TOWER_TEXT; ++value_index)
        if (fabs((double)other.state_data[value_index] - (double)sound.state_data[value_index]) >
            1e-6)
          same_flag = 0;
      test_true(!same_flag, "a clip that sounds different reaches a different answer");
      media_free(&other);
    }

    /* The processor does not read a clip past its budget: so many soft tokens
     * of so many milliseconds each, and a longer clip cut to fit before a frame
     * is taken from it.  The cut is of the samples rather than of the rows,
     * which is a distinction with a difference — capping the rows would agree
     * on the count and disagree on the last one, whose frames would have been
     * drawn from audio the reference never read.  So a clip that ends at the
     * budget and a clip four times as long have to come out the same, row for
     * row, and not merely the same length. */
    {
      app_media brim, over;
      int same_flag = 1;
      test_true(test_wave_write("clip.wav", 8000, SOUND_BUDGET * SOUND_SPAN * 8),
                "a clip exactly at the budget is written");
      test_true(media_audio(model, path_text, &brim) == APP_OKAY, "the clip at the budget runs");
      test_true(brim.row_count == SOUND_BUDGET && !brim.cut_flag,
                "a clip at the budget spends all of it and is not cut");
      test_true(test_wave_write("clip.wav", 8000, 4 * SOUND_BUDGET * SOUND_SPAN * 8),
                "a clip four times the budget is written");
      test_true(media_audio(model, path_text, &over) == APP_OKAY, "the clip past the budget runs");
      test_true(over.row_count == SOUND_BUDGET && over.cut_flag,
                "a clip past the budget is worth the budget and says it was cut");
      for (row_index = 0; row_index < over.row_count && row_index < brim.row_count; ++row_index)
        for (value_index = 0; value_index < TOWER_TEXT; ++value_index)
          if (fabs((double)over.state_data[row_index * TOWER_TEXT + value_index] -
                   (double)brim.state_data[row_index * TOWER_TEXT + value_index]) > 1e-6)
            same_flag = 0;
      test_true(same_flag, "the clip past the budget is the clip that ends at it");
      media_free(&brim);
      media_free(&over);
    }
    media_free(&sound);
  }

  /* -- the ids around a run -------------------------------------------- */
  {
    /* The processor does not lay a bare run of placeholders down.  It opens and
     * closes each one, and the model reads those two ids as ordinary text, so a
     * prompt built without them is a prompt the reference never sees. */
    app_media_span span_list[2];
    int32_t id_list[64];
    int open_id = -1, shut_id = -1, id_count, place_from;
    test_true(model_image_wrap(model, &open_id, &shut_id) && open_id == 20 && shut_id == 21,
              "the image brackets come from the configuration");
    test_true(model_audio_wrap(model, &open_id, &shut_id) && open_id == 22 && shut_id == 23,
              "the audio brackets come from the configuration");
    span_list[0].kind_mark = APP_MEDIA_IMAGE;
    span_list[0].row_count = 3;
    span_list[0].place_from = -1;
    span_list[1].kind_mark = APP_MEDIA_AUDIO;
    span_list[1].row_count = 2;
    span_list[1].place_from = -1;
    id_count = token_media_run(model, span_list, 2, 0, id_list, 64);
    test_true(id_count == 9, "a run is an opener, one placeholder a row, and a closer");
    test_true(id_list[0] == 20 && id_list[1] == 5 && id_list[3] == 5 && id_list[4] == 21,
              "the image run is bracketed and as long as the tower's rows");
    test_true(id_list[5] == 22 && id_list[6] == 6 && id_list[7] == 6 && id_list[8] == 23,
              "the clip's run follows the picture's, bracketed the same way");
    test_true(span_list[0].place_from == 1 && span_list[1].place_from == 6,
              "each run is reported past its opener, where the rows go");

    span_list[0].place_from = span_list[1].place_from = -1;
    id_count = token_frame_media(model, "hello", span_list, 2, id_list, 64);
    place_from = span_list[0].place_from;
    test_true(id_count > 9 && place_from > 0, "the frame carries the run and the words");
    test_true(id_list[place_from - 1] == 20 && id_list[place_from] == 5 &&
                  id_list[place_from + 3] == 21,
              "the frame brackets the picture's run where the run is reported");
    test_true(span_list[1].place_from == place_from + 5 &&
                  id_list[span_list[1].place_from - 1] == 22 &&
                  id_list[span_list[1].place_from + 2] == 23,
              "the clip's run follows it, bracketed and reported the same way");
    /* The words after a run begin a fresh chunk of text, because a special id
     * ended the one before: the tokenizer's lead mark belongs to them. */
    test_true(id_list[span_list[1].place_from + 3] == 18 &&
                  id_list[span_list[1].place_from + 4] == 4,
              "the words follow the run marked as a fresh chunk, and the turn closes");
  }

  /* -- several runs, in the order the caller gave them ------------------ */
  {
    /* A prompt can carry more than one picture or clip, and the order is the
     * caller's rather than a picture and then a clip.  A tower that produced
     * nothing keeps its place in the list, so an index into the spans stays an
     * index into the attachments they came from. */
    app_media_span span_list[4];
    int32_t id_list[64];
    int id_count, slot_index;
    for (slot_index = 0; slot_index < 4; ++slot_index) span_list[slot_index].place_from = -1;
    span_list[0].kind_mark = APP_MEDIA_AUDIO;
    span_list[0].row_count = 2;
    span_list[1].kind_mark = APP_MEDIA_IMAGE;
    span_list[1].row_count = 3;
    span_list[2].kind_mark = APP_MEDIA_IMAGE;
    span_list[2].row_count = 0;
    span_list[3].kind_mark = APP_MEDIA_IMAGE;
    span_list[3].row_count = 1;
    id_count = token_media_run(model, span_list, 4, 0, id_list, 64);
    test_true(id_count == 4 + 5 + 3, "an empty span costs no ids and the others keep their runs");
    test_true(id_list[0] == 22 && id_list[3] == 23 && id_list[4] == 20 && id_list[8] == 21,
              "the clip leads because that is where the caller put it");
    test_true(span_list[0].place_from == 1 && span_list[1].place_from == 5 &&
                  span_list[2].place_from == -1 && span_list[3].place_from == 10,
              "every run is reported where its rows go, and the empty one is not reported");
    test_true(id_list[9] == 20 && id_list[10] == 5 && id_list[11] == 21,
              "the second picture is bracketed like the first");
  }

  /* -- the pieces in the caller's order --------------------------------- */
  {
    /* A content list is an order, not a picture followed by words, and the
     * frame builder takes one. The old call is the same list with every span
     * first, so it has to write the same ids. */
    app_media_span span_list[1];
    app_part part_list[3];
    int32_t plain_list[64], part_list_ids[64], word_list[16];
    int plain_count, part_ids_count, word_count, lead_place, place_from, index;
    int same_flag = 1;

    span_list[0].kind_mark = APP_MEDIA_IMAGE;
    span_list[0].row_count = 3;
    span_list[0].place_from = -1;
    plain_count = token_frame_media(model, "and say", span_list, 1, plain_list, 64);
    lead_place = span_list[0].place_from;

    part_list[0].kind_mark = APP_PART_MEDIA;
    part_list[0].text_ref = NULL;
    part_list[0].span_index = 0;
    part_list[1].kind_mark = APP_PART_TEXT;
    part_list[1].text_ref = "and say";
    part_list[1].span_index = -1;
    span_list[0].place_from = -1;
    part_ids_count = token_frame_parts(model, part_list, 2, span_list, 1, part_list_ids, 64);
    if (part_ids_count != plain_count) same_flag = 0;
    for (index = 0; index < plain_count && index < part_ids_count; ++index)
      if (plain_list[index] != part_list_ids[index]) same_flag = 0;
    test_true(same_flag && span_list[0].place_from == lead_place,
              "the span first and the words last is what the older call already wrote");

    /* Words, the picture, more words. The frame's opening is unchanged, so the
     * first words stand exactly where the opener stood before them, and the run
     * begins that much further in. */
    part_list[0].kind_mark = APP_PART_TEXT;
    part_list[0].text_ref = "look at";
    part_list[0].span_index = -1;
    part_list[1].kind_mark = APP_PART_MEDIA;
    part_list[1].text_ref = NULL;
    part_list[1].span_index = 0;
    part_list[2].kind_mark = APP_PART_TEXT;
    part_list[2].text_ref = "and say";
    part_list[2].span_index = -1;
    span_list[0].place_from = -1;
    part_ids_count = token_frame_parts(model, part_list, 3, span_list, 1, part_list_ids, 64);
    place_from = span_list[0].place_from;
    /* The words before the run continue the line the role marker opened, which
     * is what the older call's trailing words did not do. */
    word_count = token_encode(model, "look at", 0, word_list, 16);
    same_flag = word_count > 0 && place_from == lead_place + word_count;
    for (index = 0; index < word_count; ++index)
      if (part_list_ids[lead_place - 1 + index] != word_list[index]) same_flag = 0;
    test_true(same_flag, "the words in front of a picture stand where the opener stood");
    test_true(part_list_ids[place_from - 1] == 20 && part_list_ids[place_from] == 5 &&
                  part_list_ids[place_from + 3] == 21,
              "the run is bracketed where the caller asked for it, not at the front");
    test_true(part_ids_count == plain_count + word_count,
              "words on both sides cost the words and nothing else");
    test_true(token_frame_parts(model, part_list, 3, span_list, 0, part_list_ids, 64) < 0,
              "a part naming a span that is not there is refused");
  }

  /* -- the substitution ----------------------------------------------- */
  {
    int32_t id_list[8];
    uint8_t flag_list[8];
    float *state_list = (float *)mem_clear(sizeof(float) * 8u * TOWER_TEXT);
    const float *logit_list;
    float keep_value = 0.0f;
    int slot_index;
    for (slot_index = 0; slot_index < 8; ++slot_index) {
      id_list[slot_index] = slot_index < 4 ? 5 : (int32_t)(7 + slot_index);
      flag_list[slot_index] = slot_index < 4 ? 1u : 0u;
    }
    if (state_list && media.row_count >= 4) {
      memcpy(state_list, media.state_data, sizeof(float) * 4u * TOWER_TEXT);
      test_true(session_open(model, &session) == APP_OKAY, "a session opens");
      if (session) {
        test_true(session_prime_media(session, id_list, 8, state_list, flag_list) == APP_OKAY,
                  "a prompt with placeholders primes");
        logit_list = session_step(session, id_list[7]);
        test_true(logit_list != NULL, "the prompt reaches the head");
        if (logit_list) keep_value = logit_list[0];
        session_reset(session);
        test_true(session_prime(session, id_list, 8) == APP_OKAY, "the same ids prime plainly");
        logit_list = session_step(session, id_list[7]);
        test_true(logit_list && fabs((double)logit_list[0] - keep_value) > 1e-6,
                  "a substituted embedding changes what the stack computes");
        /* The reference rewrites every media position to the pad id before it
         * reads the per-layer embedding, so nothing of the placeholder reaches
         * the stack and the id under a filled lane cannot matter. */
        {
          int32_t other_list[8];
          int slot_other;
          for (slot_other = 0; slot_other < 8; ++slot_other)
            other_list[slot_other] = slot_other < 4 ? 6 : id_list[slot_other];
          session_reset(session);
          test_true(session_prime_media(session, other_list, 8, state_list, flag_list) == APP_OKAY,
                    "another placeholder id primes the same rows");
          logit_list = session_step(session, other_list[7]);
          test_true(logit_list && fabs((double)logit_list[0] - keep_value) < 1e-9,
                    "the id under a lane a tower filled reaches nothing");
        }
        session_close(session);
      }
    }
    mem_free(state_list);
  }

  media_free(&media);
  model_free(model);
  mem_free(pack->body_data);
  mem_free(pack);
}

/* Turns after the first, and conversations beside each other on one model. */
static void test_turn(void) {
  app_setup setup = app_setup_plain();
  app_model *model = NULL;
  app_part part;
  int32_t first_list[64], next_list[64], both_list[128];
  int first_count, next_count, slot_index;
  test_open("turn");
  setup.thread_count = 2;
  if (!test_wing_write(0)) {
    test_true(0, "the synthetic checkpoint is written");
    return;
  }
  if (model_load(test_yard_path, &setup, &model) != APP_OKAY || !model) {
    test_true(0, "the checkpoint loads");
    return;
  }

  part.kind_mark = APP_PART_TEXT;
  part.text_ref = "hello world";
  part.span_index = -1;
  first_count = token_frame_parts(model, &part, 1, NULL, 0, first_list, 64);
  next_count = token_frame_next(model, &part, 1, NULL, 0, next_list, 64);
  test_true(first_count > 2, "a first turn lays a frame down");
  test_true(next_count > 2, "a later turn lays a frame down");

  /* The two frames say the same thing about the turn; what they differ in is
   * what comes before it.  A first turn opens the document, a later one closes
   * the model's turn — the id the sampler stopped on and never fed back. */
  if (first_count > 2 && next_count > 2) {
    int lead_count = next_count - (first_count - 1);
    test_true(first_list[0] == 1, "a first turn opens the document");
    test_true(next_list[0] == 4, "a later turn closes the model's turn first");
    test_true(lead_count >= 1, "the close is what a later turn adds");
    test_true(lead_count >= 1 && memcmp(next_list + lead_count, first_list + 1,
                                        sizeof(int32_t) * (size_t)(first_count - 1)) == 0,
              "past the close, a later turn is the first turn without its opening");
  }

  /* Continuing a conversation is the same arithmetic as having asked for the
   * whole transcript at once: the cache holds the turns before, and a session
   * that has been fed two turns in two calls reaches what one fed both in one
   * call reaches, to the bit. */
  if (first_count > 2 && next_count > 2 && first_count + next_count <= 128) {
    app_session *step_session = NULL, *whole_session = NULL;
    const float *logit_list;
    float *keep_list = (float *)mem_clear(sizeof(float) * (size_t)model->head_sheet.row_count);
    int okay_flag = 1;
    memcpy(both_list, first_list, sizeof(int32_t) * (size_t)first_count);
    memcpy(both_list + first_count, next_list, sizeof(int32_t) * (size_t)next_count);
    test_true(session_open(model, &step_session) == APP_OKAY, "a conversation opens");
    test_true(session_open(model, &whole_session) == APP_OKAY, "a second one opens beside it");
    if (keep_list && step_session && whole_session) {
      test_true(session_prime(step_session, first_list, first_count) == APP_OKAY,
                "the first turn primes");
      logit_list = session_step(step_session, first_list[first_count - 1]);
      test_true(logit_list != NULL, "the first turn reaches the head");
      test_true(session_fill(step_session) == first_count,
                "a conversation holds the ids it was fed");
      test_true(session_prime(step_session, next_list, next_count) == APP_OKAY,
                "the second turn primes onto the first");
      logit_list = session_step(step_session, next_list[next_count - 1]);
      if (logit_list)
        memcpy(keep_list, logit_list, sizeof(float) * (size_t)model->head_sheet.row_count);
      test_true(logit_list != NULL, "the second turn reaches the head");

      test_true(session_prime(whole_session, both_list, first_count + next_count) == APP_OKAY,
                "the whole transcript primes at once");
      logit_list = session_step(whole_session, both_list[first_count + next_count - 1]);
      for (slot_index = 0; logit_list && slot_index < model->head_sheet.row_count; ++slot_index)
        if (logit_list[slot_index] != keep_list[slot_index]) okay_flag = 0;
      test_true(logit_list && okay_flag,
                "two turns fed in turn reach what the whole transcript reaches");
      test_true(session_fill(step_session) == session_fill(whole_session),
                "and the two conversations hold the same number of ids");
    }
    mem_free(keep_list);

    /* A conversation beside another is its own: what one is fed does not reach
     * the other, which is the whole point of a model that many sessions share.
     * They run one at a time, because the kernels underneath them reach one
     * fork and join pool. */
    if (step_session && whole_session) {
      app_session *third_session = NULL;
      const float *third_list;
      float *keep_third = (float *)mem_clear(sizeof(float) * (size_t)model->head_sheet.row_count);
      int same_flag = 1;
      session_reset(whole_session);
      test_true(session_prime(whole_session, first_list, first_count) == APP_OKAY,
                "a reset conversation takes the first turn again");
      third_list = session_step(whole_session, first_list[first_count - 1]);
      if (keep_third && third_list)
        memcpy(keep_third, third_list, sizeof(float) * (size_t)model->head_sheet.row_count);
      test_true(session_open(model, &third_session) == APP_OKAY, "a third conversation opens");
      if (third_session) {
        int32_t other_list[4];
        other_list[0] = 1;
        other_list[1] = 7;
        other_list[2] = 8;
        other_list[3] = 9;
        test_true(session_prime(third_session, other_list, 4) == APP_OKAY,
                  "the third takes something else entirely");
        session_step(third_session, other_list[3]);
        test_true(session_fill(whole_session) == first_count && session_fill(third_session) == 4,
                  "each conversation counts only what it was fed");
        session_close(third_session);
      }
      session_reset(whole_session);
      session_prime(whole_session, first_list, first_count);
      third_list = session_step(whole_session, first_list[first_count - 1]);
      for (slot_index = 0; keep_third && third_list && slot_index < model->head_sheet.row_count;
           ++slot_index)
        if (third_list[slot_index] != keep_third[slot_index]) same_flag = 0;
      test_true(third_list && same_flag,
                "a conversation reaches the same place whatever ran beside it");
      mem_free(keep_third);
    }
    session_close(step_session);
    session_close(whole_session);
  }
  model_free(model);
}

/* A conversation written out and read back. */
static void test_keep(void) {
  app_setup setup = app_setup_plain();
  app_model *model = NULL;
  app_session *from_session = NULL, *into_session = NULL;
  char path_text[1024];
  int32_t id_list[24], back_list[24];
  float *keep_list = NULL;
  const float *logit_list;
  uint64_t stamp_back = 0;
  float peak_key = 0.0f, peak_value = 0.0f;
  int id_count = 12, slot_index, okay_flag = 1, kind_back = -1;
  test_open("keep");
  setup.thread_count = 2;
  if (!test_wing_write(0)) {
    test_true(0, "the synthetic checkpoint is written");
    return;
  }
  if (model_load(test_yard_path, &setup, &model) != APP_OKAY || !model) {
    test_true(0, "the checkpoint loads");
    return;
  }
  path_join(path_text, sizeof(path_text), test_yard_path, "hold.cache");
  for (slot_index = 0; slot_index < id_count; ++slot_index)
    id_list[slot_index] = (int32_t)(7 + slot_index % 7);

  test_true(session_open(model, &from_session) == APP_OKAY, "a conversation opens");
  test_true(session_open(model, &into_session) == APP_OKAY, "a second one opens");
  keep_list = (float *)mem_clear(sizeof(float) * (size_t)model->head_sheet.row_count);
  if (!from_session || !into_session || !keep_list) {
    session_close(from_session);
    session_close(into_session);
    mem_free(keep_list);
    model_free(model);
    return;
  }

  test_true(session_prime(from_session, id_list, id_count) == APP_OKAY, "a prompt primes");
  test_true(session_ids(from_session, NULL, 0) == id_count - 1,
            "a session holds every id but the one the caller still has");
  test_true(session_ids(from_session, back_list, 24) == id_count - 1 &&
                memcmp(back_list, id_list, sizeof(int32_t) * (size_t)(id_count - 1)) == 0,
            "and hands them back in the order it was fed them");
  test_true(session_save(from_session, path_text, 0x1234567890ABCDEFull, APP_KEEP_PROMPT) ==
                APP_OKAY,
            "the conversation is written out");
  peak_key = session_cache_peak(from_session, 0, 0);
  peak_value = session_cache_peak(from_session, 0, 1);
  test_true(peak_key > 0.0f && peak_value > 0.0f, "the conversation reached a peak in both caches");

  /* What the file is for: the session that reads it reaches the same place as
   * the session that wrote it, to the bit, without priming a single id. */
  logit_list = session_step(from_session, id_list[id_count - 1]);
  if (logit_list)
    memcpy(keep_list, logit_list, sizeof(float) * (size_t)model->head_sheet.row_count);
  test_true(logit_list != NULL, "the prompt reaches the head");

  test_true(session_load(into_session, path_text, &stamp_back, &kind_back) == APP_OKAY,
            "the conversation is read back");
  test_true(stamp_back == 0x1234567890ABCDEFull, "the caller's stamp comes back unread");
  test_true(kind_back == APP_KEEP_PROMPT, "and says it holds a prompt rather than a conversation");
  test_true(session_fill(into_session) == id_count - 1, "and holds what it was written with");
  /* The peaks go in the file too: what a range has to cover is a question about
   * the whole conversation, and half of it would answer it wrongly. */
  test_true(session_cache_peak(into_session, 0, 0) == peak_key &&
                session_cache_peak(into_session, 0, 1) == peak_value,
            "the peaks the conversation reached come back with it");
  logit_list = session_step(into_session, id_list[id_count - 1]);
  for (slot_index = 0; logit_list && slot_index < model->head_sheet.row_count; ++slot_index)
    if (logit_list[slot_index] != keep_list[slot_index]) okay_flag = 0;
  test_true(logit_list && okay_flag, "a restored conversation reaches the same logits, bit for bit");

  /* A file that is not one of these, or is one that stops short, is refused,
   * and a session that tried to read it is left cleared rather than half fed. */
  {
    uint8_t *file_data;
    size_t file_size = 0;
    session_reset(into_session);
    file_data = (uint8_t *)file_slurp(path_text, &file_size);
    test_true(file_data != NULL && file_size > 64, "the file is read back to be damaged");
    if (file_data) {
      char other_text[1024];
      path_join(other_text, sizeof(other_text), test_yard_path, "short.cache");
      test_true(test_file_write("short.cache", file_data, file_size / 2),
                "a truncated cache is written");
      test_true(session_load(into_session, other_text, NULL, NULL) != APP_OKAY,
                "a cache that stops short is refused");
      test_true(session_fill(into_session) == 0, "and leaves the session cleared");

      file_data[3] ^= 0xFF;
      path_join(other_text, sizeof(other_text), test_yard_path, "wrong.cache");
      test_true(test_file_write("wrong.cache", file_data, file_size), "a damaged cache is written");
      test_true(session_load(into_session, other_text, NULL, NULL) != APP_OKAY,
                "a file that is not one of these is refused");
      file_data[3] ^= 0xFF;

      /* The mark is what says a cache belongs to this model and this storage.
       * A cache whose mark disagrees is a conversation the model never had. */
      file_data[KEEP_MARK_SIZE] ^= 0x01;
      path_join(other_text, sizeof(other_text), test_yard_path, "alien.cache");
      test_true(test_file_write("alien.cache", file_data, file_size), "an alien cache is written");
      test_true(session_load(into_session, other_text, NULL, NULL) == APP_FAIL_STATE,
                "a cache written from other shapes is refused");
      mem_free(file_data);
    }
    test_true(session_load(into_session, "no_such_cache_file", NULL, NULL) == APP_FAIL_MISSING,
              "a cache that is not there is refused");
  }

  /* The other of the two things a file can be.  A conversation is written after
   * an answer rather than before one, holds every id the session was fed, and
   * says so — so a caller can tell which it has without guessing, and a session
   * that reads it is where the one that wrote it stopped rather than one id
   * short of it. */
  {
    char talk_text[1024];
    const float *step_list;
    int32_t answer_id, follow_id;
    session_reset(from_session);
    session_reset(into_session);
    path_join(talk_text, sizeof(talk_text), test_yard_path, "talk.cache");
    test_true(session_prime(from_session, id_list, id_count) == APP_OKAY, "a prompt primes again");
    step_list = session_step(from_session, id_list[id_count - 1]);
    test_true(step_list != NULL, "and is answered once");
    answer_id = (int32_t)(id_list[0] + 1);
    follow_id = (int32_t)(id_list[0] + 2);
    step_list = session_step(from_session, answer_id);
    test_true(step_list != NULL, "and the answer is fed back");
    test_true(session_ids(from_session, NULL, 0) == id_count + 1,
              "a conversation holds every id it was fed");
    test_true(session_save(from_session, talk_text, 3, APP_KEEP_TALK) == APP_OKAY,
              "a conversation is written out");
    test_true(session_load(into_session, talk_text, &stamp_back, &kind_back) == APP_OKAY &&
                  kind_back == APP_KEEP_TALK && stamp_back == 3,
              "and reads back as a conversation, with what the caller stamped it");
    test_true(session_fill(into_session) == id_count + 1,
              "and holds every id rather than one short of them");
    /* The next turn is the first id of it fed onto what the file held.  The
     * session that wrote the file and the session that read it have to reach
     * the same place from that id, which is what makes the file a conversation
     * rather than a picture of one. */
    step_list = session_step(from_session, follow_id);
    if (step_list)
      memcpy(keep_list, step_list, sizeof(float) * (size_t)model->head_sheet.row_count);
    okay_flag = step_list != NULL;
    step_list = session_step(into_session, follow_id);
    for (slot_index = 0; step_list && slot_index < model->head_sheet.row_count; ++slot_index)
      if (step_list[slot_index] != keep_list[slot_index]) okay_flag = 0;
    test_true(step_list && okay_flag,
              "and carries on from where the conversation stopped, bit for bit");
  }

  mem_free(keep_list);
  session_close(from_session);
  session_close(into_session);
  model_free(model);
}

static void test_face(void) {
  app_setup setup = app_setup_plain();
  app_taste taste = app_taste_plain();
  app_model *model = NULL;
  test_open("face");
  test_true(setup.thread_count == 0, "the plain setup asks for the host thread count");
  test_true(taste.heat_value >= 0.0f, "the plain taste has a defined temperature");
  test_true(strcmp(app_code_text(APP_OKAY), "okay") == 0, "APP_OKAY has a name");
  test_true(strcmp(app_code_text(APP_FAIL_MISSING), app_code_text(APP_FAIL_FORMAT)) != 0,
            "failure codes have distinct names");
  test_true(model_load("/no/such/folder", &setup, &model) != APP_OKAY,
            "model_load reports a missing folder");
  test_true(model == NULL, "model_load leaves no model behind on failure");
  model_free(NULL);
  session_close(NULL);
  test_true(1, "freeing a null model and session is safe");
}

/* ======================================================================== */
/* 11. the shipped checkpoint                                               */
/* ======================================================================== */

/* Everything above this line is held against a definition written out beside
 * it, or against a synthetic checkpoint built here for the purpose.  Neither
 * touches the weights the engine is actually for.  The comparison that does —
 * `app_diff.py` against the reference — wants python, torch, and a transformers
 * that knows the architecture, and a host can easily have the checkpoint and
 * none of those.  So what that comparison settled is written down here: the
 * frame the tokenizer lays down for a prompt, and the head of the distribution
 * the whole stack reaches over it.
 *
 * The numbers are the engine's own, read off the SSE2 build the seam comparison
 * judged against the reference (`CHANGES.md`, 0.7.2).  They are held to a slack
 * rather than to the last bit, because the checkpoint rounds every activation
 * onto a static grid and a build that sums in another order lands on a
 * neighbouring step.  Measured on this export, an SSE2 build against an AVX2
 * one over five prompts: rank one never moved, no id in a top eight fell below
 * rank eleven in the other build, and the top eight logits moved by at most
 * 1.30.  The slack is 2.0 — clear of what a backend change did, and well inside
 * the 2.30 to 5.33 the reference moves against itself when its own input is
 * nudged by a millionth.  What that catches is a layer wired wrong, not a
 * last-bit difference.
 *
 * The prompts are chosen for a wide rank one: the least of the three leads the
 * next id by 3.27, which is more than twice the widest move a backend change
 * was seen to make.
 *
 * No picture is in here.  The processor lifts even a thirty-two pixel image to
 * the export's full patch budget — 260 soft tokens, four and a half minutes on
 * this host — so the cheapest media shot would cost more than the whole of the
 * rest of the suite by three orders of magnitude.  `run.py parity --seam
 * --model model` is where that join is judged. */

#define TEST_SHOT_TOP   8   /* recorded ids per prompt */
#define TEST_SHOT_ROOM  16  /* ranks they are allowed to move within */
#define TEST_SHOT_SLACK 2.0 /* how far a recorded logit may move */

typedef struct test_shot_case {
  const char    *prompt_text;
  int            id_count;
  const int32_t *id_list;    /* the frame `token_frame` lays down */
  const int32_t *rank_list;  /* the highest scoring ids, in order */
  const float   *logit_list; /* and what they scored */
} test_shot_case;

static const int32_t shot_paris_id[14] = {2,    105, 2364, 107, 818, 5279, 529,
                                          7001, 563, 106,  107, 105, 4368, 107};
static const int32_t shot_paris_rank[TEST_SHOT_TOP] = {818,   50429, 1018,   236777,
                                                       10450, 48,    236798, 9366};
static const float shot_paris_logit[TEST_SHOT_TOP] = {27.329222f, 24.059254f, 21.418470f,
                                                      19.322742f, 18.975519f, 18.860050f,
                                                      18.603403f, 18.475531f};

static const int32_t shot_hello_id[12] = {2,      105, 2364, 107, 37889, 29104,
                                          236761, 106, 107,  105, 4368,  107};
static const int32_t shot_hello_rank[TEST_SHOT_TOP] = {9259,   10979, 236777, 1018,
                                                       174886, 21529, 17531,  23391};
static const float shot_hello_logit[TEST_SHOT_TOP] = {28.229704f, 23.872934f, 22.081196f,
                                                      21.774128f, 21.530983f, 21.086210f,
                                                      20.350374f, 20.348656f};

static const int32_t shot_prime_id[16] = {2,    105, 2364, 107, 1613, 506,  1171, 1806,
                                          8355, 4945, 236761, 106, 107, 105, 4368, 107};
static const int32_t shot_prime_rank[TEST_SHOT_TOP] = {818,  236770, 236777, 8291,
                                                       1018, 49190,  236800, 236829};
static const float shot_prime_logit[TEST_SHOT_TOP] = {28.001677f, 23.693737f, 21.795601f,
                                                      20.744570f, 20.577866f, 20.443802f,
                                                      20.155479f, 19.610378f};

static const test_shot_case shot_case_list[3] = {
    {"The capital of France is", 14, shot_paris_id, shot_paris_rank, shot_paris_logit},
    {"Say hello.", 12, shot_hello_id, shot_hello_rank, shot_hello_logit},
    {"List the first three prime numbers.", 16, shot_prime_id, shot_prime_rank,
     shot_prime_logit},
};

/* Where the checkpoint is: `IGLLM_MODEL` when it is set, and otherwise the
 * vendored export, as seen from the tree root and from the build folder.  The
 * variable is the only place looked at once it is set, so naming a folder that
 * holds no `config.json` — `IGLLM_MODEL=none` — is how a host that does not
 * want to spend the twenty seconds turns this section off. */
static const char *test_shot_folder(void) {
  static const char *look_list[2] = {"model", "../model"};
  const char *named_text = getenv("IGLLM_MODEL");
  char path_text[1024];
  file_map map;
  size_t look_index, look_count = sizeof(look_list) / sizeof(look_list[0]);
  for (look_index = 0; look_index < look_count; ++look_index) {
    const char *folder_text = named_text ? named_text : look_list[look_index];
    path_join(path_text, sizeof(path_text), folder_text, "config.json");
    if (file_open(path_text, &map) == APP_OKAY) {
      file_close(&map);
      return folder_text;
    }
    if (named_text) break;
  }
  return NULL;
}

/* The shape these were taken from.  A folder holding some other checkpoint is
 * skipped rather than failed: all a failure would report is that the numbers
 * below are not about it. */
static int test_shot_shape(const app_model *model) {
  return model_vocab_count(model) == 262144 && model_layer_count(model) == 35 &&
         model_state_size(model) == 1536 && model_window_limit(model) == 131072 &&
         model_vision_ready(model) && model_audio_ready(model) &&
         model_image_token(model) == 258880 && model_audio_token(model) == 258881 &&
         model_image_rows(model) == 280;
}

/* The highest scoring ids, in order, without sorting the vocabulary. */
static void test_shot_rank(const float *logit_list, int vocab_count, int *rank_list,
                           int rank_room) {
  int rank_count = 0, rank_index, vocab_index;
  for (vocab_index = 0; vocab_index < vocab_count; ++vocab_index) {
    float value = logit_list[vocab_index];
    if (rank_count < rank_room) {
      rank_list[rank_count++] = vocab_index;
    } else if (value > logit_list[rank_list[rank_count - 1]]) {
      rank_list[rank_count - 1] = vocab_index;
    } else {
      continue;
    }
    for (rank_index = rank_count - 1; rank_index > 0; --rank_index) {
      int keep_value;
      if (logit_list[rank_list[rank_index]] <= logit_list[rank_list[rank_index - 1]]) break;
      keep_value = rank_list[rank_index];
      rank_list[rank_index] = rank_list[rank_index - 1];
      rank_list[rank_index - 1] = keep_value;
    }
  }
}

static void test_shot(void) {
  const char *folder_text = test_shot_folder();
  app_setup setup = app_setup_plain();
  app_model *model = NULL;
  int case_index;
  test_open("shot");
  if (!folder_text) {
    printf("   skipped: no checkpoint; set IGLLM_MODEL or run beside model/\n");
    return;
  }
  if (model_load(folder_text, &setup, &model) != APP_OKAY || !model) {
    test_true(0, "the shipped checkpoint loads");
    model_free(model);
    return;
  }
  test_true(1, "the shipped checkpoint loads");
  if (!test_shot_shape(model)) {
    printf("   skipped: %s is not the export these were recorded from\n", folder_text);
    model_free(model);
    return;
  }
  /* The clip budget the export's processor records, which is the ceiling a
   * longer clip is cut to: 750 soft tokens of 40 milliseconds, half a minute. */
  test_true(model_audio_rows(model) == 750 && model_audio_span_ms(model) == 40,
            "the shipped export allows a clip of thirty seconds");

  /* What is resident and what a token reads, both recorded.  The second is
   * under a third of the first on this export, and the reason is external to
   * the engine: the per-layer embedding table alone is 1120 MiB of the file and
   * a step reads one row of it.  A build that swept it would land above that
   * figure on its own, so the bound below fails long before the exact totals
   * would need to be re-recorded for an export of another shape. */
  /* The 120 bytes over the figure this held before the cache scales were read
   * are those scales: thirty of the seventy the export ships, four bytes each.
   * The other forty belong to layers that read another layer's cache and own
   * none of their own, so the loader never binds them. */
  test_true(model_memory_bytes(model) == 2448245496u, "the shipped export is the size recorded");
  test_true(model_decode_bytes(model) == 796326800u, "a token reads what was recorded");
  test_true(model_decode_bytes(model) < 1120u * 1024u * 1024u,
            "a token reads less than the per-layer embedding table alone");
  test_true(model_decode_bytes(model) * 3u < model_memory_bytes(model),
            "a token reads under a third of what the export holds");

  for (case_index = 0; case_index < 3; ++case_index) {
    const test_shot_case *shot = &shot_case_list[case_index];
    app_session *session = NULL;
    int32_t id_room[64];
    int rank_room[TEST_SHOT_ROOM];
    const float *logit_list;
    int id_count, slot_index, worst_place = 1;
    double worst_gap = 0.0;

    id_count = token_frame(model, shot->prompt_text, id_room,
                           (int)(sizeof(id_room) / sizeof(id_room[0])));
    test_near((double)id_count, (double)shot->id_count, 0.0, "the frame is the length recorded");
    if (id_count != shot->id_count) continue;
    test_true(memcmp(id_room, shot->id_list, sizeof(int32_t) * (size_t)id_count) == 0,
              "the frame is the one recorded");

    if (session_open(model, &session) != APP_OKAY || !session) {
      test_true(0, "a session opens on the shipped checkpoint");
      continue;
    }
    test_true(session_prime(session, id_room, id_count) == APP_OKAY, "the prompt primes");
    logit_list = session_step(session, id_room[id_count - 1]);
    test_true(logit_list != NULL, "the stack reaches the vocabulary");
    if (!logit_list) {
      session_close(session);
      continue;
    }
    test_shot_rank(logit_list, model_vocab_count(model), rank_room, TEST_SHOT_ROOM);
    test_true(rank_room[0] == shot->rank_list[0], "rank one is the id recorded");
    /* Where each recorded id landed this time, and how far its score moved.
     * Both go through `test_near`, so a failure says by how much. */
    for (slot_index = 0; slot_index < TEST_SHOT_TOP; ++slot_index) {
      int32_t want_id = shot->rank_list[slot_index];
      int place_index, place_found = TEST_SHOT_ROOM + 1;
      double gap = (double)logit_list[want_id] - (double)shot->logit_list[slot_index];
      for (place_index = 0; place_index < TEST_SHOT_ROOM; ++place_index)
        if (rank_room[place_index] == want_id) {
          place_found = place_index + 1;
          break;
        }
      if (place_found > worst_place) worst_place = place_found;
      if (gap < 0.0) gap = -gap;
      if (gap > worst_gap) worst_gap = gap;
    }
    test_near((double)worst_place, 1.0, (double)(TEST_SHOT_ROOM - 1),
              "every recorded id is still in the top sixteen");
    test_near(worst_gap, 0.0, TEST_SHOT_SLACK, "every recorded logit is within the slack");
    session_close(session);
  }
  model_free(model);
}

/* ======================================================================== */
/* entry point                                                              */
/* ======================================================================== */

int main(void) {
  printf("%s %s tests, backend %s\n", APP_NAME, APP_VERSION, back_flavor());
  test_yard_open();

  test_platform();
  test_json();
  test_store();
  test_number();
  test_pack();
  test_plane();
  test_level();
  test_expert();
  test_gemma();
  test_kernel();
  test_rope();
  test_token();
  test_puff();
  test_image();
  test_png_wide();
  test_jpeg();
  test_scale();
  test_cache();
  test_wave();
  test_mel();
  test_wing();
  test_tower();
  test_turn();
  test_keep();
  test_face();
  test_shot();

  test_yard_close();
  printf("\n%d passed, %d failed\n", test_pass_count, test_fail_count);
  return test_fail_count == 0 ? 0 : 1;
}

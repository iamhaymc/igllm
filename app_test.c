/* app_test.c - unit tests for the engine in app_core.c.
 *
 * The suite exercises every layer that can be checked without the pretrained
 * checkpoint: the platform shims, the JSON reader, the safetensors store, the
 * packed-quantization decode, the kernels against plain references, the rotary
 * tables, and the tokenizer. Fixtures are written into a scratch folder and
 * removed on exit. */

#include "app_core.c"

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
                                    "shape.json", NULL};
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
    mem_free(row_list);
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
    test_true(plane_bind_part(model, "experts.gate_up_proj", 2, 3, &sheet) == APP_OKAY,
              "plane_bind_part accepts a stacked tensor");
    test_true(sheet.row_count == 2 && sheet.col_count == 4, "the slice drops the expert axis");
    plane_row(&sheet, 1, row_list);
    test_near(row_list[0], 20.0, 0, "the slice lands on the right expert");
    test_near(row_list[3], 23.0, 0, "the slice keeps its row stride");
    test_true(plane_bind_part(model, "experts.gate_up_proj", 0, 1, &sheet) == APP_FAIL_FORMAT,
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

  { /* Packed dot against a plain bit-stream loop, over every widened width. */
    int bit_list[4] = {2, 3, 4, 8};
    int span_list[6] = {1, 3, 15, 16, 31, 64};
    int bit_slot, span_slot;
    for (bit_slot = 0; bit_slot < 4; ++bit_slot) {
      int bit_count = bit_list[bit_slot];
      int element_count = 128;
      uint32_t mask_value = (uint32_t)((1u << bit_count) - 1u);
      uint8_t *code_data = (uint8_t *)mem_clear((size_t)(element_count * bit_count + 7) / 8 + 8);
      float *act_list = (float *)mem_clear(sizeof(float) * (size_t)element_count);
      int element_index;
      for (element_index = 0; element_index < element_count; ++element_index) {
        test_pack_write(code_data, (size_t)element_index, bit_count,
                        (uint32_t)(element_index * 11 + bit_slot * 5) & mask_value);
        act_list[element_index] = (float)sin((double)element_index * 0.19);
      }
      for (span_slot = 0; span_slot < 6; ++span_slot) {
        int span_count = span_list[span_slot];
        int from_index = span_slot * 8; /* a byte boundary in every widened width */
        double want_value = 0.0;
        int slot;
        for (slot = 0; slot < span_count; ++slot)
          want_value += (double)pack_read(code_data, (size_t)(from_index + slot), bit_count) *
                        (double)act_list[from_index + slot];
        test_near(kern_dot_code(code_data, from_index, span_count, act_list + from_index, bit_count),
                  want_value, 1e-3, "kern_dot_code matches the bit-stream reference");
      }
      mem_free(code_data);
      mem_free(act_list);
    }
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
      back_open(&desk, &pool, kit.sheet.group_count);
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
      back_open(&desk, &pool, kit.sheet.group_count);
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
    int32_t id_list[32];
    int id_count = token_encode_book(book, "hello", 0, id_list, 32);
    test_true(id_count == 1 && id_list[0] == 18,
              "a prepend normalizer marks the lead word whatever the caller asks");
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
      int id_count = token_encode_book(book, word_list[word_index], 0, id_list, 32);
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
/* 8. public surface                                                        */
/* ======================================================================== */

/* ======================================================================== */
/* 9. whole graph on a synthetic checkpoint                                 */
/* ======================================================================== */

#define TEST_WING_LIMIT 128

typedef struct test_wing_kit {
  char   name_list[TEST_WING_LIMIT][96];
  int    size_list[TEST_WING_LIMIT][3];
  size_t from_list[TEST_WING_LIMIT];
  size_t span_list[TEST_WING_LIMIT];
  int    item_count;
  size_t body_size;
} test_wing_kit;

static void test_wing_add(test_wing_kit *kit, const char *name_text, int high_size, int wide_size,
                          int deep_size) {
  int slot = kit->item_count;
  size_t value_count;
  if (slot >= TEST_WING_LIMIT) return;
  kit->item_count += 1;
  snprintf(kit->name_list[slot], sizeof(kit->name_list[slot]), "%s", name_text);
  kit->size_list[slot][0] = high_size;
  kit->size_list[slot][1] = wide_size;
  kit->size_list[slot][2] = deep_size;
  value_count = (size_t)high_size * (size_t)(wide_size > 0 ? wide_size : 1) *
                (size_t)(deep_size > 0 ? deep_size : 1);
  kit->from_list[slot] = kit->body_size;
  kit->span_list[slot] = value_count * sizeof(float);
  kit->body_size += kit->span_list[slot];
}

/* Writes a complete miniature checkpoint so the whole graph can be exercised. */
static int test_wing_write(int moe_flag) {
  static const int state_size = 16, inner_size = 24, layer_count = 2, head_count = 2;
  static const int head_size = 8, kv_count = 1, ple_size = 4, vocab_count = 27;
  static const int expert_count = 4, expert_inner = 6;
  test_wing_kit *kit = (test_wing_kit *)mem_clear(sizeof(test_wing_kit));
  char name_text[96];
  char *header_text;
  size_t header_size, pad_count, file_size, header_fill = 0;
  uint8_t *file_data;
  float *body_data;
  uint64_t header_count;
  int layer_index, slot, byte_index, okay_flag;
  char config_text[1024];

  if (!kit) return 0;
  test_wing_add(kit, "model.embed_tokens.weight", vocab_count, state_size, 0);
  test_wing_add(kit, "model.embed_tokens_per_layer.weight", 32, layer_count * ple_size, 0);
  test_wing_add(kit, "model.per_layer_model_projection.weight", layer_count * ple_size, state_size, 0);
  test_wing_add(kit, "model.per_layer_projection_norm.weight", ple_size, 0, 0);
  test_wing_add(kit, "model.norm.weight", state_size, 0, 0);
  for (layer_index = 0; layer_index < layer_count; ++layer_index) {
#define TEST_WING_NAME(leaf) \
  (snprintf(name_text, sizeof(name_text), "model.layers.%d.%s", layer_index, leaf), name_text)
    test_wing_add(kit, TEST_WING_NAME("self_attn.q_proj.weight"), head_count * head_size, state_size, 0);
    test_wing_add(kit, TEST_WING_NAME("self_attn.k_proj.weight"), kv_count * head_size, state_size, 0);
    test_wing_add(kit, TEST_WING_NAME("self_attn.v_proj.weight"), kv_count * head_size, state_size, 0);
    test_wing_add(kit, TEST_WING_NAME("self_attn.o_proj.weight"), state_size, head_count * head_size, 0);
    test_wing_add(kit, TEST_WING_NAME("self_attn.q_norm.weight"), head_size, 0, 0);
    test_wing_add(kit, TEST_WING_NAME("self_attn.k_norm.weight"), head_size, 0, 0);
    test_wing_add(kit, TEST_WING_NAME("mlp.gate_proj.weight"), inner_size, state_size, 0);
    test_wing_add(kit, TEST_WING_NAME("mlp.up_proj.weight"), inner_size, state_size, 0);
    test_wing_add(kit, TEST_WING_NAME("mlp.down_proj.weight"), state_size, inner_size, 0);
    test_wing_add(kit, TEST_WING_NAME("input_layernorm.weight"), state_size, 0, 0);
    test_wing_add(kit, TEST_WING_NAME("post_attention_layernorm.weight"), state_size, 0, 0);
    test_wing_add(kit, TEST_WING_NAME("pre_feedforward_layernorm.weight"), state_size, 0, 0);
    test_wing_add(kit, TEST_WING_NAME("post_feedforward_layernorm.weight"), state_size, 0, 0);
    test_wing_add(kit, TEST_WING_NAME("per_layer_input_gate.weight"), ple_size, state_size, 0);
    test_wing_add(kit, TEST_WING_NAME("per_layer_projection.weight"), state_size, ple_size, 0);
    test_wing_add(kit, TEST_WING_NAME("post_per_layer_input_norm.weight"), state_size, 0, 0);
    if (!moe_flag) continue;
    test_wing_add(kit, TEST_WING_NAME("router.proj.weight"), expert_count, state_size, 0);
    test_wing_add(kit, TEST_WING_NAME("router.scale"), state_size, 0, 0);
    test_wing_add(kit, TEST_WING_NAME("router.per_expert_scale"), expert_count, 0, 0);
    test_wing_add(kit, TEST_WING_NAME("experts.gate_up_proj"), expert_count, 2 * expert_inner,
                  state_size);
    test_wing_add(kit, TEST_WING_NAME("experts.down_proj"), expert_count, state_size, expert_inner);
    test_wing_add(kit, TEST_WING_NAME("post_feedforward_layernorm_1.weight"), state_size, 0, 0);
    test_wing_add(kit, TEST_WING_NAME("pre_feedforward_layernorm_2.weight"), state_size, 0, 0);
    test_wing_add(kit, TEST_WING_NAME("post_feedforward_layernorm_2.weight"), state_size, 0, 0);
#undef TEST_WING_NAME
  }

  header_size = 256 * (size_t)kit->item_count + 64;
  header_text = (char *)mem_clear(header_size);
  if (!header_text) { mem_free(kit); return 0; }
  header_fill += (size_t)snprintf(header_text, header_size, "{");
  for (slot = 0; slot < kit->item_count; ++slot) {
    char shape_text[64];
    if (kit->size_list[slot][2] > 0)
      snprintf(shape_text, sizeof(shape_text), "[%d,%d,%d]", kit->size_list[slot][0],
               kit->size_list[slot][1], kit->size_list[slot][2]);
    else if (kit->size_list[slot][1] > 0)
      snprintf(shape_text, sizeof(shape_text), "[%d,%d]", kit->size_list[slot][0],
               kit->size_list[slot][1]);
    else
      snprintf(shape_text, sizeof(shape_text), "[%d]", kit->size_list[slot][0]);
    header_fill += (size_t)snprintf(header_text + header_fill, header_size - header_fill,
                                    "%s\"%s\":{\"dtype\":\"F32\",\"shape\":%s,"
                                    "\"data_offsets\":[%lu,%lu]}",
                                    slot ? "," : "", kit->name_list[slot], shape_text,
                                    (unsigned long)kit->from_list[slot],
                                    (unsigned long)(kit->from_list[slot] + kit->span_list[slot]));
  }
  header_fill += (size_t)snprintf(header_text + header_fill, header_size - header_fill, "}");

  pad_count = (8 - (header_fill % 8)) % 8;
  header_count = (uint64_t)(header_fill + pad_count);
  file_size = 8 + (size_t)header_count + kit->body_size;
  file_data = (uint8_t *)mem_clear(file_size);
  if (!file_data) { mem_free(header_text); mem_free(kit); return 0; }
  for (byte_index = 0; byte_index < 8; ++byte_index)
    file_data[byte_index] = (uint8_t)((header_count >> (8 * byte_index)) & 0xFFu);
  memcpy(file_data + 8, header_text, header_fill);
  for (byte_index = 0; byte_index < (int)pad_count; ++byte_index)
    file_data[8 + header_fill + byte_index] = ' ';
  body_data = (float *)(file_data + 8 + (size_t)header_count);
  for (slot = 0; slot < kit->item_count; ++slot) {
    size_t from_slot = kit->from_list[slot] / sizeof(float);
    size_t value_count = kit->span_list[slot] / sizeof(float);
    size_t value_index;
    for (value_index = 0; value_index < value_count; ++value_index)
      body_data[from_slot + value_index] =
          (float)(0.6 * sin((double)(value_index * 13 + (size_t)slot * 7 + 1) * 0.37));
  }
  okay_flag = test_file_write("model.safetensors", file_data, file_size);
  mem_free(file_data);
  mem_free(header_text);
  mem_free(kit);

  snprintf(config_text, sizeof(config_text),
           "{\"vocab_size\":%d,\"hidden_size\":%d,\"intermediate_size\":%d,"
           "\"num_hidden_layers\":%d,\"num_attention_heads\":%d,\"num_key_value_heads\":%d,"
           "\"head_dim\":%d,\"global_head_dim\":%d,\"sliding_window\":4,"
           "\"vocab_size_per_layer_input\":32,\"hidden_size_per_layer_input\":%d,"
           "\"num_kv_shared_layers\":0,\"rms_norm_eps\":1e-6,\"max_position_embeddings\":64,"
           "\"bos_token_id\":1,\"eos_token_id\":2,\"pad_token_id\":0,"
           "\"layer_types\":[\"sliding_attention\",\"full_attention\"],"
           "\"enable_moe_block\":%s,\"num_experts\":%d,\"top_k_experts\":2,"
           "\"moe_intermediate_size\":%d}",
           vocab_count, state_size, inner_size, layer_count, head_count, kv_count, head_size,
           head_size, ple_size, moe_flag ? "true" : "false", expert_count, expert_inner);
  if (!test_file_write("config.json", config_text, strlen(config_text))) okay_flag = 0;
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
    mem_free(want_list);
    session_close(wide_session);
    session_close(thin_session);
    model_free(model);
  }
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
  test_expert();
  test_kernel();
  test_rope();
  test_token();
  test_wing();
  test_face();

  test_yard_close();
  printf("\n%d passed, %d failed\n", test_pass_count, test_fail_count);
  return test_fail_count == 0 ? 0 : 1;
}

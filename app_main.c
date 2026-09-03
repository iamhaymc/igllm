/* app_main.c - command line front end for the engine in app_core.c. */

#include "app_core.c"

#define MAIN_PROMPT_LIMIT 16384

typedef struct main_flag {
  const char *task_text;
  const char *model_path;
  const char *prompt_text;
  int         serve_limit;
  int         thread_count;
  int         window_limit;
  int         verbose_level;
  int         raw_flag;
  app_taste   taste;
} main_flag;

static void main_usage(void) {
  printf("%s %s - Gemma 4 E2B IT QAT inference engine\n\n", APP_NAME, APP_VERSION);
  printf("usage: igllm <task> --model <folder> [options]\n\n");
  printf("tasks:\n");
  printf("  chat       one instruction-tuned turn, chat framed\n");
  printf("  complete   raw continuation of the prompt text\n");
  printf("  bench      timed prefill and decode report\n");
  printf("  tokens     print the token ids of the prompt\n");
  printf("  logits     print the next token distribution as json\n");
  printf("  probe      print the resolved model shape\n\n");
  printf("options:\n");
  printf("  --model <folder>    checkpoint folder in huggingface layout\n");
  printf("  --prompt <text>     prompt text, defaults to a short greeting\n");
  printf("  --serve <count>     tokens to produce, default 128\n");
  printf("  --threads <count>   worker threads, default host count\n");
  printf("  --window <count>    context length cap\n");
  printf("  --heat <value>      temperature, 0 for greedy\n");
  printf("  --top-k <count>     top-k cutoff, 0 disables\n");
  printf("  --top-p <value>     top-p cutoff\n");
  printf("  --echo-penalty <v>  repetition penalty\n");
  printf("  --seed <value>      sampler seed\n");
  printf("  --raw               skip the chat frame in the chat task\n");
  printf("  --verbose           print progress detail\n");
}

static int main_flags(int argc, char **argv, main_flag *flag_out) {
  int argument_index;
  memset(flag_out, 0, sizeof(*flag_out));
  flag_out->taste = app_taste_plain();
  flag_out->serve_limit = 128;
  flag_out->prompt_text = "Hello!";
  flag_out->task_text = argc > 1 && argv[1][0] != '-' ? argv[1] : NULL;
  if (argc > 1 && argv[1][0] == '-') return 0;

  for (argument_index = 2; argument_index < argc; ++argument_index) {
    const char *name_text = argv[argument_index];
    const char *value_text = argument_index + 1 < argc ? argv[argument_index + 1] : NULL;
    if (strcmp(name_text, "--model") == 0 && value_text) flag_out->model_path = argv[++argument_index];
    else if (strcmp(name_text, "--prompt") == 0 && value_text) flag_out->prompt_text = argv[++argument_index];
    else if (strcmp(name_text, "--serve") == 0 && value_text) flag_out->serve_limit = atoi(argv[++argument_index]);
    else if (strcmp(name_text, "--threads") == 0 && value_text) flag_out->thread_count = atoi(argv[++argument_index]);
    else if (strcmp(name_text, "--window") == 0 && value_text) flag_out->window_limit = atoi(argv[++argument_index]);
    else if (strcmp(name_text, "--heat") == 0 && value_text) flag_out->taste.heat_value = (float)atof(argv[++argument_index]);
    else if (strcmp(name_text, "--top-k") == 0 && value_text) flag_out->taste.top_count = atoi(argv[++argument_index]);
    else if (strcmp(name_text, "--top-p") == 0 && value_text) flag_out->taste.top_portion = (float)atof(argv[++argument_index]);
    else if (strcmp(name_text, "--echo-penalty") == 0 && value_text) flag_out->taste.echo_penalty = (float)atof(argv[++argument_index]);
    else if (strcmp(name_text, "--seed") == 0 && value_text) flag_out->taste.seed_value = strtoull(argv[++argument_index], NULL, 10);
    else if (strcmp(name_text, "--raw") == 0) flag_out->raw_flag = 1;
    else if (strcmp(name_text, "--verbose") == 0) flag_out->verbose_level = 1;
    else if (strcmp(name_text, "--help") == 0 || strcmp(name_text, "-h") == 0) return 0;
    else {
      fprintf(stderr, "unknown option: %s\n", name_text);
      return -1;
    }
  }
  return flag_out->task_text ? 1 : 0;
}

/* `logits` frames the same turn `chat` would, because what it reports is the
 * distribution the chat task samples from; comparing it against the reference
 * is only meaningful when both sides read the same prompt. `--raw` opts out. */
static int main_prompt_ids(app_model *model, const main_flag *flag, int32_t *id_list, int id_limit) {
  if ((strcmp(flag->task_text, "chat") == 0 || strcmp(flag->task_text, "logits") == 0) &&
      !flag->raw_flag)
    return token_frame(model, flag->prompt_text, id_list, id_limit);
  {
    int id_count = 0;
    int start_id = token_start_id(model);
    if (start_id >= 0) id_list[id_count++] = start_id;
    {
      int part_count = token_encode(model, flag->prompt_text, 1, id_list + id_count, id_limit - id_count);
      if (part_count > 0) id_count += part_count;
    }
    return id_count;
  }
}

static void main_emit(app_model *model, int32_t id_value) {
  char text_room[64];
  int text_count = token_decode(model, id_value, text_room, (int)sizeof(text_room));
  if (text_count > 0) {
    fwrite(text_room, 1, (size_t)text_count, stdout);
    fflush(stdout);
  }
}

static int main_serve(app_model *model, const main_flag *flag, int quiet_flag) {
  app_session *session = NULL;
  int32_t *id_list;
  int id_count, serve_index;
  int32_t id_value;
  const float *logit_list;
  app_code code;

  code = session_open(model, &session);
  if (code != APP_OKAY) {
    fprintf(stderr, "session: %s\n", app_code_text(code));
    return 1;
  }
  id_list = (int32_t *)malloc(sizeof(int32_t) * MAIN_PROMPT_LIMIT);
  if (!id_list) { session_close(session); return 1; }
  id_count = main_prompt_ids(model, flag, id_list, MAIN_PROMPT_LIMIT);
  if (id_count < 1) {
    fprintf(stderr, "prompt: empty\n");
    free(id_list);
    session_close(session);
    return 1;
  }
  if (flag->verbose_level) fprintf(stderr, "[prompt %d tokens]\n", id_count);

  code = session_prime(session, id_list, id_count);
  if (code != APP_OKAY) {
    fprintf(stderr, "prime: %s\n", app_code_text(code));
    free(id_list);
    session_close(session);
    return 1;
  }
  id_value = id_list[id_count - 1];
  for (serve_index = 0; serve_index < flag->serve_limit; ++serve_index) {
    logit_list = session_step(session, id_value);
    if (!logit_list) break;
    id_value = session_pick(session, logit_list, &flag->taste);
    if (token_is_close(model, id_value)) break;
    if (!quiet_flag) main_emit(model, id_value);
  }
  if (!quiet_flag) printf("\n");

  {
    app_tally tally = session_tally(session);
    double prime_rate = tally.prime_seconds > 0.0 ? (double)tally.prime_tokens / tally.prime_seconds : 0.0;
    double serve_rate = tally.serve_seconds > 0.0 ? (double)tally.serve_tokens / tally.serve_seconds : 0.0;
    fprintf(stderr, "prefill %.2f tok/s over %zu tokens\n", prime_rate, tally.prime_tokens);
    fprintf(stderr, "decode  %.2f tok/s over %zu tokens\n", serve_rate, tally.serve_tokens);
    fprintf(stderr, "memory  %.1f MiB\n", (double)tally.memory_bytes / (1024.0 * 1024.0));
  }
  free(id_list);
  session_close(session);
  return 0;
}

static int main_logits(app_model *model, const main_flag *flag) {
  app_session *session = NULL;
  int32_t *id_list;
  int id_count, show_count, slot_index;
  const float *logit_list;
  int vocab_count = model_vocab_count(model);
  app_code code = session_open(model, &session);

  if (code != APP_OKAY) {
    fprintf(stderr, "session: %s\n", app_code_text(code));
    return 1;
  }
  id_list = (int32_t *)malloc(sizeof(int32_t) * MAIN_PROMPT_LIMIT);
  if (!id_list) { session_close(session); return 1; }
  id_count = main_prompt_ids(model, flag, id_list, MAIN_PROMPT_LIMIT);
  if (id_count < 1 || session_prime(session, id_list, id_count) != APP_OKAY) {
    fprintf(stderr, "prime failed\n");
    free(id_list);
    session_close(session);
    return 1;
  }
  logit_list = session_step(session, id_list[id_count - 1]);
  if (!logit_list) {
    free(id_list);
    session_close(session);
    return 1;
  }

  show_count = flag->serve_limit > 0 && flag->serve_limit < vocab_count ? flag->serve_limit : 16;
  printf("{\"tokens\":[");
  for (slot_index = 0; slot_index < id_count; ++slot_index)
    printf("%s%d", slot_index ? "," : "", id_list[slot_index]);
  printf("],\"top\":[");
  {
    /* Partial selection of the highest scoring ids, without sorting the vocabulary. */
    int *rank_list = (int *)malloc(sizeof(int) * (size_t)show_count);
    int rank_count = 0, rank_index, vocab_index;
    for (vocab_index = 0; vocab_index < vocab_count; ++vocab_index) {
      float value = logit_list[vocab_index];
      if (rank_count < show_count) {
        rank_list[rank_count++] = vocab_index;
      } else if (value > logit_list[rank_list[rank_count - 1]]) {
        rank_list[rank_count - 1] = vocab_index;
      } else {
        continue;
      }
      for (rank_index = rank_count - 1; rank_index > 0; --rank_index) {
        if (logit_list[rank_list[rank_index]] <= logit_list[rank_list[rank_index - 1]]) break;
        {
          int keep_value = rank_list[rank_index];
          rank_list[rank_index] = rank_list[rank_index - 1];
          rank_list[rank_index - 1] = keep_value;
        }
      }
    }
    for (rank_index = 0; rank_index < rank_count; ++rank_index)
      printf("%s{\"id\":%d,\"logit\":%.6f}", rank_index ? "," : "", rank_list[rank_index],
             (double)logit_list[rank_list[rank_index]]);
    free(rank_list);
  }
  printf("]}\n");
  free(id_list);
  session_close(session);
  return 0;
}

static int main_tokens(app_model *model, const main_flag *flag) {
  int32_t id_list[MAIN_PROMPT_LIMIT];
  int id_count = main_prompt_ids(model, flag, id_list, MAIN_PROMPT_LIMIT);
  int id_index;
  for (id_index = 0; id_index < id_count; ++id_index) {
    char text_room[64];
    token_decode(model, id_list[id_index], text_room, (int)sizeof(text_room));
    printf("%6d  %s\n", id_list[id_index], text_room);
  }
  printf("%d tokens\n", id_count);
  return 0;
}

static int main_probe(app_model *model) {
  printf("model    %s\n", model_name(model));
  printf("layers   %d\n", model_layer_count(model));
  printf("vocab    %d\n", model_vocab_count(model));
  printf("window   %d\n", model_window_limit(model));
  printf("weights  %.1f MiB\n", (double)model_memory_bytes(model) / (1024.0 * 1024.0));
  return 0;
}

int main(int argc, char **argv) {
  main_flag flag;
  app_setup setup;
  app_model *model = NULL;
  app_code code;
  int result_code;
  int flag_state = main_flags(argc, argv, &flag);

  if (flag_state <= 0) {
    main_usage();
    return flag_state == 0 ? 0 : 1;
  }
  if (!flag.model_path) {
    fprintf(stderr, "missing --model <folder>\n");
    return 1;
  }

  setup = app_setup_plain();
  setup.thread_count = flag.thread_count;
  setup.window_limit = flag.window_limit;
  setup.verbose_level = flag.verbose_level;

  code = model_load(flag.model_path, &setup, &model);
  if (code != APP_OKAY) {
    fprintf(stderr, "load: %s\n", app_code_text(code));
    return 1;
  }

  if (strcmp(flag.task_text, "chat") == 0 || strcmp(flag.task_text, "complete") == 0)
    result_code = main_serve(model, &flag, 0);
  else if (strcmp(flag.task_text, "bench") == 0)
    result_code = main_serve(model, &flag, 1);
  else if (strcmp(flag.task_text, "logits") == 0)
    result_code = main_logits(model, &flag);
  else if (strcmp(flag.task_text, "tokens") == 0)
    result_code = main_tokens(model, &flag);
  else if (strcmp(flag.task_text, "probe") == 0)
    result_code = main_probe(model);
  else {
    main_usage();
    result_code = 1;
  }

  model_free(model);
  return result_code;
}

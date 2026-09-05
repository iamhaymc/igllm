/* app_main.c - command line front end for the engine in app_core.c. */

#include "app_core.c"

#define MAIN_PROMPT_LIMIT 16384

typedef struct main_flag {
  const char *task_text;
  const char *model_path;
  const char *prompt_text;
  const char *image_path;
  const char *audio_path;
  int         serve_limit;
  int         thread_count;
  int         window_limit;
  int         verbose_level;
  int         raw_flag;
  app_taste   taste;
} main_flag;

/* A prompt after both towers have run: the ids, one embedding row for every id
 * a tower filled, and the marks that say which those are.  The rows are laid
 * out against the ids rather than packed, so a lane index is an id index and
 * nothing downstream has to re-derive the mapping. */
typedef struct main_reel {
  int32_t *id_list;
  float   *state_list;
  uint8_t *state_flag;
  int      id_count;
  int      state_size;
} main_reel;

static void main_reel_free(main_reel *reel) {
  free(reel->id_list);
  free(reel->state_list);
  free(reel->state_flag);
  memset(reel, 0, sizeof(*reel));
}

static int main_reel_open(main_reel *reel, int state_size) {
  memset(reel, 0, sizeof(*reel));
  reel->state_size = state_size;
  reel->id_list = (int32_t *)calloc((size_t)MAIN_PROMPT_LIMIT, sizeof(int32_t));
  reel->state_list = (float *)calloc((size_t)MAIN_PROMPT_LIMIT * (size_t)state_size, sizeof(float));
  reel->state_flag = (uint8_t *)calloc((size_t)MAIN_PROMPT_LIMIT, 1);
  if (!reel->id_list || !reel->state_list || !reel->state_flag) {
    main_reel_free(reel);
    return 0;
  }
  return 1;
}

/* Copies one clip's rows onto the placeholder run the framer left for them. */
static void main_reel_lay(main_reel *reel, int from_index, const app_media *media) {
  int row_index;
  for (row_index = 0; row_index < media->row_count; ++row_index) {
    int slot_index = from_index + row_index;
    if (slot_index < 0 || slot_index >= reel->id_count) break;
    reel->state_flag[slot_index] = 1;
    memcpy(reel->state_list + (size_t)slot_index * (size_t)reel->state_size,
           media->state_data + (size_t)row_index * (size_t)media->state_size,
           sizeof(float) * (size_t)reel->state_size);
  }
}

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
  printf("  --image <path>      a png, pnm or bmp shown before the prompt\n");
  printf("  --audio <path>      a riff wave played before the prompt\n");
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
    else if (strcmp(name_text, "--image") == 0 && value_text) flag_out->image_path = argv[++argument_index];
    else if (strcmp(name_text, "--audio") == 0 && value_text) flag_out->audio_path = argv[++argument_index];
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

/* Runs whichever towers the caller asked for and reports how many placeholder
 * ids each of them wants.  A tower the checkpoint does not carry, or a file it
 * cannot read, is a failure rather than a silently dropped attachment: a prompt
 * that quietly lost its picture would answer the wrong question. */
static int main_media_load(app_model *model, const main_flag *flag, app_media *image_out,
                           app_media *audio_out) {
  app_code code;
  memset(image_out, 0, sizeof(*image_out));
  memset(audio_out, 0, sizeof(*audio_out));
  if (flag->image_path) {
    if (model_image_token(model) < 0) {
      fprintf(stderr, "image: this checkpoint has no image placeholder token\n");
      return 0;
    }
    code = media_image(model, flag->image_path, image_out);
    if (code != APP_OKAY) {
      fprintf(stderr, "image: %s (%s)\n", app_code_text(code), flag->image_path);
      return 0;
    }
  }
  if (flag->audio_path) {
    if (model_audio_token(model) < 0) {
      fprintf(stderr, "audio: this checkpoint has no audio placeholder token\n");
      media_free(image_out);
      return 0;
    }
    code = media_audio(model, flag->audio_path, audio_out);
    if (code != APP_OKAY) {
      fprintf(stderr, "audio: %s (%s)\n", app_code_text(code), flag->audio_path);
      media_free(image_out);
      return 0;
    }
  }
  return 1;
}

/* `logits` frames the same turn `chat` would, because what it reports is the
 * distribution the chat task samples from; comparing it against the reference
 * is only meaningful when both sides read the same prompt. `--raw` opts out.
 *
 * A media run sits between the turn opener and the user's text, bracketed the
 * way the processor brackets it, and the rows the towers produced are laid down
 * on the placeholders inside that bracket. */
static int main_reel_build(app_model *model, const main_flag *flag, main_reel *reel) {
  app_media image_media, audio_media;
  app_media_span span_list[2];
  int span_count = 0;
  int image_span = -1, audio_span = -1;
  int frame_flag = (strcmp(flag->task_text, "chat") == 0 ||
                    strcmp(flag->task_text, "logits") == 0) && !flag->raw_flag;

  if (!main_reel_open(reel, model_state_size(model))) return 0;
  if (!main_media_load(model, flag, &image_media, &audio_media)) {
    main_reel_free(reel);
    return 0;
  }
  /* One span per attachment, in the order a content list puts them: the
   * picture, then the clip, then the words. */
  if (image_media.row_count > 0) {
    span_list[span_count].kind_mark = APP_MEDIA_IMAGE;
    span_list[span_count].row_count = image_media.row_count;
    span_list[span_count].place_from = -1;
    image_span = span_count++;
  }
  if (audio_media.row_count > 0) {
    span_list[span_count].kind_mark = APP_MEDIA_AUDIO;
    span_list[span_count].row_count = audio_media.row_count;
    span_list[span_count].place_from = -1;
    audio_span = span_count++;
  }

  if (frame_flag) {
    reel->id_count = token_frame_media(model, flag->prompt_text, span_list, span_count,
                                       reel->id_list, MAIN_PROMPT_LIMIT);
  } else {
    int id_count = 0;
    int start_id = token_start_id(model);
    int part_count;
    if (start_id >= 0) id_count = 1, reel->id_list[0] = (int32_t)start_id;
    part_count = token_media_run(model, span_list, span_count, id_count, reel->id_list + id_count,
                                 MAIN_PROMPT_LIMIT - id_count);
    if (part_count > 0) id_count += part_count;
    part_count = token_encode(model, flag->prompt_text, 1, reel->id_list + id_count,
                              MAIN_PROMPT_LIMIT - id_count);
    if (part_count > 0) id_count += part_count;
    reel->id_count = id_count;
  }
  if (image_span >= 0 && span_list[image_span].place_from >= 0)
    main_reel_lay(reel, span_list[image_span].place_from, &image_media);
  if (audio_span >= 0 && span_list[audio_span].place_from >= 0)
    main_reel_lay(reel, span_list[audio_span].place_from, &audio_media);
  media_free(&image_media);
  media_free(&audio_media);
  if (reel->id_count < 1) {
    fprintf(stderr, "prompt: empty\n");
    main_reel_free(reel);
    return 0;
  }
  return 1;
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
  main_reel reel;
  int serve_index, last_index;
  int32_t id_value;
  const float *logit_list;
  const float *state_data;
  app_code code;

  code = session_open(model, &session);
  if (code != APP_OKAY) {
    fprintf(stderr, "session: %s\n", app_code_text(code));
    return 1;
  }
  if (!main_reel_build(model, flag, &reel)) {
    session_close(session);
    return 1;
  }
  if (flag->verbose_level) fprintf(stderr, "[prompt %d tokens]\n", reel.id_count);

  code = session_prime_media(session, reel.id_list, reel.id_count, reel.state_list, reel.state_flag);
  if (code != APP_OKAY) {
    fprintf(stderr, "prime: %s\n", app_code_text(code));
    main_reel_free(&reel);
    session_close(session);
    return 1;
  }
  last_index = reel.id_count - 1;
  id_value = reel.id_list[last_index];
  state_data = reel.state_flag[last_index]
                   ? reel.state_list + (size_t)last_index * (size_t)reel.state_size
                   : NULL;
  for (serve_index = 0; serve_index < flag->serve_limit; ++serve_index) {
    logit_list = session_step_state(session, id_value, state_data);
    state_data = NULL; /* only the prompt's last id can have come from a tower */
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
  main_reel_free(&reel);
  session_close(session);
  return 0;
}

static int main_logits(app_model *model, const main_flag *flag) {
  app_session *session = NULL;
  main_reel reel;
  int32_t *id_list;
  int id_count, show_count, slot_index;
  const float *logit_list;
  int vocab_count = model_vocab_count(model);
  app_code code = session_open(model, &session);

  if (code != APP_OKAY) {
    fprintf(stderr, "session: %s\n", app_code_text(code));
    return 1;
  }
  if (!main_reel_build(model, flag, &reel)) {
    session_close(session);
    return 1;
  }
  id_list = reel.id_list;
  id_count = reel.id_count;
  if (session_prime_media(session, id_list, id_count, reel.state_list, reel.state_flag) !=
      APP_OKAY) {
    fprintf(stderr, "prime failed\n");
    main_reel_free(&reel);
    session_close(session);
    return 1;
  }
  logit_list = session_step_state(session, id_list[id_count - 1],
                                  reel.state_flag[id_count - 1]
                                      ? reel.state_list + (size_t)(id_count - 1) *
                                                              (size_t)reel.state_size
                                      : NULL);
  if (!logit_list) {
    main_reel_free(&reel);
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
  main_reel_free(&reel);
  session_close(session);
  return 0;
}

static int main_tokens(app_model *model, const main_flag *flag) {
  main_reel reel;
  int id_index;
  if (!main_reel_build(model, flag, &reel)) return 1;
  for (id_index = 0; id_index < reel.id_count; ++id_index) {
    char text_room[64];
    token_decode(model, reel.id_list[id_index], text_room, (int)sizeof(text_room));
    printf("%6d  %s%s\n", reel.id_list[id_index], text_room,
           reel.state_flag[id_index] ? "  [tower]" : "");
  }
  printf("%d tokens\n", reel.id_count);
  main_reel_free(&reel);
  return 0;
}

static int main_probe(app_model *model) {
  const char *name_text = model_name(model);
  printf("model    %s\n", name_text ? name_text : "");
  printf("layers   %d\n", model_layer_count(model));
  printf("vocab    %d\n", model_vocab_count(model));
  printf("window   %d\n", model_window_limit(model));
  printf("state    %d\n", model_state_size(model));
  printf("vision   %s\n", model_vision_ready(model) ? "yes" : "no");
  if (model_vision_ready(model))
    printf("  rows   %d, placeholder id %d\n", model_image_rows(model), model_image_token(model));
  printf("audio    %s\n", model_audio_ready(model) ? "yes" : "no");
  if (model_audio_ready(model)) printf("  placeholder id %d\n", model_audio_token(model));
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

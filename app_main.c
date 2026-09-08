/* app_main.c - command line front end for the engine in app_core.c. */

#include "app_core.c"

#define MAIN_PROMPT_LIMIT 16384

#define MAIN_MEDIA_LIMIT 8

/* One piece of the turn, kept where the command line put it.  The order the
 * flags are given is the order the pieces are laid down in, which is what a
 * content list means by order; before this the picture always led because there
 * could only be one of each, and the words always came last because there was
 * no way to say otherwise.
 *
 * `--prompt` still means the words after the attachments, wherever it is
 * written, so every invocation that worked before works the same way. `--text`
 * is the one that takes its place in the order. */
#define MAIN_SHOW_IMAGE APP_MEDIA_IMAGE
#define MAIN_SHOW_AUDIO APP_MEDIA_AUDIO
#define MAIN_SHOW_TEXT  2

typedef struct main_show {
  const char *path_text; /* the file, or the words when this is a text piece */
  int         kind_mark; /* MAIN_SHOW_IMAGE, MAIN_SHOW_AUDIO or MAIN_SHOW_TEXT */
} main_show;

typedef struct main_flag {
  const char *task_text;
  const char *model_path;
  const char *prompt_text;
  main_show   show_list[MAIN_MEDIA_LIMIT];
  int         show_count;
  int         text_count;   /* how many of those are words */
  int         prompt_flag;  /* --prompt was given rather than defaulted */
  int         serve_limit;
  int         thread_count;
  int         window_limit;
  int         verbose_level;
  int         cache_bits;   /* 0 keeps the key and value cache in float, 8 quantizes it */
  int         raw_flag;
  int         loop_flag;
  /* Lanes a speculative round may carry, zero for none.  One is the same as
   * none: a block of one lane is an ordinary step with the block path's
   * bookkeeping around it. */
  int         guess_span;
  /* Soft tokens one picture may cost, zero for the checkpoint's own maximum.
   * It buys most of what a picture costs and it costs detail, so it is a flag
   * and never a default. */
  int         image_rows;
  /* A file the pictures' rows are read from at the start and written to at the
   * end, so that a photograph asked about in two runs is encoded in one. */
  const char *image_keep;
  const char *keep_path;
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
  printf("  chat       one instruction-tuned turn, chat framed; --loop for more\n");
  printf("  complete   raw continuation of the prompt text\n");
  printf("  bench      timed prefill and decode report\n");
  printf("  tokens     print the token ids of the prompt\n");
  printf("  logits     print the next token distribution as json\n");
  printf("  probe      print the resolved model shape\n");
  printf("  cache      print the calibrated cache ranges against a prompt\n");
  printf("  guess      what a block of guesses is worth, against a proposer's ceiling\n\n");
  printf("options:\n");
  printf("  --model <folder>    checkpoint folder in huggingface layout\n");
  printf("  --prompt <text>     prompt text, defaults to a short greeting\n");
  printf("  --text <text>       words in the order the flags give them, repeatable\n");
  printf("  --image <path>      a png, jpeg, pnm or bmp shown before the prompt, repeatable\n");
  printf("  --audio <path>      a riff wave played before the prompt, repeatable\n");
  printf("  --serve <count>     tokens to produce, default 128\n");
  printf("  --threads <count>   worker threads, default host count\n");
  printf("  --window <count>    context length cap\n");
  printf("  --image-tokens <n>  soft tokens a picture may cost, fewer for a faster read\n");
  printf("  --image-keep <path> hold pictures' rows here, and reuse them next run\n");
  printf("  --cache <bits>      key and value cache storage, 0 float or 8 quantized\n");
  printf("  --heat <value>      temperature, 0 for greedy\n");
  printf("  --top-k <count>     top-k cutoff, 0 disables\n");
  printf("  --top-p <value>     top-p cutoff\n");
  printf("  --echo-penalty <v>  repetition penalty\n");
  printf("  --seed <value>      sampler seed\n");
  printf("  --guess <lanes>     speculative block, 0 or 1 for none\n");
  printf("  --loop              keep the chat task open for more turns\n");
  printf("  --keep <path>       hold the prompt's cache here, and reuse it next time\n");
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
    else if (strcmp(name_text, "--prompt") == 0 && value_text) {
      flag_out->prompt_text = argv[++argument_index];
      flag_out->prompt_flag = 1;
    }
    else if ((strcmp(name_text, "--image") == 0 || strcmp(name_text, "--audio") == 0 ||
              strcmp(name_text, "--text") == 0) && value_text) {
      if (flag_out->show_count >= MAIN_MEDIA_LIMIT) {
        fprintf(stderr, "prompt: at most %d pieces in one turn\n", MAIN_MEDIA_LIMIT);
        return -1;
      }
      flag_out->show_list[flag_out->show_count].kind_mark =
          name_text[2] == 'i' ? MAIN_SHOW_IMAGE
                              : (name_text[2] == 'a' ? MAIN_SHOW_AUDIO : MAIN_SHOW_TEXT);
      if (flag_out->show_list[flag_out->show_count].kind_mark == MAIN_SHOW_TEXT)
        ++flag_out->text_count;
      flag_out->show_list[flag_out->show_count++].path_text = argv[++argument_index];
    }
    else if (strcmp(name_text, "--serve") == 0 && value_text) flag_out->serve_limit = atoi(argv[++argument_index]);
    else if (strcmp(name_text, "--guess") == 0 && value_text) flag_out->guess_span = atoi(argv[++argument_index]);
    else if (strcmp(name_text, "--threads") == 0 && value_text) flag_out->thread_count = atoi(argv[++argument_index]);
    else if (strcmp(name_text, "--window") == 0 && value_text) flag_out->window_limit = atoi(argv[++argument_index]);
    else if (strcmp(name_text, "--image-tokens") == 0 && value_text) flag_out->image_rows = atoi(argv[++argument_index]);
    else if (strcmp(name_text, "--image-keep") == 0 && value_text) flag_out->image_keep = argv[++argument_index];
    else if (strcmp(name_text, "--cache") == 0 && value_text) flag_out->cache_bits = atoi(argv[++argument_index]);
    else if (strcmp(name_text, "--heat") == 0 && value_text) flag_out->taste.heat_value = (float)atof(argv[++argument_index]);
    else if (strcmp(name_text, "--top-k") == 0 && value_text) flag_out->taste.top_count = atoi(argv[++argument_index]);
    else if (strcmp(name_text, "--top-p") == 0 && value_text) flag_out->taste.top_portion = (float)atof(argv[++argument_index]);
    else if (strcmp(name_text, "--echo-penalty") == 0 && value_text) flag_out->taste.echo_penalty = (float)atof(argv[++argument_index]);
    else if (strcmp(name_text, "--seed") == 0 && value_text) flag_out->taste.seed_value = strtoull(argv[++argument_index], NULL, 10);
    else if (strcmp(name_text, "--keep") == 0 && value_text) flag_out->keep_path = argv[++argument_index];
    else if (strcmp(name_text, "--loop") == 0) flag_out->loop_flag = 1;
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
static int main_media_load(app_model *model, const main_flag *flag, app_media *media_list) {
  int show_index;
  for (show_index = 0; show_index < MAIN_MEDIA_LIMIT; ++show_index)
    memset(&media_list[show_index], 0, sizeof(media_list[show_index]));
  for (show_index = 0; show_index < flag->show_count; ++show_index) {
    const main_show *show = &flag->show_list[show_index];
    int audio_flag = show->kind_mark == MAIN_SHOW_AUDIO;
    const char *kind_text = audio_flag ? "audio" : "image";
    app_code code;
    if (show->kind_mark == MAIN_SHOW_TEXT) continue; /* words open no file */
    if ((audio_flag ? model_audio_token(model) : model_image_token(model)) < 0) {
      fprintf(stderr, "%s: this checkpoint has no %s placeholder token\n", kind_text, kind_text);
      break;
    }
    code = audio_flag ? media_audio(model, show->path_text, &media_list[show_index])
                      : media_image(model, show->path_text, &media_list[show_index]);
    if (code != APP_OKAY) {
      fprintf(stderr, "%s: %s (%s)\n", kind_text, app_code_text(code), show->path_text);
      break;
    }
    /* A clip past the processor's budget is cut to it, which is what the
     * reference does with one.  Saying so is the difference between an answer
     * about the whole clip and an answer about its first half, and with more
     * than one clip in a prompt the line has to say which of them was cut. */
    /* What the picture actually came to, which is not the budget: both sides of
     * the resize round down to a whole pooling window, so the rows land under
     * whatever cap was in force rather than on it. */
    if (!audio_flag && flag->verbose_level > 0)
      fprintf(stderr, "[image %s: %d rows of at most %d]\n", show->path_text,
              media_list[show_index].row_count, model_image_rows(model));
    if (audio_flag && media_list[show_index].cut_flag)
      fprintf(stderr, "audio: the clip runs past the budget of %d s and is cut to it (%s)\n",
              (model_audio_rows(model) * model_audio_span_ms(model) + 999) / 1000,
              show->path_text);
  }
  if (show_index >= flag->show_count) return 1;
  /* Whatever loaded before the failure is the caller's to lose, not to leak. */
  while (show_index-- > 0) media_free(&media_list[show_index]);
  return 0;
}

/* `logits` frames the same turn `chat` would, because what it reports is the
 * distribution the chat task samples from; comparing it against the reference
 * is only meaningful when both sides read the same prompt. `--raw` opts out.
 *
 * A media run sits between the turn opener and the user's text, bracketed the
 * way the processor brackets it, and the rows the towers produced are laid down
 * on the placeholders inside that bracket. */
static int main_reel_build(app_model *model, const main_flag *flag, main_reel *reel,
                           int first_flag) {
  app_media media_list[MAIN_MEDIA_LIMIT];
  app_media_span span_list[MAIN_MEDIA_LIMIT];
  app_part part_list[MAIN_MEDIA_LIMIT + 1];
  int span_count = flag->show_count;
  int span_index, part_count = 0;
  int frame_flag = (strcmp(flag->task_text, "chat") == 0 ||
                    strcmp(flag->task_text, "logits") == 0) && !flag->raw_flag;

  if (!main_reel_open(reel, model_state_size(model))) return 0;
  if (!main_media_load(model, flag, media_list)) {
    main_reel_free(reel);
    return 0;
  }
  /* One span per attachment, in the order the command line gave them, which is
   * the order a content list puts them in.  A tower that produced no rows keeps
   * its place rather than being dropped, and so does a piece that is words, so
   * a span index stays an attachment index and every run gets its own rows
   * back. */
  for (span_index = 0; span_index < span_count; ++span_index) {
    int text_flag = flag->show_list[span_index].kind_mark == MAIN_SHOW_TEXT;
    span_list[span_index].kind_mark = text_flag ? MAIN_SHOW_IMAGE
                                                : flag->show_list[span_index].kind_mark;
    span_list[span_index].row_count = text_flag ? 0 : media_list[span_index].row_count;
    span_list[span_index].place_from = -1;
    part_list[part_count].kind_mark = text_flag ? APP_PART_TEXT : APP_PART_MEDIA;
    part_list[part_count].text_ref = text_flag ? flag->show_list[span_index].path_text : NULL;
    part_list[part_count].span_index = text_flag ? -1 : span_index;
    ++part_count;
  }
  /* `--prompt` is the words after the attachments, wherever on the line it was
   * written, so every invocation that predates `--text` lays down what it always
   * did.  The greeting it defaults to stands in only when nothing else spoke. */
  if (flag->prompt_flag || flag->text_count == 0) {
    part_list[part_count].kind_mark = APP_PART_TEXT;
    part_list[part_count].text_ref = flag->prompt_text;
    part_list[part_count].span_index = -1;
    ++part_count;
  }

  if (frame_flag) {
    /* A turn that follows one the model has answered picks the frame up where
     * the last one stopped rather than opening the document again. */
    reel->id_count = first_flag ? token_frame_parts(model, part_list, part_count, span_list,
                                                   span_count, reel->id_list, MAIN_PROMPT_LIMIT)
                                : token_frame_next(model, part_list, part_count, span_list,
                                                   span_count, reel->id_list, MAIN_PROMPT_LIMIT);
  } else {
    int id_count = 0;
    int start_id = token_start_id(model);
    int part_index;
    if (start_id >= 0) id_count = 1, reel->id_list[0] = (int32_t)start_id;
    for (part_index = 0; part_index < part_count; ++part_index) {
      const app_part *part = &part_list[part_index];
      int wrote_count = part->kind_mark == APP_PART_MEDIA
                            ? token_media_run(model, span_list + part->span_index, 1, id_count,
                                              reel->id_list + id_count,
                                              MAIN_PROMPT_LIMIT - id_count)
                            : token_encode(model, part->text_ref, 1, reel->id_list + id_count,
                                           MAIN_PROMPT_LIMIT - id_count);
      if (wrote_count > 0) id_count += wrote_count;
    }
    reel->id_count = id_count;
  }
  for (span_index = 0; span_index < span_count; ++span_index)
    if (span_list[span_index].place_from >= 0)
      main_reel_lay(reel, span_list[span_index].place_from, &media_list[span_index]);
  for (span_index = 0; span_index < span_count; ++span_index) media_free(&media_list[span_index]);
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

/* What a kept cache is stamped with: the embedding rows a tower filled, which
 * the ids cannot speak for.  Two pictures lay down the same placeholder ids, so
 * without this a cache kept for one prompt would be restored for the other. */
static uint64_t main_keep_stamp(const main_reel *reel, int id_count) {
  uint64_t stamp_value = 0xCBF29CE484222325ull;
  int id_index, value_index;
  for (id_index = 0; id_index < id_count; ++id_index) {
    if (!reel->state_flag[id_index]) continue;
    for (value_index = 0; value_index < reel->state_size; ++value_index) {
      float value_now = reel->state_list[(size_t)id_index * (size_t)reel->state_size +
                                         (size_t)value_index];
      uint32_t raw_value;
      memcpy(&raw_value, &value_now, sizeof(raw_value));
      stamp_value = (stamp_value ^ raw_value) * 0x100000001B3ull;
    }
  }
  return stamp_value;
}

/* Primes a prompt, reusing a cache kept from an earlier run where the file
 * holds a prompt this one begins with.
 *
 * What is reused is a prefix and not a match: a kept cache of six hundred ids
 * in front of a prompt of six hundred and four is six hundred ids that need not
 * be primed again, and the four are primed onto it.  A file that does not
 * begin this prompt is a different conversation and is replaced. */
static app_code main_keep_prime(const main_flag *flag, app_session *session,
                                const main_reel *reel) {
  int32_t *held_list = NULL;
  int held_count = 0, same_count = 0, held_kind = APP_KEEP_PROMPT;
  uint64_t stamp_value = 0, held_stamp = 0;
  app_code code;

  if (!flag->keep_path) return session_prime_media(session, reel->id_list, reel->id_count,
                                                   reel->state_list, reel->state_flag);
  /* Only a prompt file is a prompt cache.  A conversation holds an answer and
   * every id of it, so priming a prompt onto it would be a turn the model never
   * had; the loop's `/open` is what a conversation is read with. */
  if (session_load(session, flag->keep_path, &held_stamp, &held_kind) == APP_OKAY &&
      held_kind == APP_KEEP_PROMPT) {
    held_count = session_ids(session, NULL, 0);
    held_list = (int32_t *)calloc((size_t)(held_count > 0 ? held_count : 1), sizeof(int32_t));
    if (held_list) session_ids(session, held_list, held_count);
    /* The last id of the prompt is the one `session_step` is fed, so a cache
     * that holds every id of it has nothing left to prime and is one too far. */
    while (held_list && same_count < held_count && same_count < reel->id_count - 1 &&
           held_list[same_count] == reel->id_list[same_count])
      ++same_count;
    stamp_value = main_keep_stamp(reel, same_count);
    if (same_count != held_count || held_stamp != stamp_value) same_count = 0;
    free(held_list);
  }
  if (same_count < 1) {
    session_reset(session);
    same_count = 0;
  } else if (flag->verbose_level) {
    fprintf(stderr, "[kept %d of %d ids]\n", same_count, reel->id_count);
  }
  code = session_prime_media(session, reel->id_list + same_count, reel->id_count - same_count,
                             reel->state_list
                                 ? reel->state_list + (size_t)same_count * (size_t)reel->state_size
                                 : NULL,
                             reel->state_flag ? reel->state_flag + same_count : NULL);
  if (code != APP_OKAY) return code;
  /* The file holds the prompt rather than the answer, so it is written before
   * a token is sampled and a rerun of the same prompt starts where this one
   * did. */
  if (same_count < reel->id_count - 1) {
    app_code keep_code = session_save(session, flag->keep_path,
                                      main_keep_stamp(reel, reel->id_count - 1),
                                      APP_KEEP_PROMPT);
    if (keep_code != APP_OKAY)
      fprintf(stderr, "keep: %s\n", app_code_text(keep_code));
  }
  return APP_OKAY;
}

/* Samples until the model closes its turn or the caller's budget runs out,
 * printing each piece as it lands.  The state row belongs to the prompt's last
 * id and only that id can have one, because only a prompt's ids come from a
 * tower. */
/* The largest of a row of logits, which is what a greedy step picks and what
 * a greedy verification compares against. */
static int32_t main_guess_top(const float *logit_list, int vocab_count) {
  int32_t best_id = 0;
  int slot;
  for (slot = 1; slot < vocab_count; ++slot)
    if (logit_list[slot] > logit_list[best_id]) best_id = (int32_t)slot;
  return best_id;
}

/* What a run of speculative rounds bought, for the line printed after it. */
typedef struct main_bet {
  long round_count;   /* blocks run */
  long draw_count;    /* guesses proposed */
  long take_count;    /* guesses the model agreed with */
  long token_count;   /* tokens the answer produced */
} main_bet;

/* One answer, a block of guesses at a time.
 *
 * The scout proposes what the stream did the last time it said this, the block
 * verifies every position in one pass, and the leading guesses the model agrees
 * with are kept.  The token after the last accepted guess is the model's own
 * and is true whatever was guessed, so a round always commits at least one:
 * this cannot loop without progress and cannot produce a token a plain decode
 * would not have.
 *
 * **Greedy only, and the caller has already checked it.** Verification here is
 * an argmax comparison, which is the right rule at a temperature of zero and
 * the wrong one above it — speculative decoding under a temperature needs the
 * modified rejection rule, accept with probability `min(1, p/q)` and resample
 * from the difference, and an n-gram proposer has no `q` to divide by.  Rather
 * than sample from a distribution the block would have skewed, `--guess` is
 * refused where the taste is not greedy.
 *
 * Where the scout has nothing to propose the round is a block of one lane.
 * That is a little more than a plain step rather than a little less, and it is
 * why `--guess` is a flag and not the default: on free generation the scout
 * draws rarely and the flag costs a few percent, while on text that quotes its
 * prompt it is worth more than half again. */
static void main_answer_guess(app_model *model, const main_flag *flag, app_session *session,
                              app_scout *scout, const main_reel *reel, int quiet_flag,
                              main_bet *bet) {
  int32_t block_list[16];
  int32_t id_value = reel->id_list[reel->id_count - 1];
  int32_t after_id = 0;
  int block_want = flag->guess_span;
  int shut_flag = 0;
  /* Greedy verification is an argmax comparison; anything else is the
   * rejection rule. */
  int taste_flag = flag->taste.heat_value > 0.0f;
  if (block_want > session_guess_limit(session)) block_want = session_guess_limit(session);
  if (block_want < 1) block_want = 1;
  /* This turn's ids, which in a conversation are the new turn's alone: the
   * scout is the caller's and already holds every turn before it. */
  if (!scout || scout_note(scout, reel->id_list, reel->id_count) != APP_OKAY) return;
  while (bet->token_count < flag->serve_limit && !shut_flag) {
    const float *rows;
    int span_count, slot, take_count = 0;
    int want_count = block_want - 1;
    if (want_count > flag->serve_limit - (int)bet->token_count - 1)
      want_count = flag->serve_limit - (int)bet->token_count - 1;
    if (want_count < 0) want_count = 0;
    block_list[0] = id_value;
    span_count = 1 + scout_draw(scout, block_list + 1, want_count);
    if (span_count == 1) {
      /* Nothing proposed, so nothing to verify: an ordinary step, which is
       * cheaper than a block of one lane and is the whole of what `--guess`
       * costs on a stream the scout cannot help with. */
      const float *logit_list = session_step(session, id_value);
      int32_t made_id;
      if (!logit_list) break;
      bet->round_count += 1;
      made_id = taste_flag ? session_pick(session, logit_list, &flag->taste)
                           : main_guess_top(logit_list, model_vocab_count(model));
      if (token_is_close(model, made_id)) break;
      id_value = made_id;
      if (!quiet_flag) main_emit(model, made_id);
      bet->token_count += 1;
      if (scout_note(scout, &made_id, 1) != APP_OKAY) break;
      continue;
    }
    rows = session_guess(session, block_list, span_count);
    if (!rows) break;
    bet->round_count += 1;
    bet->draw_count += span_count - 1;
    if (taste_flag) {
      /* Under a temperature the block is verified by speculative sampling's
       * modified rejection rule, which hands back both the prefix it kept and
       * the token that follows it. */
      take_count = session_guess_taste(session, rows, block_list, span_count, &flag->taste,
                                       &after_id);
      if (take_count < 0) break;
    } else {
      while (take_count + 1 < span_count &&
             main_guess_top(rows + (size_t)take_count * (size_t)model_vocab_count(model),
                            model_vocab_count(model)) == block_list[take_count + 1])
        take_count += 1;
      after_id = main_guess_top(rows + (size_t)take_count * (size_t)model_vocab_count(model),
                                model_vocab_count(model));
    }
    bet->take_count += take_count;
    for (slot = 0; slot <= take_count && bet->token_count < flag->serve_limit; ++slot) {
      int32_t made_id = slot < take_count ? block_list[slot + 1] : after_id;
      if (token_is_close(model, made_id)) {
        shut_flag = 1;
        /* The block is kept up to the token before the close, so the cache
         * carries exactly what was emitted. */
        take_count = slot;
        break;
      }
      id_value = made_id;
      if (!quiet_flag) main_emit(model, made_id);
      bet->token_count += 1;
      if (scout_note(scout, &made_id, 1) != APP_OKAY) { shut_flag = 1; break; }
    }
    if (session_guess_keep(session, take_count + 1) != APP_OKAY) break;
  }
}

static void main_answer(app_model *model, const main_flag *flag, app_session *session,
                        int32_t id_value, const float *state_data, int quiet_flag) {
  int serve_index;
  for (serve_index = 0; serve_index < flag->serve_limit; ++serve_index) {
    const float *logit_list = session_step_state(session, id_value, state_data);
    state_data = NULL;
    if (!logit_list) break;
    id_value = session_pick(session, logit_list, &flag->taste);
    if (token_is_close(model, id_value)) break;
    if (!quiet_flag) main_emit(model, id_value);
  }
}

/* What one clock read costs on this host, so the timer can be held to account
 * in the units it reports in rather than assumed to be free.
 *
 * The loop is written so the compiler cannot hoist the call: each read is
 * consumed.  A hundred thousand of them is a few milliseconds and is paid once,
 * after the run it describes, so it is not in the numbers it qualifies. */
static double main_clock_cost(void) {
  double from_time, keep_value = 0.0;
  int read_index;
  const int read_limit = 100000;
  from_time = time_now();
  for (read_index = 0; read_index < read_limit; ++read_index) keep_value += time_now();
  if (keep_value == 0.0) return 0.0; /* never, and the compiler cannot know it */
  return (time_now() - from_time) / (double)read_limit;
}

/* Where a decode step goes, a part at a time.
 *
 * The parts are a partition of the step and not a sample of it, so they sum to
 * the step by construction; what the last two lines report is the two ways that
 * claim could still be wrong.  `unnamed` is the step's wall clock less the sum,
 * which is the accounting either side of the pass and should be microseconds.
 * `timer` is the clock reads the division itself spent, priced at what a read
 * measured just now — the part of every number above that is the measuring. */
static void main_phases(const app_session *session) {
  app_phase_book book = session_phases(session);
  size_t byte_list[APP_PHASE_COUNT];
  double total_seconds = 0.0, pass_seconds, step_each, read_cost;
  size_t total_bytes = 0;
  int phase_slot, order_list[APP_PHASE_COUNT], order_index, order_scan;

  if (!book.step_count) return;
  session_phase_bytes(session, byte_list);
  for (phase_slot = 0; phase_slot < APP_PHASE_COUNT; ++phase_slot) {
    total_seconds += book.seconds[phase_slot];
    total_bytes += byte_list[phase_slot];
  }
  if (total_seconds <= 0.0) return;
  /* The sampler runs after `session_step` returns, so it is in the token but
   * not in the step the step's clock timed.  Everything else is the pass. */
  pass_seconds = total_seconds - book.seconds[APP_PHASE_PICK];
  step_each = book.step_seconds / (double)book.step_count;

  /* Largest first: the point of the report is which part to look at next, and
   * a list in graph order buries it. */
  for (phase_slot = 0; phase_slot < APP_PHASE_COUNT; ++phase_slot)
    order_list[phase_slot] = phase_slot;
  for (order_index = 1; order_index < APP_PHASE_COUNT; ++order_index) {
    int keep_slot = order_list[order_index];
    for (order_scan = order_index;
         order_scan > 0 && book.seconds[order_list[order_scan - 1]] < book.seconds[keep_slot];
         --order_scan)
      order_list[order_scan] = order_list[order_scan - 1];
    order_list[order_scan] = keep_slot;
  }

  fprintf(stderr, "\nphases  %zu decode steps, %.2f ms a step, %.1f MiB swept\n", book.step_count,
          step_each * 1000.0, (double)total_bytes / (1024.0 * 1024.0));
  fprintf(stderr, "%-24s %10s %7s %10s %8s %7s\n", "part", "ms a step", "share", "MiB a step",
          "GiB/s", "a step");
  for (order_index = 0; order_index < APP_PHASE_COUNT; ++order_index) {
    double phase_seconds, phase_mib;
    phase_slot = order_list[order_index];
    if (book.counts[phase_slot] == 0) continue;
    phase_seconds = book.seconds[phase_slot] / (double)book.step_count;
    phase_mib = (double)byte_list[phase_slot] / (1024.0 * 1024.0);
    fprintf(stderr, "%-24s %10.3f %6.1f%% %10.1f ", app_phase_text(phase_slot),
            phase_seconds * 1000.0, 100.0 * book.seconds[phase_slot] / total_seconds, phase_mib);
    /* A part that reads no weights has no rate to quote, and quoting one would
     * invite the reader to compare it with the memory's. */
    if (byte_list[phase_slot] > 0 && phase_seconds > 0.0)
      fprintf(stderr, "%8.2f", phase_mib / 1024.0 / phase_seconds);
    else
      fprintf(stderr, "%8s", "-");
    fprintf(stderr, " %7.1f\n", (double)book.counts[phase_slot] / (double)book.step_count);
  }
  fprintf(stderr, "%-24s %10.3f %6.1f%% %10.1f %8.2f\n", "named, in all",
          total_seconds / (double)book.step_count * 1000.0, 100.0,
          (double)total_bytes / (1024.0 * 1024.0),
          (double)total_bytes / (1024.0 * 1024.0 * 1024.0) /
              (total_seconds / (double)book.step_count));

  read_cost = main_clock_cost();
  fprintf(stderr, "unnamed %.3f ms a step: the step's own clock, less the pass's parts\n",
          (book.step_seconds - pass_seconds) / (double)book.step_count * 1000.0);
  fprintf(stderr, "timer   %.3f ms a step of the above, %.0f reads at %.0f ns\n",
          read_cost * (double)book.read_count / (double)book.step_count * 1000.0,
          (double)book.read_count / (double)book.step_count, read_cost * 1e9);
}

static int main_serve(app_model *model, const main_flag *flag, int quiet_flag) {
  app_session *session = NULL;
  main_reel reel;
  main_bet bet;
  app_scout *scout = NULL;
  int last_index;
  int guess_flag;
  int32_t id_value;
  const float *state_data;
  app_code code;

  memset(&bet, 0, sizeof(bet));
  guess_flag = flag->guess_span > 1;

  code = session_open(model, &session);
  if (code != APP_OKAY) {
    fprintf(stderr, "session: %s\n", app_code_text(code));
    return 1;
  }
  if (!main_reel_build(model, flag, &reel, 1)) {
    session_close(session);
    return 1;
  }
  if (flag->verbose_level) fprintf(stderr, "[prompt %d tokens]\n", reel.id_count);

  code = main_keep_prime(flag, session, &reel);
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
  /* A block cannot carry an embedding row a tower filled, so a prompt whose
   * last id is a soft token takes the plain loop whatever the flag says. */
  if (guess_flag && !state_data && scout_open(&scout) == APP_OKAY) {
    main_answer_guess(model, flag, session, scout, &reel, quiet_flag, &bet);
    scout_close(scout);
  } else {
    main_answer(model, flag, session, id_value, state_data, quiet_flag);
    guess_flag = 0;
  }
  if (!quiet_flag) printf("\n");

  {
    app_tally tally = session_tally(session);
    double prime_rate = tally.prime_seconds > 0.0 ? (double)tally.prime_tokens / tally.prime_seconds : 0.0;
    double serve_rate = tally.serve_seconds > 0.0 ? (double)tally.serve_tokens / tally.serve_seconds : 0.0;
    double read_rate =
        tally.serve_seconds > 0.0
            ? (double)tally.serve_bytes / tally.serve_seconds / (1024.0 * 1024.0 * 1024.0)
            : 0.0;
    double read_each = tally.serve_tokens > 0
                           ? (double)tally.serve_bytes / (double)tally.serve_tokens /
                                 (1024.0 * 1024.0)
                           : 0.0;
    fprintf(stderr, "prefill %.2f tok/s over %zu tokens\n", prime_rate, tally.prime_tokens);
    fprintf(stderr, "decode  %.2f tok/s over %zu tokens\n", serve_rate, tally.serve_tokens);
    /* The weights and cache the decode steps actually read, and the rate that
     * comes to.  Dividing the resident total by the same seconds would name a
     * bandwidth the run never asked the machine for. */
    fprintf(stderr, "reads   %.1f MiB a token, %.2f GiB/s\n", read_each, read_rate);
    fprintf(stderr, "weights %.1f MiB mapped\n",
            (double)model_memory_bytes(model) / (1024.0 * 1024.0));
    fprintf(stderr, "memory  %.1f MiB allocated\n",
            (double)tally.memory_bytes / (1024.0 * 1024.0));
    /* What the block path actually bought, in the two numbers that decide it:
     * how many tokens a round carried, and how often the scout was right.  A
     * round always commits at least one, so a figure at 1.00 means the guessing
     * never paid and the flag cost whatever the extra lanes cost. */
    if (guess_flag && bet.round_count > 0)
      fprintf(stderr, "guess   %.2f tokens a round over %ld rounds, %.0f%% of %ld guesses kept\n",
              (double)bet.token_count / (double)bet.round_count, bet.round_count,
              bet.draw_count > 0 ? 100.0 * (double)bet.take_count / (double)bet.draw_count : 0.0,
              bet.draw_count);
  }
  if (flag->verbose_level) main_phases(session);
  main_reel_free(&reel);
  session_close(session);
  return 0;
}

/* -- the talk loop -------------------------------------------------------- */

/* Several conversations against one loaded model.
 *
 * This is the property the split between `app_model` and `app_session` exists
 * for, and the single turn tasks never showed it: the weights are mapped once
 * and every conversation costs only its own cache and scratch.  What a
 * conversation costs is what `/list` reports.
 *
 * The turns run one at a time.  A model's kernels reach the same thread pool,
 * which is a fork and join rather than a queue, so two conversations stepping
 * at once would be two callers inside it — the sessions are independent, the
 * pool underneath them is not. */
#define MAIN_TALK_LIMIT 16
#define MAIN_LINE_LIMIT 8192

typedef struct main_talk {
  app_session *session;
  /* The conversation's own proposer, opened the first time `--guess` needs it.
   * It belongs to the conversation rather than to the turn, and that is the
   * whole reason it is here: a follow-up question about the same document
   * quotes both the document and the answer before it, which is where prompt
   * lookup is at its best and what a per-turn scout would throw away. */
  app_scout   *scout;
  int          turn_count;
  int          open_flag;  /* whether the model has already answered in it */
  /* What `/image` and `/audio` have put in front of the next turn, and the room
   * their paths are kept in, because the line they were typed on is read over. */
  main_show    show_list[MAIN_MEDIA_LIMIT];
  int          show_count;
  char         path_room[MAIN_MEDIA_LIMIT][1024];
} main_talk;

/* One line of standard input, with the newline dropped.  Returns 0 at the end
 * of the input, which is what ends the loop. */
static int main_line_read(char *line_out, int line_limit) {
  int fill_count = 0;
  int letter;
  for (;;) {
    letter = fgetc(stdin);
    if (letter == EOF) return fill_count > 0;
    if (letter == '\n') break;
    if (fill_count + 1 < line_limit) line_out[fill_count++] = (char)letter;
  }
  line_out[fill_count] = '\0';
  return 1;
}

/* Whether a line is only spaces, which is not a turn. */
static int main_line_bare(const char *line_text) {
  while (*line_text) {
    if (*line_text != ' ' && *line_text != '\t' && *line_text != '\r') return 0;
    ++line_text;
  }
  return 1;
}

static void main_talk_help(void) {
  printf("  /image <path> put a picture in front of the next turn\n");
  printf("  /audio <path> put a clip in front of the next turn\n");
  printf("  /new          start another conversation on the same model\n");
  printf("  /talk <n>     switch to conversation n\n");
  printf("  /list         the conversations, their turns and what they hold\n");
  printf("  /drop         close the current conversation\n");
  printf("  /save <path>  write this conversation out\n");
  printf("  /open <path>  read one back into this conversation\n");
  printf("  /quit         leave\n");
  printf("  anything else is a turn in the current conversation\n");
}

/* One turn of a conversation: the pieces the caller has put in front of it and
 * the words they typed, framed, primed and answered.  A turn that is not the
 * first picks the frame up where the last one stopped. */
static void main_talk_turn(app_model *model, const main_flag *flag, main_talk *talk,
                           const char *word_text) {
  main_flag turn_flag = *flag;
  main_reel reel;
  app_code code;
  const float *state_data;

  turn_flag.prompt_text = word_text;
  turn_flag.prompt_flag = 1;
  turn_flag.text_count = 0;
  turn_flag.show_count = talk->show_count;
  memcpy(turn_flag.show_list, talk->show_list, sizeof(turn_flag.show_list));
  if (!main_reel_build(model, &turn_flag, &reel, !talk->open_flag)) return;

  /* A conversation that has filled its window is refused before a single id is
   * consumed, so saying so and staying open is safe: the conversation is
   * exactly where it was, and `/new` or `/drop` is the way on. */
  code = session_prime_media(talk->session, reel.id_list, reel.id_count, reel.state_list,
                             reel.state_flag);
  if (code != APP_OKAY) {
    fprintf(stderr, "prime: %s\n", app_code_text(code));
    main_reel_free(&reel);
    return;
  }
  state_data = reel.state_flag[reel.id_count - 1]
                   ? reel.state_list + (size_t)(reel.id_count - 1) * (size_t)reel.state_size
                   : NULL;
  if (flag->guess_span > 1 && !state_data &&
      (talk->scout || scout_open(&talk->scout) == APP_OKAY)) {
    main_bet bet;
    memset(&bet, 0, sizeof(bet));
    main_answer_guess(model, flag, talk->session, talk->scout, &reel, 0, &bet);
    if (flag->verbose_level && bet.round_count > 0)
      fprintf(stderr, "\nguess   %.2f tokens a round over %ld rounds, %.0f%% of %ld kept\n",
              (double)bet.token_count / (double)bet.round_count, bet.round_count,
              bet.draw_count > 0 ? 100.0 * (double)bet.take_count / (double)bet.draw_count : 0.0,
              bet.draw_count);
  } else {
    main_answer(model, flag, talk->session, reel.id_list[reel.id_count - 1], state_data, 0);
  }
  printf("\n");
  fflush(stdout);
  main_reel_free(&reel);
  talk->turn_count += 1;
  talk->open_flag = 1;
  talk->show_count = 0;
}

static int main_loop(app_model *model, const main_flag *flag) {
  main_talk talk_list[MAIN_TALK_LIMIT];
  char line_text[MAIN_LINE_LIMIT];
  int talk_slot = 0, talk_index;

  /* The same refusal the single turn path makes, and for the same reason:
   * verification is an argmax comparison and there is no rejection rule here
   * for a temperature to be sampled under. */
  if (flag->guess_span > 1 && flag->taste.heat_value > 0.0f) {
    fprintf(stderr, "--guess is greedy only; use --heat 0\n");
    return 1;
  }
  memset(talk_list, 0, sizeof(talk_list));
  if (session_open(model, &talk_list[0].session) != APP_OKAY) {
    fprintf(stderr, "session: %s\n", app_code_text(APP_FAIL_MEMORY));
    return 1;
  }

  /* Whatever the command line carried — words, pictures, clips — is the first
   * turn, so a loop is the single turn task with more turns after it. */
  if (flag->prompt_flag || flag->show_count > 0) {
    talk_list[0].show_count = flag->show_count;
    memcpy(talk_list[0].show_list, flag->show_list, sizeof(talk_list[0].show_list));
    main_talk_turn(model, flag, &talk_list[0], flag->prompt_text);
  } else {
    printf("%s %s - a conversation on a loaded model; /help for the rest\n", APP_NAME,
           APP_VERSION);
  }

  for (;;) {
    printf("%d> ", talk_slot + 1);
    fflush(stdout);
    if (!main_line_read(line_text, (int)sizeof(line_text))) break;
    if (main_line_bare(line_text)) continue;

    /* A line that opens with a slash is an instruction to the loop; everything
     * else is a turn. */
    if (line_text[0] == '/') {
      if (strcmp(line_text, "/quit") == 0 || strcmp(line_text, "/exit") == 0) break;
      if (strcmp(line_text, "/help") == 0) {
        main_talk_help();
        continue;
      }
      if (strcmp(line_text, "/new") == 0) {
        int free_slot = -1;
        for (talk_index = 0; talk_index < MAIN_TALK_LIMIT; ++talk_index)
          if (!talk_list[talk_index].session) { free_slot = talk_index; break; }
        if (free_slot < 0) {
          printf("at most %d conversations at once\n", MAIN_TALK_LIMIT);
          continue;
        }
        /* Every conversation carries a cache of its own at the window's full
         * span, which is what `/list` reports and what `--window` sizes. */
        if (session_open(model, &talk_list[free_slot].session) != APP_OKAY) {
          printf("no room for another conversation; try a smaller --window\n");
          continue;
        }
        talk_slot = free_slot;
        printf("conversation %d\n", talk_slot + 1);
        continue;
      }
      if (strncmp(line_text, "/talk ", 6) == 0) {
        int want_slot = atoi(line_text + 6) - 1;
        if (want_slot < 0 || want_slot >= MAIN_TALK_LIMIT || !talk_list[want_slot].session) {
          printf("no conversation %d\n", want_slot + 1);
          continue;
        }
        talk_slot = want_slot;
        continue;
      }
      if (strcmp(line_text, "/list") == 0) {
        for (talk_index = 0; talk_index < MAIN_TALK_LIMIT; ++talk_index) {
          app_session *other = talk_list[talk_index].session;
          if (!other) continue;
          printf("%s%d  %d turns, %d ids, %.1f MiB of cache\n",
                 talk_index == talk_slot ? "* " : "  ", talk_index + 1,
                 talk_list[talk_index].turn_count, session_fill(other),
                 (double)session_cache_room(other) / (1024.0 * 1024.0));
        }
        continue;
      }
      if (strcmp(line_text, "/drop") == 0) {
        int next_slot = -1;
        session_close(talk_list[talk_slot].session);
        scout_close(talk_list[talk_slot].scout);
        memset(&talk_list[talk_slot], 0, sizeof(talk_list[talk_slot]));
        for (talk_index = 0; talk_index < MAIN_TALK_LIMIT; ++talk_index)
          if (talk_list[talk_index].session) { next_slot = talk_index; break; }
        if (next_slot < 0) break; /* the last one closed is the end of the loop */
        talk_slot = next_slot;
        continue;
      }
      /* A conversation written out and read back.  What `--keep` holds is a
       * prompt, written before its first token is sampled so that a rerun
       * starts where the last run started; what these two hold is a
       * conversation, written after an answer so that the next turn is framed
       * onto it.  The file says which of the two it is and the other is
       * refused, because the cache alone cannot tell them apart and reading one
       * for the other drops an id or repeats a turn.
       *
       * The turns a conversation holds ride in the stamp the engine hands back
       * unread: a conversation restored whole has nothing to match a stamp
       * against, and whether the model has already answered is what decides
       * how the next turn is framed. */
      if (strncmp(line_text, "/save ", 6) == 0) {
        main_talk *talk = &talk_list[talk_slot];
        app_code keep_code =
            session_save(talk->session, line_text + 6, (uint64_t)talk->turn_count, APP_KEEP_TALK);
        if (keep_code != APP_OKAY)
          printf("save: %s\n", app_code_text(keep_code));
        else
          printf("conversation %d written, %d turns, %d ids\n", talk_slot + 1, talk->turn_count,
                 session_fill(talk->session));
        continue;
      }
      if (strncmp(line_text, "/open ", 6) == 0) {
        main_talk *talk = &talk_list[talk_slot];
        uint64_t held_stamp = 0;
        int held_kind = APP_KEEP_PROMPT;
        app_code keep_code = session_load(talk->session, line_text + 6, &held_stamp, &held_kind);
        if (keep_code != APP_OKAY) {
          printf("open: %s\n", app_code_text(keep_code));
          continue;
        }
        if (held_kind != APP_KEEP_TALK) {
          /* A prompt read here would leave the session one id short of what it
           * says it holds, so it is put back rather than continued. */
          session_reset(talk->session);
          printf("open: that file holds a prompt rather than a conversation\n");
          continue;
        }
        talk->turn_count = (int)held_stamp;
        talk->open_flag = talk->turn_count > 0;
        talk->show_count = 0;
        /* The scout's stream is the one this conversation was in, and the file
         * has just replaced it.  It is forgotten rather than carried, because a
         * proposer reading a conversation it is no longer in would be wrong
         * without being caught — every guess is verified, so it would cost
         * lanes rather than correctness, which is exactly the failure a test
         * would not find. */
        scout_clear(talk->scout);
        printf("conversation %d restored, %d turns, %d ids\n", talk_slot + 1, talk->turn_count,
               session_fill(talk->session));
        continue;
      }
      if (strncmp(line_text, "/image ", 7) == 0 || strncmp(line_text, "/audio ", 7) == 0) {
        main_talk *talk = &talk_list[talk_slot];
        if (talk->show_count >= MAIN_MEDIA_LIMIT) {
          printf("at most %d pieces in one turn\n", MAIN_MEDIA_LIMIT);
          continue;
        }
        /* The path is kept rather than copied, so it has to outlive the turn:
         * the line it points into is not read again until the turn is done. */
        talk->show_list[talk->show_count].kind_mark =
            line_text[1] == 'i' ? MAIN_SHOW_IMAGE : MAIN_SHOW_AUDIO;
        talk->show_list[talk->show_count].path_text = talk->path_room[talk->show_count];
        text_fill(talk->path_room[talk->show_count], sizeof(talk->path_room[0]), line_text + 7);
        talk->show_count += 1;
        printf("%d in front of the next turn\n", talk->show_count);
        continue;
      }
      printf("unknown: %s, /help for what there is\n", line_text);
      continue;
    }

    main_talk_turn(model, flag, &talk_list[talk_slot], line_text);
  }

  for (talk_index = 0; talk_index < MAIN_TALK_LIMIT; ++talk_index) {
    if (talk_list[talk_index].session) session_close(talk_list[talk_index].session);
    scout_close(talk_list[talk_index].scout);
  }
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
  if (!main_reel_build(model, flag, &reel, 1)) {
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
  if (!main_reel_build(model, flag, &reel, 1)) return 1;
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

/* What the export's calibrated cache ranges hold, judged against a prompt.
 *
 * The scales are per tensor and static, so the only question that decides
 * whether a quantized cache is safe is how the range compares with what the
 * cache is actually asked to carry: a peak above the range clips, and a peak
 * far below it spends levels on room nothing uses.  Both are printed a layer,
 * with the fill the ratio comes to. */
/* -- what a block of guesses would be worth ------------------------------- */

/* Speculative decoding is a bet: propose a short block, check the whole block
 * in one pass, and keep the guesses the model agrees with.  Whether it pays
 * turns on two numbers that multiply — what a block costs against the tokens it
 * commits, and how often a proposer is right — and only the first of them is a
 * property of this engine.
 *
 * So this task measures the first alone and brackets the second rather than
 * guessing at it.  Two proposers that could never be written for real work are
 * exactly what is wanted: one that is always right, which is the most any
 * proposer could ever be worth, and one that is always wrong, which is what a
 * proposer costs when it never helps.  Every real proposer lands between them.
 *
 * Both runs are held to the tokens a plain greedy decode produced, so this
 * checks the block path as much as it measures it: a block that verified
 * wrongly, or an undo that left a guess behind in the cache, shows up as a
 * stream that does not match. */

/* Guesses the model agreed with, as a share of guesses drawn.  A proposer that
 * drew none — which is the null row's opposite and the scout's early rounds —
 * has no share to report rather than a share of zero. */
static const char *main_guess_share(long take_count, long draw_count) {
  static char text_room[16];
  if (draw_count < 1) return "-";
  snprintf(text_room, sizeof(text_room), "%.0f%%",
           100.0 * (double)take_count / (double)draw_count);
  return text_room;
}

/* Which proposer a run is measured with.  The first two are cheats and could
 * never be written for real work; the third is the one that ships. */
#define MAIN_GUESS_NULL   0
#define MAIN_GUESS_ORACLE 1
#define MAIN_GUESS_SCOUT  2

static const char *main_guess_name(int mode_value) {
  if (mode_value == MAIN_GUESS_ORACLE) return "oracle";
  if (mode_value == MAIN_GUESS_SCOUT) return "n-gram";
  return "null";
}

/* One speculative run.  `mode_value` picks the proposer: the true continuation,
 * an id that is deliberately not it, or the n-gram scout, which is the only one
 * of the three a caller could actually have.
 *
 * The scout is told the prompt before the clock starts and every committed
 * token after that, and it is never told a guess — so what it proposes is only
 * ever built out of what the model has agreed to.  Where it has nothing to say
 * the round is a block of one lane, which is what a caller without a proposal
 * pays and is a little more than a plain step. */
static double main_guess_run(app_session *session, main_reel *reel, int block_span,
                             const int32_t *true_list, int serve_count, int mode_value,
                             int32_t *made_list, int vocab_count, long *round_out,
                             long *commit_out, long *draw_out) {
  int32_t block_list[16];
  int32_t last_id = reel->id_list[reel->id_count - 1];
  int made_count = 0;
  long round_count = 0, commit_count = 0, draw_count = 0;
  app_scout *scout = NULL;
  double from_time;

  session_reset(session);
  if (session_prime_media(session, reel->id_list, reel->id_count, reel->state_list,
                          reel->state_flag) != APP_OKAY)
    return -1.0;
  if (mode_value == MAIN_GUESS_SCOUT) {
    if (scout_open(&scout) != APP_OKAY) return -1.0;
    if (scout_note(scout, reel->id_list, reel->id_count) != APP_OKAY) {
      scout_close(scout);
      return -1.0;
    }
  }
  from_time = time_now();
  while (made_count < serve_count) {
    const float *rows;
    int span_count = block_span;
    int slot, take_count = 0;
    if (span_count > serve_count - made_count + 1) span_count = serve_count - made_count + 1;
    if (span_count < 1) span_count = 1;
    block_list[0] = last_id;
    if (mode_value == MAIN_GUESS_SCOUT) {
      span_count = 1 + scout_draw(scout, block_list + 1, span_count - 1);
    } else {
      for (slot = 1; slot < span_count; ++slot) {
        /* The oracle proposes what the plain run produced; the other proposes
         * something that is not it, so the first guess of every block fails and
         * the block commits exactly the one token it was always going to. */
        int32_t want_id =
            made_count + slot - 1 < serve_count ? true_list[made_count + slot - 1] : 0;
        block_list[slot] =
            mode_value == MAIN_GUESS_ORACLE ? want_id : (int32_t)((want_id + 1) % vocab_count);
      }
    }
    draw_count += span_count - 1;
    if (span_count == 1 && mode_value == MAIN_GUESS_SCOUT) {
      /* What the shipped path does with nothing to verify: an ordinary step,
       * which is cheaper than a block of one lane.  The oracle and the null
       * always draw, so only this row is affected and only where the scout has
       * nothing — which is what a caller would actually pay. */
      const float *logit_list = session_step(session, last_id);
      if (!logit_list) { scout_close(scout); return -1.0; }
      round_count += 1;
      last_id = main_guess_top(logit_list, vocab_count);
      made_list[made_count++] = last_id;
      commit_count += 1;
      if (scout_note(scout, &last_id, 1) != APP_OKAY) { scout_close(scout); return -1.0; }
      continue;
    }
    rows = session_guess(session, block_list, span_count);
    if (!rows) { scout_close(scout); return -1.0; }
    round_count += 1;
    /* The leading guesses the model agrees with, and then the token after the
     * last of those, which is true whatever the guesses were. */
    while (take_count + 1 < span_count &&
           main_guess_top(rows + (size_t)take_count * (size_t)vocab_count, vocab_count) ==
               block_list[take_count + 1])
      take_count += 1;
    for (slot = 0; slot <= take_count && made_count < serve_count; ++slot) {
      int32_t made_id = slot < take_count
                            ? block_list[slot + 1]
                            : main_guess_top(rows + (size_t)slot * (size_t)vocab_count, vocab_count);
      made_list[made_count++] = made_id;
      last_id = made_id;
      commit_count += 1;
      if (scout && scout_note(scout, &made_id, 1) != APP_OKAY) { scout_close(scout); return -1.0; }
    }
    if (session_guess_keep(session, take_count + 1) != APP_OKAY) {
      scout_close(scout);
      return -1.0;
    }
  }
  *round_out = round_count;
  *commit_out = commit_count;
  *draw_out = draw_count;
  scout_close(scout);
  return time_now() - from_time;
}

/* Where a speculative round goes, and what a lane of it costs.
 *
 * `main_phases` divides a decode step; this divides a *round*, which is the
 * same graph several lanes wide, and it divides two of them — the narrowest
 * block and the widest — so the difference names the marginal lane part by
 * part.  That is the number `TODO.md`'s speculative entry is now about: a block
 * shares the weight sweep across its lanes and cannot share the arithmetic, and
 * what an extra lane costs is what holds the ceiling down.
 *
 * Both books come from the oracle, because it commits every lane it is given
 * and so runs full width every round.  A proposer that draws nothing takes an
 * ordinary step, and its rounds would be a mixture of two shapes. */
static void main_guess_phases(const app_phase_book *thin_book, int thin_span,
                              const app_phase_book *wide_book, int wide_span) {
  int phase_slot, order_list[APP_PHASE_COUNT], order_index, order_scan;
  double thin_total = 0.0, wide_total = 0.0;
  double lane_span = (double)(wide_span - thin_span);

  if (!thin_book->step_count || !wide_book->step_count || lane_span <= 0.0) return;
  for (phase_slot = 0; phase_slot < APP_PHASE_COUNT; ++phase_slot) {
    thin_total += thin_book->seconds[phase_slot];
    wide_total += wide_book->seconds[phase_slot];
  }
  if (wide_total <= 0.0) return;

  /* Ordered by what the lane costs rather than by what the round costs, because
   * the lane is the question. */
  for (phase_slot = 0; phase_slot < APP_PHASE_COUNT; ++phase_slot)
    order_list[phase_slot] = phase_slot;
  for (order_index = 1; order_index < APP_PHASE_COUNT; ++order_index) {
    int keep_slot = order_list[order_index];
    double keep_gap = wide_book->seconds[keep_slot] / (double)wide_book->step_count -
                      thin_book->seconds[keep_slot] / (double)thin_book->step_count;
    for (order_scan = order_index; order_scan > 0; --order_scan) {
      int other = order_list[order_scan - 1];
      double other_gap = wide_book->seconds[other] / (double)wide_book->step_count -
                         thin_book->seconds[other] / (double)thin_book->step_count;
      if (other_gap >= keep_gap) break;
      order_list[order_scan] = other;
    }
    order_list[order_scan] = keep_slot;
  }

  printf("\nround   a block of %d against a block of %d, both with the oracle\n", thin_span,
         wide_span);
  {
    char thin_text[24], wide_text[24];
    sprintf(thin_text, "ms at %d", thin_span);
    sprintf(wide_text, "ms at %d", wide_span);
    printf("%-24s %11s %11s %13s %8s\n", "part", thin_text, wide_text, "ms a lane", "share");
  }
  for (order_index = 0; order_index < APP_PHASE_COUNT; ++order_index) {
    double thin_each, wide_each, lane_each;
    phase_slot = order_list[order_index];
    if (!wide_book->counts[phase_slot] && !thin_book->counts[phase_slot]) continue;
    thin_each = thin_book->seconds[phase_slot] / (double)thin_book->step_count;
    wide_each = wide_book->seconds[phase_slot] / (double)wide_book->step_count;
    lane_each = (wide_each - thin_each) / lane_span;
    printf("%-24s %11.3f %11.3f %13.3f %7.1f%%\n", app_phase_text(phase_slot), thin_each * 1000.0,
           wide_each * 1000.0, lane_each * 1000.0,
           100.0 * (wide_each - thin_each) / (wide_total / (double)wide_book->step_count -
                                              thin_total / (double)thin_book->step_count));
  }
  printf("%-24s %11.3f %11.3f %13.3f %7.1f%%\n", "named, in all",
         thin_total / (double)thin_book->step_count * 1000.0,
         wide_total / (double)wide_book->step_count * 1000.0,
         (wide_total / (double)wide_book->step_count -
          thin_total / (double)thin_book->step_count) / lane_span * 1000.0,
         100.0);
  printf("a lane is what a block cannot share: the weights are swept once a round however\n");
  printf("wide it is, so every millisecond above is arithmetic, cache or staging that each\n");
  printf("lane pays for itself.  Halve it and the ceiling in the table above doubles.\n");
}

static int main_guess(app_model *model, const main_flag *flag) {
  static const int span_list[4] = {2, 4, 8, 16};
  app_session *session = NULL;
  main_reel reel;
  int vocab_count = model_vocab_count(model);
  int serve_count = flag->serve_limit > 0 ? flag->serve_limit : 32;
  int span_slot, slot;
  int32_t *true_list;
  int32_t *made_list;
  double plain_seconds;
  int32_t last_id;
  app_phase_book thin_book, wide_book;
  int thin_span = 0, wide_span = 0;

  if (session_open(model, &session) != APP_OKAY || !session) {
    fprintf(stderr, "session failed\n");
    return 1;
  }
  if (!main_reel_build(model, flag, &reel, 1)) { session_close(session); return 1; }
  true_list = (int32_t *)calloc((size_t)serve_count, sizeof(int32_t));
  made_list = (int32_t *)calloc((size_t)serve_count, sizeof(int32_t));
  if (!true_list || !made_list) {
    free(true_list); free(made_list); main_reel_free(&reel); session_close(session);
    return 1;
  }

  /* The plain run, which is both the baseline and the answer the two cheats are
   * held to. */
  if (session_prime_media(session, reel.id_list, reel.id_count, reel.state_list,
                          reel.state_flag) != APP_OKAY) {
    fprintf(stderr, "prime failed\n");
    free(true_list); free(made_list); main_reel_free(&reel); session_close(session);
    return 1;
  }
  last_id = reel.id_list[reel.id_count - 1];
  plain_seconds = time_now();
  for (slot = 0; slot < serve_count; ++slot) {
    const float *logit_list = session_step(session, last_id);
    if (!logit_list) { fprintf(stderr, "step failed\n"); break; }
    last_id = main_guess_top(logit_list, vocab_count);
    true_list[slot] = last_id;
  }
  plain_seconds = time_now() - plain_seconds;

  printf("prompt  %d ids, %d tokens greedily\n", reel.id_count, serve_count);
  printf("plain   %.2f tok/s, %.2f ms a token\n", (double)serve_count / plain_seconds,
         plain_seconds / (double)serve_count * 1000.0);
  printf("block   at most %d lanes\n\n", session_guess_limit(session));
  printf("%-6s %-8s %8s %11s %10s %8s %9s  %s\n", "block", "proposer", "tok/s", "ms a round",
         "committed", "of drawn", "vs plain", "stream");
  thin_book = session_phases_block(session);  /* zeroed until an oracle row fills one */
  wide_book = thin_book;
  for (span_slot = 0; span_slot < 4; ++span_slot) {
    int block_span = span_list[span_slot];
    int mode_slot;
    /* The ceiling, then the one that ships, then the floor: a reader compares
     * the middle row with the two it sits between. */
    static const int mode_list[3] = {MAIN_GUESS_ORACLE, MAIN_GUESS_SCOUT, MAIN_GUESS_NULL};
    if (block_span > session_guess_limit(session)) continue;
    for (mode_slot = 0; mode_slot < 3; ++mode_slot) {
      int mode_value = mode_list[mode_slot];
      long round_count = 0, commit_count = 0, draw_count = 0;
      double seconds;
      int same_flag = 1;
      if (mode_value == MAIN_GUESS_ORACLE) session_phase_clear(session);
      seconds = main_guess_run(session, &reel, block_span, true_list, serve_count, mode_value,
                               made_list, vocab_count, &round_count, &commit_count, &draw_count);
      if (mode_value == MAIN_GUESS_ORACLE) {
        if (!thin_span) { thin_book = session_phases_block(session); thin_span = block_span; }
        wide_book = session_phases_block(session);
        wide_span = block_span;
      }
      if (seconds < 0.0 || round_count < 1) {
        printf("%-6d %-8s   run failed\n", block_span, main_guess_name(mode_value));
        continue;
      }
      for (slot = 0; slot < serve_count; ++slot)
        if (made_list[slot] != true_list[slot]) { same_flag = 0; break; }
      printf("%-6d %-8s %8.2f %11.2f %10.2f %8s %8.2fx  %s\n", block_span,
             main_guess_name(mode_value), (double)serve_count / seconds,
             seconds / (double)round_count * 1000.0,
             (double)commit_count / (double)round_count,
             main_guess_share(commit_count - round_count, draw_count), plain_seconds / seconds,
             same_flag ? "matches plain" : "DIVERGES");
    }
  }
  if (flag->verbose_level) main_guess_phases(&thin_book, thin_span, &wide_book, wide_span);
  printf("\noracle is the ceiling of any proposer and null is its floor; n-gram is the one\n");
  printf("that ships, reads only the stream it has already seen, and takes a plain step\n");
  printf("where it has nothing to propose, exactly as --guess does.  A proposer pays where\n");
  printf("its accepted guesses a round carry the round's cost past a plain step.\n");
  session_close(session);
  free(true_list);
  free(made_list);
  main_reel_free(&reel);
  return 0;
}

static int main_cache(app_model *model, const main_flag *flag) {
  app_session *session = NULL;
  main_reel reel;
  int layer_count = model_layer_count(model);
  int layer_index;
  app_code code = session_open(model, &session);

  if (code != APP_OKAY) {
    fprintf(stderr, "session: %s\n", app_code_text(code));
    return 1;
  }
  if (!main_reel_build(model, flag, &reel, 1)) {
    session_close(session);
    return 1;
  }
  if (session_prime_media(session, reel.id_list, reel.id_count, reel.state_list, reel.state_flag) !=
      APP_OKAY) {
    fprintf(stderr, "prime failed\n");
    main_reel_free(&reel);
    session_close(session);
    return 1;
  }
  session_step_state(session, reel.id_list[reel.id_count - 1],
                     reel.state_flag[reel.id_count - 1]
                         ? reel.state_list +
                               (size_t)(reel.id_count - 1) * (size_t)reel.state_size
                         : NULL);

  {
    int byte_flag = flag->cache_bits == 8;
    double room = (double)session_cache_room(session) / (1024.0 * 1024.0);
    double other = (double)session_cache_room_at(session, byte_flag ? 0 : 8) / (1024.0 * 1024.0);
    printf("prompt   %d ids\n", reel.id_count);
    printf("room     %.1f MiB of cache at full span, held as %s (%.1f MiB as %s)\n", room,
           byte_flag ? "bytes" : "floats", other, byte_flag ? "floats" : "bytes");
  }
  printf("cache    %s\n\n", flag->cache_bits == 8 ? "quantized through the scales" : "float");
  printf("%5s  %12s %10s %6s   %12s %10s %6s\n", "layer", "k range", "k peak", "fill",
         "v range", "v peak", "fill");
  for (layer_index = 0; layer_index < layer_count; ++layer_index) {
    float key_scale = model_cache_scale(model, layer_index, 0);
    float value_scale = model_cache_scale(model, layer_index, 1);
    float key_peak = session_cache_peak(session, layer_index, 0);
    float value_peak = session_cache_peak(session, layer_index, 1);
    float key_span = key_scale * CACHE_FP8_TOP;
    float value_span = value_scale * CACHE_FP8_TOP;
    printf("%5d  %12.6f %10.4f ", layer_index, (double)key_span, (double)key_peak);
    if (key_span > 0.0f) printf("%5.1f%%", (double)(100.0f * key_peak / key_span));
    else printf("%6s", "-");
    printf("   %12.6f %10.4f ", (double)value_span, (double)value_peak);
    if (value_span > 0.0f) printf("%5.1f%%", (double)(100.0f * value_peak / value_span));
    else printf("%6s", "-");
    printf("\n");
  }
  main_reel_free(&reel);
  session_close(session);
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
  /* A checkpoint that records no budget is not one that allows no clip: it
   * is one the engine reads all of, so say that rather than print a zero. */
  if (model_audio_ready(model) && model_audio_rows(model) > 0)
    printf("  rows   %d at %d ms, %.1f s of clip, placeholder id %d\n", model_audio_rows(model),
           model_audio_span_ms(model),
           (double)model_audio_rows(model) * (double)model_audio_span_ms(model) / 1000.0,
           model_audio_token(model));
  else if (model_audio_ready(model))
    printf("  rows   no budget recorded, placeholder id %d\n", model_audio_token(model));
  printf("weights  %.1f MiB mapped\n", (double)model_memory_bytes(model) / (1024.0 * 1024.0));
  /* What a token costs is not what the file weighs: the embedding tables are
   * read a row at a time and the towers are not in the token loop at all. */
  printf("decode   %.1f MiB a token\n", (double)model_decode_bytes(model) / (1024.0 * 1024.0));
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
  setup.cache_bits = flag.cache_bits;

  code = model_load(flag.model_path, &setup, &model);
  if (code != APP_OKAY) {
    fprintf(stderr, "load: %s\n", app_code_text(code));
    return 1;
  }

  /* The picture budget is the model's rather than a turn's, so it is set once
   * here and every task below reads its pictures under it.  A checkpoint with
   * no vision tower cannot honour it, and a caller that asked is told so rather
   * than given a run that quietly ignored the flag. */
  if (flag.image_rows > 0) {
    code = model_image_budget(model, flag.image_rows);
    if (code != APP_OKAY) {
      fprintf(stderr, "image-tokens: %s\n", app_code_text(code));
      model_free(model);
      return 1;
    }
  }

  /* Pictures kept from an earlier run.  A file that is not there is the first
   * run and is not worth a word; one that is there and is refused is worth
   * saying, because it means the tower is about to run over pictures the caller
   * thought were already encoded — but it is not a failure, since running the
   * tower is exactly what the engine would have done without the flag. */
  if (flag.image_keep) {
    code = media_store_load(model, flag.image_keep);
    if (code != APP_OKAY && code != APP_FAIL_MISSING)
      fprintf(stderr, "image-keep: %s, so the pictures are encoded again (%s)\n",
              app_code_text(code), flag.image_keep);
  }

  if (strcmp(flag.task_text, "chat") == 0 && flag.loop_flag)
    result_code = main_loop(model, &flag);
  else if (strcmp(flag.task_text, "chat") == 0 || strcmp(flag.task_text, "complete") == 0)
    result_code = main_serve(model, &flag, 0);
  else if (strcmp(flag.task_text, "bench") == 0)
    result_code = main_serve(model, &flag, 1);
  else if (strcmp(flag.task_text, "logits") == 0)
    result_code = main_logits(model, &flag);
  else if (strcmp(flag.task_text, "tokens") == 0)
    result_code = main_tokens(model, &flag);
  else if (strcmp(flag.task_text, "probe") == 0)
    result_code = main_probe(model);
  else if (strcmp(flag.task_text, "cache") == 0)
    result_code = main_cache(model, &flag);
  else if (strcmp(flag.task_text, "guess") == 0)
    result_code = main_guess(model, &flag);
  else {
    main_usage();
    result_code = 1;
  }

  /* And back out again, whatever the task made of them.  Writing is worth a
   * word when it fails: the caller asked for a file and there will not be one,
   * so the next run pays the tower again without knowing why. */
  if (flag.image_keep && model_vision_ready(model)) {
    code = media_store_save(model, flag.image_keep);
    if (code != APP_OKAY)
      fprintf(stderr, "image-keep: %s, so nothing is held for the next run (%s)\n",
              app_code_text(code), flag.image_keep);
  }

  model_free(model);
  return result_code;
}

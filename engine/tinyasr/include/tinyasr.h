/* tinyasr: on-device speech recognition (streaming with pause / end-of-turn detection, or a whole file).
 * Audio: mono float, 16 kHz, [-1, 1]. Not thread-safe per object; models can serve several sessions.
 * Returned strings are UTF-8 JSON owned by the object, valid until its next call.
 * Model files are checked when loaded: a corrupted file, or one for another version, makes tinyasr_load return NULL
 * with the reason; it never crashes. Every function accepts NULL for its object and then does nothing: it returns
 * NULL (pointers), 0 (tinyasr_chunk_ms) or -1 (tinyasr_file_progress). So a failed load needs no special path. */
#ifndef TINYASR_H
#define TINYASR_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct tinyasr_models tinyasr_models;
typedef struct tinyasr_session tinyasr_session;
typedef struct tinyasr_file tinyasr_file;

/* Version of the library (its API and the model files it reads): 2. */
uint32_t tinyasr_version(void);

/* Model files from memory (copied): the recognizer (.tzf, int4, uncompressed; with its end-of-turn head, else a
 * sentence ends after 1.6 s of silence) and silero_vad.onnx. NULL on failure, with the reason in err. */
tinyasr_models *tinyasr_load(const uint8_t *asr, size_t asr_len, const uint8_t *vad, size_t vad_len, char *err,
                             size_t err_len);
void tinyasr_free(tinyasr_models *m);
uint32_t tinyasr_chunk_ms(const tinyasr_models *m); /* 0 for NULL */

/* Live audio. langs: the languages to recognize in, comma-separated ("en", "en,ru"; NULL or "": all), e.g. from a
 * language ID the host runs. One fixes the language; with several every full-context pass runs in each of them and
 * the surest wins (that many times the cost). Starts from start_lang (may be NULL).
 * clock: monotonic seconds, or NULL. beam: beam width of the full-context passes (0: the default, 4; 1: greedy,
 * fastest; streaming text is always greedy). push / finish return a JSON array of events:
 *   {"type":"commit","words":[{"text":"Hello,","start":0.52,"end":0.9}, ...]}   final words, appended
 *   {"type":"partial","open":[...],"tail":[...]}   words not final yet (replace the previous partial)
 *   {"type":"language","lang":"uk"}      several languages: the recognizer switched to this one
 *   {"type":"eou","at":3.1,"p":0.93}   the end-of-turn head at a pause (> 0.5: the sentence ends)
 */
/* NULL if m is NULL. */
tinyasr_session *tinyasr_session_new(const tinyasr_models *m, const char *langs, const char *start_lang,
                                     double (*clock)(void), uint32_t beam);
const char *tinyasr_session_push(tinyasr_session *s, const float *pcm, size_t n);
const char *tinyasr_session_finish(tinyasr_session *s);
const char *tinyasr_session_lang(const tinyasr_session *s);
void tinyasr_session_free(tinyasr_session *s);

/* A recorded file (copied; langs and beam as for sessions): only the full-context pass, piece by piece. next returns the next piece's final
 * words as a JSON array of {"text","start","end"} (seconds), or NULL at the end. NULL if m is NULL. */
tinyasr_file *tinyasr_file_new(const tinyasr_models *m, const float *pcm, size_t n, const char *langs, uint32_t beam);
const char *tinyasr_file_lang(const tinyasr_file *f);
const char *tinyasr_file_next(tinyasr_file *f);
float tinyasr_file_progress(const tinyasr_file *f); /* 0..1; -1 for NULL */
void tinyasr_file_free(tinyasr_file *f);

#ifdef __cplusplus
}
#endif
#endif

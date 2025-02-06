import numpy as np


def log_sum_exp(a, b):
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    m = max(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def ctc_beam_search_decoder_single(log_probs, beam_width=10, blank=0):
    T, nclass = log_probs.shape
    beam = {}
    beam[()] = (log_probs[0, blank], float("-inf"))
    for s in range(nclass):
        if s == blank:
            continue
        beam[(s,)] = (float("-inf"), log_probs[0, s])
    for t in range(1, T):
        new_beam = {}
        for prefix, (p_b, p_nb) in beam.items():
            total = log_sum_exp(p_b, p_nb)
            new_score = total + log_probs[t, blank]
            if prefix in new_beam:
                old_p_b, old_p_nb = new_beam[prefix]
                new_beam[prefix] = (log_sum_exp(old_p_b, new_score), old_p_nb)
            else:
                new_beam[prefix] = (new_score, float("-inf"))
            for s in range(nclass):
                if s == blank:
                    continue
                new_prefix = prefix + (s,)
                if len(prefix) > 0 and prefix[-1] == s:
                    new_score = p_b + log_probs[t, s]
                    if prefix in new_beam:
                        old_p_b, old_p_nb = new_beam[prefix]
                        new_beam[prefix] = (old_p_b, log_sum_exp(old_p_nb, new_score))
                    else:
                        new_beam[prefix] = (float("-inf"), new_score)
                else:
                    new_score = total + log_probs[t, s]
                    if new_prefix in new_beam:
                        old_p_b, old_p_nb = new_beam[new_prefix]
                        new_beam[new_prefix] = (old_p_b, log_sum_exp(old_p_nb, new_score))
                    else:
                        new_beam[new_prefix] = (float("-inf"), new_score)
        beam_items = list(new_beam.items())
        beam_items.sort(key=lambda x: log_sum_exp(x[1][0], x[1][1]), reverse=True)
        beam = dict(beam_items[:beam_width])
    best_prefix = max(beam.items(), key=lambda x: log_sum_exp(x[1][0], x[1][1]))[0]
    decoded = []
    prev = None
    for s in best_prefix:
        if s != prev:
            decoded.append(s)
        prev = s
    return decoded


def decode_predictions_beam(preds, idx_to_char, beam_width=10, blank=0):
    preds = preds.cpu().detach().numpy()  # (T, batch, nclass)
    T, batch_size, nclass = preds.shape
    decoded_strings = []
    for b in range(batch_size):
        log_probs = preds[:, b, :]  # (T, nclass)
        best_seq = ctc_beam_search_decoder_single(log_probs, beam_width=beam_width, blank=blank)
        decoded_str = "".join(idx_to_char.get(s, "") for s in best_seq)
        decoded_strings.append(decoded_str)
    return decoded_strings

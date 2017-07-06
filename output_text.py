import numpy as np
import logging


def output_text(model, text_idx, vocab, batch_size):
    ofile = open('logging/test_output.txt', 'wt', encoding='utf-8')
    generated_texts = model.predict(text_idx, batch_size=32)
    inverse_vocab = {v: k for (k, v) in vocab.items()}

    for i, text in enumerate(generated_texts):
        txt_list = [inverse_vocab.get(x, '') for x in text.tolist()]
        oline = ''.join(txt_list)
        ofile.write('{}: {}'.format(i, oline) + '\n')
    ofile.close()
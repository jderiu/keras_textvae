import numpy as np
import logging


def output_text(model, text_idx, vocab, step='final', delimiter='', fname='logging/test_output'):
    ofile = open('{}_{}.txt'.format(fname, step), 'wt', encoding='utf-8')
    generated_texts = model.predict(text_idx, batch_size=32)
    inverse_vocab = {v: k for (k, v) in vocab.items()}

    for i, text in enumerate(generated_texts):
        list_txt_idx = [int(x) for x in text.tolist()]
        txt_list = [inverse_vocab.get(int(x), '') for x in list_txt_idx]
        oline = delimiter.join(txt_list)
        ofile.write('{}'.format(oline) + '\n')
    ofile.close()


def output_lex_text(model, text_idx, text_lex, vocab, step='final', delimiter='', fname='logging/test_output' ):
    ofile = open('{}_{}.txt'.format(fname, step), 'wt', encoding='utf-8')
    generated_texts = model.predict(text_idx, batch_size=32)
    inverse_vocab = {v: k for (k, v) in vocab.items()}

    for i, text in enumerate(generated_texts):
        list_txt_idx = [int(x) for x in text.tolist()]
        txt_list = [inverse_vocab.get(int(x), '') for x in list_txt_idx]
        oline = delimiter.join(txt_list)
        for lex_key in text_lex.keys():
            val = text_lex[lex_key][i]
            if val:
                oline = oline.replace(lex_key, val)
        ofile.write('{}'.format(oline) + '\n')

    ofile.close()


def output_word_text(model, text_idx, vocab, step='final', delimiter=''):
    ofile = open('logging/test_output_{}.txt'.format(step), 'wt', encoding='utf-8')
    generated_sentence_matrix = model.predict(text_idx, batch_size=32) #n_samles, sample_size, 200
    inverse_vocab = {v: k for (k, v) in vocab.items()}




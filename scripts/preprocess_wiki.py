from nltk.tokenize import sent_tokenize
import os
import re
from xml.etree.cElementTree import iterparse
from tqdm import tqdm
import spacy

nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])

#wiki_file = bz2file.open('F:/Wikipedia Embeddings/enwiki-20170120-pages-articles-multistream.xml.bz2', 'rb')
wiki_file = open('F:/Wikipedia Embeddings/enwiki-20170120-pages-articles-multistream.xml', 'rt', encoding='utf-8')

RE_P0 = re.compile('<!--.*?-->', re.DOTALL | re.UNICODE)  # comments
RE_P1 = re.compile('<ref([> ].*?)(</ref>|/>)', re.DOTALL | re.UNICODE)  # footnotes
RE_P2 = re.compile("(\n\[\[[a-z][a-z][\w-]*:[^:\]]+\]\])+$", re.UNICODE)  # links to languages
RE_P3 = re.compile("{{([^}{]*)}}", re.DOTALL | re.UNICODE)  # template
RE_P4 = re.compile("{{([^}]*)}}", re.DOTALL | re.UNICODE)  # template
RE_P5 = re.compile('\[(\w+):\/\/(.*?)(( (.*?))|())\]', re.UNICODE)  # remove URL, keep description
RE_P6 = re.compile("\[([^][]*)\|([^][]*)\]", re.DOTALL | re.UNICODE)  # simplify links, keep description
RE_P7 = re.compile('\n\[\[[iI]mage(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)  # keep description of images
RE_P8 = re.compile('\n\[\[[fF]ile(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)  # keep description of files
RE_P9 = re.compile('<nowiki([> ].*?)(</nowiki>|/>)', re.DOTALL | re.UNICODE)  # outside links
RE_P10 = re.compile('<math([> ].*?)(</math>|/>)', re.DOTALL | re.UNICODE)  # math content
RE_P11 = re.compile('<(.*?)>', re.DOTALL | re.UNICODE)  # all other tags
RE_P12 = re.compile('\n(({\|)|(\|-)|(\|}))(.*?)(?=\n)', re.UNICODE)  # table formatting
RE_P13 = re.compile('\n(\||\!)(.*?\|)*([^|]*?)', re.UNICODE)  # table cell formatting
RE_P14 = re.compile('\[\[Category:[^][]*\]\]', re.UNICODE)  # categories
# Remove File and Image template
RE_P15 = re.compile('\[\[([fF]ile:|[iI]mage)[^]]*(\]\])', re.UNICODE)
RE_TITLE = re.compile(".*?\((.*?)\)")
# MediaWiki namespaces (https://www.mediawiki.org/wiki/Manual:Namespace) that
# ought to be ignored
IGNORED_NAMESPACES = ['Wikipedia', 'Category', 'File', 'Portal', 'Template',
                      'MediaWiki', 'User', 'Help', 'Book', 'Draft',
                      'WikiProject', 'Special', 'Talk']

def remove_markup(text):
    text = re.sub(RE_P2, "", text)  # remove the last list (=languages)
    # the wiki markup is recursive (markup inside markup etc)
    # instead of writing a recursive grammar, here we deal with that by removing
    # markup in a loop, starting with inner-most expressions and working outwards,
    # for as long as something changes.
    text = remove_template(text)
    text = remove_file(text)
    iters = 0
    while True:
        old, iters = text, iters + 1
        text = re.sub(RE_P0, "", text)  # remove comments
        text = re.sub(RE_P1, '', text)  # remove footnotes
        text = re.sub(RE_P9, "", text)  # remove outside links
        text = re.sub(RE_P10, "", text)  # remove math content
        text = re.sub(RE_P11, "", text)  # remove all remaining tags
        text = re.sub(RE_P14, '', text)  # remove categories
        text = re.sub(RE_P5, '\\3', text)  # remove urls, keep description
        text = re.sub(RE_P6, '\\2', text)  # simplify links, keep description only
        # remove table markup
        text = text.replace('||', '\n|')  # each table cell on a separate line
        text = re.sub(RE_P12, '\n', text)  # remove formatting lines
        text = re.sub(RE_P13, '\n\\3', text)  # leave only cell content
        # remove empty mark-up
        text = text.replace('[]', '')
        if old == text or iters > 2:  # stop if nothing changed between two iterations or after a fixed number of iterations
            break

    # the following is needed to make the tokenizer see '[[socialist]]s' as a single word 'socialists'
    # TODO is this really desirable?
    #text = text.replace('[', '').replace(']', '')  # promote all remaining markup to plain text
    return text

def remove_template(s):
    """Remove template wikimedia markup.

    Return a copy of `s` with all the wikimedia markup template removed. See
    http://meta.wikimedia.org/wiki/Help:Template for wikimedia templates
    details.

    Note: Since template can be nested, it is difficult remove them using
    regular expresssions.
    """

    # Find the start and end position of each template by finding the opening
    # '{{' and closing '}}'
    n_open, n_close = 0, 0
    starts, ends = [], []
    in_template = False
    prev_c = None
    for i, c in enumerate(iter(s)):
        if not in_template:
            if c == '{' and c == prev_c:
                starts.append(i - 1)
                in_template = True
                n_open = 1
        if in_template:
            if c == '{':
                n_open += 1
            elif c == '}':
                n_close += 1
            if n_open == n_close:
                ends.append(i)
                in_template = False
                n_open, n_close = 0, 0
        prev_c = c

    # Remove all the templates
    s = ''.join([s[end + 1:start] for start, end in
                 zip(starts + [None], [-1] + ends)])

    return s


def remove_file(s):
    """Remove the 'File:' and 'Image:' markup, keeping the file caption.

    Return a copy of `s` with all the 'File:' and 'Image:' markup replaced by
    their corresponding captions. See http://www.mediawiki.org/wiki/Help:Images
    for the markup details.
    """
    # The regex RE_P15 match a File: or Image: markup
    for match in re.finditer(RE_P15, s):
        m = match.group(0)
        caption = m[:-2].split('|')[-1]
        s = s.replace(m, caption, 1)
    return s

def get_namespace(tag):
    """Returns the namespace of tag."""
    m = re.match("^{(.*?)}", tag)
    namespace = m.group(1) if m else ""
    if not namespace.startswith("http://www.mediawiki.org/xml/export-"):
        raise ValueError("%s not recognized as MediaWiki dump namespace"
                         % namespace)
    return namespace
_get_namespace = get_namespace

elems = (elem for _, elem in iterparse(wiki_file, events=("end",)))
root = next(elems)
namespace = get_namespace(root.tag)
ns_mapping = {"ns": namespace}
page_tag = "{%(ns)s}page" % ns_mapping
text_path = "./{%(ns)s}revision/{%(ns)s}text" % ns_mapping
title_path = "./{%(ns)s}title" % ns_mapping
ns_path = "./{%(ns)s}ns" % ns_mapping
pageid_path = "./{%(ns)s}id" % ns_mapping

title_regex = re.compile(r"==.*==")

fname = 'E:/Wikipedia Embeddings/processed_articles.txt'
if os.path.exists(fname):
    pr_file = open(fname, 'rt')
    line = pr_file.readlines()[-1].replace('\n', '')
    if line:
        already_processed_articles = int(line)
    else:
        already_processed_articles = 0
    pr_file.close()
else:
    already_processed_articles = 0


pr_file = open('F:/Wikipedia Embeddings/simple_processed_articles_en.txt', 'at')
ofile = open('F:/Wikipedia Embeddings/wiki_articles_{}.en.txt'.format(already_processed_articles), 'wt', encoding='utf-8')
ofile_sentence = open('F:/Wikipedia Embeddings/wiki_sentences_{}.en.txt'.format(already_processed_articles), 'wt', encoding='utf-8')
ofile_postag = open('F:/Wikipedia Embeddings/wiki_pos_sentences_{}.en.txt'.format(already_processed_articles), 'wt', encoding='utf-8')

processed_articles = 0
elemlist = []
for elem in tqdm(elems):
    if elem.tag == page_tag:
        title = elem.find(title_path).text
        text = elem.find(text_path).text
        if not text or not title:
            elem.clear()
            root.clear()
            [el.clear() for el in elemlist]
            elemlist = []
            continue

        if len(text.split(' ')) > 50 and len(text.split(' ')) < 50000:
            processed_articles += 1
            if processed_articles <= already_processed_articles:
                elem.clear()
                root.clear()
                [el.clear() for el in elemlist]
                elemlist = []
                continue
            if processed_articles % 100000 == 0:
                print('Processed {} articles:'.format(processed_articles))

            text = remove_markup(text)
            text = text.replace('[', '').replace(']', '')
            text = text.replace('\n', ' ')
            text = text.replace('\r', ' ')
            text = text.replace('\'\'\'', '')
            text = text.replace('\'\'', '').strip()

            pr_file.write('{}\n'.format(processed_articles))
            ofile.write(text + '\n')

            for sentence in sent_tokenize(text):
                osentence = re.sub(title_regex, '', sentence)
                ofile_sentence.write(osentence.strip() + '\n')

            if processed_articles % 1000:
                ofile.flush()
                pr_file.flush()
                ofile_sentence.flush()

        elem.clear()
        root.clear()
        [el.clear() for el in elemlist]
        elemlist = []
    else:
        elemlist.append(elem)
        root.clear()

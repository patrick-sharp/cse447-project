import sys
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import wikipedia
from google_trans_new import google_translator

"""
    Need articles in:
        * Mandarin - zh
        * Cantonese - zh-yue
        * English - en
        * Italian - it
        * German - de
        * Japanese - ja
        * Russian - ru
        * Spanish - es
        * French - fr
        * Norwegian - no
        * Dutch - nl
        * Danish - da
        * Swedish - sv
    Possible articles:
        * space
        * ISS
        * spaceship
        * space travel
"""
languages = ['zh', 'zh-yue', 'en', 'it', 'de', 'ja', \
            'ru', 'es', 'fr', 'no', 'nl', 'da', 'sv']
trans_lang = ['zh-tw', 'zh-cn', 'en', 'it', 'de', 'ja', \
            'ru', 'es', 'fr', 'no', 'nl', 'da', 'sv']
wip2language = {'zh': 'chinese (traditional)', 'zh-yue': 'chinese (simplified)', 'en': 'English', 'it' : 'Italian', 'de': 'German', 'ja': 'Japanese', \
            'ru': 'Russian', 'es': 'Spanish', 'fr': 'French', 'no': 'Norwegan', 'nl' : 'Dutch', 'da': 'Danish', 'sv': 'Swedish'}
topics = ['Mars', 'Moon', 'Moon landing', 'astronaut', 'outer space', 'Interstellar travel', 'Interplanetary spaceflight', 'International Space Station', \
        'galaxy', 'solar system', 'spaceship', 'Milky way galaxy', 'planetary orbit']

translator = google_translator()

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', help='where to save the data', default='../data')
args = parser.parse_args()
data_dir = args.data_dir

try:
    os.mkdir(data_dir)
except FileExistsError:
    pass

print("Begin Downloading Articles...")
# saving article_lang.txt
for i, lang in enumerate(languages):
    for topic in topics:
        wikipedia.set_lang(lang)
        string = ""
        topic_tokens = topic.split()
        for i, token in enumerate(topic_tokens):
            if i < (len(topic_tokens) - 1):
                string += token + "-"
            else:
                string += token

        try:
            translated_topic = translator.translate(topic, lang_tgt=trans_lang[i])
            page = wikipedia.page(translated_topic).content
            with open(os.path.join(data_dir, string + "_" + lang + ".txt"), 'w') as f:
                print(wikipedia.page(translated_topic).content, file = f)
            print(f'Successfully downloaded an article on {topic} in {wip2language[lang]}')
        except wikipedia.DisambiguationError:
            print(f'Disambiguation error on {topic} in {wip2language[lang]}')
            continue
        except wikipedia.PageError:
            print(f'Page error on {topic} in {wip2language[lang]}')
            continue
                
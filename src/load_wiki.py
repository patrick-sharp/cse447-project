import wikipedia
import sys
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
        * Norergian - no
        * Dutch - nl
        * Danish - da
        * Sweedish - sv
    Possible articles:
        * space
        * ISS
        * spaceship
        * space travel
"""
languages = ['zh', 'zh-yue', 'en', 'it', 'de', 'ja', \
            'ru', 'es', 'fr', 'no', 'nl', 'da', 'sv']
wip2language = {'zh': 'Mandarin', 'zh-yue': 'Cantonese', 'en': 'English', 'it' : 'Italian', 'de': 'German', 'ja': 'Japanese', \
            'ru': 'Russian', 'es': 'Spanish', 'fr': 'French', 'no': 'Norwegan', 'nl' : 'Dutch', 'da': 'Danish', 'sv': 'Sweedish'}
topics = ['outer space', 'Interstellar travel', 'Interplanetary spaceflight', 'International Space Station']
print("Saving Articles...")
# saving files as article_lang.txt
for lang in languages:
    for topic in topics:
        wikipedia.set_lang(lang)
        # take spaces out of topics
        txt_filename = ""
        topic_tokens = topic.split()
        for i, token in enumerate(topic_tokens):
            if i < (len(topic_tokens) - 1):
                txt_filename += token + "-"
            else:
                txt_filename += token

        try:
            page_content = wikipedia.page(topic).content
            with open(txt_filename + "_" + lang + ".txt", 'w') as f:
                print(wikipedia.page(topic).content, file = f)
        except wikipedia.DisambiguationError:
            print(f'Disambiguation error on {topic} in {wip2language[lang]}')
            continue
        except wikipedia.PageError:
            print(f'Page error on {topic} in {wip2language[lang]}')
            continue

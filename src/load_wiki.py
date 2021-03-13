import os
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
        * Hindi - hi
        * Arabic - ar
"""
def download_dataset(small=False):
  languages = ['zh', 'zh-yue', 'en', 'it', 'de', 'ja',
              'ru', 'es', 'fr', 'no', 'nl', 'da', 'sv', 
                'hi', 'ar']

  trans_lang = ['zh-tw', 'zh-cn', 'en', 'it', 'de', 'ja',
              'ru', 'es', 'fr', 'no', 'nl', 'da', 'sv', 
              'hi', 'ar']

  wip2language = {'zh': 'Chinese (traditional)', 'zh-yue': 'Chinese (simplified)', 
                  'en': 'English', 'it' : 'Italian', 'de': 'German', 'ja': 'Japanese',
                  'ru': 'Russian', 'es': 'Spanish', 'fr': 'French', 'no': 'Norwegian', 
                  'nl' : 'Dutch', 'da': 'Danish', 'sv': 'Swedish', 'hi': 'Hindi',
                  'ar': 'Arabic'}

  topics = ['International Space Station', 'Interplanetary spaceflight',
            'Interstellar travel', 'Mars', 'Milky way galaxy', 'Moon', 'Moon landing', 
            'Retrograde and prograde motion', 'Satellite', 'Space station', 'asteroid', 
            'astronaut', 'astronomy', 'atmosphere', 'galaxy', 'geosynchronous', 
            'ion engine', 'outer space', 'planet', 'planetary orbit', 'quaternion', 
            'radiation', 'rocket', 'solar panel', 'solar system', 'space capsule', 
            'space telescope', 'spacecraft', 'spacecraft propulsion', 'spaceship', 'star', 
            'gyroscope', 'plutonium', 'oxygen', 'plasma', 'space suit', 'heat shield',
            'ablation', 'actuator', 'altimeter', 'aperture', 'horizon', 'inertia', 'gravity',
            'light', 'electromagnetic radiation', 'electromagnetism', 'mercury', 'venus', 
            'jupiter', 'saturn', 'neptune', 'pluto', 'meteorite', 'uranus', 'oxidizer',
            'parsec', 'lightyear', 'photon', 'radar', 'radiometry', 'nebula', 'solar flare',
            'spectrometer', 'supernova', 'thruster', 'weightlessness', 'wavelength',
            'universe', 'telemetry', 'pulsar', 'parachute', 'velocity', 'meteor', 'meteoroid',
            'magnetron', 'speed of sound', 'speed of light', 'NASA']

  translator = google_translator()

  DATA_DIR = '../data'

  try:
    os.mkdir(DATA_DIR)
    print(f"Created directory {DATA_DIR}")
  except FileExistsError:
    pass

  chars_per_lang = {}
  for lang in languages:
    chars_per_lang[lang] = 0

  articles_per_lang = {}
  for lang in languages:
    articles_per_lang[lang] = 0

  num_successes = 0
  num_total = 0
  print("Begin Downloading Articles...")
  # saving article_lang.txt
  for i, lang in enumerate(languages):
    print(f'Downloading articles for {wip2language[lang]}')
    wikipedia.set_lang(lang)
    for topic in topics:
      # if articles_per_lang[lang] >= 50 if not small else 2:
      #     break
      if chars_per_lang[lang] >= 1e6 if not small else 1.5e4:
        break
      string = ""
      topic_tokens = topic.split()
      for i, token in enumerate(topic_tokens):
        if i < (len(topic_tokens) - 1):
          string += token + "-"
        else:
          string += token
      
      if lang != 'en':
        translated_topic = translator.translate(topic, lang_tgt=trans_lang[i])
      else:
        translated_topic = topic
      num_total += 1
      try:
        page = wikipedia.page(translated_topic, auto_suggest=False)
      except wikipedia.DisambiguationError as disambiguation:
        try:
          page = wikipedia.page(disambiguation.options[0], auto_suggest=False)
        except:
          try:
            page = wikipedia.page(disambiguation.options[0], auto_suggest=True)
          except:
            pass
            # print(f'Error on {topic} in {wip2language[lang]}')
      except wikipedia.PageError:
        try:
          page = wikipedia.page(translated_topic, auto_suggest=True)
          # print(f'Page error on {topic} in {wip2language[lang]}, downloaded article "{page.title}" instead')
        except:
          # print(f'Page error on {topic} in {wip2language[lang]}')
          continue
      with open(os.path.join(DATA_DIR, string + "_" + lang + ".txt"), 'w') as f:
        print(page.content, file = f)
      chars_per_lang[lang] += len(page.content)
      articles_per_lang[lang] += 1
      # print(f'Successfully downloaded an article on {topic} in {wip2language[lang]}')
      num_successes += 1

  print("Successfully downloaded {}/{} articles".format(num_successes, num_total))
  print(chars_per_lang)
  print(articles_per_lang)

if __name__ == '__main__':
  download_dataset()
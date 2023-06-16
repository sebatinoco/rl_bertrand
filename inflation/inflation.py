import pandas as pd
import requests

request = 'https://search.worldbank.org/api/v2/wds?format=xml&fct=count_exact,lang_exact&rows=0'

requests.get(request)
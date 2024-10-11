import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

cur_time = datetime.now()
print(f"время запуска программы: {cur_time.strftime("%Y-%m-%d %H:%M:%S")}")

regions_url = 'https://api.hh.ru/areas'
response = requests.get(regions_url)
regions_code = {}
if response.ok:
    regions_resp = response.json()
    regions_code = {region['name']: int(region['id']) 
                        for country in regions_resp if country['name'] == 'Россия'
                        for region in country['areas']}
else:
    print(f"err {response.status_code}: {response.text}")

regions_code = dict(sorted(regions_code.items(), key=lambda item: item[1]))
def get_vacancies_count(region, position, level):
    params = {
        'text': position + level,  # по заголовку ищем, отдельной фильтрации на джунов миддлов и синьоров нет на сколько знаю, поэтому его тоже буду в название заносить
        'area': region,  
    }
    response = requests.get(BASE_URL, headers=HEADERS, params=params)
    if response.status_code == 200:
        data = response.json()
        return data['found'] 
    else:
        return 0

BASE_URL = 'https://api.hh.ru/vacancies'
HEADERS = {'User-Agent': 'Mozilla/5.0'}  # для апи заголовок 
positions = ['Data Analyst', 'Data Scientist', 'Data Engineer']  
levels = ['junior', 'middle', 'senior']  
regions = list(regions_code.values())[:2]

data = []
for region in regions:
    for position in positions:
        for level in levels:
            count = get_vacancies_count(str(region), position, level)
            data.append([region, position, level, count])


df = pd.DataFrame(data, columns=['Регион', 'Направление', 'Уровень', 'Количество вакансий'])
print(df)
df.to_json('data.json', orient='records', force_ascii=False, lines=True)

table = df.pivot_table(index='Направление', columns='Регион', values='Количество вакансий', aggfunc='sum')
table.plot(kind='bar', figsize=(10, 6))
plt.title('Количество вакансий по направлениям и регионам')
plt.ylabel('Количество вакансий')
plt.xlabel('Направление')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


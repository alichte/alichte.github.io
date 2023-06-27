# PokeAPI

In this notebook, I'm going to be walking through the use of [pokebase](https://pypi.org/project/pokebase/), a Python wrapper for [PokeAPI](https://pokeapi.co/). In a [previous post](https://alichte.github.io/2023/03/13/PokeClustering.html), you can read about the analysis that I did with this data, so this notebook will focus on the use of the API and data cleaning. I start by importing all the necessary packages.


```python
import pokebase as pb
import pandas as pd
import seaborn as sns
```

I start with a look at Ivysaur (#2 in the Pokedex) to see what type of information I can get back from the API. Is this somewhere in the documentation? Maybe, but in this case it was easier to just explore myself. There are a lot of potentially interesting categories that could be used to enrich the dataset beyond names and stats. I may come back to this for a future post, because the evo chains and egg groups I think are potentially very useful for finding nuance that the original clusters I found could not.


```python
MIN_ID = 1
MAX_ID_GEN_1 = 151
p = pb.pokemon(2)
p.name
```


```python
import requests
URL = "https://pokeapi.co/api/v2/pokemon-species/2/"
  
# sending get request and saving the response as response object
r = requests.get(url = URL)
  
# extracting data in json format
data = r.json()

#What is coming back?
dex_no = data['id']
name = data['name']
baby = data['is_baby']
legend = data ['is_legendary']
myth = data['is_mythical']
egg_group = data['egg_groups'] #is list
evo_chain = data['evolution_chain']
```

I dig in a little bit more to the evolution chain here. Again, lots of potentially useful data. 


```python
ev = requests.get(url = evo_chain['url'])
ev_data = ev.json()
```


```python
ev_data['chain']['species']['name']
```




    'bulbasaur'




```python
len(ev_data['chain']['evolves_to'])
```




    1




```python
ev_data['chain']['evolves_to'][0]
```




    {'evolution_details': [{'gender': None,
       'held_item': None,
       'item': None,
       'known_move': None,
       'known_move_type': None,
       'location': None,
       'min_affection': None,
       'min_beauty': None,
       'min_happiness': None,
       'min_level': 16,
       'needs_overworld_rain': False,
       'party_species': None,
       'party_type': None,
       'relative_physical_stats': None,
       'time_of_day': '',
       'trade_species': None,
       'trigger': {'name': 'level-up',
        'url': 'https://pokeapi.co/api/v2/evolution-trigger/1/'},
       'turn_upside_down': False}],
     'evolves_to': [{'evolution_details': [{'gender': None,
         'held_item': None,
         'item': None,
         'known_move': None,
         'known_move_type': None,
         'location': None,
         'min_affection': None,
         'min_beauty': None,
         'min_happiness': None,
         'min_level': 32,
         'needs_overworld_rain': False,
         'party_species': None,
         'party_type': None,
         'relative_physical_stats': None,
         'time_of_day': '',
         'trade_species': None,
         'trigger': {'name': 'level-up',
          'url': 'https://pokeapi.co/api/v2/evolution-trigger/1/'},
         'turn_upside_down': False}],
       'evolves_to': [],
       'is_baby': False,
       'species': {'name': 'venusaur',
        'url': 'https://pokeapi.co/api/v2/pokemon-species/3/'}}],
     'is_baby': False,
     'species': {'name': 'ivysaur',
      'url': 'https://pokeapi.co/api/v2/pokemon-species/2/'}}




```python
ev_data['chain']['evolves_to'][0]['evolution_details'][0]['min_level']
```




    16




```python
ev_data['chain']['evolves_to'][0]['species']['name']
```




    'ivysaur'



With that exploration done, I dig into usint the API to bring back a set of pokemon, their types, and their stats. You could run this for any set, but I think it makes most sense to break into generations. A more robust function might allow you to set which generations you want to bring back, but in this case, you need to know the national dex numebr you want to start and stop with. Correct examples for all current generations are below. Note that the `pokebase` wrapper for PokeAPI is a bit slow. Running a generation will take at least half an hour. 


```python
def get_gen(gen_start, gen_stop, filename, save = True):
    
    tups_list = [] 
    
    for i in range(gen_start,gen_stop + 1):
        p = pb.pokemon(i)
        #print(i) #I would recommend uncommenting this when you run- the API is pretty slow and this is a sanity check that it's still going
        stats = [stat.base_stat for stat in p.stats]
        types = [t.type for t in p.types]
        if len(types) == 1:
            types.append('none')

        uid = i
        name = p.name
        weight = p.weight
        height = p.height

        stats = tuple(stats)
        types = tuple(types)
        inwh = (uid, name, weight, height)

        out = inwh + types + stats
        tups_list.append(out)
        
    cols = ["id","name","weight","height","primary_type","secondary_type","hp","attack","defense","spa","spd","speed"]
    gen = pd.DataFrame(tups_list, columns = cols)
    
    if save:
        gen.to_csv(filename)
    return gen
    
    
```


```python
#gen1 = get_gen(1,151, 'gen1.csv')
#gen2 = get_gen(152,251, 'gen2.csv')
#gen3 = get_gen(252,386, 'gen3.csv')
#gen4 = get_gen(387,493, 'gen4.csv')
#gen5 = get_gen(494,649, 'gen5.csv')
#gen6 = get_gen(650,721, 'gen6.csv')
#gen7 = get_gen(722,809, 'gen7.csv')
#gen8 = get_gen(810,905, 'gen8.csv')
#gen9 = get_gen(906,1008, 'gen9.csv')
```

Please see my post from back in March with some EDA and clustering analysis I was able to perform on this data! 


```python

```

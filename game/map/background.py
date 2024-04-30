import numpy as np
import pandas as pd
# blocks
block_numbers = 10
block_material = ['dirt', 'wood', 'stone', 'iron']
block_kind = ['none', 'building', 'building','building','building',
              'functional', 'transport', 'functional', 'functional', 'functional', ]
block_func = ['workbench', 'forge', 'barrier', 'attacker', 'transport', 'road', 
              'ATbooster', 'HPrestorer', 'DFbooster', 'vehicles']
block_name = ['nothing', 'dirt block', 'wood block', 'stone block', 'iron block', 
              'workbench', 'water pipe', 'forge', 'tower', 'town center']
block_hp = [0, 5, 10, 20, 50, 15, 10, 30, 30, 40]
block_df = [0, 0,  2,  3,  5,  2,  2,  3,  3,  0]

block_id = np.arange(block_numbers)
block_data = pd.DataFrame(columns = block_id, 
                          index = ['block name', 'block kind', 'block HP', 'block DF'], 
                          data = [block_name, 
                                  block_kind,
                                  block_hp, 
                                  block_df])
# print(block_data)

# floor
floor_numbers = 3
floor_kind = ['ground', 'water', 'rock']
floor_id = np.arange(floor_numbers)
floor_data = pd.DataFrame(columns = floor_id, index = ['floor name'], data = [floor_kind])
# print(floor_data)

# tarrain
tarrain_number = 5
tarrain_kind = ['plain', 'hill', 'mountain', 'river', 'lake']
tarrain_id = np.arange(tarrain_number)
tarrain_data = pd.DataFrame(columns = tarrain_id, index = ['floor name'], data = [tarrain_kind])
# print(tarrain_data)

# tool
tool_number = 10
tool_kind = ['weapon', 'weapon', 'weapon', 'weapon', 'shield', 
             'armor', 'tool', 'tool', 'tool', 'tool']
tool_name = ['stick', 'stone', 'sword', 'wand', 'shield', 
             'cuirass', 'hammer', 'axe', 'pickaxe', 'bandage']
tool_at =    [1, 2, 4, 5, 1, 1, 3, 3, 3, -1]
tool_range = [2, 1, 3, 5, 1, 1, 3, 3, 3,  3]
tool_id = np.arange(tool_number)
tool_data = pd.DataFrame(columns = tool_id, 
                         index = ['tool name', 'tool kind', 'tool AT', 'tool range'], 
                         data = [tool_name, tool_kind, tool_at, tool_range])
# print(tool_data)

# creature
creature_number = 6
creature_kind = ['tiny monster', 'normal monster', 'huge monster', 'traverser', 'villager', 'soldier']
creature_hp =    [5, 10, 25, 100, 20, 30]
creature_df =    [0,  1,  1,   1,  1,  2]
creature_at =    [1,  1,  3,   1,  1,  2]
creature_range = [1,  1,  1,   2,  1,  2]
creature_id = np.arange(creature_number)
creature_data = pd.DataFrame(columns = creature_id, 
                             index = ['creature name', 'creature HP', 'creature DF', 'creature AT', 'creature range'],
                             data = [creature_kind, creature_hp, creature_df, creature_at, creature_range])
# print(creature_data)


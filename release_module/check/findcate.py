"""
 找到3Dcompat数据集中 bottle的语义label ； 和partnet中的bottle的语义label
"""
# in partnet
[{"text": "Pot", "children": [{"text": "Pot Body", "children": [{"text": "Lid", "objs": ["new-1"], "id": 2, "name": "lid"}, {"text": "Container", "objs": ["new-2"], "id": 3, "name": "container"}, [{"text": "Pot", "children": [{"text": "Pot Body", "children": [{"text": "Lid",,
 "objs": ["new-1"], "id": 2, "name": "lid"}, {"text": "Container", "objs": ["new-2"], "id": 3, "name": "container"}, {"text": "Bottom", "objs": ["new-0"], "id": 4, "name": "bottom"}], "id": 1, "name": "body"}], "id": 0, "name": "pot"}]


# in 3Dcompat
class: bottle
partnames: ['lid', 'body', 'stripe']
class: bottle
partnames: ['body', 'straw']
class: bottle
partnames: ['lid', 'body', 'handle']
class: bottle
partnames: ['lid', 'body']
class: bottle
partnames: ['lid', 'body']
class: bottle

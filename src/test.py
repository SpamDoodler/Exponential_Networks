import map as mp

map1 = mp.path_map((-1, 1), (-1, 1), (1000, 1000))

a = map1.new_index(0.3 - 0.7j)
print(map1.get_coordinates(a))

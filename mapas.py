from asyncio.windows_events import NULL
import osmnx as ox
G = ox.graph_from_address('Puerta del Sol, Madrid, Comunidad de Madrid, 28013, España', dist=1000, network_type='walk')
#ox.plot_graph(G)

#id = 5301853067
id = 25906743 
print('Inicio Nodos:', G)

print('Nodos sucesores')
for node in G.successors(id):
    print(G.nodes[node])

print(G.out_edges(id, data = 'length'))
edges = list(G.out_edges(id))
print(edges)
print(edges[0])
print(edges[0][1])

print('Other Data')
G2 = ox.speed.add_edge_speeds(G, precision=3)
print(G2.out_edges(id, data = 'speed_kph'))
G3 = ox.speed.add_edge_travel_times(G, precision=3)
print(G3.out_edges(id, data = 'travel_time'))



def calculate(G, path):
    resultTime = 0
    resultDistance = 0
    routes = ox.utils_graph.get_route_edge_attributes(G, path, attribute=None, retrieve_default=None)
    for route in routes:
        resultTime += route['travel_time'] 
        resultDistance += route['length']
    return resultTime, resultDistance    

def printResults(G, path):
    ox.plot_graph_route(G, path)
    time, distance = calculate(G,path)
    print('Calculate distance: ', distance)
    print('Calculate time: ', time)

   

def backtrace(parents, start, end):
    path = []
    current = end
    path.append(end)
    
    while current != start:
    #for i in range(1000):
        path.insert(0, parents[current])
        current = parents[current]
    return path


def bfs(graph, node, dst): #function for BFS
  visited = [] # List for visited nodes.
  queue = []     #Initialize a queue
  parent = {}
  visited.append(node)
  queue.append(node)

  while queue:          # Creating loop to visit each node
    current = queue.pop(0) 
    if current == dst: return True, backtrace(parent, node, dst)
    #print (current, end = " S ") 

    edges = list(graph.out_edges(current))

    for neighbour in edges:
      if neighbour[1] not in visited:
        parent[neighbour[1]] = current
        visited.append(neighbour[1])
        queue.append(neighbour[1])
  return False

def dfsSearch(graph, node, dst, depth, limit, visited, parent):
    if not limit or depth <= limit :
        depth += 1
        if(node == dst):
            print('Econtre nodo: ',node)
            return True, parent
        if node not in visited:
            visited.append(node)
            edges = list(graph.out_edges(node))
            for neighbour in edges:
                if neighbour[1] not in visited:
                    parent[neighbour[1]]=node
                    if dfsSearch(graph, neighbour[1], dst, depth, limit, visited, parent): return True, parent

def dfs(graph, node, dst, limit=0):
    visited = []
    parent = {}
    depth = 0
    try:
        exito, parent = dfsSearch(graph, node, dst, depth, limit, visited, parent)
        if exito: return exito, backtrace(parent, node , dst)
    except:
        return False, False
    
    
    




# Driver Code
print("Following is the Breadth-First Search")
#e = bfs(G, id, 26341673)    # function calling.
#printResults(G, path)
#exito, path = dfs(G, id, 26341673)
#print(exito)
#print(path)
#printResults(G, path)

exito, path = dfs(G, id, 26341673, 25)
if exito: printResults(G, path)
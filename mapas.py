from asyncio.windows_events import NULL
import osmnx as ox
import time



# Calcular tiempo y distancia de cada ruta
# Recibe el grafo y secuencia de nodos del camino
def calculate(G, path):
    resultTime = 0
    resultDistance = 0
    # Obtener los resultados de la ruta
    routes = ox.utils_graph.get_route_edge_attributes(G, path, attribute=None, retrieve_default=None)
    for route in routes:
        resultTime += route['travel_time']
        resultDistance += route['length']
    return resultTime, resultDistance

# Imprimir los resultados de cada ruta
# Regresa los valores para obtener el promedio
# de cada de tiempo de ejecucion, distancia y 
# tiempo de recorrido de cada algoritmo
def printResults(G, path):
    time, distance = calculate(G, path)
    print('Calculate distance: ', distance)
    print('Calculate time: ', time)
    ox.plot_graph_route(G, path)
    return time, distance

# Obtiene el camino desde el nodo origen
# al nodo destino, mediante el uso del
# diccionario parents, que tiene cada
# nodo y su padre, de acuerdp con la
# busqueda de cada algoritmo
def backtrace(parents, start, end):
    path = []
    current = end
    path.append(end)

    while current != start:
        # for i in range(1000):
        path.insert(0, parents[current])
        current = parents[current]
    return path

# Algoritmo BFS
def bfs(graph, node, dst):  # function for BFS
    visited = []  # List for visited nodes.
    queue = []  # Initialize a queue
    parent = {}
    visited.append(node)
    queue.append(node)

    while queue:  
        current = queue.pop(0)
        if current == dst:
            return True, backtrace(parent, node, dst)

        edges = list(graph.out_edges(current))

        for neighbour in edges:
            if neighbour[1] not in visited:
                parent[neighbour[1]] = current
                visited.append(neighbour[1])
                queue.append(neighbour[1])
    return False

# Algoritmo DFS y DLS
def dfsSearch(graph, node, dst, depth, limit, visited, parent):
    if not limit or depth <= limit:
        depth += 1
        if (node == dst):
            return True, parent
        if node not in visited:
            visited.append(node)
            edges = list(graph.out_edges(node))
            for neighbour in edges:
                if neighbour[1] not in visited:
                    parent[neighbour[1]] = node
                    if dfsSearch(graph, neighbour[1], dst, depth, limit, visited, parent): return True, parent

# Funcion de preparacion para algoritmo DFS y DLS
def dfs(graph, node, dst, limit=0):
    visited = []
    parent = {}
    depth = 0
    try:
        exito, parent = dfsSearch(graph, node, dst, depth, limit, visited, parent)
        if exito: return exito, backtrace(parent, node, dst)
    except:
        # Permite incrementar el numero limite
        # del nivel en caso de no encontrar 
        # una ruta con el limite establecido
        return dfs(graph, node, dst, limit+10)


def ucs(graph, start, goal, data='length'):
 
    parent = {}
    answer = []
    queue = []
    for i in range(len(goal)):
        answer.append(10 ** 8)
    queue.append([0, start])
    visited = {}
    count = 0
    while (len(queue) > 0):
        queue = sorted(queue)
        p = queue[-1]
        del queue[-1]
        p[0] *= -1
        if (p[1] in goal):
            index = goal.index(p[1])
            if (answer[index] == 10 ** 8):
                count += 1
            if (answer[index] > p[0]):
                answer[index] = p[0]
            del queue[-1]
            queue = sorted(queue)
            if (count == len(goal)):
                return answer, backtrace(parent, start, goal[0])
        if (p[1] not in visited):
            edges = list(graph.out_edges(p[1], data=data))
            for i in range(len(edges)):
                if(edges[i][1] not in visited):
                    parent[edges[i][1]] = edges[i][0]
                    queue.append([p[0] + edges[i][2], edges[i][1]])
        visited[p[1]] = 1

    return answer, parent

# Prueba con todos los algoritmos una pareja de nodos
def testProblem1(Graph, start, end):
    timeExec = []
    distance = []
    timeA = []
    # BFS
    print(" - Calculating BFS Route - ")
    start_time = time.time()
    exito, path = bfs(Graph, start, end)
    print("Tiempo empleado")
    texe = (time.time() - start_time)
    print("--- %s seconds ---" % texe)
    tmp1, tmp2 = printResults(Graph, path)
    timeExec.append(texe)
    timeA.append(tmp1)
    distance.append(tmp2)
    print()
    # DFS
    print(" - Calculating DFS Route - ")
    start_time = time.time()
    exito, path = dfs(Graph, start, end)
    texe = (time.time() - start_time)
    print("--- %s seconds ---" % texe)
    tmp1, tmp2 = printResults(Graph, path)
    timeExec.append(texe)
    timeA.append(tmp1)
    distance.append(tmp2)
    print()
    # DLS
    print(" - Calculating DLS Route - ")
    start_time = time.time()
    exito, path = dfs(Graph, start, end, limit=30)
    print("Tiempo empleado")
    texe = (time.time() - start_time)
    print("--- %s seconds ---" % texe)
    tmp1, tmp2 = printResults(Graph, path)
    timeExec.append(texe)
    timeA.append(tmp1)
    distance.append(tmp2)
    print()
    # UCS Distance
    print(" - Calculating UCS Distance Route - ")
    start_time = time.time()
    exito, path = ucs(Graph, start, [end], data='length')
    print("Tiempo empleado")
    texe = (time.time() - start_time)
    print("--- %s seconds ---" % texe)
    tmp1, tmp2 = printResults(Graph, path)
    timeExec.append(texe)
    timeA.append(tmp1)
    distance.append(tmp2)
    print()
    # UCS Time
    print(" - Calculating UCS Time Route - ")
    start_time = time.time()
    exito, path = ucs(Graph, start, [end], data='travel_time')
    print("Tiempo empleado")
    texe = (time.time() - start_time)
    print("--- %s seconds ---" % texe)
    tmp1, tmp2 = printResults(Graph, path)
    timeExec.append(texe)
    timeA.append(tmp1)
    distance.append(tmp2)
    print()
    return timeExec, timeA, distance

# Imprime los resultados promedio de toda la ejecucion
def printAverageResult(algoritm, texe, time, dist):
    print("Average Results ", algoritm, " : Ratio 1km")
    print("Time execution: ", texe / 10)
    print("Time : ", time / 10)
    print("Distance: ", dist / 10)

# Ejecucion de las 10 parejas de puntos
def runProblem1(Graph, locations):
    algoritmNames = ['BFS', 'DFS', 'DLS', 'UCS Distance', 'UCS Time']
    timeExecA = [0, 0, 0, 0, 0]
    distanceA = [0, 0, 0, 0, 0]
    timeA = [0, 0, 0, 0, 0]
    for i in range(len(locations)):
        print('Pareja de putnos #', i+1)
        tmp1, tmp2, tmp3 = testProblem1(Graph, locations[i][0], locations[i][1])
        for j in range(5):

            timeExecA[j] = timeExecA[j] + tmp1[j]
            distanceA[j] = distanceA[j] + tmp2[j]
            timeA[j] = timeA[j] + tmp3[j]
    for i in range(5):
        printAverageResult(algoritmNames[i], timeExecA[i], distanceA[i], timeA[i])


# Data
def main():
    # Graph
    G = ox.graph_from_address('Puerta del Sol, Madrid, Comunidad de Madrid, 28013, España', dist=1000, network_type='walk')
    G2 = ox.speed.add_edge_speeds(G, precision=3)
    G3 = ox.speed.add_edge_travel_times(G, precision=3)

    # 1 km Ratio distance
    problemOneLocations = []
    problemOneLocations.append([3246224685, 21734250])
    problemOneLocations.append([21734250, 25906743])
    problemOneLocations.append([25906743, 26341673])
    problemOneLocations.append([26341673, 25906273])
    problemOneLocations.append([25906273, 26486643])
    problemOneLocations.append([26486643, 26487779])
    problemOneLocations.append([26487779, 21941563])
    problemOneLocations.append([21941563, 21947369])
    problemOneLocations.append([21947369, 25906743])
    problemOneLocations.append([21947369, 21734250])
    # Run Problem 1
    runProblem1(G, problemOneLocations)
if __name__ == '__main__':
    main()

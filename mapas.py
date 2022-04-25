from asyncio.windows_events import NULL
import osmnx as ox
import time




def calculate(G, path):
    resultTime = 0
    resultDistance = 0
    routes = ox.utils_graph.get_route_edge_attributes(G, path, attribute=None, retrieve_default=None)
    for route in routes:
        resultTime += route['travel_time']
        resultDistance += route['length']
    return resultTime, resultDistance


def printResults(G, path):
    time, distance = calculate(G, path)
    print('Calculate distance: ', distance)
    print('Calculate time: ', time)
    ox.plot_graph_route(G, path)

def backtrace(parents, start, end):
    path = []
    current = end
    path.append(end)

    while current != start:
        # for i in range(1000):
        path.insert(0, parents[current])
        current = parents[current]
    return path


def bfs(graph, node, dst):  # function for BFS
    visited = []  # List for visited nodes.
    queue = []  # Initialize a queue
    parent = {}
    visited.append(node)
    queue.append(node)

    while queue:  # Creating loop to visit each node
        current = queue.pop(0)
        if current == dst:
            return True, backtrace(parent, node, dst)
        # print (current, end = " S ")

        edges = list(graph.out_edges(current))

        for neighbour in edges:
            if neighbour[1] not in visited:
                parent[neighbour[1]] = current
                visited.append(neighbour[1])
                queue.append(neighbour[1])
    return False


def dfsSearch(graph, node, dst, depth, limit, visited, parent):
    if not limit or depth <= limit:
        depth += 1
        if (node == dst):
            print('Econtre nodo: ', node)
            return True, parent
        if node not in visited:
            visited.append(node)
            edges = list(graph.out_edges(node))
            for neighbour in edges:
                if neighbour[1] not in visited:
                    parent[neighbour[1]] = node
                    if dfsSearch(graph, neighbour[1], dst, depth, limit, visited, parent): return True, parent


def dfs(graph, node, dst, limit=0):
    visited = []
    parent = {}
    depth = 0
    try:
        exito, parent = dfsSearch(graph, node, dst, depth, limit, visited, parent)
        if exito: return exito, backtrace(parent, node, dst)
    except:
        return dfs(graph, node, dst, limit+10)


def ucs(graph, start, goal, data='length'):
    # minimum cost upto
    # goal state from starting
    parent = {}
    answer = []
    # create a priority queue
    queue = []
    # set the answer vector to max value
    for i in range(len(goal)):
        answer.append(10 ** 8)
    # insert the starting index
    queue.append([0, start])
    # map to store visited node
    visited = {}
    # count
    count = 0
    # while the queue is not empty
    while (len(queue) > 0):
        # get the top element of the
        queue = sorted(queue)
        p = queue[-1]
        # pop the element
        del queue[-1]
        # get the original value
        p[0] *= -1
        # check if the element is part of
        # the goal list
        if (p[1] in goal):
            # get the position
            index = goal.index(p[1])

            # if a new goal is reached
            if (answer[index] == 10 ** 8):
                count += 1
            # if the cost is less
            if (answer[index] > p[0]):
                answer[index] = p[0]
            # pop the element
            del queue[-1]
            queue = sorted(queue)
            if (count == len(goal)):
                return answer, backtrace(parent, start, goal[0])
        # check for the non visited nodes
        # which are adjacent to present node
        if (p[1] not in visited):

            edges = list(graph.out_edges(p[1], data=data))
            for i in range(len(edges)):
                # value is multiplied by -1 so that
                # least priority is at the top
                if(edges[i][1] not in visited):
                    parent[edges[i][1]] = edges[i][0]
                    queue.append([p[0] + edges[i][2], edges[i][1]])
        # mark as visited
        visited[p[1]] = 1

    return answer, parent


# Driver Code
def testProblem1(Graph, start, end):
    # BFS
    print(" - Calculating BFS Route - ")
    start_time = time.time()
    exito, path = bfs(Graph, start, end)
    print("Tiempo empleado")
    print("--- %s seconds ---" % (time.time() - start_time))
    printResults(Graph, path)
    print()
    # DFS
    print(" - Calculating DFS Route - ")
    start_time = time.time()
    exito, path = dfs(Graph, start, end)
    print("Tiempo empleado")
    print("--- %s seconds ---" % (time.time() - start_time))
    printResults(Graph, path)
    print()
    # DLS
    print(" - Calculating DLS Route - ")
    start_time = time.time()
    exito, path = dfs(Graph, start, end, limit=30)
    print("Tiempo empleado")
    print("--- %s seconds ---" % (time.time() - start_time))
    printResults(Graph, path)
    print()
    # UCS Distance
    print(" - Calculating UCS Distance Route - ")
    start_time = time.time()
    exito, path = ucs(Graph, start, [end], data='length')
    print("Tiempo empleado")
    print("--- %s seconds ---" % (time.time() - start_time))
    printResults(Graph, path)
    print()
    # UCS Time
    print(" - Calculating UCS Time Route - ")
    start_time = time.time()
    exito, path = ucs(Graph, start, [end], data='travel_time')
    print("Tiempo empleado")
    print("--- %s seconds ---" % (time.time() - start_time))
    printResults(Graph, path)
    print()

def runProblem1(Graph, locations):
    for location in locations:
        testProblem1(Graph, location[0], location[1])

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

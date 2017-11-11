#############################
# STWalk2
# Learns spatial representation and temporal representation for nodes in graph separately and
# combines them by simple arithmetic addition to represent nodes' trajectory.
#############################

import random
from gensim.models import Word2Vec
import networkx as nx
import sys
import time
from multiprocessing import cpu_count
import argparse

def createSpaceTimeGraph(G_list, time_window, start_node, time_step):
    """
     time step is necessary because we want representation only for last time step and
     we will create the spa-time graph for [time_step, time_step-1,time_step-2,...,time_step-time_window]
    """
    start = time.time()
    G = G_list[-1]

    for time1 in range(1, time_window + 1):
        past_node = start_node.split("_")[0] + "_" + str(time_step - time1)
        if past_node not in G_list:
            continue
        else:
            G.add_edge(start_node, past_node)

            G_past = G_list[-time1 - 1]

            # considering first level neighbors
            past_neighbors = list(G_past.neighbors(past_node))
            temp = []

            # considering second level neighbors
            for elt in past_neighbors:
                temp = temp + list(G_past.neighbors(elt))

            # merging list of level-1 and level-2 neighbors
            past_neighbors = past_neighbors + temp
            past_neighbors.append(past_node)

            # subgraph of G_past containing nodes from "past_neighbors" and edges between those nodes.
            past_subgraph = G_past.subgraph(past_neighbors)

            # merge current graph with past subgraphs
            G = nx.compose(G, past_subgraph)
            start_node = past_node
    return G


def random_walk(SpaceTimegraph, path_length, rand=random.Random(0), start=None):  # alpha = prob of restart
    """ Returns a truncated random walk.
        path_length: Length of the random walk.
        removed = alpha: probability of restarts.
        start: the start node of the random walk.
    """
    G = SpaceTimegraph
    if start:
        path = [start]
    else:
        sys.exit("ERROR: Start node not mentioned for random_walk")

    while len(path) < path_length:
        cur = path[-1]
        if len(G[cur]) > 0:
            path.append(rand.choice(list(G[cur])))
        else:
            break
    return path


def create_vocab(G_list, num_restart, path_length, nodes, time_step, isTimeWalk, rand=random.Random(0), time_window=1):
    walks = []
    nodes = list(nodes)

    # number of path = number of restarts per node
    for cnt in range(num_restart):
        rand.shuffle(nodes)
        start = time.time()
        for node in list(nodes):
            if not isTimeWalk:  # if it is space walk
                G = G_list[-1]  # then consider only current time-step
            else:
                G = createSpaceTimeGraph(G_list, time_window, node, time_step)
            walks.append(random_walk(SpaceTimegraph=G, path_length=path_length, rand=rand, start=node))
    return walks


def combineEmbedding(spaceFilePath, timeFilePath, time_step, output_file):
    print("Generating spatiotemporal representation from spaceWalk and timeWalk...")
    spaceDict = dict()
    timeDict = dict()
    finalEmb = dict()

    with open(spaceFilePath, "r") as inFile:
        for line in inFile:
            inFile.readline()
            token = line.strip().split(" ")
            if "_" + str(time_step) in token[0]:
                rep = token[1].strip().split(" ")
                spaceDict[token[0]] = [rep[i] for i in range(0, len(rep))]

    with open(timeFilePath, "r") as inFile:
        for line in inFile:
            inFile.readline()
            token = line.strip().split(" ")
            if "_" + str(time_step) in token[0]:
                rep = token[1].strip().split(" ")
                timeDict[token[0]] = [rep[i] for i in range(0, len(rep))]

    for spaceKey in spaceDict:
        if spaceKey in timeDict:
            finalEmb[spaceKey] = []
            for i in range(len(spaceDict[spaceKey])):
                lst = finalEmb[spaceKey]
                spacelist = spaceDict[spaceKey]
                timelist = timeDict[spaceKey]
                lst.append((float(spacelist[i]) + float(timelist[i])))
                finalEmb[spaceKey] = lst

    # store each time file as separate
    path = output_file+"/spatiotemporal_window_"+str(time_step)+".stwalktwo"
    with open(path, "w") as inFile:
        for key in finalEmb:
            inFile.write(key + " ")
            for val in finalEmb[key]:
                inFile.write(str(val) + " ")
            inFile.write("\n")
    print("Combined representation file saved: " + path)
    return


def combineSpaceaTimeWalk(input_direc, output_file, number_restart, walk_length_time, walk_length_space,
                          representation_size, time_step,
                          time_window_size, workers, vocab_window_size):
    if time_window_size > time_step:
        sys.exit("ERROR: time_window_size(=" + str(time_window_size) + ") cannot be more than time_step(=" + str(
            time_step) + "):")

    if walk_length_space < vocab_window_size:
        print("WARNING: space_walk length cannot be less than window size. Setting space_walk_length=window_size")
        walk_length_space = vocab_window_size
    if walk_length_time < vocab_window_size:
        print("WARNING: time_walk length cannot be less than window size. Setting time_walk_length=window_size")
        walk_length_time = vocab_window_size

    G_list = [nx.read_graphml(input_direc + "/graph_" + str(i) + ".graphml") for i in
              range(time_step - time_window_size, time_step + 1)]

    ###############
    #  spaceWalk  #
    ###############
    # get list of nodes
    nodes = G_list[-1].nodes()
    spacewalks = create_vocab(G_list, num_restart=number_restart, path_length=walk_length_space, nodes=nodes,
                              rand=random.Random(0), time_step=time_step, isTimeWalk=False, time_window=0)

    # time-step is decremented by 1, because, time steps are from 0 to time_step-1=total time_step length
    print("Generating representation from spaceWalk...")
    model = Word2Vec(spacewalks, size=representation_size, window=vocab_window_size, min_count=0, workers=workers)
    space_output_file = output_file+"/space_window_"+str(time_step)+".stwalktwo"
    model.wv.save_word2vec_format(space_output_file)
    print("Space Representation File saved: " + space_output_file)

    ################
    #  timeWalk  ##
    ###############
    # get list of nodes
    nodes = G_list[-1].nodes()
    timewalks = create_vocab(G_list, num_restart=number_restart, path_length=walk_length_space, nodes=nodes,
                             rand=random.Random(0), time_step=time_step, isTimeWalk=True, time_window=time_window_size)

    # time-step is decremented by 1, because, time steps are from 0 to time_step-1=total time_step length
    print("Generating representation from timeWalk")
    model = Word2Vec(timewalks, size=representation_size, window=vocab_window_size, min_count=0, workers=workers)
    time_output_file = output_file+"/time_window_"+str(time_step)+".stwalktwo"
    model.wv.save_word2vec_format(time_output_file)
    print("Time Representation File saved: " + time_output_file)

    ################################
    # combining two representation #
    ################################
    combineEmbedding(space_output_file,
                     time_output_file,
                     time_step, output_file)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', action='store', dest='dataset', help='Dataset')
    arg = parser.parse_args()
    print('dataset =', arg.dataset)

    if arg.dataset not in ["epinion","dblp",'ciao']:
        print("Invalid dataset.\nAllowed datsets are epinion, dblp, ciao")
        sys.exit(0)
    # by default we use Epinion dataset for experiment
    direc = "../epinion"
    max_timestep = 109  # Epinion dataset has 0 to 109 graphs

    if arg.dataset == "ciao":
        direc = "../" + arg.dataset
        max_timestep = 114
    elif arg.dataset == "dblp":
        direc = "../" + arg.dataset
        max_timestep = 44

    number_restart = 40
    walk_length_time = 30
    walk_length_space = 10
    representation_size = 64
    vocab_window_size = 5
    time_window_size = 5  # number of previous time steps graphs plus current time step graph

    workers = cpu_count()
    seq = []
    for t in range(0, max_timestep+1):
        if (t + 1) % time_window_size == 0:
            seq.append(t)

    for t in seq:
        print("\nGenerating " + str(representation_size) + " dimension embeddings for nodes at t=" + str(t))
        time_step = t
        combineSpaceaTimeWalk(input_direc=direc + "/input_graphs",
                              output_file=direc + "/output_stwalktwo",
                              number_restart=number_restart,
                              walk_length_time=walk_length_time, walk_length_space=walk_length_space,
                              vocab_window_size=vocab_window_size,
                              representation_size=representation_size, time_step=time_step,
                              time_window_size=time_window_size - 1,
                              workers=workers)

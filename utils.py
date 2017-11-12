# -*- coding: utf-8 -*-
import numpy

def direction(data):
    results = []
    gradient = numpy.gradient(data)
    s = 0
    for i in range(len(gradient)):
        if gradient[i] < 0:
            s = -1
        elif 0 < gradient[i]:
            s = 1
        results.append(s)
    return results

def rewards(data):
    results = []
    direction_data = direction(data)
    dist = 0
    for i in range(len(direction_data) - 1):
        dist += 1
        if direction_data[i] != direction_data[i+1]:
            results.append((direction_data[i], dist))
            dist = 0
    results.append((0, dist+1))
    feature = []
    for r in results:
        bs = 0 if r[0] == -1 else 1
        f = [abs(bs - 1/r[1] * x) for x in range(r[1])]
        feature.extend(f)
    return feature


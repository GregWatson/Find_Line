import math
import numpy as np

def edge(a, b):
    """Return vector from point a to point b."""
    return (b[0] - a[0], b[1] - a[1])

def dot(u, v):
    """Return the dot product of vectors u and v."""
    return u[0]*v[0] + u[1]*v[1]

def cross(u, v):
    """Return the cross product of vectors u and v."""
    return u[0]*v[1] - u[1]*v[0]

def length(u):
    """Return the length of vector u."""
    return math.hypot(u[0], u[1])

def angle_between(u, v):
    """Return the unsigned angle between vectors u and v in degrees."""
    cos_theta = dot(u, v) / (length(u) * length(v))
    cos_theta = max(-1.0, min(1.0, cos_theta)) # clamp for safety
    return math.degrees(math.acos(cos_theta))

def get_polygon_angles(vertices):
    """
    Given a list of vertices in CW order, return a list of (inner, outer) angles.
    """
    n = len(vertices)
    results = []

    for i in range(n):
        p_prev = vertices[i - 1]
        p_curr = vertices[i]
        p_next = vertices[(i + 1) % n]

        u = edge(p_curr, p_prev) # incoming edge
        v = edge(p_curr, p_next) # outgoing edge

        theta = np.radians(angle_between(u, v))
        turn = cross(u, v)

        if turn > 0:
            # left turn → convex vertex
            outer = theta
            inner = np.pi * 2 - theta
        else:
            # right turn → convex vertex
            outer = np.pi * 2 - theta
            inner = theta

        results.append((inner, outer))

    return results
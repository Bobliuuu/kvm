"""Plan the scripted player's route through E1M1 from the WAD's map data and write it as a C header (route_e1m1.h).

The map is cut into 16-unit cells. A cell is walkable if its floor does not hurt (no slime), it has room for the player (56 units of
headroom; doors count as open), and its centre is at least 20 units from walls, blocking lines and solid objects (lamps, barrels,
columns). A* joins the goals below in order, never crossing a wall, a blocking line or a step higher than 24, and prefers the middle
of corridors; the path is then cut to the points where it turns. The header also lists the doors the route crosses (with the sector
each one moves) and the exit switch next to the last goal.

  python3 plan_route.py [--wad ../doom1.wad] [--out route_e1m1.h] [--png route.png]   (needs numpy; opencv-python for --png)
"""
import argparse
import heapq
import math
import struct

import numpy as np

# Visited in order from the player start: the armor room, back through the start room, the computer room, the zig-zag walkway, the
# side corridor with the medikit, the last two rooms and the exit switch.
GOALS = ("-200,-3240;1280,-2650;1700,-2500;1800,-2150;2450,-2150;2450,-2700;2950,-2950;3020,-3700;2950,-3800;2600,-3860;"
         "2304,-4032;2600,-3860;2950,-3800;3008,-4250;2800,-4450;3250,-4450;3008,-4700;2960,-4768")

ap = argparse.ArgumentParser()
ap.add_argument("--wad", default="../doom1.wad")
ap.add_argument("--goals", default=GOALS, help="semicolon-separated x,y map points visited in order after the start")
ap.add_argument("--out", default="route_e1m1.h")
ap.add_argument("--png", default="", help="also draw the route over the walkable cells")
a = ap.parse_args()

b = open(a.wad, "rb").read()
_, n, off = struct.unpack_from("<4sii", b, 0)
lumps = [struct.unpack_from("<ii8s", b, off + 16 * i) for i in range(n)]
names = [l[2].rstrip(b"\0").decode() for l in lumps]
i0 = names.index("E1M1")
L = {names[i0 + k]: lumps[i0 + k] for k in range(1, 11)}


def rec(nm, fmt):
    p, sz, _ = L[nm]
    s = struct.calcsize(fmt)
    return [struct.unpack_from(fmt, b, p + s * k) for k in range(sz // s)]


V = rec("VERTEXES", "<hh"); LD = rec("LINEDEFS", "<hhhhhhh"); SD = rec("SIDEDEFS", "<hh8s8s8sh")
SEC = rec("SECTORS", "<hh8s8shhh"); SEGS = rec("SEGS", "<hhhhhh"); SS = rec("SSECTORS", "<hh"); NODES = rec("NODES", "<hhhh4h4hHH")
THINGS = rec("THINGS", "<hhhhh")
DOOR_SPECIALS = {1, 26, 27, 28, 31, 32, 33, 34, 46, 117, 118}
DAMAGE = {4, 5, 7, 11, 16}                          # sector specials that hurt
# solid decorations by thing type -> radius
SOLID = {2035: 10, 2028: 16, 35: 16, 48: 16, 30: 16, 31: 16, 32: 16, 33: 16, 36: 16, 37: 16, 41: 16, 42: 16, 43: 16, 44: 16, 45: 16,
         46: 16, 47: 16, 54: 32, 55: 16, 56: 16, 57: 16, 70: 16, 85: 16, 86: 16, 25: 16, 26: 16, 27: 16}


def sector_at(x, y):
    """The sector containing map point (x, y), found by walking the BSP tree like the engine does."""
    nd = len(NODES) - 1
    while True:
        nx, ny, ndx, ndy = NODES[nd][0:4]
        rc, lc = NODES[nd][12], NODES[nd][13]
        c = rc if (y - ny) * ndx < ndy * (x - nx) else lc
        if c & 0x8000:
            seg = SEGS[SS[c & 0x7FFF][1]]
            ld = LD[seg[3]]
            return SD[ld[5] if seg[4] == 0 else ld[6]][5]
        nd = c


def is_door_sector(s):
    return any(ld[3] in DOOR_SPECIALS and ld[6] != -1 and SD[ld[6]][5] == s for ld in LD)


xs = [v[0] for v in V]; ys = [v[1] for v in V]
X0, X1, Y0, Y1 = min(xs) - 32, max(xs) + 32, min(ys) - 32, max(ys) + 32
C = 16
NX, NY = (X1 - X0) // C + 1, (Y1 - Y0) // C + 1
cx = lambda i: X0 + i * C + C // 2
cy = lambda j: Y0 + j * C + C // 2
door_secs = {s for s in range(len(SEC)) if is_door_sector(s)}
sec = np.full((NX, NY), -1, np.int32)
ok = np.zeros((NX, NY), bool)
for i in range(NX):
    for j in range(NY):
        s = sector_at(cx(i), cy(j)); sec[i, j] = s
        fh, ch, _, _, _, sp, _ = SEC[s]
        ok[i, j] = sp not in DAMAGE and (ch - fh >= 56 or s in door_secs)

walls = [(V[v1], V[v2]) for (v1, v2, flags, special, tag, s1, s2) in LD if s2 == -1 or flags & 1]   # one-sided or blocking


def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / float(dx * dx + dy * dy or 1)))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


near = np.full((NX, NY), 1e9)                       # distance from each cell centre to the nearest wall or solid object
for (ax, ay), (bx, by) in walls:
    i0_, i1_ = max(0, (min(ax, bx) - 80 - X0) // C), min(NX - 1, (max(ax, bx) + 80 - X0) // C)
    j0_, j1_ = max(0, (min(ay, by) - 80 - Y0) // C), min(NY - 1, (max(ay, by) + 80 - Y0) // C)
    for i in range(i0_, i1_ + 1):
        for j in range(j0_, j1_ + 1):
            near[i, j] = min(near[i, j], seg_dist(cx(i), cy(j), ax, ay, bx, by))
for (tx_, ty_, _, ttype, tflags) in THINGS:         # every skill's copy counts; multiplayer-only things (flag 16) are not spawned
    if ttype in SOLID and not tflags & 16:
        r_ = SOLID[ttype]
        for i in range(max(0, (tx_ - 80 - X0) // C), min(NX - 1, (tx_ + 80 - X0) // C) + 1):
            for j in range(max(0, (ty_ - 80 - Y0) // C), min(NY - 1, (ty_ + 80 - Y0) // C) + 1):
                near[i, j] = min(near[i, j], math.hypot(cx(i) - tx_, cy(j) - ty_) - r_)
ok &= near >= 20                                    # the player's radius is 16


def segments_cross(p, q, a1, a2):
    d = (q[0] - p[0]) * (a2[1] - a1[1]) - (q[1] - p[1]) * (a2[0] - a1[0])
    if d == 0:
        return False
    t = ((a1[0] - p[0]) * (a2[1] - a1[1]) - (a1[1] - p[1]) * (a2[0] - a1[0])) / d
    u = ((a1[0] - p[0]) * (q[1] - p[1]) - (a1[1] - p[1]) * (q[0] - p[0])) / d
    return 0 <= t <= 1 and 0 <= u <= 1


def crosses_wall(p, q):
    return any(segments_cross(p, q, w[0], w[1]) for w in walls)


wall_index = {}
for k, ((ax, ay), (bx, by)) in enumerate(walls):
    for i in range((min(ax, bx) - X0) // C - 1, (max(ax, bx) - X0) // C + 2):
        for j in range((min(ay, by) - Y0) // C - 1, (max(ay, by) - Y0) // C + 2):
            wall_index.setdefault((i, j), []).append(k)


def blocked(i, j, i2, j2):
    """Can the player not step from cell (i, j) to the neighbouring cell (i2, j2)?"""
    s1, s2 = sec[i, j], sec[i2, j2]
    if SEC[s2][0] - SEC[s1][0] > 24 and s2 not in door_secs:
        return True
    p, q = (cx(i), cy(j)), (cx(i2), cy(j2))
    for k in set(wall_index.get((i, j), [])) | set(wall_index.get((i2, j2), [])):
        if segments_cross(p, q, walls[k][0], walls[k][1]):
            return True
    return False


def cell(x, y):
    return int((x - X0) // C), int((y - Y0) // C)


def astar(src, dst):
    (si, sj), (ti, tj) = cell(*src), cell(*dst)
    for (pi, pj) in ((si, sj), (ti, tj)):
        if not ok[pi, pj]:
            raise SystemExit(f"point ({cx(pi)},{cy(pj)}) is not walkable")
    g = {(si, sj): 0.0}; prev = {}
    pq = [(0.0, si, sj)]
    while pq:
        f, i, j = heapq.heappop(pq)
        if (i, j) == (ti, tj):
            break
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            i2, j2 = i + di, j + dj
            if not (0 <= i2 < NX and 0 <= j2 < NY) or not ok[i2, j2] or blocked(i, j, i2, j2):
                continue
            ng = g[(i, j)] + math.hypot(di, dj) * C + 40.0 / max(1.0, near[i2, j2] - 12)   # extra cost near walls
            if ng < g.get((i2, j2), 1e18):
                g[(i2, j2)] = ng; prev[(i2, j2)] = (i, j)
                heapq.heappush(pq, (ng + math.hypot(ti - i2, tj - j2) * C, i2, j2))
    if (ti, tj) not in prev and (ti, tj) != (si, sj):
        raise SystemExit(f"no path {src} -> {dst}")
    path = [(ti, tj)]
    while path[-1] != (si, sj):
        path.append(prev[path[-1]])
    return [(cx(i), cy(j)) for i, j in reversed(path)]


def simplify(pts):
    """Keep only the points where the path must turn: drop points while the straight line between the kept ones stays on walkable
    cells at least 28 units from walls and crosses no wall."""
    out = [pts[0]]
    k = 0
    while k < len(pts) - 1:
        best = k + 1
        for m in range(len(pts) - 1, k + 1, -1):
            p, q = pts[k], pts[m]
            n_ = max(2, int(math.hypot(q[0] - p[0], q[1] - p[1]) / 8))
            good = True
            for t in range(n_ + 1):
                ci, cj = cell(p[0] + (q[0] - p[0]) * t / n_, p[1] + (q[1] - p[1]) * t / n_)
                if not ok[ci, cj] or near[ci, cj] < 28:
                    good = False; break
            if good and not crosses_wall(p, q):
                best = m; break
        out.append(pts[best]); k = best
    return out


start = next((t[0], t[1]) for t in THINGS if t[3] == 1)
route = [start]
for g in (tuple(int(v) for v in g.split(",")) for g in a.goals.split(";")):
    route += simplify(astar(route[-1], g))[1:]
length = sum(math.hypot(route[k + 1][0] - route[k][0], route[k + 1][1] - route[k][1]) for k in range(len(route) - 1))
print(f"{len(route)} waypoints, {length:.0f} map units")

doors, door_sec = [], []
for k in range(len(route) - 1):
    for (v1, v2, flags, special, tag, s1, s2) in LD:
        if special in DOOR_SPECIALS and s2 != -1 and segments_cross(route[k], route[k + 1], V[v1], V[v2]):
            mid = ((V[v1][0] + V[v2][0]) // 2, (V[v1][1] + V[v2][1]) // 2)
            if mid not in doors:
                doors.append(mid); door_sec.append(SD[s2][5])      # a door line's back side is the door sector
exits = [((V[v1][0] + V[v2][0]) // 2, (V[v1][1] + V[v2][1]) // 2) for (v1, v2, flags, special, tag, s1, s2) in LD if special == 11]
exit_ = min(exits, key=lambda m: math.hypot(m[0] - route[-1][0], m[1] - route[-1][1]))
print("doors on the route:", doors, "exit switch:", exit_)
with open(a.out, "w") as f:
    f.write("// E1M1 route for the scripted player (player.c). Generated by plan_route.py from the map data in doom1.wad.\n")
    f.write("static const short route_pts[][2] = {\n" + "".join(f"  {{{x}, {y}}},\n" for x, y in route) + "};\n")
    f.write("// middles of the door lines the route crosses, and the sector each door moves\n")
    f.write("static const short route_doors[][2] = {\n" + "".join(f"  {{{x}, {y}}},\n" for x, y in doors) + "};\n")
    f.write("static const short route_door_sec[] = {" + ", ".join(str(v) for v in door_sec) + "};\n")
    f.write("// middle of the exit switch next to the last waypoint\n")
    f.write(f"static const short route_exit[2] = {{{exit_[0]}, {exit_[1]}}};\n")

if a.png:
    import cv2
    S = 0.5
    img = np.zeros((int((Y1 - Y0) * S) + 1, int((X1 - X0) * S) + 1, 3), np.uint8)
    P = lambda x, y: (int((x - X0) * S), int((Y1 - y) * S))
    for i in range(NX):
        for j in range(NY):
            if ok[i, j]:
                x, y = P(cx(i) - C // 2, cy(j) + C // 2)
                cv2.rectangle(img, (x, y), (x + int(C * S) - 1, y + int(C * S) - 1), (45, 45, 45), -1)
    for (ax, ay), (bx, by) in walls:
        cv2.line(img, P(ax, ay), P(bx, by), (200, 200, 200), 1)
    for k in range(len(route) - 1):
        cv2.line(img, P(*route[k]), P(*route[k + 1]), (0, 200, 255), 2)
    for k, p in enumerate(route):
        cv2.circle(img, P(*p), 4, (0, 255, 0), -1)
    cv2.imwrite(a.png, img)

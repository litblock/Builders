import json
import math
import queue
import random
import sys
import threading
import time
from array import array

q = queue.Queue()

CFG = {}
RNG = random.Random(0)

known = {}       # weapon pickup id -> pickup dict (remembered after it leaves view)
known_cubes = {}  # cube id -> {"c": cube dict, "t": last seen match time}
ignore = {}      # (type, id) -> match time until which we skip that target
chase = {"key": None, "best": 1e9, "t": 0.0, "start": 0.0, "tz": None}
wander = {"pt": None, "until": 0.0}

EARLY_S = 120     # "early game": avoid fights against healthier enemies before this match time
HP_AVOID = 0.8    # avoid an enemy whose HP is more than 1/0.8 = 1.25x ours
FLEE_KEEP_S = 3.0  # keep running from the last threat this long after losing sight of it
HOP_ON_HIT = True  # hop while fleeing when shot (weak: our hitbox is 9 m tall; set False to disable)
flee_mem = {"until": 0.0, "from": None}
COVER_RADII = (10.0, 20.0, 35.0, 50.0, 70.0)   # how far from us to look for cover (m)
COVER_ANGLES = 24                  # directions tried per radius
HEAL_HOLD_S = 25.0  # after a retreat, wait in cover this long (or until ~90% HP)
cover = {"pt": None, "plan_t": -99.0, "hold_until": 0.0, "hidden": False}
dodge = {"until": 0.0, "side": 1}
MAP = {"ok": False}
mob = {"blocked": 0, "jump_cd": 0.0, "climb_until": 0.0}
nav = {"side": 1, "until": 0.0, "last_pos": None, "stuck": 0, "last_t": -1.0}

WEAPON_RANK = {"laser": 0, "lightning": 1, "mortar": 2}  # lower = preferred
# Upgrade ranking by id (higher = better), from the real hello.config.upgrades list.
# Survival first (placement is what scores), then gun sustain, then held-weapon upgrades.
UP_RANK = {
    "plating": 100, "nanites": 92, "gun_mag": 88, "armor": 84,
    "laser_damage": 80, "gyros": 76, "gun_damage": 74, "lightning_efficiency": 70,
    "lightning_range": 68, "gun_rate": 66, "mortar_splash": 66, "mortar_reload": 64,
    "laser_charge": 62, "gun_velocity": 60, "servos": 55, "mortar_velocity": 52,
    "sensors": 50, "lightning_chain": 45, "capacitor": 40, "laser_pierce": 30,
    "reactor": 30, "jets": 5,
}
# fallback for ids not listed above: (keyword in id/desc, score)
UP_SCORE = [("armor", 5), ("damage taken", 5), ("max hp", 4), ("regen", 3),
            ("damage", 4), ("fire rate", 4), ("sensor", 2), ("speed", 2)]


def reader():
    for line in sys.stdin:
        line = line.strip()
        if line:
            try:
                q.put(json.loads(line))
            except ValueError:
                pass
    q.put(None)


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def owned_counts(me):
    """How many stacks of each upgrade we already have (format-tolerant)."""
    counts = {}
    try:
        ups = me.get("upgrades", [])
        if isinstance(ups, dict):
            return {str(k): int(v) for k, v in ups.items()}
        for u in ups:
            uid = u if isinstance(u, str) else (u.get("id") if isinstance(u, dict) else None)
            if uid:
                counts[uid] = counts.get(uid, 0) + 1
    except Exception:
        pass
    return counts


def pick_upgrade(offer, held, me):
    owned = owned_counts(me)
    best, bi = -1e9, 0
    for i, c in enumerate(offer.get("choices", [])):
        uid = str(c.get("id", ""))
        if uid in UP_RANK:
            score = UP_RANK[uid]
        else:
            txt = (uid + " " + str(c.get("desc", ""))).lower()
            score = 0
            for k, v in UP_SCORE:
                if k in txt:
                    score = max(score, v)
            for w in held:
                if w and w in txt:
                    score = max(score, 4)
        score -= 8 * owned.get(uid, 0)    # mild preference for variety over stacking
        if score > best:
            best, bi = score, i
    return bi


def aim_at(me, tpos, tvel, speed, zoff, lead=True):
    """yaw, pitch to hit a target; leads moving targets unless lead=False."""
    px, py, pz = me["pos"]
    mz = pz + me["stats"].get("muzzle_height", 7)
    dist = math.hypot(tpos[0] - px, tpos[1] - py)
    ft = dist / speed if lead else 0.0
    tx = tpos[0] + tvel[0] * ft
    ty = tpos[1] + tvel[1] * ft
    tz = tpos[2] + zoff
    yaw = math.atan2(ty - py, tx - px)
    pitch = math.atan2(tz - mz, math.hypot(tx - px, ty - py))
    return yaw, pitch


def fire_second(me, w1, dist, yaw_direct_err):
    """Whether to hold the slot-1 trigger this decision."""
    if not w1:
        return False
    kind = w1["kind"]
    if kind == "lightning":
        rng = w1["stats"].get("range", 45)
        return dist < 0.9 * rng and yaw_direct_err < 0.25 and me["energy"] > 10
    if kind == "laser":
        if w1["cooldown"] > 0.05 or dist > w1["stats"].get("range", 280) - 10:
            return False
        full = w1["stats"].get("charge", 0.75)
        charge = w1.get("charge", 0)
        tight = yaw_direct_err < max(0.01, min(0.15, math.atan(2.0 / max(dist, 1))))
        if charge >= full - 0.05:
            return False              # let go -> fires now
        if charge > 0.3 and not tight:
            return False              # lost the target, release early
        return tight or charge > 0
    return False


def update_memory(o, px, py, pz):
    visible = set()
    for p in o.get("pickups", []):
        known[p["id"]] = p
        visible.add(p["id"])
    for pid, p in list(known.items()):
        d = math.hypot(p["pos"][0] - px, p["pos"][1] - py)
        if pid not in visible and d < 25 and abs(p["pos"][2] - pz) < 4:
            del known[pid]            # we are next to where it was and it is gone
    now = o["t"]
    seen_cubes = set()
    for c in o.get("cubes", []):
        known_cubes[c["id"]] = {"c": c, "t": now}
        seen_cubes.add(c["id"])
    for cid, rec in list(known_cubes.items()):
        cp = rec["c"]["pos"]
        d = math.hypot(cp[0] - px, cp[1] - py)
        if cid not in seen_cubes and (d < 20 or now - rec["t"] > 45):
            del known_cubes[cid]      # gone (taken) or too old to trust
    for e in o.get("events", []):
        if e.get("type") == "pickup" and known_cubes:
            near = min(known_cubes.values(),
                       key=lambda r: math.hypot(r["c"]["pos"][0] - px, r["c"]["pos"][1] - py))
            if math.hypot(near["c"]["pos"][0] - px, near["c"]["pos"][1] - py) < 6:
                known_cubes.pop(near["c"]["id"], None)
    for e in o.get("events", []):
        if e.get("type") == "took":
            known.pop(e["id"], None)
        elif e.get("type") == "take_failed":
            if e.get("reason") == "gone":
                known.pop(e["id"], None)
            elif e.get("reason") == "have":
                ignore[("p", e["id"])] = 1e9


def loot(o, me, act, held, z, cx, cy, enemies_none=True):
    """V7A: conservative loot scoring with target commitment, based on V4 combat."""
    px, py, pz = me["pos"]
    now = o["t"]
    chase["tz"] = None
    need_weapon = me["weapons"][1] is None
    # Only seek loot inside a future-safe area, or close enough to return before
    # the first shrink. The latter lets edge spawns loot instead of marching
    # across the arena empty-handed for the first minute.
    nr = z["next_radius"]
    shrink_start = z.get("shrink_start", 0)
    dist_to_next = math.hypot(px - cx, py - cy)
    travel_buffer = max(0.0, (dist_to_next - 0.78 * nr) / 11.7)
    can_detour = me.get("in_zone", False) and now + travel_buffer + 15 < shrink_start

    def valid(pos, key, kind):
        if ignore.get(key, 0) > now:
            return False
        d = math.hypot(pos[0] - px, pos[1] - py)
        dz = abs(pos[2] - pz)
        if dz > (2.5 + 0.22 * d if kind == "cube" else 3.0):
            return False
        next_dist = math.hypot(pos[0] - cx, pos[1] - cy)
        if next_dist > 0.90 * nr:
            if not can_detour or d > 65 or now + d / 10 + max(0, (next_dist - 0.78 * nr) / 11.7) + 12 >= shrink_start:
                return False
        # Reject immediately blocked targets except close targets, for which
        # V4's obstacle steering can find a short approach.
        if d > 22 and not path_clear(px, py, pos[0], pos[1], max_len=min(d, 130)):
            return False
        return True

    enemies = [m for m in o.get("mechs", []) if m.get("kind") == "mech" and m.get("id") != me["id"]]
    def danger(pos):
        if not enemies:
            return 0.0
        nearest = min(math.hypot(pos[0]-m["pos"][0], pos[1]-m["pos"][1]) for m in enemies)
        if nearest < 35: return 500.0
        if nearest < 60: return 160.0
        if nearest < 95: return 55.0
        return 0.0

    cands = []  # score, key, target, kind, stopping distance
    if need_weapon:
        for pid, p in known.items():
            if p["kind"] in held or not valid(p["pos"], ("p", pid), "pickup"):
                continue
            d = math.hypot(p["pos"][0]-px, p["pos"][1]-py)
            if d < 130:
                cands.append((d + 18 * WEAPON_RANK.get(p["kind"], 3) - 45 + danger(p["pos"]), ("p", pid), p, "pickup", 2.5))
    for rec in known_cubes.values():
        c = rec["c"]
        d = math.hypot(c["pos"][0]-px, c["pos"][1]-py)
        if d < 105 and valid(c["pos"], ("c", c["id"]), "cube"):
            cands.append((d - 22 + danger(c["pos"]), ("c", c["id"]), c, "cube", 1.5))
    for c in o.get("crates", []):
        d = math.hypot(c["pos"][0]-px, c["pos"][1]-py)
        if d < 100 and valid(c["pos"], ("k", c["id"]), "crate"):
            cands.append((d + (35 if need_weapon else 65) + danger(c["pos"]), ("k", c["id"]), c, "crate", 18.0))
    for m in o.get("mechs", []):
        if m.get("kind") == "npc":
            d = math.hypot(m["pos"][0]-px, m["pos"][1]-py)
            if d < 75 and valid(m["pos"], ("n", m["id"]), "npc"):
                cands.append((d + 105 + danger(m["pos"]), ("n", m["id"]), m, "npc", 28.0))

    if cands:
        cands.sort(key=lambda v: v[0])
        # Keep pursuing a reasonable target: otherwise scores change slightly
        # each frame and the mech oscillates between several nearby objects.
        chosen = cands[0]
        for candidate in cands:
            if candidate[1] == chase["key"] and candidate[0] <= chosen[0] + 40:
                chosen = candidate
                break
        _, key, tgt, kind, standoff = chosen
        tpos = tgt["pos"]
        dx, dy = tpos[0] - px, tpos[1] - py
        d = math.hypot(dx, dy)
        if chase["key"] != key:
            chase.update(key=key, best=d, t=now, start=now)
        elif d < chase["best"] - 1.0:
            chase.update(best=d, t=now)
        elif d > standoff + 5 and now - chase["t"] > 3.5:
            ignore[key] = now + 22
            chase["key"] = None
            return False
        if kind in ("crate", "npc") and now - chase["start"] > 16:
            ignore[key] = now + 30
            chase["key"] = None
            return False
        chase["tz"] = tpos[2]
        if kind == "pickup":
            if d < 3.5:
                act["take"] = tgt["id"]
            act["move"] = [dx / d, dy / d] if d > 2.2 else [0, 0]
            return True
        if kind == "cube":
            act["move"] = [dx / d, dy / d] if d > 1.5 else [0, 0]
            return True
        gun = me["weapons"][0]
        speed = gun["stats"].get("speed", 252) if gun else 252
        yaw, pitch = aim_at(me, tpos, tgt.get("vel", [0, 0, 0]), speed, 1.6 if kind == "npc" else 1.2)
        act["aim"] = [yaw, pitch]
        yaw_ok = abs(wrap(yaw - me["aim"][0])) < 0.075
        pitch_ok = abs(pitch - me["aim"][1]) < 0.085
        clear = path_clear(px, py, tpos[0], tpos[1], max_len=120)
        act["fire"] = [bool(gun and gun.get("reload", 0) <= 0 and gun.get("ammo", 0) > 0
                            and yaw_ok and pitch_ok and d < 110 and clear), False]
        act["move"] = [dx / d, dy / d] if d > standoff and d > 0.01 else [0, 0]
        return True

    # Keep V4's exploration behavior when no reachable loot is visible.
    if need_weapon and now < 150:
        wp = wander["pt"]
        if wp is None or now > wander["until"] or math.hypot(wp[0] - px, wp[1] - py) < 10:
            a = RNG.uniform(0, 2 * math.pi)
            r = 0.75 * nr * math.sqrt(RNG.random())
            wander["pt"] = (cx + r * math.cos(a), cy + r * math.sin(a))
            wander["until"] = now + 10
            wp = wander["pt"]
        dx, dy = wp[0] - px, wp[1] - py
        d = math.hypot(dx, dy) or 1.0
        act["move"] = [dx / d, dy / d]
        return True
    return False


# ----------------------------------------------------------------------------
# Map: terrain heights + building prisms (v2/world.json, v2/terrain.i16)
# ----------------------------------------------------------------------------
CELL = 32.0


def pip(x, y, pts):
    """Point in polygon (ray casting)."""
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi:
            inside = not inside
        j = i
    return inside


def _pairs(flat):
    if flat and isinstance(flat[0], (list, tuple)):
        return [(p[0], p[1]) for p in flat]
    return list(zip(flat[0::2], flat[1::2]))


def load_map(hello):
    try:
        d = hello["map"]["dir"]
        with open(os.path.join(d, "v2", "world.json")) as f:
            w = json.load(f)
        t = w["terrain"]
        arr = array("h")
        with open(os.path.join(d, "v2", "terrain.i16"), "rb") as f:
            arr.frombytes(f.read())
        if sys.byteorder == "big":
            arr.byteswap()
        nx, ny = int(t["nx"]), int(t["ny"])
        if len(arr) < nx * ny:
            raise ValueError("terrain file too small")
        prisms, grid = [], {}
        for pr in w.get("prisms", []):
            pts = _pairs(pr["ring"])
            if len(pts) < 3:
                continue
            hraw = pr.get("holes") or []
            if hraw and isinstance(hraw[0], (int, float)):
                hraw = [hraw]                          # a single flat ring
            holes = [_pairs(h) for h in hraw if h]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            idx = len(prisms)
            prisms.append((min(xs), min(ys), max(xs), max(ys), pr["base"], pr["top"], pts, holes))
            for gx in range(int(min(xs) // CELL), int(max(xs) // CELL) + 1):
                for gy in range(int(min(ys) // CELL), int(max(ys) // CELL) + 1):
                    grid.setdefault((gx, gy), []).append(idx)
        MAP.update(ok=True, arr=arr, nx=nx, ny=ny, minX=float(t["minX"]), minY=float(t["minY"]),
                   step=float(t.get("step", 1.0)), prisms=prisms, grid=grid)
    except Exception as e:
        MAP["ok"] = False


def in_terrain(x, y):
    m = MAP
    return (m["minX"] <= x <= m["minX"] + (m["nx"] - 1) * m["step"]
            and m["minY"] <= y <= m["minY"] + (m["ny"] - 1) * m["step"])


def ground(x, y):
    m = MAP
    nx, ny = m["nx"], m["ny"]
    fx = min(max((x - m["minX"]) / m["step"], 0.0), nx - 1.001)
    fy = min(max((y - m["minY"]) / m["step"], 0.0), ny - 1.001)
    i, j = int(fx), int(fy)
    tx, ty = fx - i, fy - j
    a, b = m["arr"], j * nx + i
    return ((a[b] * (1 - tx) + a[b + 1] * tx) * (1 - ty)
            + (a[b + nx] * (1 - tx) + a[b + nx + 1] * tx) * ty) / 100.0


def ground_z(x, y, default):
    return ground(x, y) if MAP["ok"] and in_terrain(x, y) else default


def inside_prism(x, y, z):
    for idx in MAP["grid"].get((int(x // CELL), int(y // CELL)), ()):
        x0, y0, x1, y1, base, top, pts, holes = MAP["prisms"][idx]
        if x < x0 or x > x1 or y < y0 or y > y1 or z < base or z > top:
            continue
        if pip(x, y, pts) and not any(pip(x, y, h) for h in holes):
            return True
    return False


def los_blocked(a, b):
    """True if terrain or a building is between points a and b (trees ignored: unreliable cover)."""
    dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    n = max(2, int(math.sqrt(dx * dx + dy * dy + dz * dz) / 1.5))
    for k in range(1, n):
        f = k / n
        x, y, z = a[0] + dx * f, a[1] + dy * f, a[2] + dz * f
        if in_terrain(x, y) and z < ground(x, y):
            return True
        if inside_prism(x, y, z):
            return True
    return False


def solid_at(x, y):
    g = ground(x, y)
    return inside_prism(x, y, g + 1.5) or inside_prism(x, y, g + 8.5)


def path_clear(ax, ay, bx, by, max_len=None):
    """Check the requested segment without stepping past its endpoint."""
    full = math.hypot(bx - ax, by - ay)
    if full < 1e-6:
        return True
    d = min(full, max_len) if max_len is not None else full
    n = max(1, int(math.ceil(d / 2.0)))
    ux, uy = (bx - ax) / full, (by - ay) / full
    for k in range(1, n + 1):
        step = d * k / n
        x, y = ax + ux * step, ay + uy * step
        if not in_terrain(x, y) or solid_at(x, y):
            return False
    return True


def nav_passable(x, y, z0):
    """Conservative collision test: mech radius, height, and terrain steps."""
    if not in_terrain(x, y):
        return False
    gz = ground(x, y)
    if gz - z0 > 1.45:  # world.rules.maxStep is 1.5 m
        return False
    for ox, oy in ((0, 0), (2.7, 0), (-2.7, 0), (0, 2.7), (0, -2.7),
                   (1.9, 1.9), (-1.9, 1.9), (1.9, -1.9), (-1.9, -1.9)):
        xx, yy = x + ox, y + oy
        if not in_terrain(xx, yy):
            return False
        if inside_prism(xx, yy, gz + 1.5) or inside_prism(xx, yy, gz + 8.5):
            return False
    return True


def steer_around(o, me, desired):
    """Local obstacle avoidance for every movement mode, with a stable detour side.

    Tests short candidate arcs instead of requiring a straight path to a far goal.
    This is local steering, not a full global pathfinder.
    """
    if not MAP["ok"]:
        return desired
    vx, vy = desired
    mag = math.hypot(vx, vy)
    if mag < 0.1:
        return desired
    vx, vy = vx / mag, vy / mag
    px, py, pz = me["pos"]
    now = o["t"]
    if nav["last_pos"] is not None and now - nav["last_t"] >= 0.45:
        travelled = math.hypot(px - nav["last_pos"][0], py - nav["last_pos"][1])
        nav["stuck"] = nav["stuck"] + 1 if travelled < 1.3 else 0
        nav["last_pos"], nav["last_t"] = (px, py), now
    elif nav["last_pos"] is None:
        nav["last_pos"], nav["last_t"] = (px, py), now

    # Avoid steering into the imminent zone edge, but allow escape if already outside.
    z = o["zone"]
    cx, cy = z["next_center"]
    limit = max(0.0, z["next_radius"] - 5.0)
    currently_outside = math.hypot(px - cx, py - cy) > limit
    base = math.atan2(vy, vx)
    offsets = (0, 0.30, -0.30, 0.60, -0.60, 0.95, -0.95,
               1.35, -1.35, 1.75, -1.75, 2.3, -2.3, math.pi)
    best = None
    for off in offsets:
        angle = base + off
        ux, uy = math.cos(angle), math.sin(angle)
        free = 0.0
        prev_z = ground_z(px, py, pz)
        for dist in (3.5, 7.0, 11.0, 16.0):
            x, y = px + ux * dist, py + uy * dist
            if not nav_passable(x, y, prev_z):
                break
            if not currently_outside and math.hypot(x - cx, y - cy) > limit:
                break
            if currently_outside and math.hypot(x - cx, y - cy) > math.hypot(px - cx, py - cy) + 2:
                break
            prev_z = ground(x, y)
            free = dist
        if free < 3.5:
            continue
        # Favor forward progress; commit to a side when detouring to avoid oscillation.
        score = free * 0.11 + 5.0 * math.cos(off) - 0.5 * abs(off)
        if now < nav["until"] and off * nav["side"] > 0:
            score += 1.5
        if nav["stuck"] >= 2 and abs(off) > 0.6:
            score += 3.0
        if best is None or score > best[0]:
            best = (score, ux, uy, off)
    if best is None:
        return [0.0, 0.0]
    if abs(best[3]) > 0.3:
        nav["side"] = 1 if best[3] > 0 else -1
        nav["until"] = now + 2.0
    return [best[1] * mag, best[2] * mag]


def plan_cover(o, me, threats, cx, cy, nr, force=False):
    """Pick a nearby spot hidden from every threat (else the farthest reachable one)."""
    if not MAP["ok"] or not threats:
        return None
    now = o["t"]
    if (not force) and cover["pt"] is not None and now - cover["plan_t"] < 0.8:
        return cover["pt"]
    t0 = time.perf_counter()
    px, py, pz = me["pos"]
    sz = 1 + 0.05 * me.get("cubes", 0)
    thr = sorted(threats, key=lambda t: math.hypot(t[0] - px, t[1] - py))[:2]
    d0 = math.hypot(thr[0][0] - px, thr[0][1] - py)
    best_h = best_f = None
    for r in COVER_RADII:
        for k in range(COVER_ANGLES):
            if time.perf_counter() - t0 > 0.02:
                break                                   # keep inside the 50 ms decision budget
            a = k * 2 * math.pi / COVER_ANGLES
            x, y = px + r * math.cos(a), py + r * math.sin(a)
            if math.hypot(x - cx, y - cy) > 0.85 * nr or not in_terrain(x, y) or solid_at(x, y):
                continue
            dt = min(math.hypot(x - t[0], y - t[1]) for t in thr)
            if dt < d0 - 5:
                continue                                # never retreat toward the threat
            if not path_clear(px, py, x, y):
                continue
            g = ground(x, y)
            hidden = all(los_blocked(t, (x, y, g + 4.5 * sz)) and los_blocked(t, (x, y, g + 9 * sz))
                         for t in thr)
            if hidden:
                cost = r + 0.3 * max(0.0, 80.0 - dt)
                if best_h is None or cost < best_h[0]:
                    best_h = (cost, (x, y))
            else:
                far = dt - 0.2 * r
                if best_f is None or far > best_f[0]:
                    best_f = (far, (x, y))
    pick = best_h or best_f
    cover["plan_t"] = now
    cover["pt"] = pick[1] if pick else None
    cover["hidden"] = best_h is not None
    return cover["pt"]


def retreat_vec(o, me, threats, cx, cy, nr):
    """Move vector toward cover, [0,0] when already there, None if no plan (caller falls back)."""
    pt = plan_cover(o, me, threats, cx, cy, nr)
    if pt is None:
        return None
    dx, dy = pt[0] - me["pos"][0], pt[1] - me["pos"][1]
    d = math.hypot(dx, dy)
    return [0, 0] if d < 3.0 else [dx / d, dy / d]


def laser_dodge(o, me, enemies):
    """Sidestep (boosted) when an enemy's charging laser is lined up on us."""
    px, py = me["pos"][0], me["pos"][1]
    now = o["t"]
    near = None
    for m in enemies:
        if not m.get("charging"):
            continue
        dx, dy = px - m["pos"][0], py - m["pos"][1]
        d = math.hypot(dx, dy) or 1.0
        if d < 290 and abs(wrap(m["aim"][0] - math.atan2(dy, dx))) < math.atan2(4.0, d) + 0.08:
            if near is None or d < near[0]:
                near = (d, m)
    if near is None:
        return None
    m = near[1]
    ux, uy = math.cos(m["aim"][0]), math.sin(m["aim"][0])
    if now > dodge["until"]:                    # pick a side at the start of a dodge, then commit
        best_side, best_free = dodge["side"], -1.0
        for side in (1, -1):
            vx, vy = -uy * side, ux * side
            free = 12.0
            if MAP["ok"]:
                free = 0.0
                while free < 12.0 and in_terrain(px + vx * (free + 2), py + vy * (free + 2)) \
                        and not solid_at(px + vx * (free + 2), py + vy * (free + 2)):
                    free += 2.0
            if free > best_free:
                best_side, best_free = side, free
        dodge["side"] = best_side
    dodge["until"] = now + 0.5
    s = dodge["side"]
    return [-uy * s, ux * s]


def mobility(o, me, act, want_up, fleeing, hit_now):
    """Jump when we are pushing against something and not moving; jet if the goal is higher."""
    now = o["t"]
    mv = act.get("move", [0, 0])
    wants_move = math.hypot(mv[0], mv[1]) > 0.5
    spd = math.hypot(me["vel"][0], me["vel"][1])
    if me["grounded"]:
        if wants_move and spd < 2.0:
            mob["blocked"] += 1
        else:
            mob["blocked"] = 0
        if now >= mob["jump_cd"]:
            if mob["blocked"] >= 4:                  # ~0.4 s of pushing without moving
                act["jump"] = True
                mob["jump_cd"] = now + 1.0
                mob["blocked"] = 0
                if want_up:
                    mob["climb_until"] = now + 1.2
            elif HOP_ON_HIT and fleeing and hit_now and spd > 4.0:
                act["jump"] = True
                mob["jump_cd"] = now + 2.0
    elif now < mob["climb_until"] and me["energy"] > 25:
        act["boost"] = True                          # in the air: jet until the feet clear the edge


def decide(o):
    me = o["self"]
    px, py, pz = me["pos"]
    act = {"tick": o["tick"]}
    held = [w["kind"] if w else None for w in me["weapons"]]

    if o.get("offer"):
        act["pick"] = pick_upgrade(o["offer"], held, me)

    update_memory(o, px, py, pz)
    now = o["t"]
    hit_now = any(e.get("type") == "damaged" and e.get("by") is not None
                  for e in o.get("events", []))
    fleeing = False
    want_up = False

    # --- zone: head for the next circle, stop once well inside it ---
    z = o["zone"]
    cx, cy = z["next_center"]
    zd = math.hypot(cx - px, cy - py) or 1.0
    inside = me["in_zone"] and zd < 0.85 * z["next_radius"]
    if (not inside) or zd > 0.5 * z["next_radius"]:
        act["move"] = [(cx - px) / zd, (cy - py) / zd]
    else:
        act["move"] = [0, 0]

    gun = me["weapons"][0]
    w1 = me["weapons"][1]
    enemies = [m for m in o["mechs"] if m["kind"] == "mech" and m["id"] != me["id"]]
    firing_gun = False

    if enemies:
        t = min(enemies, key=lambda m: math.hypot(m["pos"][0] - px, m["pos"][1] - py))
        dx, dy = t["pos"][0] - px, t["pos"][1] - py
        dist = math.hypot(dx, dy) or 1.0
        speed = gun["stats"]["speed"] if gun else 252
        kind1 = w1["kind"] if w1 else None

        # Baseline V1 survival thresholds retained; do not repeat V2's broad retreat policy.
        # Distant enemies should not prevent safe farming and collecting upgrades.
        if dist > 75 and not hit_now and now < 180 and (inside or (me["in_zone"] and now < z.get("shrink_start", 0) - 20)):
            if loot(o, me, act, held, z, cx, cy):
                act["move"] = steer_around(o, me, act.get("move", [0, 0]))
                mobility(o, me, act, chase["tz"] is not None and chase["tz"] - pz > 1.0,
                         False, hit_now)
                if gun and gun["reload"] == 0 and gun["ammo"] == 0:
                    act["reload"] = True
                return act

        # early game: avoid enemies that are healthier than us (needs enemy hp in obs)
        threats = [m for m in enemies
                   if m.get("hp") is not None and me["hp"] < HP_AVOID * m["hp"]
                   and math.hypot(m["pos"][0] - px, m["pos"][1] - py) < 150]
        avoid = o["t"] < EARLY_S and bool(threats)

        yaw_direct = math.atan2(dy, dx)
        yaw_err = abs(wrap(yaw_direct - me["aim"][0]))

        # laser is hitscan: aim straight at the target instead of leading
        lead = not (kind1 == "laser" and w1["cooldown"] < 0.3)
        yaw, pitch = aim_at(me, t["pos"], t["vel"], speed, 4.5, lead)
        lob = False
        if kind1 == "mortar" and w1["cooldown"] <= 0.05 and 15 < dist < 200:
            v = w1["stats"].get("speed", 72)
            g = CFG.get("weapons", {}).get("mortar", {}).get("gravity", 22)
            s = g * dist / (v * v)
            if s < 1:
                th = 0.5 * math.asin(s)
                ft = dist / (v * math.cos(th))
                yaw = math.atan2(t["pos"][1] + t["vel"][1] * ft - py,
                                 t["pos"][0] + t["vel"][0] * ft - px)
                pitch = th
                lob = True
        act["aim"] = [yaw, pitch]

        fire1 = fire_second(me, w1, dist, yaw_err)
        if lob:
            fire1 = (abs(wrap(yaw - me["aim"][0])) < 0.08 and abs(pitch - me["aim"][1]) < 0.05)
            firing_gun = False
            fire0 = False
        else:
            fire0 = abs(wrap(yaw - me["aim"][0])) < 0.1
            firing_gun = fire0
        if avoid and dist >= 40:       # don't provoke or reset regen unless they are on top of us
            fire0 = fire1 = False
        act["fire"] = [fire0, fire1]

        # strafe, close in with lightning, or back off when low on HP
        if inside:
            if avoid:
                fleeing = True
                tt = min(threats, key=lambda m: math.hypot(m["pos"][0] - px, m["pos"][1] - py))
                flee_mem["until"] = now + FLEE_KEEP_S
                flee_mem["from"] = (tt["pos"][0], tt["pos"][1])
                fx, fy = px - tt["pos"][0], py - tt["pos"][1]
                fn = math.hypot(fx, fy) or 1.0
                fx, fy = fx / fn, fy / fn
                if zd > 0.6 * z["next_radius"]:        # keep drifting toward the circle center
                    fx += 0.8 * (cx - px) / zd
                    fy += 0.8 * (cy - py) / zd
                    fn = math.hypot(fx, fy) or 1.0
                    fx, fy = fx / fn, fy / fn
                tl = [(m["pos"][0], m["pos"][1], m["pos"][2] + 8 * (1 + 0.05 * m.get("cubes", 0)))
                      for m in threats]
                rv = retreat_vec(o, me, tl, cx, cy, z["next_radius"])
                if rv is not None:
                    cover["hold_until"] = now + HEAL_HOLD_S
                act["move"] = rv if rv is not None else [fx, fy]
            elif me["hp"] < 0.35 * me["max_hp"]:
                fleeing = True
                flee_mem["until"] = now + FLEE_KEEP_S
                flee_mem["from"] = (t["pos"][0], t["pos"][1])
                tl = [(m["pos"][0], m["pos"][1], m["pos"][2] + 8 * (1 + 0.05 * m.get("cubes", 0)))
                      for m in enemies
                      if m is t or math.hypot(m["pos"][0] - px, m["pos"][1] - py) < 150]
                rv = retreat_vec(o, me, tl, cx, cy, z["next_radius"])
                if rv is not None:
                    cover["hold_until"] = now + HEAL_HOLD_S
                act["move"] = rv if rv is not None else [-dx / dist, -dy / dist]
            else:
                want_up = t["pos"][2] - pz > 1.5       # enemy on higher ground: climb if blocked
                side = 1 if (o["tick"] // 15) % 2 else -1  # flip every 1.5 s
                cubes_near = [r["c"] for r in known_cubes.values()
                              if math.hypot(r["c"]["pos"][0] - px, r["c"]["pos"][1] - py) < 15
                              and abs(r["c"]["pos"][2] - pz) < 3]
                if cubes_near:            # strafe toward a nearby cube
                    cc = min(cubes_near, key=lambda c: math.hypot(c["pos"][0] - px, c["pos"][1] - py))
                    side = 1 if (-dy / dist * (cc["pos"][0] - px) + dx / dist * (cc["pos"][1] - py)) >= 0 else -1
                sx, sy = -dy / dist * side, dx / dist * side
                lrng = w1["stats"].get("range", 45) if kind1 == "lightning" else 45
                if kind1 == "lightning" and dist > 0.7 * lrng and me["hp"] > 0.5 * me["max_hp"]:
                    mx, my = 0.8 * dx / dist + 0.6 * sx, 0.8 * dy / dist + 0.6 * sy
                    n = math.hypot(mx, my) or 1.0
                    act["move"] = [mx / n, my / n]
                else:
                    act["move"] = [sx, sy]
        dv = laser_dodge(o, me, enemies)
        if dv is not None:                  # a charging laser is lined up on us: sprint sideways
            act["move"] = dv
            if me["energy"] > 25:
                act["boost"] = True
    else:
        # shot by a mech we cannot see: treat the bearing as a threat and take cover
        if inside:
            for e in o.get("events", []):
                if (e.get("type") == "damaged" and e.get("bearing") is not None
                        and e.get("by") is not None and e["by"] < 16):   # ids 0-15 are mechs
                    flee_mem["until"] = now + FLEE_KEEP_S
                    flee_mem["from"] = (px + 80 * math.cos(e["bearing"]),
                                        py + 80 * math.sin(e["bearing"]))
                    cover["plan_t"] = -99.0               # force a fresh plan
                    cover["hold_until"] = now + HEAL_HOLD_S
        if inside and flee_mem["from"] is not None and now < flee_mem["until"]:
            # lost sight of a threat: go to cover (or keep running from where it was)
            fleeing = True
            gx, gy = flee_mem["from"]
            rv = retreat_vec(o, me, [(gx, gy, ground_z(gx, gy, pz) + 8)], cx, cy, z["next_radius"])
            if rv is None:
                fx, fy = px - gx, py - gy
                fn = math.hypot(fx, fy) or 1.0
                fx, fy = fx / fn, fy / fn
                if zd > 0.6 * z["next_radius"]:
                    fx += 0.8 * (cx - px) / zd
                    fy += 0.8 * (cy - py) / zd
                    fn = math.hypot(fx, fy) or 1.0
                    fx, fy = fx / fn, fy / fn
                rv = [fx, fy]
            act["move"] = rv
            acted = True
        elif (inside and cover["pt"] is not None and now < cover["hold_until"]
              and me["hp"] < 0.9 * me["max_hp"]):
            # heal in cover: walk to the spot and wait there (regen needs ~5 s without damage)
            ddx, ddy = cover["pt"][0] - px, cover["pt"][1] - py
            dd = math.hypot(ddx, ddy)
            act["move"] = [ddx / dd, ddy / dd] if dd > 3.0 else [0, 0]
            acted = True
        else:
            acted = loot(o, me, act, held, z, cx, cy) if (inside or (me["in_zone"] and now < z.get("shrink_start", 0) - 20)) else False
            want_up = acted and chase["tz"] is not None and chase["tz"] - pz > 1.0
        if "fire" in act:
            firing_gun = act["fire"][0]
        if not acted:
            # nothing to do: face whoever just hit us
            hits = [e for e in o.get("events", [])
                    if e.get("type") == "damaged" and e.get("bearing") is not None]
            if hits:
                act["aim"] = [hits[-1]["bearing"], 0]

    # Apply obstacle avoidance to zone travel, loot chasing, combat and retreat.
    # Do not override intentional stationary actions (healing or shooting from cover).
    act["move"] = steer_around(o, me, act.get("move", [0, 0]))
    mobility(o, me, act, want_up, fleeing, hit_now)

    # reload when empty, or when idle and under half a magazine
    if gun and gun["reload"] == 0 and (
            gun["ammo"] == 0 or (not firing_gun and gun["ammo"] < gun["stats"]["mag"] // 2)):
        act["reload"] = True

    return act


def main():
    global CFG, RNG
    threading.Thread(target=reader, daemon=True).start()
    hello = q.get()
    if hello is None:
        return
    CFG = hello.get("config", {})
    RNG = random.Random(hello.get("seed", 0))
    load_map(hello)
    send({"type": "ready", "name": "simple-bot"})

    while True:
        m = q.get()
        # skip ahead to the newest queued message
        while m is not None and m.get("type") != "end":
            try:
                n = q.get_nowait()
            except queue.Empty:
                break
            m = n
            if n is None:
                break
        if m is None or m.get("type") == "end":
            break
        if m.get("type") == "obs":
            try:
                act = decide(m)
            except Exception as e:
                act = {"tick": m["tick"]}
            send(act)


if __name__ == "__main__":
    main()
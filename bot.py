import json
import math
import os
import queue
import random
import sys
import threading
import time
from array import array

q = queue.Queue()

# Offline debugging only: set MECH_LOG=log.jsonl when running check.py.
# Unset in real matches, so nothing is opened on the read-only disk.
LOG = open(os.environ["MECH_LOG"], "a") if os.environ.get("MECH_LOG") else None

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
    """Weapon pickups > cubes > NPC/crate farming > wander. Returns True if it acted."""
    px, py, pz = me["pos"]
    now = o["t"]
    chase["tz"] = None
    need_weapon = me["weapons"][1] is None
    safe_r = 0.9 * z["next_radius"]

    def ok(pos, key, slope=False):
        if ignore.get(key, 0) > now:
            return False
        dz = abs(pos[2] - pz)
        if slope:                     # allow gentle terrain over distance
            dz -= 0.25 * math.hypot(pos[0] - px, pos[1] - py)
        if dz > 2.5:                  # roofs etc.: we cannot walk up there
            return False
        return math.hypot(pos[0] - cx, pos[1] - cy) < safe_r

    cands = []   # (score, key, pos, kind, standoff)
    if need_weapon:
        for pid, p in known.items():
            if p["kind"] in held or not ok(p["pos"], ("p", pid)):
                continue
            d = math.hypot(p["pos"][0] - px, p["pos"][1] - py)
            if d < 150:
                cands.append((d + 40 * WEAPON_RANK.get(p["kind"], 3), ("p", pid), p, "pickup", 1.0))
    for rec in known_cubes.values():
        c = rec["c"]
        d = math.hypot(c["pos"][0] - px, c["pos"][1] - py)
        if d < 80 and ok(c["pos"], ("c", c["id"]), slope=True):
            cands.append((200 + d, ("c", c["id"]), c, "cube", 0.5))
    for m in o.get("mechs", []):
        if m["kind"] == "npc":
            d = math.hypot(m["pos"][0] - px, m["pos"][1] - py)
            if d < 70 and ok(m["pos"], ("n", m["id"])):
                cands.append((300 + d, ("n", m["id"]), m, "npc", 30.0))
    for c in o.get("crates", []):
        d = math.hypot(c["pos"][0] - px, c["pos"][1] - py)
        if d < 70 and ok(c["pos"], ("k", c["id"])):
            cands.append((310 + d, ("k", c["id"]), c, "crate", 20.0))

    if cands:
        cands.sort(key=lambda c: c[0])
        _, key, tgt, kind, standoff = cands[0]
        tpos = tgt["pos"]
        chase["tz"] = tpos[2]
        dx, dy = tpos[0] - px, tpos[1] - py
        d = math.hypot(dx, dy) or 1.0

        # stuck / timeout detection (no pathfinding yet)
        if chase["key"] != key:
            chase.update(key=key, best=d, t=now, start=now)
        elif d < chase["best"] - 1.5:
            chase.update(best=d, t=now)
        elif d > standoff + 5 and now - chase["t"] > 3.0:
            ignore[key] = now + 30
            chase["key"] = None
        if kind in ("npc", "crate") and now - chase["start"] > 12.0:
            ignore[key] = now + 40
            chase["key"] = None

        if kind == "pickup":
            if d < 3.0:
                act["take"] = tgt["id"]
            act["move"] = [dx / d, dy / d] if d > 2.0 else [0, 0]
            return True
        if kind == "cube":
            act["move"] = [dx / d, dy / d]
            return True
        # npc / crate: close to standoff range and shoot the gun
        gun = me["weapons"][0]
        speed = gun["stats"]["speed"] if gun else 252
        zoff = 1.75 if kind == "npc" else 1.5
        tv = tgt.get("vel", [0, 0, 0])
        yaw, pitch = aim_at(me, tpos, tv, speed, zoff)
        act["aim"] = [yaw, pitch]
        act["fire"] = [abs(wrap(yaw - me["aim"][0])) < 0.1, False]
        act["move"] = [dx / d, dy / d] if d > standoff else [0, 0]
        return True

    # nothing to go for: look for a second weapon by wandering inside the zone
    if need_weapon and now < 150:
        wp = wander["pt"]
        if wp is None or now > wander["until"] or math.hypot(wp[0] - px, wp[1] - py) < 10:
            a = RNG.uniform(0, 2 * math.pi)
            r = 0.75 * z["next_radius"] * math.sqrt(RNG.random())
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
        print("map loaded: %d prisms, terrain %dx%d" % (len(prisms), nx, ny), file=sys.stderr)
    except Exception as e:
        MAP["ok"] = False
        print("map NOT loaded (retreat uses plain flee):", repr(e), file=sys.stderr)


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
    full = math.hypot(bx - ax, by - ay) or 1.0
    d = min(full, max_len) if max_len is not None else full
    n = max(1, int(d / 2.0))
    ux, uy = (bx - ax) / full, (by - ay) / full
    for k in range(1, n + 1):
        x, y = ax + ux * 2.0 * k, ay + uy * 2.0 * k
        if not in_terrain(x, y) or solid_at(x, y):
            return False
    return True


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
            acted = loot(o, me, act, held, z, cx, cy) if inside else False
            want_up = acted and chase["tz"] is not None and chase["tz"] - pz > 1.0
        if "fire" in act:
            firing_gun = act["fire"][0]
        if not acted:
            # nothing to do: face whoever just hit us
            hits = [e for e in o.get("events", [])
                    if e.get("type") == "damaged" and e.get("bearing") is not None]
            if hits:
                act["aim"] = [hits[-1]["bearing"], 0]

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
    if LOG:
        LOG.write(json.dumps({"hello": hello, "map_ok": MAP["ok"]}) + "\n")
        LOG.flush()
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
                print("error:", repr(e), file=sys.stderr)
                act = {"tick": m["tick"]}
            send(act)
            if LOG:
                LOG.write(json.dumps({"obs": m, "act": act}) + "\n")
                LOG.flush()


if __name__ == "__main__":
    main()
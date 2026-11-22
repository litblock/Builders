import heapq
import json
import math
import os
import queue
import random
import sys
import threading
import time
from array import array
from collections import deque
from typing import NamedTuple

q = queue.Queue()

CFG = {}
RNG = random.Random(0)

known = {}       # weapon pickup id -> pickup dict (remembered after it leaves view)
known_cubes = {}  # cube id -> {"c": cube dict, "t": last seen match time}
ignore = {}      # (type, id) -> match time until which we skip that target
chase = {"key": None, "best": 1e9, "t": 0.0, "start": 0.0, "tz": None, "goal": None}
wander = {"pt": None, "until": 0.0}

EARLY_S = 120     # "early game": avoid fights against healthier enemies before this match time
HP_AVOID = 0.8    # avoid an enemy whose HP is more than 1/0.8 = 1.25x ours
FLEE_KEEP_S = 3.0  # keep running from the last threat this long after losing sight of it
flee_mem = {"until": 0.0, "from": None}
COVER_RADII = (10.0, 20.0, 35.0, 50.0, 70.0)   # how far from us to look for cover (m)
COVER_ANGLES = 24                  # directions tried per radius
HEAL_HOLD_S = 25.0  # after a retreat, wait in cover this long (or until ~90% HP)
cover = {"pt": None, "plan_t": -99.0, "hold_until": 0.0, "hidden": False}
dodge = {"until": 0.0, "side": 1}
MAP = {"ok": False}
mob = {"blocked": 0, "jump_cd": 0.0, "climb_until": 0.0,
       "sprint_until": 0.0, "sprint_ready": 0.0,
       "strafe_side": 1, "strafe_until": 0.0}
nav = {"side": 1, "until": 0.0, "last_pos": None, "stuck": 0, "last_t": -1.0}
route = {"key": None, "goal": None, "waypoints": deque(), "retry": 0.0,
         "best": math.inf, "progress": 0.0, "next_limit": 0.0}
zone_loot = {"id": None, "until": 0.0, "retry": 0.0, "hit_until": 0.0}
ZONE_LOOT_S = 1.0
ZONE_LOOT_RESERVE_S = 4.0

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


def aim_toward(me, pos, height):
    """Face a target while no equipped weapon has a valid shot."""
    px, py, pz = me["pos"]
    x, y, z = pos
    yaw = math.atan2(y - py, x - px)
    pitch = math.atan2(z + height - pz - me["stats"].get("muzzle_height", 7),
                       math.hypot(x - px, y - py))
    return yaw, pitch


class Shot(NamedTuple):
    yaw: float
    pitch: float
    yaw_limit: float
    pitch_limit: float


def aligned(shot, aim):
    return (abs(wrap(shot.yaw - aim[0])) <= shot.yaw_limit
            and abs(shot.pitch - aim[1]) <= shot.pitch_limit)


def target_position(target, flight, height):
    x, y, z = target["pos"]
    vx, vy, vz = target.get("vel", (0, 0, 0))
    x, y = x + vx * flight, y + vy * flight
    # Vertical velocity is noisy after a fall or a jump. Limit extrapolation.
    if flight:
        z += max(-7.0, min(7.0, vz * min(flight, 0.4)))
        z = max(z, ground_z(x, y, z))
    return x, y, z + height


def arc_clear(origin, direction, offset, speed, gravity, flight):
    if not MAP["ok"]:
        return True
    # Include the barrel sweep and then sample the shell's curved flight.
    muzzle_tip = tuple(origin[i] + offset * direction[i] for i in range(3))
    if los_blocked(origin, muzzle_tip):
        return False
    steps = max(3, math.ceil(flight * speed / 2.0))
    for i in range(steps):
        t = flight * i / steps
        travel = offset + speed * t
        x = origin[0] + travel * direction[0]
        y = origin[1] + travel * direction[1]
        z = origin[2] + travel * direction[2] - 0.5 * gravity * t * t
        if ((in_terrain(x, y) and ground(x, y) > z) or inside_prism(x, y, z)):
            return False
    return True


def weapon_shot(me, weapon, target):
    """Determine an aim and tolerance for the equipped weapon's own geometry."""
    kind = weapon["kind"]
    config = CFG.get("weapons", {}).get(kind, {})
    stats = {**config, **weapon["stats"]}
    body = me["stats"]
    target_kind = target.get("kind", "crate")
    scale = 1.0 + 0.05 * target.get("cubes", 0)
    radius, height = {"mech": (2.5, 9.0), "npc": (1.8, 3.5),
                      "crate": (1.4, 2.4)}[target_kind]
    radius *= scale
    height *= scale
    offset = body.get("radius", 2.5) + 0.5
    px, py, pz = me["pos"]
    muzzle_height = (body.get("height", 9.0) + 0.5 if kind == "mortar"
                     else body.get("muzzle_height", 7.0))
    origin = px, py, pz + muzzle_height
    pitch_min, pitch_max = body.get("min_pitch", -0.7), body.get("max_pitch", 1.3)

    if kind == "mortar":
        horizontal = math.dist(origin[:2], target["pos"][:2])
        splash = stats.get("splash_radius", 7.0)
        # Give impact prediction room for the bot's movement during flight.
        if horizontal < splash + 8.0:
            return None
        speed, gravity = stats.get("speed", 72.0), stats.get("gravity", 22.0)
        vx, vy, _ = target.get("vel", (0, 0, 0))
        hit_height = min(1.5, 0.5 * height)

        def displacement(t):
            x, y, z = target_position(target, t, hit_height)
            return x - px, y - py, z - origin[2] + 0.5 * gravity * t * t

        def gap(t):
            return math.hypot(*displacement(t)) - (offset + speed * t)

        # Find low and high arcs by bracketing both roots of the flight equation.
        last_t = 0.0
        last_gap = gap(last_t)
        max_flight = min(stats.get("ttl", 7.0), 7.0)
        first_t = max(0.06, (horizontal - offset) / (speed + math.hypot(vx, vy)))
        count = max(0, math.ceil((max_flight - first_t) / 0.15))
        for i in range(count + 1):
            t = min(max_flight, first_t + 0.15 * i)
            if t < first_t:
                break
            error = gap(t)
            if (error <= 0) != (last_gap <= 0):
                low, high = last_t, t
                for _ in range(9):
                    middle = (low + high) / 2
                    if (gap(middle) <= 0) == (last_gap <= 0):
                        low = middle
                    else:
                        high = middle
                flight = (low + high) / 2
                dx, dy, dz = displacement(flight)
                yaw = math.atan2(dy, dx)
                pitch = math.atan2(dz, math.hypot(dx, dy))
                length = math.hypot(dx, dy, dz)
                direction = dx / length, dy / length, dz / length
                safe_impact = math.hypot(dx, dy) > splash + 8.0
                if (safe_impact and pitch_min <= pitch <= pitch_max
                        and arc_clear(origin, direction, offset, speed, gravity, flight)):
                    # Pitch errors move the impact point more than yaw errors.
                    tolerance = min(0.07, math.atan2(0.55 * splash, horizontal))
                    return Shot(yaw, pitch, tolerance, max(0.012, tolerance * 0.55))
            last_t, last_gap = t, error
        return None

    if kind == "lightning":
        point = target_position(target, 0.0, 0.55 * height)
        dx, dy, dz = (point[i] - origin[i] for i in range(3))
        if math.hypot(dx, dy) > stats.get("range", 45.0) - 1.5:
            return None
        if MAP["ok"] and los_blocked(origin, point):
            return None
        pitch = max(pitch_min, min(pitch_max, math.atan2(dz, math.hypot(dx, dy))))
        return Shot(math.atan2(dy, dx), pitch, math.radians(stats.get("cone_deg", 30) * 0.43),
                    math.pi)

    if kind not in ("gun", "laser"):
        return None
    point = target_position(target, 0.0, 0.55 * height)
    if kind == "gun":
        speed = stats.get("speed", 252.0)
        for _ in range(4):
            flight = max(0.0, (math.dist(origin, point) - offset) / speed)
            point = target_position(target, flight, 0.55 * height)
    max_range = stats.get("range", 230.0 if kind == "gun" else 280.0)
    if math.dist(origin, point) - offset > max_range:
        return None
    dx, dy, dz = (point[i] - origin[i] for i in range(3))
    horizontal = math.hypot(dx, dy)
    pitch = math.atan2(dz, horizontal)
    if not pitch_min <= pitch <= pitch_max or (MAP["ok"] and los_blocked(origin, point)):
        return None
    return Shot(math.atan2(dy, dx), pitch,
                min(0.18, math.atan2(0.95 * radius, horizontal)),
                min(0.18, math.atan2(0.42 * height, horizontal)))


def fire_weapons(me, act, target, allow_fire=True, allow_charge=True):
    """Coordinate the shared torso aim and independent trigger for each slot."""
    weapons = me["weapons"]
    current_aim = me["aim"]
    act["fire"] = [False, False]
    default_height = 4.5 if target.get("kind") == "mech" else 1.4
    act["aim"] = aim_toward(me, target["pos"], default_height)
    if not allow_fire:
        return

    shots = []
    ready = []
    for weapon in weapons:
        kind = weapon["kind"] if weapon else None
        available = bool(weapon and weapon["cooldown"] <= 0.06)
        if kind == "gun":
            available = available and weapon["ammo"] > 0 and weapon["reload"] <= 0
        elif kind == "lightning":
            available = available and me["energy"] >= weapon["stats"].get("energy", 3) + 10
        elif kind == "laser":
            available = available and (allow_charge or weapon.get("charge", 0) > 0)
        ready.append(available)
        # Wait until a shell is nearly ready before tracing an expensive arc.
        shots.append(weapon_shot(me, weapon, target)
                     if weapon and (kind != "mortar" or weapon["cooldown"] <= 0.25) else None)

    candidates = [i for i, shot in enumerate(shots) if shot and ready[i]]
    # An already-charging laser owns the aim through release. A compatible gun
    # or lightning shot can still fire on the same decision.
    charging = next((i for i in candidates if weapons[i].get("charge", 0) > 0), None)
    weights = {"gun": 1.0, "lightning": 1.5, "mortar": 2.6, "laser": 1.35}

    def score(i):
        shot = shots[i]
        compatible = sum(weights[weapons[j]["kind"]] for j in candidates
                         if aligned(shots[j], (shot.yaw, shot.pitch)))
        slew = abs(wrap(shot.yaw - current_aim[0])) + abs(shot.pitch - current_aim[1])
        # Use precise aim when lightning can share it with a narrow-angle weapon.
        return compatible - 0.55 * slew + (0.08 if weapons[i]["kind"] != "lightning" else 0.0)

    if charging is not None:
        selected = charging
    elif candidates:
        selected = max(candidates, key=score)
    else:
        # Track a valid shot through reload or cooldown.
        shot = next((shot for shot in shots if shot), None)
        if shot:
            act["aim"] = [shot.yaw, shot.pitch]
        return

    shot = shots[selected]
    act["aim"] = [shot.yaw, shot.pitch]
    for i in candidates:
        shot = shots[i]
        if not aligned(shot, act["aim"]):
            continue
        weapon = weapons[i]
        on_target = aligned(shot, current_aim)
        if weapon["kind"] == "laser":
            charge = weapon.get("charge", 0)
            full = weapon["stats"].get("charge", 0.75)
            if charge > 0:
                # Release an aligned beam early if we need to resume full-speed movement.
                act["fire"][i] = allow_charge and (charge < full - 0.06 or not on_target)
            else:
                act["fire"][i] = on_target and allow_charge
        else:
            act["fire"][i] = on_target


def reload_gun(me, act):
    fire = act.get("fire", (False, False))
    for i, weapon in enumerate(me["weapons"]):
        if weapon and weapon["kind"] == "gun" and weapon["reload"] == 0:
            ammo = weapon["ammo"]
            if ammo == 0 or (not fire[i] and ammo < weapon["stats"]["mag"] // 2):
                act["reload"] = True


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
    """Prioritize reachable pickups/cubes, actively break crates for loot, then farm NPCs."""
    px, py, pz = me["pos"]
    now = o["t"]
    chase["tz"] = None
    chase["goal"] = None
    need_weapon = None in me["weapons"]
    safe_r = 0.92 * z["next_radius"]

    def ok(pos, key, slope=False):
        if ignore.get(key, 0) > now:
            return False
        dz = abs(pos[2] - pz)
        if slope:                     # allow gentle terrain over distance
            dz -= 0.25 * math.hypot(pos[0] - px, pos[1] - py)
        if dz > 2.5:                  # roofs etc.: we cannot walk up there
            return False
        return math.hypot(pos[0] - cx, pos[1] - cy) < safe_r

    # Small, localized loot risk penalty; unlike V2, do not reject loot outright.
    nearby_players = [m for m in o.get("mechs", []) if m["kind"] == "mech"]
    def risk(pos):
        nearest = min((math.hypot(pos[0]-m["pos"][0], pos[1]-m["pos"][1])
                       for m in nearby_players), default=999.0)
        return 110.0 if nearest < 25 else (35.0 if nearest < 45 else 0.0)

    cands = []   # (score, key, pos, kind, standoff)
    if need_weapon:
        for pid, p in known.items():
            if p["kind"] in held or not ok(p["pos"], ("p", pid)):
                continue
            d = math.hypot(p["pos"][0] - px, p["pos"][1] - py)
            if d < 165:
                cands.append((10 + d + 18 * WEAPON_RANK.get(p["kind"], 3) + risk(p["pos"]), ("p", pid), p, "pickup", 1.0))
    # V8A single experiment: modestly prefer already-dropped cubes over
    # shooting crates. Preserve all V4 combat, navigation, and survival logic.
    for rec in known_cubes.values():
        c = rec["c"]
        d = math.hypot(c["pos"][0] - px, c["pos"][1] - py)
        if d < 125 and ok(c["pos"], ("c", c["id"]), slope=True):
            # V8C: a cube is only worth chasing when the pickup is not contested.
            # Visible mechs are the only reliable threats in this observation;
            # do not mistake NPCs for opposing players.
            enemy_to_cube = min((math.hypot(c["pos"][0] - m["pos"][0],
                                             c["pos"][1] - m["pos"][1])
                                 for m in nearby_players), default=999.0)
            enemy_to_self = min((math.hypot(px - m["pos"][0],
                                             py - m["pos"][1])
                                 for m in nearby_players), default=999.0)
            # Keep looting from overriding combat or retreat when a threat is close.
            if enemy_to_self < 45 or enemy_to_cube < 30:
                continue
            # Deprioritize moderately contested cubes; don't chase them first.
            cube_danger = 55.0 if enemy_to_cube < 65 else 0.0
            cands.append((20 + d + risk(c["pos"]) + cube_danger,
                          ("c", c["id"]), c, "cube", 0.5))
    for m in o.get("mechs", []):
        if m["kind"] == "npc":
            d = math.hypot(m["pos"][0] - px, m["pos"][1] - py)
            if d < 90 and ok(m["pos"], ("n", m["id"])):
                cands.append((135 + d + risk(m["pos"]), ("n", m["id"]), m, "npc", 28.0))
    # V8B experiment: avoid committing to a crate when enemies can contest it.
    # Only visible enemy mechs count; retain V8A's behavior when none are visible.
    for c in o.get("crates", []):
        d = math.hypot(c["pos"][0] - px, c["pos"][1] - py)
        if d < 120 and ok(c["pos"], ("k", c["id"])):
            enemy_to_crate = min((math.hypot(c["pos"][0] - m["pos"][0],
                                              c["pos"][1] - m["pos"][1])
                                  for m in nearby_players), default=999.0)
            enemy_to_self = min((math.hypot(px - m["pos"][0], py - m["pos"][1])
                                 for m in nearby_players), default=999.0)
            # A crate takes several gunshots to open. Avoid exposing ourselves
            # at short range, but do not disable looting due to distant enemies.
            if enemy_to_self < 55 or enemy_to_crate < 40:
                continue
            crate_danger = 65.0 if enemy_to_crate < 75 else 0.0
            cands.append((100 + d + risk(c["pos"]) + crate_danger,
                          ("k", c["id"]), c, "crate", 22.0))

    if cands:
        cands.sort(key=lambda c: c[0])
        _, key, tgt, kind, standoff = cands[0]
        tpos = tgt["pos"]
        chase["tz"] = tpos[2]
        dx, dy = tpos[0] - px, tpos[1] - py
        d = math.hypot(dx, dy) or 1.0

        # A detour can temporarily increase the straight-line distance.
        following_route = (route["key"] == (key, tuple(z["next_center"]), z["next_radius"])
                           and route["waypoints"] and now - route["progress"] < 2.0)
        if chase["key"] != key:
            chase.update(key=key, best=d, t=now, start=now)
        elif d < chase["best"] - 1.5:
            chase.update(best=d, t=now)
        elif following_route:
            chase["t"] = now
        timed_out = kind in ("npc", "crate") and now - chase["start"] > 18.0
        stalled = now - chase["t"] > 3.0 and (d > standoff + 5 or kind in ("cube", "pickup"))
        if timed_out or stalled:
            ignore[key] = now + (40 if timed_out else 30)
            chase.update(key=None, tz=None)
            return False

        chase["goal"] = tuple(tpos[:2])

        if kind == "pickup":
            if d < 3.0:
                act["take"] = tgt["id"]
            act["move"] = [dx / d, dy / d] if d > 2.0 else [0, 0]
            return True
        if kind == "cube":
            act["move"] = [dx / d, dy / d]
            return True
        # Crates MUST be shot to break them: 300 HP -> 1 cube and 50% weapon drop.
        # Shoot while approaching; keep a safe standoff so the mech does not
        # walk into the crate. NPCs also drop cubes when killed.
        clear = not MAP["ok"] or path_clear(px, py, tpos[0], tpos[1], max_len=120)
        if d <= standoff and not clear and now - chase["t"] > 3.0:
            ignore[key] = now + 30
            chase.update(key=None, tz=None, goal=None)
            return False
        fire_weapons(me, act, tgt, allow_fire=d < 115.0,
                     allow_charge=d <= standoff + 4.0 and
                     not any(m.get("charging") for m in nearby_players))
        # Don't run into targets; advance until within accurate gun range.
        act["move"] = [dx / d, dy / d] if d > standoff or not clear else [0, 0]
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


def inside_prism(x, y, z, height=0.0):
    for idx in MAP["grid"].get((int(x // CELL), int(y // CELL)), ()):
        x0, y0, x1, y1, base, top, pts, holes = MAP["prisms"][idx]
        if x < x0 or x > x1 or y < y0 or y > y1 or z + height < base or z > top:
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


ROUTE_GRID = 6.0
ROUTE_HORIZON = 72.0
ROUTE_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (1, -1), (-1, 1), (-1, -1))


def walkable_segment(start, end, resolution=2.0):
    """Sample the mech's footprint along a ground route."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    steps = max(1, math.ceil(math.hypot(dx, dy) / resolution))
    height = ground(*start)
    for i in range(1, steps + 1):
        x, y = start[0] + dx * i / steps, start[1] + dy * i / steps
        if not nav_passable(x, y, height):
            return False
        height = ground(x, y)
    return True


def segment_separation(start, end, point):
    dx, dy = end[0] - start[0], end[1] - start[1]
    rx, ry = point[0] - start[0], point[1] - start[1]
    fraction = max(0.0, min(1.0, (rx * dx + ry * dy) / (dx * dx + dy * dy or 1.0)))
    return math.hypot(rx - fraction * dx, ry - fraction * dy)


def contested_segment(start, end, enemies):
    for enemy in enemies:
        point = enemy["pos"][:2]
        if segment_separation(start, end, point) + 2.5 < min(22.5, math.dist(start, point)):
            return True
    return False


def plan_route(start, goal, circles, enemies, partial):
    """Bounded A* on a local grid anchored at the mech's current position."""
    deadline = time.perf_counter() + 0.012
    goal_distance = math.dist(start, goal)
    threats = [tuple(m["pos"][:2]) for m in enemies if math.dist(start, m["pos"][:2]) < 135]
    frontier = [(goal_distance, 0.0, (0, 0))]
    parents = {(0, 0): None}
    costs = {(0, 0): 0.0}
    best, best_distance, reached = (0, 0), goal_distance, False

    def point(node):
        return start[0] + ROUTE_GRID * node[0], start[1] + ROUTE_GRID * node[1]

    def inside_limits(p):
        return all(math.dist(p, center) <= radius for center, radius in circles)

    for _ in range(320):
        if not frontier or time.perf_counter() >= deadline:
            break
        _, cost, node = heapq.heappop(frontier)
        if cost != costs[node]:
            continue
        here = point(node)
        remaining = math.dist(here, goal)
        if remaining < best_distance:
            best, best_distance = node, remaining
        if (node != (0, 0) and remaining <= 9.0 and inside_limits(goal)
                and walkable_segment(here, goal)):
            best, reached = node, True
            break
        for ix, iy in ROUTE_NEIGHBORS:
            if time.perf_counter() >= deadline:
                break
            other = node[0] + ix, node[1] + iy
            if max(abs(other[0]), abs(other[1])) > 16:
                continue
            there = point(other)
            step = ROUTE_GRID * math.hypot(ix, iy)
            new_cost = cost + step
            if new_cost >= costs.get(other, math.inf) or not inside_limits(there):
                continue
            separation = min((segment_separation(here, there, p) for p in threats),
                             default=math.inf)
            new_cost += step * max(0.0, 45.0 - separation) / 25.0
            if new_cost >= costs.get(other, math.inf) or not walkable_segment(here, there):
                continue
            costs[other] = new_cost
            parents[other] = node
            estimate = new_cost + 1.2 * math.dist(there, goal)
            heapq.heappush(frontier, (estimate, new_cost, other))

    if not reached and (not partial or goal_distance - best_distance < ROUTE_GRID):
        return deque()
    waypoints = deque([goal] if reached and math.dist(point(best), goal) > 0.5 else [])
    while parents[best] is not None:
        waypoints.appendleft(point(best))
        best = parents[best]
    return waypoints


def move_with_route(o, me, desired, goal, key, enemies):
    """Follow cached waypoints when direct ground movement is blocked."""
    speed = math.hypot(*desired)
    if goal is None or not MAP["ok"] or not me["grounded"] or speed < 0.1:
        return steer_around(o, me, desired)

    start = tuple(me["pos"][:2])
    z = o["zone"]
    center = tuple(z["next_center"])
    current_center = tuple(z["center"])
    key = (key, center, z["next_radius"])
    now = o["t"]
    if route["key"] != key or math.dist(route["goal"], goal) > 9.0:
        route.update(key=key, goal=goal, waypoints=deque(), retry=now,
                     best=math.inf, progress=now)

    # A route may skirt the next circle's boundary, but it must stay in the
    # current circle or avoid taking the mech farther outside it.
    current_distance = math.dist(start, current_center)
    next_distance = math.dist(start, center)
    next_radius = z["next_radius"]
    circles = ((current_center, max(z["radius"], current_distance)),
               (center, max(next_radius, next_distance + 2.0)))
    waypoints = route["waypoints"]
    if waypoints and now - route["progress"] >= 2.0:
        waypoints.clear()
        route["retry"] = now + 0.8
    if not waypoints and now >= route["retry"]:
        dx, dy = (component / speed for component in desired)
        distance = math.dist(start, goal)
        look = min(16.0, distance)
        direct = start[0] + dx * look, start[1] + dy * look
        if look > 2.0 and not walkable_segment(start, direct):
            if distance > ROUTE_HORIZON:
                goal = (start[0] + dx * ROUTE_HORIZON, start[1] + dy * ROUTE_HORIZON)
            waypoints = plan_route(start, goal, circles, enemies,
                                   partial=key[0] == "zone" or distance > ROUTE_HORIZON)
            route.update(waypoints=waypoints, retry=now + 0.8,
                         best=math.inf, progress=now, next_limit=circles[1][1])
    while waypoints and math.dist(start, waypoints[0]) < 1.7:
        waypoints.popleft()
        route.update(best=math.inf, progress=now)
    if waypoints:
        # Keep the detour's original limit until we enter the next circle.
        next_limit = next_radius if next_distance <= next_radius else route["next_limit"]

        def usable(point):
            return (math.dist(point, center) <= next_limit
                    and math.dist(point, current_center) <= circles[0][1]
                    and not contested_segment(start, point, enemies)
                    and walkable_segment(start, point, resolution=0.8))

        # Cut a corner only when the full shortcut has clearance.
        if len(waypoints) > 1 and math.dist(start, waypoints[1]) <= 14.0 and usable(waypoints[1]):
            waypoints.popleft()
            route.update(best=math.inf, progress=now)
        point = waypoints[0]
        if not usable(point):
            waypoints.clear()
            route["retry"] = now + 0.8
        else:
            distance = math.dist(start, point)
            if distance < route["best"] - 1.0:
                route.update(best=distance, progress=now)
            return [(point[i] - start[i]) * speed / distance for i in range(2)]
    return steer_around(o, me, desired)


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


def combat_move(o, me, target, enemies):
    """Choose a strafe with enough clearance and room to use the equipped weapon."""
    px, py, _ = me["pos"]
    dx, dy = target["pos"][0] - px, target["pos"][1] - py
    distance = max(1.0, math.hypot(dx, dy))
    fx, fy = dx / distance, dy / distance
    stats = me["stats"]
    arc_range = max((w["stats"].get("range", 45) for w in me["weapons"]
                     if w and w["kind"] == "lightning"), default=0)
    alone = all(m is target or math.dist((px, py), m["pos"][:2]) > 75 for m in enemies)
    close_with_lightning = (arc_range and distance < arc_range + 14 and alone
                            and me["hp"] > 0.65 * me["max_hp"] and me["energy"] > 28)
    if close_with_lightning:
        radial = max(-0.55, min(0.55, (distance - 0.72 * arc_range) / 20))
    else:
        radial = -max(0.0, min(0.60, (40.0 - distance) / 27.0))

    center = o["zone"]["next_center"]
    limit = max(math.dist((px, py), center), 0.88 * o["zone"]["next_radius"])
    best = None
    side_now = mob["strafe_side"]
    for side in (side_now, -side_now):
        move = (radial * fx - side * fy, radial * fy + side * fx)
        mag = math.hypot(*move) or 1.0
        vx, vy = move[0] / mag, move[1] / mag
        end = px + 9 * vx, py + 9 * vy
        separation = min(math.dist(end, m["pos"][:2]) for m in enemies)
        room = not MAP["ok"] or walkable_segment((px, py), end)
        continuity = 2.5 if side == side_now and o["t"] < mob["strafe_until"] else 0.0
        momentum = (vx * me["vel"][0] + vy * me["vel"][1]) / max(1, stats.get("walk_speed", 11.7))
        score = (6.0 * room + min(separation, 80) * 0.18 + continuity + momentum
                 - 0.8 * max(0.0, math.dist(end, center) - limit))
        if best is None or score > best[0]:
            best = score, side, [vx, vy]
    _, side, move = best
    if side != side_now or o["t"] >= mob["strafe_until"]:
        mob.update(strafe_side=side, strafe_until=o["t"] + 0.6)
    return move


def projectile_dodge(o, me):
    """React to a near-term bullet intersection with the mech's cylinder."""
    px, py, pz = me["pos"]
    vx, vy, vz = me["vel"]
    body = me["stats"]
    radius = body.get("radius", 2.5) + 0.5
    height = body.get("height", 9.0)
    closest = None
    for p in o.get("projectiles", ())[:48]:
        if p.get("kind") not in ("gun", "npc"):
            continue
        x, y, z = p["pos"]
        dx, dy, dz = p["vel"]
        rx, ry = x - px, y - py
        ux, uy = dx - vx, dy - vy
        denom = ux * ux + uy * uy
        if not denom or rx * ux + ry * uy >= 0:
            continue
        t = -(rx * ux + ry * uy) / denom
        if not 0.07 <= t <= 0.38 or math.hypot(rx + ux * t, ry + uy * t) > radius:
            continue
        height_at_impact = z + (dz - vz) * t - pz
        if not -0.3 <= height_at_impact <= height + 0.3:
            continue
        # Ignore bullets on the other side of a wall.
        impact = (x + dx * t, y + dy * t, z + dz * t)
        if MAP["ok"] and los_blocked(p["pos"], impact):
            continue
        if closest is None or t < closest[0]:
            closest = t, height_at_impact, ux, uy
    if closest is None:
        return None, False

    t, hit_height, ux, uy = closest
    length = math.hypot(ux, uy)
    side = -uy / length, ux / length
    center = o["zone"]["center"]
    limit = max(math.dist((px, py), center), o["zone"]["radius"])

    def score(direction):
        dx, dy = direction
        end = px + dx * 7, py + dy * 7
        room = not MAP["ok"] or walkable_segment((px, py), end)
        return 5 * room - 2 * max(0, math.dist(end, center) - limit) + 0.1 * (vx * dx + vy * dy)

    movement = max((side, (-side[0], -side[1])), key=score)
    rise = body.get("jump_speed", 11.0) * (t - 0.05) - 11.0 * (t - 0.05) ** 2
    jump = hit_height + 0.45 < rise
    return movement, jump


def air_clear(me, x, y, feet):
    """Check the mech's footprint and torso clearance at a raised position."""
    radius = me["stats"].get("radius", 2.5)
    height = me["stats"].get("height", 9.0)
    for dx, dy in ((0, 0), (radius, 0), (-radius, 0), (0, radius), (0, -radius)):
        xx, yy = x + dx, y + dy
        if not in_terrain(xx, yy) or ground(xx, yy) > feet + 0.15:
            return False
        if inside_prism(xx, yy, feet + 0.1, height - 0.2):
            return False
    return True


def mobility(o, me, act, desired, want_up, fleeing, evading, jump_dodge, urgent):
    """Use energy for short sprints or intentional climbs."""
    now = o["t"]
    px, py, pz = me["pos"]
    vx, vy, vz = me["vel"]
    energy = me["energy"]
    mag = math.hypot(*desired)
    if not me["grounded"]:
        # In the air boost adds lift, so keep it for an active climb.
        if (now < mob["climb_until"] and energy > 35
                and (not MAP["ok"] or air_clear(me, px, py, pz + 1.2))):
            act["boost"] = True
        return

    if mag > 0.5 and vx * desired[0] + vy * desired[1] < 1.8:
        mob["blocked"] += 1
    else:
        mob["blocked"] = 0
    planned_speed = math.hypot(*act["move"])
    can_try_jump = jump_dodge or (mob["blocked"] >= 4 and math.hypot(vx, vy) < 2.4)
    if can_try_jump and now >= mob["jump_cd"]:
        jump_ok = not MAP["ok"] or air_clear(me, px, py, pz + 1.2)
        if jump_ok and jump_dodge and not urgent:
            act["jump"] = True
            mob["jump_cd"] = now + 1.15
        elif jump_ok and mob["blocked"] >= 4:
            dx, dy = (component / mag for component in desired)
            ahead = px + 4 * dx, py + 4 * dy
            blocked = not MAP["ok"] or not walkable_segment((px, py), ahead)
            hop_height = me["stats"].get("jump_speed", 11.0) ** 2 / 44.0 - 0.35
            low_ledge = MAP["ok"] and blocked and all(
                air_clear(me, px + d * dx, py + d * dy, pz + hop_height) for d in (2, 4, 6))
            climb = (MAP["ok"] and want_up and energy > 60 and blocked
                     and all(air_clear(me, px + d * dx, py + d * dy, pz + 4.0)
                             for d in (0, 2, 4, 6)))
            if low_ledge or climb or (not MAP["ok"] and want_up):
                act["jump"] = True
                if planned_speed < 0.4 or blocked:
                    act["move"] = [dx, dy]
                mob.update(blocked=0, jump_cd=now + 1.3,
                           climb_until=now + 0.65 if climb and not low_ledge else now)

    reserve = 31 if evading or urgent else 49
    sprint = (evading or urgent or fleeing) and energy >= reserve and planned_speed > 0.55
    if sprint and now >= mob["sprint_ready"]:
        mob.update(sprint_until=now + 0.46, sprint_ready=now + 0.9)
    if sprint and now < mob["sprint_until"] and not act.get("jump"):
        act["boost"] = True


def safe_zone_route(o, me, enemies, cx, cy, radius):
    """Pick a short local zone-bound heading that avoids visible mechs.

    Never let a large detour jeopardize reaching the shrinking circle.
    """
    px, py, _ = me["pos"]
    gx, gy = cx - px, cy - py
    gd = math.hypot(gx, gy)
    if gd < 2:
        return [0.0, 0.0]
    ux, uy = gx / gd, gy / gd
    if not enemies:
        return [ux, uy]
    # Threats outside ~115m do not justify a detour.
    nearby = [m for m in enemies if math.hypot(m["pos"][0]-px, m["pos"][1]-py) < 115]
    if not nearby:
        return [ux, uy]
    z = o["zone"]
    # If already taking zone damage, prioritize the fastest route to safety.
    urgent = not me.get("in_zone", True) or o["t"] >= z.get("shrink_start", 1e9)
    max_turn = 0.42 if urgent else 0.85
    look = min(32.0, max(12.0, gd * 0.3))
    best, best_dir = -1e20, [ux, uy]
    for turn in (-max_turn, -max_turn/2, 0, max_turn/2, max_turn):
        ca, sa = math.cos(turn), math.sin(turn)
        vx, vy = ux*ca-uy*sa, ux*sa+uy*ca
        tx, ty = px + look*vx, py + look*vy
        if MAP["ok"] and (not in_terrain(tx, ty) or solid_at(tx, ty)):
            continue
        # Penalize passing close to an enemy, including midway along the route.
        danger = 0.0
        for e in nearby:
            ex, ey = e["pos"][:2]
            rx, ry = ex-px, ey-py
            along = max(0.0, min(look, rx*vx + ry*vy))
            sep = math.hypot(rx-along*vx, ry-along*vy)
            if sep < 75:
                danger += ((75-sep)/75)**2 * (2.4 if sep < 35 else 1.4)
            # Discourage heading directly toward a mech even beyond lookahead.
            if along > 0 and math.hypot(rx,ry) < 60:
                danger += max(0, (60-math.hypot(rx,ry))/60) * 0.6
        progress = (gx*vx + gy*vy) / gd
        score = (5.0 if urgent else 2.8)*progress - danger
        if score > best:
            best, best_dir = score, [vx, vy]
    return best_dir


def zone_weapon_detour(o, me, act, held, goal, walk_speed, distance_to_zone):
    """Take an in-range weapon or briefly walk to one during safe zone travel."""
    now = o["t"]
    target = zone_loot["id"]
    # An interrupted approach should not switch straight to another pickup.
    # A validated approach below restores the target with its original deadline.
    if target is not None:
        zone_loot.update(id=None, retry=now + 3.0)
    if (goal is None or held[1] is not None or not me["grounded"]
            or now < zone_loot["hit_until"]
            or (target is None and now < zone_loot["retry"])):
        return None

    pickups = o.get("pickups", ())
    if not pickups:
        return None
    z = o["zone"]
    slack = z["shrink_end"] - now - distance_to_zone / walk_speed - ZONE_LOOT_RESERVE_S
    if slack <= 0:
        return None
    start = tuple(me["pos"][:2])
    pz = me["pos"][2]
    pickup_range = CFG.get("loot", {}).get("pickup_radius", 4.0)
    threats = [(m["pos"][:2], 65.0 if m["kind"] == "mech" else 45.0)
               for m in o["mechs"] if m["id"] != me["id"]]
    if any(math.dist(start, pos) < radius for pos, radius in threats):
        return None
    gone = {e["id"] for e in o.get("events", ())
            if e.get("type") == "take_failed" and e.get("reason") == "gone"}
    candidates = []
    for pickup in pickups:
        pid = pickup["id"]
        if ((target is not None and pid != target) or pid in gone
                or pickup["kind"] in held or ignore.get(("p", pid), 0) > now):
            continue
        distance = math.dist(start, pickup["pos"][:2])
        candidates.append((distance, WEAPON_RANK.get(pickup["kind"], 3), pid, pickup["pos"]))
    candidates.sort()
    for distance, _, pid, pos in candidates:
        if distance <= pickup_range and abs(pos[2] - pz) <= 4.0:
            act["take"] = pid
            zone_loot["id"] = target
            return None

    remaining = ZONE_LOOT_S if target is None else zone_loot["until"] - now
    approach_range = max(0.0, pickup_range - 0.35)
    max_walk = remaining * walk_speed * 0.85
    look = min(18.0, math.dist(start, goal))
    if (not candidates or remaining <= 0 or not MAP["ok"] or look < 1.0
            or candidates[0][0] > max_walk + approach_range):
        return None
    if abs(ground(*start) - pz) > 0.6:
        return None  # use the terrain route only from ground level, not a roof
    rejoin = tuple(start[i] + look * act["move"][i] for i in range(2))
    if not walkable_segment(start, rejoin, resolution=0.8):
        return None  # leave obstacle routing to the zone planner

    # Check the approaching edge at the end of the pickup window. The circle
    # contracts linearly from the current observation during the shrink.
    shrink_now = max(now, z["shrink_start"])
    future_time = min(z["shrink_end"], now + remaining + 0.3)
    fraction = 0.0
    if future_time > shrink_now:
        fraction = (future_time - shrink_now) / (z["shrink_end"] - shrink_now)
    future_center = tuple(a + fraction * (b - a) for a, b in zip(z["center"], goal))
    future_radius = z["radius"] + fraction * (z["next_radius"] - z["radius"]) - 2.0
    if max(math.dist(start, future_center), math.dist(rejoin, future_center)) > future_radius:
        return None

    best = None
    for distance, rank, pid, pos in candidates:
        if distance <= pickup_range:
            continue  # already within horizontal range but above or below us
        to_range = distance - approach_range
        if to_range > max_walk:
            break
        direction = tuple((pos[i] - start[i]) / distance for i in range(2))
        entry = tuple(start[i] + to_range * direction[i] for i in range(2))
        extra_time = max(0.0, to_range + math.dist(entry, rejoin) - look) / walk_speed
        if (extra_time > min(ZONE_LOOT_S, slack)
                or math.dist(entry, goal) > math.dist(start, goal) + 1.0
                or math.dist(entry, future_center) > future_radius
                or any(min(segment_separation(start, entry, point),
                           segment_separation(entry, rejoin, point)) < radius
                       for point, radius in threats)
                or not walkable_segment(start, entry, resolution=0.8)
                or abs(pos[2] - ground(*entry)) > 3.5
                or not walkable_segment(entry, rejoin, resolution=0.8)):
            continue
        candidate = (extra_time, distance, rank, pid, direction)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None
    zone_loot["id"] = best[3]
    if target is None:
        zone_loot["until"] = now + ZONE_LOOT_S
    return best[4]


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
    if hit_now:
        zone_loot["hit_until"] = now + 2.0
    fleeing = False
    want_up = False

    # --- zone: head for the next circle, stop once well inside it ---
    z = o["zone"]
    cx, cy = z["next_center"]
    goal = (cx, cy)
    route_key = "zone"
    zd = math.hypot(cx - px, cy - py) or 1.0
    inside = me["in_zone"] and zd < 0.85 * z["next_radius"]
    if inside:
        zone_loot["id"] = None
    if (not inside) or zd > 0.5 * z["next_radius"]:
        act["move"] = [(cx - px) / zd, (cy - py) / zd]
    else:
        act["move"] = [0, 0]

    enemies = [m for m in o["mechs"] if m["kind"] == "mech" and m["id"] != me["id"]]
    dv = laser_dodge(o, me, enemies)
    bullet_move, jump_dodge = projectile_dodge(o, me) if dv is None else (None, False)
    evade = dv if dv is not None else bullet_move
    walk_speed = max(1.0, me["stats"].get("walk_speed", 11.7))
    distance_to_zone = max(0.0, zd - z["next_radius"] + 5.0)
    urgent = (not me["in_zone"] or
              (distance_to_zone > 0 and now >= z["shrink_start"]
               and distance_to_zone / walk_speed + 2.0 > z["shrink_end"] - now))

    if enemies:
        t = min(enemies, key=lambda m: math.hypot(m["pos"][0] - px, m["pos"][1] - py))
        dx, dy = t["pos"][0] - px, t["pos"][1] - py
        dist = math.hypot(dx, dy) or 1.0

        # Baseline V1 survival thresholds retained; do not repeat V2's broad retreat policy.
        # Distant enemies should not prevent safe farming and collecting upgrades.
        if inside and dist > 75 and not hit_now and evade is None and now < 180:
            if loot(o, me, act, held, z, cx, cy):
                desired = act["move"]
                act["move"] = move_with_route(o, me, desired, chase["goal"],
                                              chase["key"], enemies)
                mobility(o, me, act, desired,
                         chase["tz"] is not None and chase["tz"] - pz > 1.0,
                         fleeing=False, evading=False, jump_dodge=False, urgent=False)
                reload_gun(me, act)
                return act

        # early game: avoid enemies that are healthier than us (needs enemy hp in obs)
        threats = [m for m in enemies
                   if m.get("hp") is not None and me["hp"] < HP_AVOID * m["hp"]
                   and math.hypot(m["pos"][0] - px, m["pos"][1] - py) < 150]
        avoid = o["t"] < EARLY_S and bool(threats)

        # strafe, close in with lightning, or back off when low on HP
        if inside:
            goal = None
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
                act["move"] = combat_move(o, me, t, enemies)
            if fleeing and rv is not None:
                goal, route_key = cover["pt"], "cover"
        else:
            act["move"] = safe_zone_route(o, me, enemies, cx, cy, z["next_radius"])
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
            goal = None
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
            if cover["pt"] is not None and cover["hidden"]:
                goal, route_key = cover["pt"], "cover"
            acted = True
        elif (inside and cover["pt"] is not None and now < cover["hold_until"]
              and me["hp"] < 0.9 * me["max_hp"]):
            goal = None
            # heal in cover: walk to the spot and wait there (regen needs ~5 s without damage)
            ddx, ddy = cover["pt"][0] - px, cover["pt"][1] - py
            dd = math.hypot(ddx, ddy)
            act["move"] = [ddx / dd, ddy / dd] if dd > 3.0 else [0, 0]
            acted = True
        else:
            acted = inside and evade is None and loot(o, me, act, held, z, cx, cy)
            want_up = acted and chase["tz"] is not None and chase["tz"] - pz > 1.0
            if acted:
                goal, route_key = chase["goal"], chase["key"]
        if not acted:
            # nothing to do: face whoever just hit us
            hits = [e for e in o.get("events", [])
                    if e.get("type") == "damaged" and e.get("bearing") is not None]
            if hits:
                act["aim"] = [hits[-1]["bearing"], 0]

    if evade is not None:
        act["move"] = evade
        goal = None
    detour = None
    if not inside:
        detour = zone_weapon_detour(o, me, act, held, None if urgent else goal,
                                    walk_speed, distance_to_zone)
    desired = act["move"] if detour is None else list(detour)
    act["move"] = (move_with_route(o, me, desired, goal, route_key, enemies)
                   if detour is None else desired)
    mobility(o, me, act, desired, want_up, fleeing, evade is not None,
             jump_dodge, urgent and route_key == "zone")
    if enemies:
        fire_weapons(me, act, t, allow_fire=not (avoid and dist >= 40),
                     allow_charge=evade is None and not (fleeing or urgent or hit_now))
    reload_gun(me, act)

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
    send({"type": "ready", "name": "mech-royale-v8m"})

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


if __name__ == "__main__":
    main()

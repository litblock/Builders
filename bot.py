import json
import math
import os
import queue
import random
import sys
import threading

q = queue.Queue()

# Offline debugging only: set MECH_LOG=log.jsonl when running check.py.
# Unset in real matches, so nothing is opened on the read-only disk.
LOG = open(os.environ["MECH_LOG"], "a") if os.environ.get("MECH_LOG") else None

CFG = {}
RNG = random.Random(0)

known = {}       # weapon pickup id -> pickup dict (remembered after it leaves view)
known_cubes = {}  # cube id -> {"c": cube dict, "t": last seen match time}
ignore = {}      # (type, id) -> match time until which we skip that target
chase = {"key": None, "best": 1e9, "t": 0.0, "start": 0.0}
wander = {"pt": None, "until": 0.0}

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


def decide(o):
    me = o["self"]
    px, py, pz = me["pos"]
    act = {"tick": o["tick"]}
    held = [w["kind"] if w else None for w in me["weapons"]]

    if o.get("offer"):
        act["pick"] = pick_upgrade(o["offer"], held, me)

    update_memory(o, px, py, pz)

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
        act["fire"] = [fire0, fire1]

        # strafe, close in with lightning, or back off when low on HP
        if inside:
            if me["hp"] < 0.35 * me["max_hp"]:
                act["move"] = [-dx / dist, -dy / dist]
            else:
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
    else:
        acted = loot(o, me, act, held, z, cx, cy) if inside else False
        if "fire" in act:
            firing_gun = act["fire"][0]
        if not acted:
            # nothing to do: face whoever just hit us
            hits = [e for e in o.get("events", [])
                    if e.get("type") == "damaged" and e.get("bearing") is not None]
            if hits:
                act["aim"] = [hits[-1]["bearing"], 0]

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
    if LOG:
        LOG.write(json.dumps({"hello": hello}) + "\n")
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
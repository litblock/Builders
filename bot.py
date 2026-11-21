import json
import math
import queue
import sys
import threading

q = queue.Queue()


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


def decide(o):
    me = o["self"]
    px, py, pz = me["pos"]
    act = {"tick": o["tick"]}

    if o.get("offer"):
        act["pick"] = 0

    # movement: head for the next zone circle, stop once well inside it
    z = o["zone"]
    cx, cy = z["next_center"]
    d = math.hypot(cx - px, cy - py)
    if d > 0.5 * z["next_radius"] or not me["in_zone"]:
        act["move"] = [(cx - px) / d, (cy - py) / d]
    else:
        act["move"] = [0, 0]

    # combat: nearest visible enemy mech, lead the shot
    enemies = [m for m in o["mechs"] if m["kind"] == "mech" and m["id"] != me["id"]]
    gun = me["weapons"][0]
    if enemies:
        t = min(enemies, key=lambda m: math.hypot(m["pos"][0] - px, m["pos"][1] - py))
        speed = gun["stats"]["speed"] if gun else 252
        dist = math.hypot(t["pos"][0] - px, t["pos"][1] - py)
        ft = dist / speed
        tx = t["pos"][0] + t["vel"][0] * ft
        ty = t["pos"][1] + t["vel"][1] * ft
        tz = t["pos"][2] + 4.5
        hd = math.hypot(tx - px, ty - py)
        yaw = math.atan2(ty - py, tx - px)
        pitch = math.atan2(tz - (pz + 7), hd)
        act["aim"] = [yaw, pitch]
        act["fire"] = [abs(wrap(yaw - me["aim"][0])) < 0.1, False]
    elif gun and gun["ammo"] < gun["stats"]["mag"] // 2 and gun["reload"] == 0:
        act["reload"] = True

    return act


def main():
    threading.Thread(target=reader, daemon=True).start()
    hello = q.get()
    if hello is None:
        return
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
                send(decide(m))
            except Exception as e:
                print("error:", repr(e), file=sys.stderr)
                send({"tick": m["tick"]})


if __name__ == "__main__":
    main()
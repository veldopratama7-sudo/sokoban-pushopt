// sokoban-pushopt: push-optimal per-room solver.
// Levels: one 1-wide hallway row (the player's row) with 1-4 rooms opening onto it through one door each;
// every box stays in its room, so the level's push-optimal count is the sum of the rooms' optima. Each room is
// solved independently by a breadth-first search over (box configuration, player region) where walking costs
// nothing and a push costs one: the first solved configuration reachable from the hallway gives the optimum.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

type Room = { cells: number[]; boxes: number[]; goals: number[]; doors: number[]; start: bigint; goalMask: bigint };
type Level = {
  w: number; size: number;
  wall: Uint8Array; goal: Uint8Array;
  player: number;
  hallway: number[];
  rooms: Room[];
};

function parse(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const size = w * h;
  const wall = new Uint8Array(size).fill(1);
  const goal = new Uint8Array(size);
  const boxAt = new Uint8Array(size);
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') boxAt[i] = 1;
      if (ch === '@' || ch === '+') player = i;
    }
  });
  const floor: number[] = [];
  for (let i = 0; i < size; i++) if (!wall[i]) floor.push(i);
  const hallRow = Math.floor(player / w);
  const hallway = floor.filter((i) => Math.floor(i / w) === hallRow);
  const hallSet = new Set(hallway);
  // rooms: connected floor components outside the hallway row, never crossing it
  const seen = new Set<number>(hallway);
  const rooms: Room[] = [];
  for (const start of floor) {
    if (seen.has(start)) continue;
    const comp: number[] = [];
    const stack = [start];
    seen.add(start);
    while (stack.length > 0) {
      const c = stack.pop()!;
      comp.push(c);
      for (const d of [-1, 1, -w, w]) {
        const n = c + d;
        if (n < 0 || n >= size || wall[n] === 1 || seen.has(n) || hallSet.has(n)) continue;
        seen.add(n);
        stack.push(n);
      }
    }
    const cells = comp.sort((a, b) => a - b);
    const boxes = cells.filter((c) => boxAt[c] === 1);
    const goals = cells.filter((c) => goal[c] === 1);
    const doors: number[] = [];
    for (const c of cells) {
      for (const d of [-w, w]) {
        const n = c + d;
        if (n >= 0 && n < size && hallSet.has(n)) { doors.push(c); break; }
      }
    }
    let sb = 0n;
    for (const b of boxes) sb |= 1n << BigInt(cells.indexOf(b));
    let gm = 0n;
    for (const g of goals) gm |= 1n << BigInt(cells.indexOf(g));
    rooms.push({ cells, boxes, goals, doors: doors.sort((a, b) => a - b), start: sb, goalMask: gm });
  }
  return { w, size, wall, goal, player, hallway, rooms };
}


/** Push-optimal search for one room; returns the pushes as (global cell, dir) or null. */
function solveRoom(lv: Level, room: Room, deadlineNodes: number): { cell: number; dir: number }[] | null {
  const w = lv.w;
  const D4 = [-w, w, -1, 1];
  const cells = room.cells;
  const n = cells.length;
  const cellIdx = new Map<number, number>();
  cells.forEach((c, k) => cellIdx.set(c, k));
  const roomSet = new Set(cells);
  const walkable = new Uint8Array(lv.size);
  for (const c of lv.hallway) walkable[c] = 1;
  for (const c of cells) walkable[c] = 1;

  // static pull-reachability from the goals: a box can rest only on cells from which some goal is
  // reachable through pushes (both neighbours along the axis must be floor); the rest are dead.
  const canMove = new Uint8Array(lv.size);
  for (const c of cells) {
    for (const d of D4) {
      const a = c + d, b = c - d;
      if (a < 0 || a >= lv.size || b < 0 || b >= lv.size) continue;
      if (lv.wall[a] === 0 && lv.wall[b] === 0) { canMove[c] = 1; break; }
    }
  }
  const reach = new Uint8Array(lv.size);
  const rq: number[] = [];
  for (const g of room.goals) { reach[g] = 1; rq.push(g); }
  while (rq.length > 0) {
    const c = rq.pop()!;
    // reverse of the optimistic push rule (box at p can move to p+d iff p-d and p+d are floor):
    // a box that reached c came from p = c-d, pushed by a player standing at p-d = c-2d.
    for (const d of D4) {
      const p = c - d, pl = c - 2 * d;
      if (p < 0 || p >= lv.size || pl < 0 || pl >= lv.size) continue;
      if (lv.wall[p] === 0 && lv.wall[pl] === 0 && roomSet.has(p) && reach[p] === 0) { reach[p] = 1; rq.push(p); }
    }
  }

  const has = (m: bigint, k: number): boolean => ((m >> BigInt(k)) & 1n) === 1n;

  /** The cells the player can walk to with this box set, from `from`; also the region's smallest cell. */
  function flood(m: bigint, from: number): { min: number; cells: Set<number> } {
    const blocked = new Set<number>();
    for (let k = 0; k < n; k++) if (has(m, k)) blocked.add(cells[k]);
    const seen = new Set<number>([from]);
    const st = [from];
    let min = from;
    while (st.length > 0) {
      const c = st.pop()!;
      if (c < min) min = c;
      for (const d of D4) {
        const nb = c + d;
        if (nb < 0 || nb >= lv.size) continue;
        if (walkable[nb] === 0 || blocked.has(nb) || seen.has(nb)) continue;
        seen.add(nb);
        st.push(nb);
      }
    }
    return { min, cells: seen };
  }

  const boxesOf = (m: bigint): Set<number> => {
    const s = new Set<number>();
    for (let k = 0; k < n; k++) if (has(m, k)) s.add(cells[k]);
    return s;
  };

  const keyOf = (m: bigint, region: number): string => m.toString(36) + '|' + region;

  if (room.boxes.length === 0) return [];
  const start = flood(room.start, lv.player);
  const startKey = keyOf(room.start, start.min);
  const dist = new Map<string, number>([[startKey, 0]]);
  const parent = new Map<string, { prev: string; box: number; dir: number }>();
  let layer: { m: bigint; region: number; key: string }[] = [{ m: room.start, region: start.min, key: startKey }];
  let nodes = 1;
  let bestKey: string | null = null;
  for (let depth = 0; depth <= 400 && layer.length > 0; depth++) {
    const next: typeof layer = [];
    for (const { m, region, key } of layer) {
      if ((m & room.goalMask) === room.goalMask) {
        // solved: needs the hallway reachable so the player can leave for other rooms
        const f = flood(m, region);
        let ok = false;
        for (const c of f.cells) { if (lv.hallway.includes(c)) { ok = true; break; } }
        if (ok) { bestKey = key; break; }
        continue;
      }
      const regionCells = flood(m, region).cells;
      const boxes = boxesOf(m);
      for (let k = 0; k < n; k++) {
        if (!has(m, k)) continue;
        const c = cells[k];
        for (const d of D4) {
          const pside = c - d;
          const target = c + d;
          if (!regionCells.has(pside)) continue;
          if (!roomSet.has(target) || boxes.has(target)) continue;
          if (reach[target] === 0 && lv.goal[target] === 0) continue; // dead square
          const m2 = (m & ~(1n << BigInt(k))) | (1n << BigInt(cellIdx.get(target)!));
          const f2 = flood(m2, c);
          const k2 = keyOf(m2, f2.min);
          if (dist.has(k2)) continue;
          dist.set(k2, depth + 1);
          parent.set(k2, { prev: key, box: k, dir: d });
          next.push({ m: m2, region: f2.min, key: k2 });
          nodes++;
        }
      }
      if (nodes > deadlineNodes) return null;
    }
    if (bestKey !== null) break;
    layer = next;
  }
  if (bestKey === null) return null;
  // backtrack pushes
  const pushes: { cell: number; dir: number }[] = [];
  let k = bestKey;
  while (parent.has(k)) {
    const p = parent.get(k)!;
    pushes.push({ cell: cells[p.box], dir: p.dir });
    k = p.prev;
  }
  pushes.reverse();
  return pushes;
}

function dirLetter(d: number): string {
  return d === -1 ? 'l' : d === 1 ? 'r' : d < 0 ? 'u' : 'd';
}

/** Shortest walk from a to b over floor minus boxes; deterministic order u, d, l, r. */
function walkPath(lv: Level, box: Uint8Array, from: number, to: number): string {
  const w = lv.w;
  if (from === to) return '';
  const prev = new Map<number, number>([[from, -1]]);
  const q: number[] = [from];
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi];
    for (const d of [-w, w, -1, 1]) {
      const n = c + d;
      if (n < 0 || n >= lv.size || lv.wall[n] === 1 || box[n] === 1 || prev.has(n)) continue;
      prev.set(n, c);
      if (n === to) {
        const out: string[] = [];
        let cur = to;
        while (cur !== from) {
          const p = prev.get(cur)!;
          const dd = cur - p;
          out.push(dd === -w ? 'u' : dd === w ? 'd' : dd === -1 ? 'l' : 'r');
          cur = p;
        }
        return out.reverse().join('');
      }
      q.push(n);
    }
  }
  return '';
}

const solve = (level: string): string => {
  const lv = parse(level);
  const box = new Uint8Array(lv.size);
  for (const r of lv.rooms) for (const b of r.boxes) box[b] = 1;
  let out = '';
  let cur = lv.player;
  for (const room of lv.rooms) {
    if (room.boxes.length === 0) continue;
    const pushes = solveRoom(lv, room, 4_000_000);
    if (pushes === null) return ''; // give up on the whole level (cannot emit a partial safe plan)
    for (const push of pushes) {
      const pside = push.cell - push.dir;
      out += walkPath(lv, box, cur, pside);
      box[push.cell] = 0;
      box[push.cell + push.dir] = 1;
      out += dirLetter(push.dir);
      cur = push.cell;
    }
  }
  return out;
};

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}

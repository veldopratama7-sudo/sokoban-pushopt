// sokoban-pushopt: push-optimal per-room Sokoban solver.
//
// Every instance is a hallway one square wide with 1-4 rooms opening onto it through one door each. A box
// pushed through a door into the hallway can never come back (pushing it through the door again would need
// the player standing on the hallway's far side, which is wall), so every box is solved inside its own room
// and the level's push-optimal count P* is the sum of the rooms' optima.
//
// Each room is solved on its own by a breadth-first search by pushes over states (box set, player region):
// walking costs nothing, a push costs one, so the first solved state the search pops is push-optimal. The
// state's key is the room's box bitmask together with the player region's smallest cell, which names the
// region canonically. The pruning never removes a needed state:
//   - dead squares: a cell from which no goal is reachable by pushes at all, reverse-propagated from the
//     room's goals under the optimistic rule "a box at p moves to p+d when p-d and p+d are both floor";
//   - frozen 2x2 blocks of walls and boxes with a box among them off a goal: no box of such a block can ever
//     be pushed (every push needs the square beyond it or the player's square inside the block).
// A solved state is accepted only when the player's region still reaches the hallway; the generator lays the
// goals so this always holds, and the check costs nothing.
// The answer's steps are then played out: between pushes the player walks by a shortest BFS over the whole
// board around the boxes' current squares. Pure and deterministic: no clock, no randomness, fixed tie order.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Work bound per room's search, in states; the wall budget is never read. A room the generator kept is
 *  solved by the reference search within 250,000 states, and this search never leaves its state space, so
 *  the bound is only a freak-case backstop. */
const MAX_STATES = 1_000_000;
/** Move letters by direction index; the direction order is the fixed tie order everywhere. */
const LETTER = 'udlr';

interface Room {
  cells: number[]; // the room's floor squares, sorted; the door cell is one of them
  idx: Int16Array; // board square -> index in cells, -1 outside the room
  startMask: number; // the room's starting boxes as a bitmask over cells
  goalMask: number; // the room's goals as a bitmask over cells
  live: Uint8Array; // board square -> 1 when some goal is reachable from it by pushes at all
  walk: Uint8Array; // board square -> 1 when the player may stand there during this room's solve
}

interface Level {
  w: number;
  size: number;
  wall: Uint8Array;
  goal: Uint8Array;
  box: Uint8Array;
  player: number;
  hall: number[];
  rooms: Room[];
}

function parse(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const size = w * h;
  const wall = new Uint8Array(size).fill(1);
  const goal = new Uint8Array(size);
  const box = new Uint8Array(size);
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') box[i] = 1;
      if (ch === '@' || ch === '+') player = i;
    }
  });
  const hallRow = Math.floor(player / w);
  const hall: number[] = [];
  for (let i = 0; i < size; i++) if (wall[i] === 0 && Math.floor(i / w) === hallRow) hall.push(i);
  // rooms: the connected floor components outside the hallway row, met in scan order
  const seen = new Uint8Array(size);
  for (const c of hall) seen[c] = 1;
  const idx = new Int16Array(size).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] === 1 || seen[s] === 1) continue;
    const comp: number[] = [];
    const stack = [s];
    seen[s] = 1;
    while (stack.length > 0) {
      const c = stack.pop()!;
      comp.push(c);
      for (const d of [-w, w, -1, 1]) {
        const q = c + d;
        if (q < 0 || q >= size || wall[q] === 1 || seen[q] === 1) continue;
        seen[q] = 1;
        stack.push(q);
      }
    }
    comp.sort((a, b) => a - b);
    const n = comp.length;
    let startMask = 0;
    let goalMask = 0;
    for (let k = 0; k < n; k++) {
      const c = comp[k];
      idx[c] = k;
      if (box[c] === 1) startMask |= 1 << k;
      if (goal[c] === 1) goalMask |= 1 << k;
    }
    const walk = new Uint8Array(size);
    for (const c of hall) walk[c] = 1;
    for (const c of comp) walk[c] = 1;
    rooms.push({ cells: comp, idx, startMask, goalMask, live: new Uint8Array(size), walk });
  }
  return { w, size, wall, goal, box, player, hall, rooms };
}

/** Marks the room's live squares: a cell stays dead when no goal is reachable from it by pushes at all. The
 *  check uses the optimistic rule (a box at p moves to p+d when p-d and p+d are both floor), run backwards
 *  from the goals: a box that reached c came from c-d, pushed by a player standing at c-2d. The rule only
 *  ever keeps cells that may still reach a goal, so it never prunes a needed state. */
function fillLive(lv: Level, room: Room): void {
  const { w, size, wall, goal } = lv;
  const live = room.live;
  const queue: number[] = [];
  for (const c of room.cells) {
    if (goal[c] === 1) {
      live[c] = 1;
      queue.push(c);
    }
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const c = queue[qi];
    for (const d of [-w, w, -1, 1]) {
      const p = c - d;
      const pl = c - 2 * d;
      if (p < 0 || p >= size || pl < 0 || pl >= size) continue;
      if (wall[p] === 0 && wall[pl] === 0 && live[p] === 0) {
        live[p] = 1;
        queue.push(p);
      }
    }
  }
}

/** Whether a box just pushed onto `t` sits in a frozen 2x2 block of walls and boxes with a box among them off
 *  a goal. Such a block never moves again: each push of a box in it needs the square beyond it, or the
 *  player's square, inside the block. `box` holds every room's boxes; only this room's move. */
function frozenAt(lv: Level, mask: number, room: Room, t: number): boolean {
  const { w, size, wall, goal, box } = lv;
  const idx = room.idx;
  const hasBox = (q: number): boolean => {
    if (idx[q] >= 0) return ((mask >>> idx[q]) & 1) === 1;
    return box[q] === 1;
  };
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    if (a < 0 || a + w + 1 >= size) continue;
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (wall[a] === 0 && !hasBox(a)) continue;
    if (wall[b] === 0 && !hasBox(b)) continue;
    if (wall[c] === 0 && !hasBox(c)) continue;
    if (wall[d] === 0 && !hasBox(d)) continue;
    if (
      (hasBox(a) && goal[a] === 0) || (hasBox(b) && goal[b] === 0) ||
      (hasBox(c) && goal[c] === 0) || (hasBox(d) && goal[d] === 0)
    ) return true;
  }
  return false;
}

/** One room's push-optimal pushes as [box index in cells, direction index 0..3 = u,d,l,r], or null when the
 *  search exceeds its work bound (a room the generator kept always solves far inside it). */
function solveRoom(lv: Level, room: Room): [number, number][] | null {
  const { w, size, player } = lv;
  const cells = room.cells;
  const n = cells.length;
  if (n > 31) return null; // the bitmask key needs a room floor of at most 31 squares: the slot carve
  // opens at most target(<=27)+3 squares plus the door cell
  fillLive(lv, room);
  const SPAN = 2 ** n;
  const DELTA = [-w, w, -1, 1];
  const idx = room.idx;
  const walk = room.walk;
  const live = room.live;
  const goalMask = room.goalMask;
  const seen = new Int32Array(size); // the expanded state's region, one stamp per pop
  const mark = new Int32Array(size); // each successor's region, one stamp per flood
  const stack = new Int32Array(size);
  let stamp = 0;
  let markStamp = 0;
  const hasBox = (q: number, mask: number): boolean => idx[q] >= 0 && ((mask >>> idx[q]) & 1) === 1;

  /** Marks the region the player walks over from `from` with `mask` boxes in the way, in `into` with `s`;
   *  returns its least square. The expanded state's region and the successors' regions use separate arrays,
   *  so later floods never disturb an earlier region's marks. */
  const flood = (mask: number, from: number, into: Int32Array, s: number): number => {
    let min = from;
    let sp = 0;
    stack[sp++] = from;
    into[from] = s;
    while (sp > 0) {
      const p = stack[--sp];
      if (p < min) min = p;
      for (let k = 0; k < 4; k++) {
        const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
        if (q < 0 || q >= size || walk[q] === 0 || into[q] === s || hasBox(q, mask)) continue;
        into[q] = s;
        stack[sp++] = q;
      }
    }
    return min;
  };

  stamp++;
  const startKey = flood(room.startMask, player, seen, stamp) * SPAN + room.startMask;
  const visited = new Set<number>([startKey]);
  const parents = new Map<number, number>(); // key -> prevKey * 4n + boxIndex * 4 + direction
  let layer = [startKey];
  while (layer.length > 0) {
    const next: number[] = [];
    for (const key of layer) {
      const low = Math.floor(key / SPAN);
      const mask = key - low * SPAN;
      stamp++;
      flood(mask, low, seen, stamp); // the expanded state's region, stable while its pushes are tried
      if (mask === goalMask) {
        for (const c of lv.hall) {
          if (seen[c] === stamp) {
            const pushes: [number, number][] = [];
            let k = key;
            while (parents.has(k)) {
              const pv = parents.get(k)!;
              pushes.push([Math.floor(pv / 4) % n, pv % 4]);
              k = Math.floor(pv / (4 * n));
            }
            pushes.reverse();
            return pushes;
          }
        }
        // solved but the player lost the hallway: another route may still reach a solved state that keeps it
      }
      for (let k = 0; k < n; k++) {
        if (((mask >>> k) & 1) !== 1) continue; // boxes are the set bits; only a box is pushed
        const cell = cells[k];
        for (let dir = 0; dir < 4; dir++) {
          const d = DELTA[dir];
          const pside = cell - d;
          if (pside < 0 || pside >= size || seen[pside] !== stamp) continue;
          const t = cell + d;
          if (t < 0 || t >= size) continue;
          const ti = idx[t];
          if (ti < 0 || ((mask >>> ti) & 1) === 1) continue; // the target is a room square free of boxes
          if (live[t] === 0) continue; // a dead square never reaches a goal
          const mask2 = (mask & ~(1 << k)) | (1 << ti);
          if (frozenAt(lv, mask2, room, t)) continue;
          markStamp++;
          const key2 = flood(mask2, cell, mark, markStamp) * SPAN + mask2;
          if (visited.has(key2)) continue;
          visited.add(key2);
          parents.set(key2, key * (4 * n) + k * 4 + dir);
          if (visited.size > MAX_STATES) return null;
          next.push(key2);
        }
      }
    }
    layer = next;
  }
  return null;
}

/** The shortest walk from `from` to `to` over the board's floor around the boxes, or null when unreachable.
 *  Neighbours are tried in the fixed order u, d, l, r, so the path is always the same. */
function walkPath(lv: Level, box: Uint8Array, from: number, to: number): string | null {
  const { w, size, wall } = lv;
  if (from === to) return '';
  const prev = new Int32Array(size).fill(-2);
  prev[from] = -1;
  const queue = [from];
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (q < 0 || q >= size || wall[q] === 1 || box[q] === 1 || prev[q] !== -2) continue;
      prev[q] = p;
      if (q === to) {
        let s = '';
        for (let c = to; c !== from; c = prev[c]) {
          const d = c - prev[c];
          s = (d === -w ? 'u' : d === w ? 'd' : d === -1 ? 'l' : 'r') + s;
        }
        return s;
      }
      queue.push(q);
    }
  }
  return null;
}

/** The level's whole answer: every room solved in turn, the player walking between pushes. */
function solveLevel(lv: Level): string {
  const box = lv.box.slice();
  const DELTA = [-lv.w, lv.w, -1, 1];
  let out = '';
  let cur = lv.player;
  for (const room of lv.rooms) {
    if (room.startMask === 0) continue;
    const pushes = solveRoom(lv, room);
    if (pushes === null) return ''; // no answer rather than a partial one; only a solved replay scores
    for (const [k, dir] of pushes) {
      const cell = room.cells[k];
      const d = DELTA[dir];
      const pside = cell - d;
      const walk = walkPath(lv, box, cur, pside);
      if (walk === null) return '';
      out += walk;
      box[cell] = 0;
      box[cell + d] = 1;
      out += LETTER[dir];
      cur = cell;
    }
  }
  return out;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  let text: string;
  try {
    text = readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8');
  } catch {
    continue; // a set's root may carry seed.json beside the instances
  }
  let moves = '';
  try {
    moves = solveLevel(parse(text));
  } catch {
    moves = ''; // an empty answer stays valid
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), moves + '\n');
}

// Scripted Doom player (no cheat codes). The engine asks it for the player's command once per game tic (a hook at the end of
// G_BuildTiccmd), so it plays through the same input path as a person at the keyboard:
//  - fight: pick the nearest live monster it can see, turn toward it at a human turning speed, fire only when the aim is within
//    6 degrees, and side-step while firing;
//  - otherwise run along a route planned from the map geometry (route_e1m1.h, made by plan_route.py), steering at a point a little
//    ahead on the current leg, and press USE at doors on the route (only when that opens them);
//  - if it stops making progress, side-step and rejoin the route; after the last waypoint, flip the exit switch.
// It reads engine state and calls engine functions only (sight checks, angles), so the same file builds for the GPU (build_doom.sh)
// and for the CPU check (cpu_build.sh), and both play the same game tic for tic.
#include "doomdef.h"
#include "doomstat.h"
#include "d_player.h"
#include "d_ticcmd.h"
#include "d_event.h"
#include "p_local.h"
#include "p_spec.h"
#include "r_main.h"
#include "r_state.h"
#include "tables.h"
#include "route_e1m1.h"

#define NWP ((int)(sizeof(route_pts) / sizeof(route_pts[0])))
#define NDOOR ((int)(sizeof(route_doors) / sizeof(route_doors[0])))
#define ANG(deg) ((angle_t)((deg) * (ANG90 / 90)))

extern unsigned int doom_input_word;
static int s_wp, s_tic, s_stuck, s_unstick, s_side_dir = 1, s_was_dead;
static fixed_t s_px, s_py;
static mobj_t *s_target, *s_ignore;
static int s_target_since, s_ignore_until;

static int iabs_(int v) { return v < 0 ? -v : v; }

// Turn toward `want` by at most `maxturn` angleturn units (65536 = 360 degrees, positive = left). Returns the remaining error.
static int turn_to(ticcmd_t* cmd, angle_t cur, angle_t want, int maxturn) {
  int diff = (int)(want - cur);
  int t = diff >> 16;
  if (t > maxturn) t = maxturn;
  if (t < -maxturn) t = -maxturn;
  cmd->angleturn = (short)t;
  return diff;
}

// The nearest live monster within 1400 map units that the player can see.
static mobj_t* pick_target(mobj_t* me) {
  mobj_t* best = 0;
  fixed_t bestd = 1400 * FRACUNIT;
  for (thinker_t* th = thinkercap.next; th != &thinkercap; th = th->next) {
    if (th->function.acp1 != (actionf_p1)P_MobjThinker) continue;
    mobj_t* m = (mobj_t*)th;
    if (!(m->flags & MF_COUNTKILL) || m->health <= 0) continue;
    if (m == s_ignore && s_tic < s_ignore_until) continue;
    fixed_t d = P_AproxDistance(m->x - me->x, m->y - me->y);
    if (d < bestd && P_CheckSight(me, m)) { best = m; bestd = d; }
  }
  return best;
}

// After a fight or a side-step: the route leg (waypoint k-1 -> k) nearest to the player among the next few; head for its far end,
// so a fight that carried the player forward does not send it back to a waypoint it has passed.
static int rejoin(mobj_t* me, int from) {
  if (from < 1) from = 1;
  int best = from;
  long long bd = -1, px = me->x >> FRACBITS, py = me->y >> FRACBITS;
  for (int k = from; k < NWP && k < from + 5; k++) {
    long long ax = route_pts[k - 1][0], ay = route_pts[k - 1][1], dx = route_pts[k][0] - ax, dy = route_pts[k][1] - ay;
    long long l2 = dx * dx + dy * dy, t = (px - ax) * dx + (py - ay) * dy;
    if (t < 0) t = 0;
    if (t > l2) t = l2;
    long long qx = l2 ? ax + dx * t / l2 : ax, qy = l2 ? ay + dy * t / l2 : ay;
    long long d = (px - qx) * (px - qx) + (py - qy) * (py - qy);
    if (bd < 0 || d <= bd) { bd = d; best = k; }
  }
  return best;
}

// USE opens a closed door and re-opens a closing one, but sends an opening or open door back down: press only when it helps.
static int door_wants_use(int k) {
  sector_t* sec = &sectors[route_door_sec[k]];
  if (sec->specialdata) return ((vldoor_t*)sec->specialdata)->direction == -1;
  return sec->ceilingheight - sec->floorheight < 56 * FRACUNIT;
}

void player_ticcmd(ticcmd_t* cmd) {
  player_t* p = &players[consoleplayer];
  mobj_t* me = p->mo;
  cmd->forwardmove = 0; cmd->sidemove = 0; cmd->angleturn = 0; cmd->buttons = 0;
  if (doom_input_word != 0xFFFFFFFFu) {            // host-written input for this tic (include/kvm_ops_game.cuh): the keyboard path, no script
    cmd->forwardmove = (signed char)(doom_input_word & 255); cmd->sidemove = (signed char)((doom_input_word >> 8) & 255);
    cmd->angleturn = (short)((signed char)((doom_input_word >> 16) & 255) * 32); cmd->buttons = (byte)(doom_input_word >> 24);
    return;
  }
  if (gamestate != GS_LEVEL || !me) return;
  s_tic++;
  if (p->playerstate == PST_DEAD) {                          // dead: press USE after a second to restart the level
    s_was_dead = 1;
    if (s_tic % 35 == 0) cmd->buttons |= BT_USE;
    return;
  }
  if (s_was_dead) { s_was_dead = 0; s_wp = 0; s_target = 0; }

  // ---- fight
  if (s_tic % 4 == 0 || (s_target && s_target->health <= 0)) {
    mobj_t* had = s_target;
    s_target = pick_target(me);
    if (s_target != had) s_target_since = s_tic;
    if (had && !s_target) s_wp = rejoin(me, s_wp - 1);
  }
  if (s_target && s_target->health > 0 && s_tic - s_target_since > 140) {     // no kill in 4 s: leave that one alone for 5 s
    s_ignore = s_target; s_ignore_until = s_tic + 175; s_target = 0;
    s_wp = rejoin(me, s_wp - 1);
  }
  if (s_target && s_target->health > 0) {
    int err = turn_to(cmd, me->angle, R_PointToAngle2(me->x, me->y, s_target->x, s_target->y), 1100);
    if (iabs_(err) < (int)ANG(6)) cmd->buttons |= BT_ATTACK;
    cmd->sidemove = (signed char)(((s_tic / 45) & 1) ? 24 : -24);
    fixed_t d = P_AproxDistance(s_target->x - me->x, s_target->y - me->y);
    if (d > 700 * FRACUNIT && iabs_(err) < (int)ANG(30)) cmd->forwardmove = 25;   // close in on far targets
    s_stuck = 0;
    return;
  }

  // ---- route done: face the exit switch, step up to it and flip it
  if (s_wp >= NWP) {
    fixed_t ex = route_exit[0] * FRACUNIT, ey = route_exit[1] * FRACUNIT;
    int err = turn_to(cmd, me->angle, R_PointToAngle2(me->x, me->y, ex, ey), 1280);
    fixed_t d = P_AproxDistance(ex - me->x, ey - me->y);
    if (d > 48 * FRACUNIT && iabs_(err) < (int)ANG(30)) cmd->forwardmove = 25;
    if (d < 80 * FRACUNIT && iabs_(err) < (int)ANG(8) && (s_tic & 7) == 0) cmd->buttons |= BT_USE;
    return;
  }

  // ---- follow the route: steer at a point 56 map units ahead of the player's projection on the current leg
  if (s_wp == 0) s_wp = 1;
  long long px = me->x >> FRACBITS, py = me->y >> FRACBITS, ax, ay, dx, dy, len2, t;
  for (;;) {
    ax = route_pts[s_wp - 1][0]; ay = route_pts[s_wp - 1][1];
    dx = route_pts[s_wp][0] - ax; dy = route_pts[s_wp][1] - ay;
    len2 = dx * dx + dy * dy;
    t = (px - ax) * dx + (py - ay) * dy;
    long long ex = route_pts[s_wp][0] - px, ey = route_pts[s_wp][1] - py;
    if (t < len2 && ex * ex + ey * ey >= 40 * 40) break;    // not at the end of this leg yet
    if (++s_wp >= NWP) return;
  }
  long long len = 1;
  while (len * len < len2) len++;
  long long ahead = (t > 0 ? t / len : 0) + 56;
  if (ahead > len) ahead = len;
  fixed_t tx = (fixed_t)((ax + dx * ahead / len) << FRACBITS), ty = (fixed_t)((ay + dy * ahead / len) << FRACBITS);
  int err = iabs_(turn_to(cmd, me->angle, R_PointToAngle2(me->x, me->y, tx, ty), 1280));
  cmd->forwardmove = err < (int)ANG(25) ? 50 : (err < (int)ANG(60) ? 20 : 0);
  for (int k = 0; k < NDOOR; k++) {                          // doors on the route: USE when close, facing it, and it helps
    fixed_t dd = P_AproxDistance(route_doors[k][0] * FRACUNIT - me->x, route_doors[k][1] * FRACUNIT - me->y);
    angle_t to = R_PointToAngle2(me->x, me->y, route_doors[k][0] * FRACUNIT, route_doors[k][1] * FRACUNIT);
    if (dd < 110 * FRACUNIT && (s_tic & 7) == 0 && iabs_((int)(to - me->angle)) < (int)ANG(60) && door_wants_use(k)) cmd->buttons |= BT_USE;
  }
  // ---- not making progress: side-step for a moment, then rejoin the route
  if (s_unstick > 0) {
    s_unstick--;
    cmd->sidemove = (signed char)(s_side_dir * 40);
    cmd->forwardmove = 0;
    if (s_unstick == 0) s_wp = rejoin(me, s_wp - 1);
  } else if (s_tic % 18 == 0) {
    fixed_t moved = P_AproxDistance(me->x - s_px, me->y - s_py);
    if (cmd->forwardmove > 0 && moved < 24 * FRACUNIT) {
      if (++s_stuck >= 2) { s_unstick = 14; s_side_dir = -s_side_dir; s_stuck = 0; }
    } else s_stuck = 0;
    s_px = me->x; s_py = me->y;
  }
}

int player_waypoint(void) { return s_wp; }

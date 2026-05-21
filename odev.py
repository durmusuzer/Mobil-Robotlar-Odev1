
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.widgets import RadioButtons, Button, Slider
import heapq, random, math, time

plt.rcParams['toolbar'] = 'None'

DARK  = "#f5efe6"                               
PANEL = "#ede4d8"                           
ACC   = "#c85f35"                                                
       
C_BRIGHT = "#2d1f0f"                                
C_MID    = "#7a5c3a"                              
C_DIM    = "#b09878"                            
       
C_OK   = "#4a8c30"           
C_WARN = "#b86820"           
C_ERR  = "#b83820"                     
                                                  
C_TRUE = "#2a5298"                              
C_EKF  = "#c85f35"                                  
C_DR   = "#2a8a48"                                   
                                 
MAP_BG  = "#faf7f2"                               
SPINE_C = "#c8b090"                 
GRID_C  = "#e0d4c0"           
OBSTACLE_COLOR = "#3d2010"                              
                                                               
CLUSTER_PAL = [
    "#e63946", "#2a9d8f", "#e9c46a", "#f4a261",
    "#a8dadc", "#c77dff", "#06d6a0", "#ffd166",
    "#ef476f", "#118ab2", "#ff9f1c", "#44cf6c",
]

def _darken(hex_col, amount=0.3):
    hex_col = hex_col.lstrip("#")
    r, g, b = (int(hex_col[i:i+2], 16) for i in (0, 2, 4))
    r = int(r * (1 - amount))
    g = int(g * (1 - amount))
    b = int(b * (1 - amount))
    return f"#{r:02x}{g:02x}{b:02x}"

def _draw_obstacle(ax, x1, y1, x2, y2, seed):
    import random as _rnd
    rng = _rnd.Random(seed * 7919 + 1337)

    base   = OBSTACLE_COLOR
    light  = _lighten(base, 0.38)                    
    dark   = _darken(base, 0.30)                   
    border = _lighten(base, 0.18)                  

    w = x2 - x1
    h = y2 - y1

    ax.add_patch(patches.FancyBboxPatch(
        (x1, y1), w, h,
        boxstyle="round,pad=0.04",
        facecolor=base, edgecolor=border, lw=1.0, zorder=2))

    th = min(h * 0.14, 0.20)
    ax.add_patch(patches.Rectangle(
        (x1 + 0.04, y2 - th), w - 0.08, th - 0.02,
        facecolor=light, edgecolor="none", zorder=3, alpha=0.75))

    sw = min(w * 0.07, 0.12)
    ax.add_patch(patches.Rectangle(
        (x2 - sw, y1 + 0.04), sw - 0.02, h - 0.08,
        facecolor=dark, edgecolor="none", zorder=3, alpha=0.55))

    n_shelves = max(1, int(h / 0.65))
    shelf_color = _lighten(base, 0.50)
    for k in range(1, n_shelves + 1):
        sy = y1 + k * h / (n_shelves + 1)
        ax.plot([x1 + 0.08, x2 - sw - 0.04], [sy, sy],
                color=shelf_color, lw=0.7, alpha=0.65, zorder=4,
                solid_capstyle="round")

    if w > 1.2:
        mid_x = x1 + w * (0.45 + rng.uniform(-0.05, 0.05))
        ax.plot([mid_x, mid_x], [y1 + 0.10, y2 - th - 0.05],
                color=shelf_color, lw=0.6, alpha=0.5, zorder=4,
                solid_capstyle="round")

def _lighten(hex_col, amount=0.3):
    hex_col = hex_col.lstrip("#")
    r, g, b = (int(hex_col[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"

class Environment:
    _CACHE_RES = 0.05                                         

    def __init__(self):
        self.width  = 20.0
        self.height = 15.0
        self.start  = (1.0, 1.0)
        self.goal   = (19.0, 14.0)
        self.obstacles, self.labels = self._make_maze()
        self._build_obstacle_cache()

    def _build_obstacle_cache(self):
        r  = self._CACHE_RES
        W  = int(math.ceil(self.width  / r)) + 2
        H  = int(math.ceil(self.height / r)) + 2
                                                                        
        g0  = np.ones((H, W), dtype=np.bool_)
        g32 = np.ones((H, W), dtype=np.bool_)
        xs  = np.arange(W, dtype=np.float32) * r
        ys  = np.arange(H, dtype=np.float32) * r
        XX, YY = np.meshgrid(xs, ys)
        mg = 0.32
        g0[ (XX < 0)  | (XX > self.width)    | (YY < 0)  | (YY > self.height)   ] = False
        g32[(XX < mg) | (XX > self.width-mg)  | (YY < mg) | (YY > self.height-mg)] = False
        for (x1, y1, x2, y2) in self.obstacles:
            j0 = max(0, int((x1 - mg) / r));   j1 = min(W-1, int((x2 + mg) / r) + 2)
            i0 = max(0, int((y1 - mg) / r));   i1 = min(H-1, int((y2 + mg) / r) + 2)
            g32[i0:i1, j0:j1] = False
            j0 = max(0, int(x1 / r));           j1 = min(W-1, int(x2 / r) + 2)
            i0 = max(0, int(y1 / r));           i1 = min(H-1, int(y2 / r) + 2)
            g0[i0:i1, j0:j1] = False
        self._cache_g0  = g0
        self._cache_g32 = g32
        self._cache_W   = W
        self._cache_H   = H

    def _make_maze(self):
        from collections import deque
        W, H = self.width, self.height
        sx, sy = self.start
        gx, gy = self.goal

        CELL    = 0.65                     
        CLEAR_R = 2.0                                     
        obs     = []

        def _ok(x1, y1, x2, y2):
            if x1 < 0.35 or y1 < 0.35 or x2 > W-0.35 or y2 > H-0.35:
                return False
            cx, cy = (x1+x2)/2, (y1+y2)/2
            if math.hypot(cx-sx, cy-sy) < CLEAR_R: return False
            if math.hypot(cx-gx, cy-gy) < CLEAR_R: return False
            for ox1,oy1,ox2,oy2 in obs:
                if x1 < ox2+0.10 and x2 > ox1-0.10 and\
                   y1 < oy2+0.10 and y2 > oy1-0.10:
                    return False
            return True

        def add_cell(ix, iy):
            x1, y1 = round(ix*CELL, 2), round(iy*CELL, 2)
            x2, y2 = round(x1+CELL, 2), round(y1+CELL, 2)
            if _ok(x1, y1, x2, y2):
                obs.append((x1, y1, x2, y2))
                return True
            return False

        def pixel_burst(cx, cy):
            ci = round(cx / CELL); cj = round(cy / CELL)
            cells = set()
                               
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    cells.add((ci+di, cj+dj))
                              
            n_arms = random.randint(3, 6)
            arm_dirs = random.sample(
                [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)], n_arms)
            for dx, dy in arm_dirs:
                length = random.randint(2, 5)
                px, py = ci, cj
                for _ in range(length):
                    px += dx; py += dy
                    cells.add((px, py))
                                                      
                    if random.random() < 0.50:
                        perp = (-dy, dx)
                        side = random.choice([-1, 1])
                        cells.add((px + perp[0]*side, py + perp[1]*side))
                                                              
            border = set()
            for (ii, jj) in cells:
                for di2, dj2 in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nb = (ii+di2, jj+dj2)
                    if nb not in cells and random.random() < 0.35:
                        border.add(nb)
            cells |= border
            for ii, jj in cells:
                add_cell(ii, jj)

        def partial_wall(x0, y0, axis, length, gap_frac=0.30):
            n = int(length / CELL)
            gap_start = random.randint(1, max(1, int(n * (0.5 - gap_frac/2))))
            gap_end   = gap_start + max(2, int(n * gap_frac))
            for k in range(n):
                if gap_start <= k < gap_end: continue
                if axis == 'h':
                    add_cell(round((x0 + k*CELL)/CELL), round(y0/CELL))
                else:
                    add_cell(round(x0/CELL), round((y0 + k*CELL)/CELL))

        def path_exists():
            res = 0.55; mg = 0.42
            def free(px, py):
                if not (mg <= px <= W-mg and mg <= py <= H-mg): return False
                for ox1,oy1,ox2,oy2 in obs:
                    if ox1-mg <= px <= ox2+mg and oy1-mg <= py <= oy2+mg:
                        return False
                return True
            s = (round(sx/res), round(sy/res))
            g = (round(gx/res), round(gy/res))
            visited = {s}; q = deque([s])
            dirs8 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
            while q:
                u = q.popleft()
                if u == g: return True
                for ddx,ddy in dirs8:
                    nb = (u[0]+ddx, u[1]+ddy)
                    if nb not in visited and free(nb[0]*res, nb[1]*res):
                        visited.add(nb); q.append(nb)
            return False

        best, best_n = [], 0
        for attempt in range(15):
            obs = []
                                  
            n_bursts = random.randint(7, 11)
            for _ in range(n_bursts):
                cx = random.uniform(2.0, W-2.0)
                cy = random.uniform(2.0, H-2.0)
                pixel_burst(cx, cy)

            n_walls = random.randint(2, 3)
            for _ in range(n_walls):
                if random.random() < 0.5:
                    y0 = random.uniform(3.0, H-3.0)
                    partial_wall(0.5, y0, 'h', W-1.0,
                                 gap_frac=random.uniform(0.25, 0.40))
                else:
                    x0 = random.uniform(3.0, W-3.0)
                    partial_wall(x0, 0.5, 'v', H-1.0,
                                 gap_frac=random.uniform(0.25, 0.40))

            if path_exists():
                if len(obs) > best_n:
                    best, best_n = list(obs), len(obs)
                break
                                                            
            if len(obs) > best_n and attempt > 3:
                best, best_n = list(obs), len(obs)

        if not best:
            obs = []
            for _ in range(5):
                cx = random.uniform(3.0, W-3.0)
                cy = random.uniform(3.0, H-3.0)
                pixel_burst(cx, cy)
            best = obs

        labels = [f"P{i+1}" for i in range(len(best))]
        return best, labels

    def in_obstacle(self, x, y, margin=0.32):
        for (x1,y1,x2,y2) in self.obstacles:
            if x1-margin<=x<=x2+margin and y1-margin<=y<=y2+margin:
                return True
        return False

    def in_bounds(self, x, y, margin=0.32):
        return margin<=x<=self.width-margin and margin<=y<=self.height-margin

    def is_free(self, x, y, margin=0.32):
        r  = self._CACHE_RES
        j  = int(x / r)
        i  = int(y / r)
        if not (0 <= j < self._cache_W and 0 <= i < self._cache_H):
            return False
                                                                       
        return bool(self._cache_g0[i, j] if margin < 0.06 else self._cache_g32[i, j])

    def segment_free(self, x1,y1,x2,y2,steps=None,margin=0.32):
        dist = math.hypot(x2-x1, y2-y1)
        if steps is None:
            steps = max(10, int(dist / 0.1))
        for i in range(steps+1):
            t=i/steps
            if not self.is_free(x1+t*(x2-x1),y1+t*(y2-y1),margin):
                return False
        return True

    def teleport_obstacle(self, robot_pos, goal_pos, clear_r=2.2):
        W, H = self.width, self.height
        rx, ry = robot_pos
        gx, gy = goal_pos
        sx, sy = self.start
        candidates = [i for i,(x1,y1,x2,y2) in enumerate(self.obstacles)
                      if math.hypot((x1+x2)/2-rx,(y1+y2)/2-ry) > clear_r*1.5
                      and math.hypot((x1+x2)/2-sx,(y1+y2)/2-sy) > clear_r]
        if not candidates:
            return None
        idx = random.choice(candidates)
        x1,y1,x2,y2 = self.obstacles[idx]
        w, h = x2-x1, y2-y1
        for _ in range(200):
            nx1 = random.uniform(0.5, W-w-0.5)
            ny1 = random.uniform(0.5, H-h-0.5)
            nx2, ny2 = nx1+w, ny1+h
            cx, cy = (nx1+nx2)/2, (ny1+ny2)/2
            if math.hypot(cx-rx,cy-ry) < clear_r: continue
            if math.hypot(cx-gx,cy-gy) < clear_r: continue
            if math.hypot(cx-sx,cy-sy) < clear_r: continue
            clash = False
            for j,(ox1,oy1,ox2,oy2) in enumerate(self.obstacles):
                if j==idx: continue
                if nx1<ox2+0.4 and nx2>ox1-0.4 and ny1<oy2+0.4 and ny2>oy1-0.4:
                    clash=True; break
            if not clash:
                self.obstacles[idx] = (round(nx1,2),round(ny1,2),
                                       round(nx2,2),round(ny2,2))
                self._build_obstacle_cache()
                return idx
        return None

    def randomize(self):
        self.obstacles, self.labels = self._make_maze()
        self._build_obstacle_cache()
        self.is_random = True

    is_random=False

class LiDAR:
    def __init__(self, env, num_beams=36, max_range=8.0, noise_std=0.10):
        self.env=env; self.num_beams=num_beams
        self.max_range=max_range; self.noise_std=noise_std
        self.angles=np.linspace(0,2*np.pi,num_beams,endpoint=False)

    def scan(self, x, y, theta):
        raw, filt = [], []
        for a in self.angles:
            d=self._cast(x,y,theta+a)
            raw.append(max(0.05, d+np.random.normal(0,self.noise_std)))
            filt.append(max(0.05,d))
        return np.array(raw), np.array(filt)

    def _cast(self, x, y, angle):
        cos_a=math.cos(angle); sin_a=math.sin(angle)
        for d in np.arange(0.1, self.max_range, 0.08):
            if not self.env.is_free(x+d*cos_a, y+d*sin_a, margin=0.0):
                return d
        return self.max_range

    def to_points(self, x, y, theta, ranges):
        return [(x+r*math.cos(theta+a), y+r*math.sin(theta+a))
                for a,r in zip(self.angles,ranges)]

    def cluster_obstacles(self, raw_ranges, x, y, theta, gap_thresh=0.5):
        pts = self.to_points(x, y, theta, raw_ranges)
        clusters, cur = [], []
        for r, pt in zip(raw_ranges, pts):
            if r >= self.max_range * 0.98:
                if cur: clusters.append(cur); cur = []
                continue
            if cur:
                if math.hypot(pt[0]-cur[-1][0], pt[1]-cur[-1][1]) > gap_thresh:
                    clusters.append(cur); cur = []
            cur.append(pt)
        if cur: clusters.append(cur)
                                                              
        if len(clusters) >= 2:
            first_pt = clusters[0][0]; last_pt = clusters[-1][-1]
            if math.hypot(first_pt[0]-last_pt[0], first_pt[1]-last_pt[1]) < gap_thresh:
                clusters[0] = clusters[-1] + clusters[0]
                clusters.pop()
        return clusters

class IMU:
    def __init__(self, noise_std=0.012, bias=0.003):
        self.noise_std=noise_std; self.bias=bias

    def measure(self, omega):
        return omega+self.bias+np.random.normal(0,self.noise_std)

class Encoder:
    def __init__(self, slip=0.01):
        self.slip=slip

    def measure(self, dl, dr):
        return (dl*(1+np.random.uniform(-self.slip,self.slip)),
                dr*(1+np.random.uniform(-self.slip,self.slip)))

def _rect_poly(cx, cy, theta, w, h):
    c,s=math.cos(theta),math.sin(theta)
    hw,hh=w/2,h/2
    pts=[(-hw,-hh),(hw,-hh),(hw,hh),(-hw,hh)]
    return [(cx+c*px-s*py, cy+s*px+c*py) for px,py in pts]

def _add_rect(ax, cx,cy,theta,w,h, fc,ec="white",lw=1.2,zorder=7,alpha=1.0):
    p=patches.Polygon(_rect_poly(cx,cy,theta,w,h),closed=True,
                      fc=fc,ec=ec,lw=lw,zorder=zorder,alpha=alpha)
    ax.add_patch(p); return p

_WR = 0.038   

class DiffDriveRobot:
    name="Differential"; L=0.5; holonomic=False

    def __init__(self,x,y,theta=0.0):
        self.x=x; self.y=y; self.theta=theta
        self.wl=0.0; self.wr=0.0

    def control(self,tx,ty):
        dx=tx-self.x; dy=ty-self.y; dist=math.hypot(dx,dy)
        des=math.atan2(dy,dx)
        aerr=math.atan2(math.sin(des-self.theta),math.cos(des-self.theta))
        
        v=min(0.7, 1.5*dist)
        
        if abs(aerr) > math.pi / 2:
            v = -v           
                                               
            aerr = math.atan2(math.sin(des-self.theta+math.pi), math.cos(des-self.theta+math.pi))
            
        if abs(aerr) > 0.5:             
            v = 0.0
            
        omega=float(np.clip(3.0*aerr,-3.0,3.0))
        return v,omega

    def step(self,v,omega,dt=0.1):
        v_L = v - (omega * self.L / 2)
        v_R = v + (omega * self.L / 2)

        dl = v_L * dt
        dr = v_R * dt
        dS = (dl + dr) / 2
        dth = (dr - dl) / self.L
        
        mid = self.theta + dth / 2
        self.x += dS * math.cos(mid)
        self.y += dS * math.sin(mid)
        self.theta += dth
        
        self.wl += dl / _WR
        self.wr += dr / _WR
        return dl, dr

    def draw(self,ax):
        arts=[]
        c,s=math.cos(self.theta),math.sin(self.theta)
        fw=np.array([c,s]); lf=np.array([-s,c])
        pos=np.array([self.x,self.y])
        arts.append(_add_rect(ax,self.x,self.y,self.theta,0.40,0.25,fc="#a84820",ec="#e8a870",lw=1.2,zorder=7))
        for side,wa in [(-1,self.wl),(1,self.wr)]:
            wc=pos+lf*side*0.16
            arts.append(_add_rect(ax,wc[0],wc[1],self.theta,0.14,0.06,fc="#2a1a0a",ec="#8a6040",lw=0.8,zorder=8))
            sp_c=math.cos(wa); sp_s=math.sin(wa)
            sp_fw=fw*0.066*sp_c; sp_lf=lf*side*0.030*sp_s
            sx,sy=wc[0]+sp_fw[0]+sp_lf[0], wc[1]+sp_fw[1]+sp_lf[1]
            ln,=ax.plot([wc[0],sx],[wc[1],sy],"-",color="#c8a070",lw=1.0,zorder=9)
            arts.append(ln)
        arts.append(ax.annotate("",xy=(self.x+0.28*c,self.y+0.28*s),xytext=(self.x,self.y),
            arrowprops=dict(arrowstyle="->",color="#e8c050",lw=1.5),zorder=10))
        return arts

    def init_draw_artists(self, ax):
        c,s=math.cos(self.theta),math.sin(self.theta)
        fw=np.array([c,s]); lf=np.array([-s,c]); pos=np.array([self.x,self.y])
        body=patches.Polygon(_rect_poly(self.x,self.y,self.theta,0.40,0.25),
                             closed=True,fc="#a84820",ec="#e8a870",lw=1.2,zorder=7)
        ax.add_patch(body)
        wl_p=pos+lf*(-1)*0.16; wr_p=pos+lf*(1)*0.16
        wheel_l=patches.Polygon(_rect_poly(wl_p[0],wl_p[1],self.theta,0.14,0.06),
                                closed=True,fc="#2a1a0a",ec="#8a6040",lw=0.8,zorder=8)
        ax.add_patch(wheel_l)
        wheel_r=patches.Polygon(_rect_poly(wr_p[0],wr_p[1],self.theta,0.14,0.06),
                                closed=True,fc="#2a1a0a",ec="#8a6040",lw=0.8,zorder=8)
        ax.add_patch(wheel_r)
        spoke_l,=ax.plot([0,0],[0,0],"-",color="#c8a070",lw=1.0,zorder=9)
        spoke_r,=ax.plot([0,0],[0,0],"-",color="#c8a070",lw=1.0,zorder=9)
        arrow=patches.FancyArrowPatch(
            (self.x,self.y),(self.x+0.28*c,self.y+0.28*s),
            arrowstyle="->",color="#e8c050",linewidth=1.5,zorder=10)
        ax.add_patch(arrow)
        return dict(body=body,wheel_l=wheel_l,wheel_r=wheel_r,
                    spoke_l=spoke_l,spoke_r=spoke_r,arrow=arrow)

    def update_draw_artists(self, arts):
        c,s=math.cos(self.theta),math.sin(self.theta)
        fw=np.array([c,s]); lf=np.array([-s,c]); pos=np.array([self.x,self.y])
        arts['body'].set_xy(_rect_poly(self.x,self.y,self.theta,0.40,0.25))
        for side,wkey,skey,wa in [(-1,'wheel_l','spoke_l',self.wl),
                                   (1,'wheel_r','spoke_r',self.wr)]:
            wc=pos+lf*side*0.16
            arts[wkey].set_xy(_rect_poly(wc[0],wc[1],self.theta,0.14,0.06))
            sp_c=math.cos(wa); sp_s=math.sin(wa)
            sp_fw=fw*0.066*sp_c; sp_lf=lf*side*0.030*sp_s
            sx=wc[0]+sp_fw[0]+sp_lf[0]; sy=wc[1]+sp_fw[1]+sp_lf[1]
            arts[skey].set_data([wc[0],sx],[wc[1],sy])
        arts['arrow'].set_positions((self.x,self.y),(self.x+0.28*c,self.y+0.28*s))

class AckermannRobot:
    name="Ackermann"; L=0.60; max_steer=math.radians(40); holonomic=False

    def __init__(self,x,y,theta=0.0):
        self.x=x; self.y=y; self.theta=theta
        self.steer=0.0; self.wangle=0.0

    def control(self,tx,ty):
        dx=tx-self.x; dy=ty-self.y
        dist=math.hypot(dx,dy)
        alpha=math.atan2(dy,dx)-self.theta
        alpha=math.atan2(math.sin(alpha),math.cos(alpha))
        
        direction = 1.0
        if abs(alpha) > math.pi / 2:
            direction = -1.0       
            alpha = math.atan2(math.sin(alpha+math.pi), math.cos(alpha+math.pi))
        
        steer_des=math.atan2(2*self.L*math.sin(alpha), max(dist, 0.1))
        self.steer=float(np.clip(steer_des,-self.max_steer,self.max_steer))
        
        speed_ratio = 1.0 - 0.6 * (abs(self.steer) / self.max_steer)
        v = 0.70 * speed_ratio * direction                   
        
        omega=v*math.tan(self.steer)/self.L
        return v,omega

    def step(self,v,omega,dt=0.1):
        dth = (v / self.L) * math.tan(self.steer) * dt
        dS = v * dt
        mid = self.theta + dth / 2
        
        self.x += dS * math.cos(mid)
        self.y += dS * math.sin(mid)
        self.theta += dth
        self.wangle += dS / _WR
        
        dl = dS - dth * self.L / 2
        dr = dS + dth * self.L / 2
        return dl, dr

    def draw(self,ax):
        arts=[]
        c,s=math.cos(self.theta),math.sin(self.theta)
        fw=np.array([c,s]); lf=np.array([-s,c])
        pos=np.array([self.x,self.y])
        arts.append(_add_rect(ax,self.x,self.y,self.theta,0.56,0.24,fc="#7a3418",ec="#d49060",lw=1.2,zorder=7))
        rpos=pos+fw*0.04
        arts.append(_add_rect(ax,rpos[0],rpos[1],self.theta,0.23,0.16,fc="#5a2410",ec="#d49060",lw=0.8,zorder=8,alpha=0.9))
        rax=pos-fw*0.19
        sp_c=math.cos(self.wangle); sp_s=math.sin(self.wangle)
        for side in [-1,1]:
            wc=rax+lf*side*0.16
            arts.append(_add_rect(ax,wc[0],wc[1],self.theta,0.12,0.06,fc="#2a1a0a",ec="#8a6040",lw=0.8,zorder=9))
            sx=wc[0]+fw[0]*0.050*sp_c+lf[0]*side*0.025*sp_s
            sy=wc[1]+fw[1]*0.050*sp_c+lf[1]*side*0.025*sp_s
            ln,=ax.plot([wc[0],sx],[wc[1],sy],"-",color="#b08050",lw=1.0,zorder=10)
            arts.append(ln)
        fax=pos+fw*0.19; fth=self.theta+self.steer
        fc2=math.cos(fth); fs2=math.sin(fth)
        fw2=np.array([fc2,fs2]); lf2=np.array([-fs2,fc2])
        for side in [-1,1]:
            wc=fax+lf*side*0.16
            arts.append(_add_rect(ax,wc[0],wc[1],fth,0.12,0.06,fc="#2a1a0a",ec="#c0a070",lw=0.8,zorder=9))
            sx=wc[0]+fw2[0]*0.050*sp_c+lf2[0]*side*0.025*sp_s
            sy=wc[1]+fw2[1]*0.050*sp_c+lf2[1]*side*0.025*sp_s
            ln,=ax.plot([wc[0],sx],[wc[1],sy],"-",color="#d0b080",lw=1.0,zorder=10)
            arts.append(ln)
        if abs(self.steer)>0.06:
            R=self.L/math.tan(self.steer)
            abs_R=abs(R)
            tc_x=self.x-R*math.sin(self.theta)
            tc_y=self.y+R*math.cos(self.theta)
            ang_rob=math.degrees(math.atan2(self.y-tc_y,self.x-tc_x))
            span=75
            t1,t2=(ang_rob,ang_rob+span) if R>0 else (ang_rob-span,ang_rob)
            arc=patches.Arc((tc_x,tc_y),2*abs_R,2*abs_R,angle=0,theta1=t1,theta2=t2,color="#e8a050",lw=1.0,ls="--",zorder=5,alpha=0.75)
            ax.add_patch(arc); arts.append(arc)
            dot,=ax.plot([tc_x],[tc_y],"+",color="#e8a050",ms=5,zorder=5,alpha=0.6)
            arts.append(dot)
        arts.append(ax.annotate("",xy=(self.x+0.36*c,self.y+0.36*s),xytext=(self.x,self.y),
            arrowprops=dict(arrowstyle="->",color="#e8c050",lw=1.5),zorder=11))
        return arts

    def init_draw_artists(self, ax):
        c,s=math.cos(self.theta),math.sin(self.theta)
        fw=np.array([c,s]); lf=np.array([-s,c]); pos=np.array([self.x,self.y])
        body=patches.Polygon(_rect_poly(self.x,self.y,self.theta,0.56,0.24),
                             closed=True,fc="#7a3418",ec="#d49060",lw=1.2,zorder=7)
        ax.add_patch(body)
        rpos=pos+fw*0.04
        cabin=patches.Polygon(_rect_poly(rpos[0],rpos[1],self.theta,0.23,0.16),
                              closed=True,fc="#5a2410",ec="#d49060",lw=0.8,zorder=8,alpha=0.9)
        ax.add_patch(cabin)
        wheel_rl=patches.Polygon([(0,0)]*4,closed=True,fc="#2a1a0a",ec="#8a6040",lw=0.8,zorder=9)
        ax.add_patch(wheel_rl)
        wheel_rr=patches.Polygon([(0,0)]*4,closed=True,fc="#2a1a0a",ec="#8a6040",lw=0.8,zorder=9)
        ax.add_patch(wheel_rr)
        spoke_rl,=ax.plot([0,0],[0,0],"-",color="#b08050",lw=1.0,zorder=10)
        spoke_rr,=ax.plot([0,0],[0,0],"-",color="#b08050",lw=1.0,zorder=10)
        wheel_fl=patches.Polygon([(0,0)]*4,closed=True,fc="#2a1a0a",ec="#c0a070",lw=0.8,zorder=9)
        ax.add_patch(wheel_fl)
        wheel_fr=patches.Polygon([(0,0)]*4,closed=True,fc="#2a1a0a",ec="#c0a070",lw=0.8,zorder=9)
        ax.add_patch(wheel_fr)
        spoke_fl,=ax.plot([0,0],[0,0],"-",color="#d0b080",lw=1.0,zorder=10)
        spoke_fr,=ax.plot([0,0],[0,0],"-",color="#d0b080",lw=1.0,zorder=10)
        arc=patches.Arc((0,0),1,1,angle=0,theta1=0,theta2=90,
                        color="#e8a050",lw=1.0,ls="--",zorder=5,alpha=0.75)
        arc.set_visible(False); ax.add_patch(arc)
        dot,=ax.plot([0],[0],"+",color="#e8a050",ms=5,zorder=5,alpha=0.6)
        dot.set_visible(False)
        arrow=patches.FancyArrowPatch(
            (self.x,self.y),(self.x+0.36*c,self.y+0.36*s),
            arrowstyle="->",color="#e8c050",linewidth=1.5,zorder=11)
        ax.add_patch(arrow)
        return dict(body=body,cabin=cabin,
                    wheel_rl=wheel_rl,wheel_rr=wheel_rr,spoke_rl=spoke_rl,spoke_rr=spoke_rr,
                    wheel_fl=wheel_fl,wheel_fr=wheel_fr,spoke_fl=spoke_fl,spoke_fr=spoke_fr,
                    arc=arc,dot=dot,arrow=arrow)

    def update_draw_artists(self, arts):
        c,s=math.cos(self.theta),math.sin(self.theta)
        fw=np.array([c,s]); lf=np.array([-s,c]); pos=np.array([self.x,self.y])
        arts['body'].set_xy(_rect_poly(self.x,self.y,self.theta,0.56,0.24))
        rpos=pos+fw*0.04
        arts['cabin'].set_xy(_rect_poly(rpos[0],rpos[1],self.theta,0.23,0.16))
        sp_c=math.cos(self.wangle); sp_s=math.sin(self.wangle)
        rax=pos-fw*0.19
        for side,wkey,skey in [(-1,'wheel_rl','spoke_rl'),(1,'wheel_rr','spoke_rr')]:
            wc=rax+lf*side*0.16
            arts[wkey].set_xy(_rect_poly(wc[0],wc[1],self.theta,0.12,0.06))
            sx=wc[0]+fw[0]*0.050*sp_c+lf[0]*side*0.025*sp_s
            sy=wc[1]+fw[1]*0.050*sp_c+lf[1]*side*0.025*sp_s
            arts[skey].set_data([wc[0],sx],[wc[1],sy])
        fax=pos+fw*0.19; fth=self.theta+self.steer
        fc2=math.cos(fth); fs2=math.sin(fth)
        fw2=np.array([fc2,fs2]); lf2=np.array([-fs2,fc2])
        for side,wkey,skey in [(-1,'wheel_fl','spoke_fl'),(1,'wheel_fr','spoke_fr')]:
            wc=fax+lf*side*0.16
            arts[wkey].set_xy(_rect_poly(wc[0],wc[1],fth,0.12,0.06))
            sx=wc[0]+fw2[0]*0.050*sp_c+lf2[0]*side*0.025*sp_s
            sy=wc[1]+fw2[1]*0.050*sp_c+lf2[1]*side*0.025*sp_s
            arts[skey].set_data([wc[0],sx],[wc[1],sy])
        if abs(self.steer)>0.06:
            R=self.L/math.tan(self.steer); abs_R=abs(R)
            tc_x=self.x-R*math.sin(self.theta); tc_y=self.y+R*math.cos(self.theta)
            ang_rob=math.degrees(math.atan2(self.y-tc_y,self.x-tc_x))
            span=75; t1,t2=(ang_rob,ang_rob+span) if R>0 else (ang_rob-span,ang_rob)
            arts['arc'].center=(tc_x,tc_y); arts['arc'].width=2*abs_R
            arts['arc'].height=2*abs_R; arts['arc'].theta1=t1; arts['arc'].theta2=t2
            arts['arc'].set_visible(True)
            arts['dot'].set_data([tc_x],[tc_y]); arts['dot'].set_visible(True)
        else:
            arts['arc'].set_visible(False); arts['dot'].set_visible(False)
        arts['arrow'].set_positions((self.x,self.y),(self.x+0.36*c,self.y+0.36*s))

ROBOT_TYPES={"Differential":DiffDriveRobot,"Ackermann":AckermannRobot}

class DeadReckoning:
    def __init__(self,x,y,theta,L=0.5):
        self.x=x; self.y=y; self.theta=theta; self.L=L
        self.history=[(x,y)]

    def update(self,dl,dr):
        dS=(dl+dr)/2; dth=(dr-dl)/self.L
        self.x+=dS*math.cos(self.theta+dth/2)
        self.y+=dS*math.sin(self.theta+dth/2)
        self.theta+=dth
        self.history.append((self.x,self.y))

class EKF:
    def __init__(self,x,y,theta,env,L=0.5):
        self.mu=np.array([x,y,theta],dtype=float)
        self.P=np.eye(3)*0.05
        self.L=L; self.env=env
        self.Q=np.diag([0.002,0.002,0.001])
        self.R_imu=np.array([[8e-4]])
        self.R_lid=0.03
        self._theta_prev=theta
        self._lid_ctr=0
        self.history=[(x,y)]

    def predict(self,dl,dr):
        self._theta_prev=self.mu[2]
        dS=(dl+dr)/2; dth=(dr-dl)/self.L
        mid=self.mu[2]+dth/2
        F=np.array([[1,0,-dS*math.sin(mid)],
                    [0,1, dS*math.cos(mid)],
                    [0,0,1]])
        self.mu[0]+=dS*math.cos(mid)
        self.mu[1]+=dS*math.sin(mid)
        self.mu[2]+=dth
        self.P=F@self.P@F.T+self.Q

    def update_imu(self,omega_meas,dt):
        z_th=self._theta_prev+omega_meas*dt
        H=np.array([[0,0,1]])
        S=H@self.P@H.T+self.R_imu
        K=(self.P@H.T)/S[0,0]
        innov=math.atan2(math.sin(z_th-self.mu[2]),math.cos(z_th-self.mu[2]))
        self.mu+=K.flatten()*innov
        self.P=(np.eye(3)-np.outer(K,H))@self.P

    def update_lidar(self,lidar_raw,lidar_angles):
        self._lid_ctr+=1
        self.history.append((float(self.mu[0]),float(self.mu[1])))
        if self._lid_ctr%4!=0: return
        step=9
        for i in range(0,len(lidar_raw),step):
            z_m=lidar_raw[i]
            if z_m>7.4: continue
            aw=lidar_angles[i]+self.mu[2]
            z_e=self._cast(self.mu[0],self.mu[1],aw)
            if z_e>7.4: continue
            innov=z_m-z_e
            if abs(innov)>1.2: continue
            eps=0.05
            h_x=(self._cast(self.mu[0]+eps,self.mu[1],aw)-z_e)/eps
            h_y=(self._cast(self.mu[0],self.mu[1]+eps,aw)-z_e)/eps
            h_t=(self._cast(self.mu[0],self.mu[1],aw+eps)-z_e)/eps
            H=np.array([[h_x,h_y,h_t]])
            S_val=float((H@self.P@H.T)[0,0])+self.R_lid
            if abs(S_val)<1e-9: continue
            K=(self.P@H.T)/S_val
            self.mu+=K.flatten()*innov
            self.P=(np.eye(3)-np.outer(K.flatten(),H))@self.P

    def _cast(self,x,y,angle):
        ca=math.cos(angle); sa=math.sin(angle)
        for d in np.arange(0.1,8.0,0.12):
            if not self.env.is_free(x+d*ca,y+d*sa,margin=0.0):
                return d
        return 8.0

class AStarPlanner:
    name="A*"
    def __init__(self,env,res=0.4,margin=0.32): self.env=env; self.res=res; self.mg=margin

    def plan(self,start,goal):
        g=lambda p:(int(round(p[0]/self.res)),int(round(p[1]/self.res)))
        w=lambda c:(c[0]*self.res,c[1]*self.res)
        s=g(start); heap=[(0.0,s)]; came={}; gs={s:0.0}; cl=set()
        dirs=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        while heap:
            _,cur=heapq.heappop(heap)
            if cur in cl: continue
            cl.add(cur); wx,wy=w(cur)
            if math.hypot(wx-goal[0],wy-goal[1])<self.res*1.5:
                path=[]; c=cur
                while c in came: path.append(w(c)); c=came[c]
                return list(reversed(path))+[goal]
            for dx,dy in dirs:
                nb=(cur[0]+dx,cur[1]+dy)
                if nb in cl: continue
                wx2,wy2=w(nb)
                if not self.env.is_free(wx2,wy2,self.mg): continue
                ng=gs[cur]+math.hypot(dx,dy)*self.res
                if ng<gs.get(nb,1e18):
                    came[nb]=cur; gs[nb]=ng
                    heapq.heappush(heap,(ng+math.hypot(wx2-goal[0],wy2-goal[1]),nb))
        print(f"Uyarı: {self.name} ile geçerli bir rota bulunamadı!")
        return [start, start]

class DijkstraPlanner:
    name="Dijkstra"
    def __init__(self,env,res=0.4,margin=0.32): self.env=env; self.res=res; self.mg=margin

    def plan(self,start,goal):
        g=lambda p:(int(round(p[0]/self.res)),int(round(p[1]/self.res)))
        w=lambda c:(c[0]*self.res,c[1]*self.res)
        s=g(start); heap=[(0.0,s)]; came={}; dist={s:0.0}; cl=set()
        dirs=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        while heap:
            d,cur=heapq.heappop(heap)
            if cur in cl: continue
            cl.add(cur); wx,wy=w(cur)
            if math.hypot(wx-goal[0],wy-goal[1])<self.res*1.5:
                path=[]; c=cur
                while c in came: path.append(w(c)); c=came[c]
                return list(reversed(path))+[goal]
            for dx,dy in dirs:
                nb=(cur[0]+dx,cur[1]+dy)
                if nb in cl: continue
                wx2,wy2=w(nb)
                if not self.env.is_free(wx2,wy2,self.mg): continue
                nd=d+math.hypot(dx,dy)*self.res
                if nd<dist.get(nb,1e18):
                    came[nb]=cur; dist[nb]=nd
                    heapq.heappush(heap,(nd,nb))
        print(f"Uyarı: {self.name} ile geçerli bir rota bulunamadı!")
        return [start, start]

class RRTPlanner:
    name="RRT"
    def __init__(self,env,res=None,step=0.6,max_iter=6000,goal_bias=0.12,margin=0.32):
        self.env=env; self.step=step
        self.max_iter=max_iter; self.goal_bias=goal_bias; self.mg=margin

    def plan(self,start,goal):
        bd=self.mg+0.05
        nodes=[start]; parent={0:-1}
        for _ in range(self.max_iter):
            samp=(goal if random.random()<self.goal_bias else
                  (random.uniform(bd,self.env.width-bd),
                   random.uniform(bd,self.env.height-bd)))
            dists=[math.hypot(n[0]-samp[0],n[1]-samp[1]) for n in nodes]
            ni=int(np.argmin(dists)); nn=nodes[ni]; d=dists[ni]
            t=min(self.step/d,1.0) if d>0 else 1.0
            nw=(nn[0]+t*(samp[0]-nn[0]),nn[1]+t*(samp[1]-nn[1]))
            if not self.env.is_free(*nw,self.mg): continue
            if not self.env.segment_free(nn[0],nn[1],nw[0],nw[1],margin=self.mg): continue
            idx=len(nodes); nodes.append(nw); parent[idx]=ni
            if math.hypot(nw[0]-goal[0],nw[1]-goal[1])<self.step:
                if not self.env.segment_free(nw[0],nw[1],goal[0],goal[1],margin=self.mg):
                    continue
                gi=len(nodes); nodes.append(goal); parent[gi]=idx
                path=[]; c=gi
                while c!=-1: path.append(nodes[c]); c=parent[c]
                return list(reversed(path))
        print(f"Uyarı: {self.name} ile geçerli bir rota bulunamadı!")
        return [start, start]

class RRTStarPlanner:
    name="RRT*"
    def __init__(self,env,res=None,step=0.6,max_iter=6000,goal_bias=0.12,margin=0.32,search_r=1.8):
        self.env=env; self.step=step; self.max_iter=max_iter
        self.goal_bias=goal_bias; self.mg=margin; self.search_r=search_r

    def plan(self,start,goal):
        bd=self.mg+0.05
        nodes=[start]; parent={0:-1}; costs={0:0.0}
        for _ in range(self.max_iter):
            samp=(goal if random.random()<self.goal_bias else
                  (random.uniform(bd,self.env.width-bd),random.uniform(bd,self.env.height-bd)))
            dists=[math.hypot(n[0]-samp[0],n[1]-samp[1]) for n in nodes]
            ni=int(np.argmin(dists)); nn=nodes[ni]; d=dists[ni]
            t=min(self.step/d,1.0) if d>0 else 1.0
            nw=(nn[0]+t*(samp[0]-nn[0]),nn[1]+t*(samp[1]-nn[1]))
            if not self.env.is_free(*nw,self.mg): continue
            if not self.env.segment_free(nn[0],nn[1],nw[0],nw[1],margin=self.mg): continue
            
            near_idx = [i for i, n in enumerate(nodes) if math.hypot(n[0]-nw[0], n[1]-nw[1]) < self.search_r]
            min_cost = costs[ni] + math.hypot(nn[0]-nw[0], nn[1]-nw[1])
            best_parent = ni
            
            for idx in near_idx:
                c = costs[idx] + math.hypot(nodes[idx][0]-nw[0], nodes[idx][1]-nw[1])
                if c < min_cost and self.env.segment_free(nodes[idx][0],nodes[idx][1],nw[0],nw[1],margin=self.mg):
                    min_cost = c; best_parent = idx
                    
            new_idx = len(nodes)
            nodes.append(nw); parent[new_idx] = best_parent; costs[new_idx] = min_cost
            
            for idx in near_idx:
                c = costs[new_idx] + math.hypot(nodes[idx][0]-nw[0], nodes[idx][1]-nw[1])
                if c < costs[idx] and self.env.segment_free(nw[0],nw[1],nodes[idx][0],nodes[idx][1],margin=self.mg):
                    parent[idx] = new_idx; costs[idx] = c
                    
            if math.hypot(nw[0]-goal[0],nw[1]-goal[1])<self.step:
                if self.env.segment_free(nw[0],nw[1],goal[0],goal[1],margin=self.mg):
                    gi = len(nodes); nodes.append(goal); parent[gi] = new_idx
                    path=[]; c=gi
                    while c!=-1: path.append(nodes[c]); c=parent[c]
                    return list(reversed(path))
        print(f"Uyarı: {self.name} ile geçerli bir rota bulunamadı!")
        return [start, start]

class PRMPlanner:
    name="PRM"
    def __init__(self,env,res=0.4,margin=0.32,num_samples=500,k_neighbors=12):
        self.env=env; self.res=res; self.mg=margin
        self.num_samples=num_samples; self.k=k_neighbors

    def plan(self,start,goal):
        bd = self.mg+0.05
        nodes = [start, goal]
        for _ in range(self.num_samples):
            x, y = random.uniform(bd,self.env.width-bd), random.uniform(bd,self.env.height-bd)
            if self.env.is_free(x, y, self.mg):
                nodes.append((x,y))

        edges = {i: [] for i in range(len(nodes))}
        visited_pairs = set()
        for i in range(len(nodes)):
            dists = sorted(
                ((math.hypot(nodes[i][0]-nodes[j][0], nodes[i][1]-nodes[j][1]), j)
                 for j in range(len(nodes)) if j != i)
            )
            connected = 0
            for d, j in dists:
                if connected >= self.k: break
                if d > 4.5: break
                pair = (min(i,j), max(i,j))
                if pair in visited_pairs: connected += 1; continue
                visited_pairs.add(pair)
                if self.env.segment_free(nodes[i][0], nodes[i][1], nodes[j][0], nodes[j][1], margin=self.mg):
                    edges[i].append((d, j)); edges[j].append((d, i))
                connected += 1

        visited = set()
        heap = [(0.0, 0)]; came = {}; costs = {0: 0.0}; goal_idx = 1
        while heap:
            d, cur = heapq.heappop(heap)
            if cur in visited: continue
            visited.add(cur)
            if cur == goal_idx:
                path = []; c = cur
                while c in came:
                    path.append(nodes[c]); c = came[c]
                path.append(nodes[0])
                return list(reversed(path))
            for cost, nxt in edges[cur]:
                nc = d + cost
                if nc < costs.get(nxt, 1e18):
                    costs[nxt] = nc; came[nxt] = cur
                    heapq.heappush(heap, (nc, nxt))
        print(f"Uyarı: {self.name} ile geçerli bir rota bulunamadı!")
        return [start, start]

class DStarLitePlanner:
    name = "D* Lite"

    def __init__(self, env, res=0.4, margin=0.32):
        self.env = env; self.res = res; self.mg = margin
        self._dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        self._g    = {}
        self._goal = None

    def _grid(self, p):
        return (int(round(p[0]/self.res)), int(round(p[1]/self.res)))

    def _world(self, c):
        return (c[0]*self.res, c[1]*self.res)

    def _free(self, c):
        wx, wy = self._world(c)
        return self.env.is_free(wx, wy, self.mg)

    def _diag_ok(self, frm, dx, dy):
        return self._free((frm[0]+dx, frm[1])) and self._free((frm[0], frm[1]+dy))

    def _backward_dijkstra(self, goal):
        s_goal = self._grid(goal)
        g = {s_goal: 0.0}
        heap = [(0.0, s_goal)]
        closed = set()
        res = self.res
        while heap:
            d, u = heapq.heappop(heap)
            if u in closed: continue
            closed.add(u)
            for dx, dy in self._dirs:
                nb = (u[0]+dx, u[1]+dy)
                if nb in closed or not self._free(nb): continue
                if dx != 0 and dy != 0 and not self._diag_ok(u, dx, dy): continue
                nd = d + math.hypot(dx, dy) * res
                if nd < g.get(nb, 1e18):
                    g[nb] = nd
                    heapq.heappush(heap, (nd, nb))
        return g

    def _extract(self, start, g, goal):
        res = self.res
        s_cur = self._grid(start)
                                                                              
        if g.get(s_cur, 1e18) >= 1e18:
            best_nb = None; best_g = 1e18
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nb = (s_cur[0]+dx, s_cur[1]+dy)
                    ng = g.get(nb, 1e18)
                    if ng < best_g and self._free(nb):
                        best_g = ng; best_nb = nb
            if best_nb is None:
                return [start, start]
            s_cur = best_nb
        path = [start]; cur = s_cur
        for _ in range(2000):
            wx, wy = self._world(cur)
            if math.hypot(wx-goal[0], wy-goal[1]) < res * 1.5:
                path.append(goal); return path
            g_cur = g.get(cur, 1e18)
            best_nb = None; best_g = g_cur - 1e-9                        
            for dx, dy in self._dirs:
                nb = (cur[0]+dx, cur[1]+dy)
                if not self._free(nb): continue
                if dx != 0 and dy != 0 and not self._diag_ok(cur, dx, dy): continue
                ng = g.get(nb, 1e18)
                if ng < best_g: best_g = ng; best_nb = nb
            if best_nb is None: break
            path.append(self._world(best_nb)); cur = best_nb
        return [start, start]

    def plan(self, start, goal):
        self._goal = goal
        self._g    = self._backward_dijkstra(goal)
        path = self._extract(start, self._g, goal)
        if len(path) <= 2:
            print("Uyarı: D* Lite ile geçerli bir rota bulunamadı!")
        return path

    def notify_position(self, robot_pos):
        return None

PLANNERS={"A*":AStarPlanner,"Dijkstra":DijkstraPlanner,
          "RRT":RRTPlanner,"RRT*":RRTStarPlanner,
          "PRM":PRMPlanner,"D* Lite":DStarLitePlanner}

ROBOT_PLAN_PARAMS={
    "Differential": {"res":0.35,"margin":0.30},
    "Ackermann":    {"res":0.35,"margin":0.32},
}

def _smooth_path(path, env, margin, steps=None):
    if len(path)<3: return path

    WINDOW    = 30                                             
    SEG_STEPS = 18                                                     

    result=[path[0]]; i=0
    while i<len(path)-1:
        best=i+1
        scan_end = min(len(path)-1, i + WINDOW)
        for j in range(scan_end, i+1, -1):
            if env.segment_free(result[-1][0],result[-1][1],
                                path[j][0],path[j][1],
                                steps=SEG_STEPS, margin=margin):
                best=j; break
        result.append(path[best]); i=best

    dense = []
    for k in range(len(result)-1):
        p1 = np.array(result[k])
        p2 = np.array(result[k+1])
        dist = np.linalg.norm(p2 - p1)
        num_pts = max(int(dist / 0.12), 2)
        for t in np.linspace(0, 1, num_pts, endpoint=False):
            dense.append((p1 + t * (p2 - p1)))
    dense.append(np.array(result[-1]))

    smoothed = []
    window = min(6, len(dense)//4)
    for idx in range(len(dense)):
        s = max(0, idx - window)
        e = min(len(dense), idx + window + 1)
        pts = dense[s:e]
        sx = sum(p[0] for p in pts) / len(pts)
        sy = sum(p[1] for p in pts) / len(pts)
        if env.is_free(sx, sy, margin=0.1):
            smoothed.append((float(sx), float(sy)))
        else:
            smoothed.append((float(dense[idx][0]), float(dense[idx][1])))

    return smoothed

class Simulation:
    def __init__(self):
        self.env=Environment()
        self.lidar=LiDAR(self.env)
        self.imu=IMU(); self.encoder=Encoder()
        self.nav_algo="A*"; self.loc_algo="EKF"
        self.robot_type="Differential"
        self.reset()

    def reset(self, initial_theta=0.0):
        sx,sy=self.env.start
        self.robot=ROBOT_TYPES[self.robot_type](sx,sy,initial_theta)
        self.dr=DeadReckoning(sx,sy,initial_theta,L=self.robot.L)
        self.ekf=EKF(sx,sy,initial_theta,self.env,L=self.robot.L)
        self.true_path=[(sx,sy)]
        self.lidar_raw=[]; self.lidar_filt=[]
        self.errors_ekf=[]; self.errors_dr=[]
        self.times=[]; self.t=0.0
        self.path_idx=0; self.done=False
        self.path_error=False
        self._escape_steps=0
        self._rep_smooth=0.0
        self._stuck_ctr=0
        self._prev_pos=(sx,sy)
        self.lidar_raw_ranges=[]

        pp=ROBOT_PLAN_PARAMS[self.robot_type]
        if self.nav_algo == "D* Lite":
                                                                     
            self._dstar = DStarLitePlanner(self.env, **pp)
            raw = self._dstar.plan(self.env.start, self.env.goal)
        else:
            self._dstar = None
            raw = PLANNERS[self.nav_algo](self.env, **pp).plan(
                self.env.start, self.env.goal)

        if len(raw) < 3:
                                                                      
            if self.env.segment_free(sx, sy, self.env.goal[0], self.env.goal[1], margin=pp["margin"]):
                raw = [self.env.start, self.env.goal]
            else:
                self.path_error = True
                self.plan_path = [self.env.start]
                self.done = True
                return
        self.path_error = False
        self.plan_path = _smooth_path(raw, self.env, pp["margin"], steps=None)

    def step(self,dt=0.1):
        if self.done: return
        r=self.robot

        raw,filt=self.lidar.scan(r.x,r.y,r.theta)
        self.lidar_raw=self.lidar.to_points(r.x,r.y,r.theta,raw)
        self.lidar_filt=self.lidar.to_points(r.x,r.y,r.theta,filt)
        self.lidar_raw_ranges=raw

        if self._dstar is not None:
            pp  = ROBOT_PLAN_PARAMS[self.robot_type]
            mg  = pp["margin"]
            self._replan_cooldown = max(0, getattr(self, '_replan_cooldown', 0) - 1)
            check_end = min(self.path_idx + 25, len(self.plan_path))
            path_blocked = any(
                not self.env.is_free(pt[0], pt[1], mg)
                for pt in self.plan_path[self.path_idx:check_end]
            )
            if path_blocked and self._replan_cooldown == 0:
                self._replan_cooldown = 20                                       
                self._dstar._g = self._dstar._backward_dijkstra(self.env.goal)
                raw = self._dstar._extract((r.x, r.y), self._dstar._g, self.env.goal)
                if len(raw) < 3:
                    raw = self._dstar.plan((r.x, r.y), self.env.goal)
                if len(raw) >= 3:
                    self.plan_path = _smooth_path(raw, self.env, mg)
                    self.path_idx  = 0
                else:
                                                           
                    self.path_error = True
                    self.done       = True
                    return

        DANGER_DIST  = 0.14                                     
        ESCAPE_STEPS = 12                                
        SAFE_RESUME  = 0.35                                              

        min_lidar = float(np.min(filt))
        near_goal = math.hypot(r.x - self.env.goal[0], r.y - self.env.goal[1]) < 2.0

        if min_lidar < DANGER_DIST and self._escape_steps == 0 and not near_goal:
            etgt = self._escape_target(filt)
            route_clear = (
                self.env.is_free(etgt[0], etgt[1], margin=0.15) and
                self.env.segment_free(r.x, r.y, etgt[0], etgt[1], margin=0.15)
            )
            if route_clear:
                self._escape_steps = ESCAPE_STEPS

        if self._escape_steps > 0 and min_lidar > SAFE_RESUME:
            self._escape_steps = 0

        if near_goal:
                                                                                  
            self._escape_steps = 0
            L_d = 1.5 if self.robot_type == "Ackermann" else 0.6
            target = self._lookahead_target(L_d=L_d)
        elif self._escape_steps > 0:
            self._escape_steps -= 1
            etgt = self._escape_target(filt)
            if (self.env.is_free(etgt[0], etgt[1], margin=0.15) and
                    self.env.segment_free(r.x, r.y, etgt[0], etgt[1], margin=0.15)):
                target = etgt
            else:
                self._escape_steps = 0
                target = self._lookahead_target(L_d=1.5 if self.robot_type=="Ackermann" else 0.6)
        elif self.robot_type=="Ackermann":
            target = self._lookahead_target(L_d=1.5)
            target = self._lidar_repulsion(target, filt)
        else:
            target = self._lookahead_target(L_d=0.6)
            target = self._lidar_repulsion(target, filt)

        GOAL_R = 0.40                                                  
        if math.hypot(r.x-self.env.goal[0], r.y-self.env.goal[1]) < GOAL_R:
            self.done=True; return

        v,omega=self.robot.control(target[0],target[1])
        dl_t,dr_t=self.robot.step(v,omega,dt)
        r.x=float(np.clip(r.x,0.1,self.env.width-0.1))
        r.y=float(np.clip(r.y,0.1,self.env.height-0.1))
        dl_e,dr_e=self.encoder.measure(dl_t,dr_t)
        om_i=self.imu.measure(omega)

        self.dr.update(dl_e,dr_e)
        self.ekf.predict(dl_e,dr_e)
        self.ekf.update_imu(om_i,dt)
        self.ekf.update_lidar(raw,self.lidar.angles)

        self.true_path.append((self.robot.x,self.robot.y))
        self.errors_ekf.append(math.hypot(self.ekf.mu[0]-self.robot.x,
                                          self.ekf.mu[1]-self.robot.y))
        self.errors_dr.append(math.hypot(self.dr.x-self.robot.x,
                                         self.dr.y-self.robot.y))
        self.t+=dt; self.times.append(self.t)

        moved=math.hypot(self.robot.x-self._prev_pos[0],
                         self.robot.y-self._prev_pos[1])
        self._prev_pos=(self.robot.x,self.robot.y)
        if moved<0.02: self._stuck_ctr+=1
        else: self._stuck_ctr=0
        if self._stuck_ctr>=25 and not self.done:
            self._replan(); self._stuck_ctr=0

    def _replan(self):
        pp=ROBOT_PLAN_PARAMS[self.robot_type]
        if self._dstar is not None:
            raw = self._dstar._extract(
                (self.robot.x, self.robot.y), self._dstar._g, self.env.goal)
        else:
            raw=PLANNERS[self.nav_algo](self.env,**pp).plan(
                (self.robot.x,self.robot.y),self.env.goal)
        if len(raw) < 3:
            if self.env.segment_free(self.robot.x, self.robot.y,
                                     self.env.goal[0], self.env.goal[1], margin=pp["margin"]):
                raw = [(self.robot.x, self.robot.y), self.env.goal]
            else:
                return
        self.plan_path=_smooth_path(raw,self.env,pp["margin"])
        self.path_idx=0

    def _lookahead_target(self, L_d=1.5):
        path = self.plan_path; r = self.robot; n = len(path)
        
        search_start = max(0, self.path_idx - 10)
        search_range = min(search_start + 50, n)
        dists = [math.hypot(path[i][0]-r.x, path[i][1]-r.y) for i in range(search_start, search_range)]
        if not dists: return path[-1]

        self.path_idx = search_start + int(np.argmin(dists))
        margin = ROBOT_PLAN_PARAMS[self.robot_type]["margin"] * 0.7           
        
        candidate = path[-1]
        for i in range(self.path_idx, n):
            pt = path[i]
            if math.hypot(pt[0]-r.x, pt[1]-r.y) >= L_d:
                candidate = pt
                break
                
        if not self.env.segment_free(r.x, r.y, candidate[0], candidate[1], margin=margin):
                                                                                                         
            safe_idx = min(self.path_idx + 3, n - 1)
            return path[safe_idx]
            
        return candidate

    def _escape_target(self, raw_ranges):
        r = self.robot
        global_angles = self.lidar.angles + r.theta
        ex = ey = 0.0
        for ga, d in zip(global_angles, raw_ranges):
            if d < 1.2:
                w = 1.0 / max(d, 0.08) ** 2
                ex -= math.cos(ga) * w
                ey -= math.sin(ga) * w
        dist = math.hypot(ex, ey)
        if dist < 1e-6:
                                                  
            ex, ey = -math.cos(r.theta), -math.sin(r.theta)
        else:
            ex /= dist; ey /= dist
        tx = float(np.clip(r.x + ex * 1.2, 0.3, self.env.width  - 0.3))
        ty = float(np.clip(r.y + ey * 1.2, 0.3, self.env.height - 0.3))
        return (tx, ty)

    def _lidar_repulsion(self, target, raw_ranges):
        REP_DIST = 0.45                                  
        REP_K    = 0.25                  
        GAP_THR  = 0.80                                                        
        MIN_FORCE = 0.15                                                              

        r = self.robot
        rx, ry = r.x, r.y
        global_angles = self.lidar.angles + r.theta

        dx, dy = target[0] - rx, target[1] - ry
        pd = math.hypot(dx, dy)
        if pd < 1e-6:
            return target
        fwd_x, fwd_y = dx / pd, dy / pd
                            
        perp_x, perp_y = -fwd_y, fwd_x

        rep_x = rep_y = 0.0
        left_close = right_close = False
        left_min = right_min = self.lidar.max_range

        for ga, d in zip(global_angles, raw_ranges):
            if d >= REP_DIST:
                continue
            bx, by = math.cos(ga), math.sin(ga)
                                                         
            side = bx * perp_x + by * perp_y
            if side > 0.15:
                left_min = min(left_min, d)
            elif side < -0.15:
                right_min = min(right_min, d)
            strength = REP_K * (1.0 / max(d, 0.15) - 1.0 / REP_DIST)
            rep_x -= bx * strength
            rep_y -= by * strength
        left_close  = left_min  < GAP_THR
        right_close = right_min < GAP_THR

        in_gap = left_close and right_close
        if in_gap or (rep_x == 0.0 and rep_y == 0.0):
            self._rep_smooth *= 0.5                               
            if abs(self._rep_smooth) < MIN_FORCE:
                return target
            dot_perp = self._rep_smooth
        else:
            raw_perp = rep_x * perp_x + rep_y * perp_y
                                                                        
            self._rep_smooth = 0.55 * self._rep_smooth + 0.45 * raw_perp
            dot_perp = self._rep_smooth

        if abs(dot_perp) < MIN_FORCE:
            return target

        tx = target[0] + dot_perp * perp_x
        ty = target[1] + dot_perp * perp_y

        if self.env.is_free(tx, ty, margin=0.10):
            return (tx, ty)
        return target

ALGO_INFO = {
    "A*": (
        "A*  (A-Star)",
        "Bilgi güdümlü en kısa yol algoritması.\n\n"
        "Nasıl çalışır:\n"
        "  • Başlangıçtan hedefe doğru ızgara üzerinde arama yapar.\n"
        "  • Her düğümü  f(n) = g(n) + h(n)  maliyetiyle sıralar:\n"
        "      g(n) = başlangıçtan n'e gerçek maliyet\n"
        "      h(n) = n'den hedefe tahmini mesafe (sezgisel)\n"
        "  • Sezgisel kabul edilebilir (gerçek maliyeti aşmaz) olduğu\n"
        "    sürece her zaman optimal yolu bulur.\n\n"
        "Karmaşıklık:  O(b^d)  — b: dallanma, d: derinlik\n"
        "Bellek:       O(b^d)\n\n"
        "Avantaj:   Optimal + Hızlı\n"
        "Dezavantaj: Büyük haritalarda bellek yoğun"
    ),
    "Dijkstra": (
        "Dijkstra Algoritması",
        "Ağırlıklı grafta en kısa yol (sezgisel yok).\n\n"
        "Nasıl çalışır:\n"
        "  • Başlangıç düğümünden tüm düğümlere en kısa mesafeyi hesaplar.\n"
        "  • Her adımda henüz işlenmemiş en düşük maliyetli düğümü seçer.\n"
        "  • A*'ın özel hali: h(n) = 0 (sezgisel yok).\n"
        "  • Tüm kenar maliyetleri ≥ 0 olduğu sürece garantili optimal.\n\n"
        "Karmaşıklık:  O((V + E) log V)\n"
        "Bellek:       O(V)\n\n"
        "Avantaj:   Her zaman optimal, sezgisele gerek yok\n"
        "Dezavantaj: A*'dan daha fazla düğüm ziyaret eder (yavaş)"
    ),
    "RRT": (
        "RRT  (Rapidly-exploring Random Tree)",
        "Rastgele örneklemeli ağaç tabanlı arama.\n\n"
        "Nasıl çalışır:\n"
        "  • Başlangıç noktasından rastgele örnekler alarak ağaç büyütür.\n"
        "  • Her adımda haritada rastgele bir nokta seçer,\n"
        "    ağaçtaki en yakın düğümden o yöne doğru küçük adım atar.\n"
        "  • Engel yoksa yeni düğümü ağaca ekler.\n"
        "  • Ağaç hedef bölgesine ulaşınca durur.\n\n"
        "Karmaşıklık:  O(n log n)  — n: iterasyon sayısı\n"
        "Olasılıksal tam: Sonsuz iterasyonda yolu bulur\n\n"
        "Avantaj:   Yüksek boyutlu uzaylarda çalışır, uygulaması kolay\n"
        "Dezavantaj: Optimal değil, yol pürüzlü çıkabilir"
    ),
    "RRT*": (
        "RRT*  (Optimal RRT)",
        "RRT'nin asimptotik olarak optimal versiyonu.\n\n"
        "Nasıl çalışır:\n"
        "  • RRT gibi ağaç büyütür, ama iki ek adım uygular:\n"
        "      1) Yeniden bağlantı (Rewire): yeni düğümün yakınındaki\n"
        "         mevcut düğümlere daha kısa yol var mı kontrol eder.\n"
        "      2) Maliyet minimizasyonu: daha iyi yol bulunursa\n"
        "         ağaç bağlantıları güncellenir.\n"
        "  • İterasyon arttıkça yol optimal limite yaklaşır.\n\n"
        "Karmaşıklık:  O(n log n)\n"
        "Asimptotik optimal: n → ∞'da optimal yolu garanti eder\n\n"
        "Avantaj:   RRT'den çok daha iyi yol kalitesi\n"
        "Dezavantaj: RRT'den ~2× daha yavaş"
    ),
    "PRM": (
        "PRM  (Probabilistic Roadmap Method)",
        "Önceden öğrenilen yol ağı üzerinde arama.\n\n"
        "Nasıl çalışır:\n"
        "  • Öğrenme aşaması: haritaya rastgele düğümler yerleştirir,\n"
        "    yakın komşuları çarpışma kontrolüyle birbirine bağlar.\n"
        "  • Sorgulama aşaması: başlangıç ve hedef bu yol ağına\n"
        "    bağlanır, ardından Dijkstra/A* ile en kısa yol bulunur.\n"
        "  • Birden fazla sorgu için aynı yol ağı yeniden kullanılır.\n\n"
        "Karmaşıklık:  Öğrenme O(n² log n), Sorgu O(log n)\n\n"
        "Avantaj:   Çok sorgulu ortamlarda çok hızlı\n"
        "Dezavantaj: Dar koridorlarda yetersiz örnekleme sorunu"
    ),
    "D* Lite": (
        "D* Lite  (Dynamic A-Star Lite)",
        "Artımlı yeniden planlama algoritması.\n\n"
        "Nasıl çalışır:\n"
        "  • İlk planda hedeften başa geriye Dijkstra uygular;\n"
        "    tüm hücrelere g-değeri (hedefe maliyet) hesaplar.\n"
        "  • Robot hareket ettikçe k_m saymacı güncellenir;\n"
        "    yalnızca etkilenen düğümler yeniden hesaplanır  O(k log n).\n"
        "  • Sıfırdan A* çalıştırmak yerine önceki hesabı günceller.\n"
        "  • Değişen ortamlar için özellikle verimlidir.\n\n"
        "Karmaşıklık:  İlk plan O(n log n), Güncelleme O(k log n)\n\n"
        "Avantaj:   Hareket sonrası çok hızlı güncelleme\n"
        "Dezavantaj: İlk kurulum maliyeti A*'dan biraz yüksek"
    ),
}

class GUI:
    def __init__(self):
        self.sim=Simulation()
        self.timer=None
        self._build()
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def _build(self):
        self.fig=plt.figure(figsize=(19,10),facecolor=DARK)
        self.fig.suptitle(
            "Depo Teslimat Robotu  —  Otonom Navigasyon Simülasyonu",
            color=C_BRIGHT,fontsize=13,fontweight="bold",y=0.98)
        gs=self.fig.add_gridspec(3,3,left=0.18,right=0.99,
                                  top=0.94,bottom=0.04,hspace=0.32,wspace=0.28)
        self.ax_map  =self.fig.add_subplot(gs[0:3,0:2])
        self.ax_lidar=self.fig.add_subplot(gs[0,2])
        self.ax_loc  =self.fig.add_subplot(gs[1,2])
        self.ax_err  =self.fig.add_subplot(gs[2,2])
        self._style(self.ax_map,self.ax_lidar,self.ax_loc,self.ax_err)
        self._build_controls()
        self._init_lidar_artists()
        self._init_loc_artists()
        self._init_err_artists()
        self._draw_map()
        self._collect_animated_arts()
                                                                            
        self._bg = None
        self.fig.canvas.mpl_connect('resize_event', self._on_resize)
        self.fig.canvas.mpl_connect('draw_event',   self._on_draw)

    def _style(self,*axes):
        for ax in axes:
            ax.set_facecolor(MAP_BG)
            ax.tick_params(colors=C_MID,labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor(SPINE_C)

    def _build_controls(self):
                                                              
        X, W   = 0.010, 0.140                             
        BX, BW = 0.013, 0.135             

        def _radio_panel(ax, title, choices, active_col):
            rb = RadioButtons(ax, choices, activecolor=active_col)
            ax.set_title(title, color=C_BRIGHT, fontsize=7.5, pad=6,
                         fontfamily='monospace', fontweight='bold')
            for lbl in rb.labels:
                lbl.set_color(C_BRIGHT); lbl.set_fontsize(8.5)
            return rb

        ax_n  = self.fig.add_axes((X, 0.717, W, 0.210), facecolor=PANEL)
        self.ax_nav = ax_n
        self.radio_nav = _radio_panel(ax_n, "Navigasyon Algoritmasi",
                                      list(PLANNERS.keys()), ACC)
                               
        ax_n.text(0.98, 0.01, "⟨sağ tık = bilgi⟩", transform=ax_n.transAxes,
                  color=C_DIM, fontsize=5.5, ha='right', va='bottom',
                  fontfamily='monospace')

        ax_rt = self.fig.add_axes((X, 0.574, W, 0.115), facecolor=PANEL)
        self.radio_robot = _radio_panel(ax_rt, "Robot Modeli",
                                        list(ROBOT_TYPES.keys()), C_OK)

        ax_l  = self.fig.add_axes((X, 0.461, W, 0.085), facecolor=PANEL)
        self.radio_loc = _radio_panel(ax_l, "Lokalizasyon",
                                      ["EKF", "Dead Reckoning"], ACC)

        ax_spd = self.fig.add_axes((X, 0.397, W-0.01, 0.038), facecolor=PANEL)
        ax_spd.set_title("Sim. Hizi", color=C_MID, fontsize=6.5,
                         loc='left', pad=3, fontfamily='monospace')
        self.slider_speed = Slider(ax_spd, "", 0.25, 4.0, valinit=0.5, color=ACC)
        self.slider_speed.valtext.set_color(ACC)
        self.slider_speed.valtext.set_fontsize(8)

        ax_nz  = self.fig.add_axes((X, 0.341, W-0.01, 0.038), facecolor=PANEL)
        ax_nz.set_title("LiDAR Gurultu", color=C_MID, fontsize=6.5,
                        loc='left', pad=3, fontfamily='monospace')
        self.slider_noise = Slider(ax_nz, "", 0.0, 0.5, valinit=0.10, color=C_WARN)
        self.slider_noise.valtext.set_color(C_WARN)
        self.slider_noise.valtext.set_fontsize(8)

        ax_s = self.fig.add_axes((BX, 0.289, BW, 0.038))
        self.btn_start = Button(ax_s, "BASLAT  ▶", color=ACC, hovercolor="#a04828")
        self.btn_start.label.set_color("#ffffff")
        self.btn_start.label.set_fontweight("bold")
        self.btn_start.label.set_fontsize(9)

        ax_r = self.fig.add_axes((BX, 0.241, BW, 0.038))
        self.btn_reset = Button(ax_r, "SIFIRLA  ↺", color="#d8ccc0", hovercolor="#c8b8a8")
        self.btn_reset.label.set_color(C_BRIGHT)
        self.btn_reset.label.set_fontsize(9)

        ax_rnd = self.fig.add_axes((BX, 0.193, BW, 0.038))
        self.btn_rand = Button(ax_rnd, "RASTGELE  ⟳", color="#d8ccc0", hovercolor="#c8b8a8")
        self.btn_rand.label.set_color(C_BRIGHT)
        self.btn_rand.label.set_fontsize(8.5)

        ax_cmp = self.fig.add_axes((BX, 0.145, BW, 0.038))
        self.btn_compare = Button(ax_cmp, "KARSILASTIR  ≡", color=ACC, hovercolor="#a04828")
        self.btn_compare.label.set_color("#ffffff")
        self.btn_compare.label.set_fontsize(8)
        self.btn_compare.label.set_fontweight("bold")

        self.ax_met = self.fig.add_axes((X, 0.020, W, 0.110), facecolor=PANEL)
        self.ax_met.axis("off")
        self._build_metric_display()

        self.radio_nav.on_clicked(self._on_nav)
        self.radio_robot.on_clicked(self._on_robot)
        self.radio_loc.on_clicked(self._on_loc)
        self.btn_start.on_clicked(self._on_start)
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_rand.on_clicked(self._on_rand)
        self.btn_compare.on_clicked(self._on_compare)
        self.slider_speed.on_changed(self._on_speed)
        self.slider_noise.on_changed(
            lambda v: (setattr(self.sim.lidar, 'noise_std', float(v)), setattr(self, '_bg', None)))

    def _build_metric_display(self):
        ax = self.ax_met; ax.clear(); ax.axis("off")
        ax.set_facecolor(PANEL)
        self._met_texts = []
                                                        
        left_slots  = 7
        right_slots = 5
        n = max(left_slots, right_slots)
        dy = 1.0 / (n + 0.5)
                                    
        for i in range(left_slots):
            t = ax.text(0.03, 1.0 - (i + 0.6) * dy, "",
                        transform=ax.transAxes, color=C_MID,
                        fontsize=6.8, va="center", fontfamily="monospace",
                        animated=True)
            self._met_texts.append(t)
                                    
        for i in range(right_slots):
            t = ax.text(0.53, 1.0 - (i + 0.6) * dy, "",
                        transform=ax.transAxes, color=C_MID,
                        fontsize=6.8, va="center", fontfamily="monospace",
                        animated=True)
            self._met_texts.append(t)
        self._update_metrics()

    def _init_lidar_artists(self):
        ax = self.ax_lidar; ax.clear(); self._style(ax)
        ax.set_title("LiDAR Nokta Bulutu  |  Ham Veri + Kume Renklendirmesi",
                     color=C_BRIGHT, fontsize=8, pad=4)
        ax.set_xlabel("Goreceli X (m)", color=C_MID, fontsize=7)
        ax.set_ylabel("Goreceli Y (m)", color=C_MID, fontsize=7)
        ax.grid(True, alpha=0.6, color=GRID_C)
        ax.set_aspect("equal")
        ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
        self._sc_raw  = ax.scatter([], [], s=4, c=C_DIM, alpha=0.30,
                                   label="Ham (gurultulu)", zorder=3, animated=True)
        self._sc_clust = ax.scatter([], [], s=10, c=C_OK, alpha=0.90,
                                    label="Kume (filtrelenmis)", zorder=4, animated=True)
        ax.plot(0, 0, color=C_BRIGHT, marker="^", ms=9, zorder=5, label="Robot")
        ax.legend(fontsize=6.5, facecolor=PANEL, labelcolor=C_BRIGHT, framealpha=0.8)

    def _init_loc_artists(self):
        ax = self.ax_loc; ax.clear(); self._style(ax)
        ax.set_title("Gercek Konum vs Lokalizasyon Tahmini",
                     color=C_BRIGHT, fontsize=8, pad=4)
        ax.set_xlabel("X (m)", color=C_MID, fontsize=7)
        ax.set_ylabel("Y (m)", color=C_MID, fontsize=7)
        ax.grid(True, alpha=0.6, color=GRID_C)
        env = self.sim.env
        ax.set_xlim(-0.5, env.width + 0.5)
        ax.set_ylim(-0.5, env.height + 0.5)
        self._ln_loc_true, = ax.plot([], [], "-",  color=C_TRUE, lw=1.8,
                                     label="Gercek Konum", animated=True)
        self._ln_loc_ekf,  = ax.plot([], [], "-",  color=C_EKF,  lw=1.0,
                                     alpha=0.9, label="EKF Tahmini", animated=True)
        self._ln_loc_dr,   = ax.plot([], [], "--", color=C_DR,   lw=1.0,
                                     alpha=0.85, label="Dead Reck.", animated=True)
        ax.legend(fontsize=6.5, facecolor=PANEL, labelcolor=C_BRIGHT, framealpha=0.8)

    def _init_err_artists(self):
        ax = self.ax_err; ax.clear(); self._style(ax)
        ax.set_title("Lokalizasyon Hata Analizi  |  EKF vs Dead Reckoning  (Zaman Serisi)",
                     color=C_BRIGHT, fontsize=9, pad=4)
        ax.set_xlabel("Gecen Sure (s)", color=C_MID, fontsize=8)
        ax.set_ylabel("Konum Hatasi (m)", color=C_MID, fontsize=8)
        ax.grid(True, alpha=0.6, color=GRID_C)
        self._ln_err_ekf, = ax.plot([], [], "-",  color=C_EKF, lw=1.5, alpha=0.9,
                                    label="EKF ani hatasi", animated=True)
        self._ln_err_dr,  = ax.plot([], [], "--", color=C_DR,  lw=1.2, alpha=0.9,
                                    label="DR ani hatasi", animated=True)
        self._hl_ekf = ax.axhline(0, color=C_EKF, ls=":", lw=1.5, alpha=0.85,
                                  visible=False, animated=True)
        self._hl_dr  = ax.axhline(0, color=C_DR,  ls=":", lw=1.5, alpha=0.85,
                                  visible=False, animated=True)
        self._tx_ekf = ax.text(0, 0, "", color=C_EKF, fontsize=7, va="bottom",
                               visible=False, animated=True)
        self._tx_dr  = ax.text(0, 0, "", color=C_DR,  fontsize=7, va="bottom",
                               visible=False, animated=True)
        ax.legend(fontsize=7, facecolor=PANEL, labelcolor=C_BRIGHT, framealpha=0.8, ncol=2)

    def _draw_map(self):
        ax=self.ax_map; ax.clear(); self._style(ax)
        env=self.sim.env
        ax.set_xlim(-0.5,env.width+0.5); ax.set_ylim(-0.5,env.height+0.5)
        ax.set_aspect("equal")
        
        title_text = (f"Ortam Haritası  |  Nav: {self.sim.nav_algo}  "
                      f"|  Robot: {self.sim.robot_type}  |  Lok: {self.sim.loc_algo}")
        
        if self.sim.path_error:
            title_text += "  |  ⚠️ HATA: GEÇERLİ BİR ROTA BULUNAMADI"
            title_color = C_ERR
        else:
            title_color = C_BRIGHT

        ax.set_title(title_text, color=title_color, fontsize=8.5, pad=4, fontweight="bold")
        
        ax.set_xlabel("X (m)",color=C_MID,fontsize=8)
        ax.set_ylabel("Y (m)",color=C_MID,fontsize=8)
        ax.grid(True,alpha=0.5,color=GRID_C,lw=0.5)
        ax.add_patch(patches.Rectangle((0,0),env.width,env.height,
                     lw=1.5,edgecolor=SPINE_C,facecolor="none",zorder=1))

        for i,(x1,y1,x2,y2) in enumerate(env.obstacles):
            _draw_obstacle(ax, x1, y1, x2, y2, seed=i)

        ax.plot(*env.start,"o",color=C_OK,ms=6,zorder=5)
        ax.plot(*env.goal, "*",color=ACC,ms=8,zorder=5)
        ax.add_patch(patches.Circle(env.goal, 0.40, fill=False,
                     edgecolor=ACC, lw=1.2, linestyle="--", zorder=5, alpha=0.7))
        ax.annotate("START",env.start,xytext=(4,6),textcoords="offset points",
                    color=C_OK,fontsize=7,fontweight="bold")
        ax.annotate("GOAL", env.goal, xytext=(4,6),textcoords="offset points",
                    color=ACC,fontsize=7,fontweight="bold")

        ax.text(0.01, 0.01, "Sol tık: engel ekle/sil  |  Sağ tık: hedef değiştir",
                transform=ax.transAxes, color=C_DIM, fontsize=6.2, va="bottom")

        self._ln_plan,=ax.plot([],[],"--",color=C_WARN,lw=1,alpha=0.4,label="Plan",zorder=2)
        if len(self.sim.plan_path)>1:
            pp=np.array(self.sim.plan_path)
            self._ln_plan.set_data(pp[:,0],pp[:,1])

        self._ln_true,=ax.plot([],[],"-",color=C_TRUE,lw=1.8,zorder=4,label="Gercek")
        self._ln_ekf, =ax.plot([],[],"-",color=C_EKF, lw=1.0,alpha=0.85,zorder=3,label="EKF")
        self._ln_dr,  =ax.plot([],[],"--",color=C_DR,  lw=0.9,alpha=0.75,zorder=3,label="DR")
        ax.legend(loc="upper left",fontsize=6.5,facecolor=PANEL,
                  labelcolor=C_BRIGHT,framealpha=0.8)

        self._lidar_lc=LineCollection([],colors=C_ERR,alpha=0.10,lw=0.5,zorder=2,animated=True)
        ax.add_collection(self._lidar_lc)
        self._ekf_ell=patches.Ellipse((0,0),0.1,0.1,angle=0,
                                      fc='none',ec=C_EKF,lw=1.2,ls='--',alpha=0.75,zorder=6,animated=True)
        self._ekf_ell.set_visible(False); ax.add_patch(self._ekf_ell)
        self._done_banner=ax.text(
            env.width/2,env.height/2,"HEDEFE ULAŞILDI  ✓",
            ha='center',va='center',fontsize=13,fontweight='bold',
            color=C_OK,zorder=20,animated=True,
            bbox=dict(boxstyle='round,pad=0.6',facecolor=PANEL,edgecolor=C_OK,lw=2,alpha=0.93))
        self._done_banner.set_visible(False)
        self._fps_text=ax.text(0.995,0.005,"FPS: 0",
            transform=ax.transAxes,ha='right',va='bottom',
            color=C_DIM,fontsize=7,fontfamily='monospace',zorder=20,animated=True)
        self._flash_rect=patches.Rectangle((0,0),0,0,lw=2.5,
            edgecolor="#ff4444",facecolor="#ff2222",alpha=0,zorder=15,animated=True)
        ax.add_patch(self._flash_rect)
        self._robot_arts=self.sim.robot.init_draw_artists(ax)
        for a in self._robot_arts.values():
            a.set_animated(True)
        for ln in [self._ln_plan,self._ln_true,self._ln_ekf,self._ln_dr]:
            ln.set_animated(True)

    def _update_map(self):
        if len(self.sim.true_path)>1:
            tp=np.array(self.sim.true_path)
            self._ln_true.set_data(tp[:,0],tp[:,1])
        if len(self.sim.ekf.history)>1:
            eh=np.array(self.sim.ekf.history)
            self._ln_ekf.set_data(eh[:,0],eh[:,1])
        if len(self.sim.dr.history)>1:
            dh=np.array(self.sim.dr.history)
            self._ln_dr.set_data(dh[:,0],dh[:,1])
        if len(self.sim.plan_path)>1:
            pp_arr=np.array(self.sim.plan_path)
            self._ln_plan.set_data(pp_arr[:,0],pp_arr[:,1])

        self.sim.robot.update_draw_artists(self._robot_arts)

        r=self.sim.robot
        if self.sim.lidar_raw:
            segs=[[(r.x,r.y),(px,py)]
                  for i,(px,py) in enumerate(self.sim.lidar_raw) if i%6==0]
            self._lidar_lc.set_segments(segs)

        try:
            P2=self.sim.ekf.P[:2,:2]
            vals,vecs=np.linalg.eigh(P2)
            vals=np.clip(vals,0,None)
            if vals[1]>1e-9:
                angle_deg=math.degrees(math.atan2(vecs[1,1],vecs[0,1]))
                scale=2.448
                self._ekf_ell.set_center((self.sim.ekf.mu[0],self.sim.ekf.mu[1]))
                self._ekf_ell.set_width(2*scale*math.sqrt(vals[1]))
                self._ekf_ell.set_height(2*scale*math.sqrt(vals[0]))
                self._ekf_ell.set_angle(angle_deg)
                self._ekf_ell.set_visible(True)
            else:
                self._ekf_ell.set_visible(False)
        except Exception:
            self._ekf_ell.set_visible(False)

        self._done_banner.set_visible(self.sim.done and not self.sim.path_error)
        self._fps_text.set_text(f"FPS: {getattr(self,'_fps',0.0):.0f}")

        fa = getattr(self, '_flash_alpha', 0.0)
        if fa > 0 and hasattr(self, '_flash_rect'):
            idx = getattr(self.sim, 'teleported_obs_idx', None)
            if idx is not None and idx < len(self.sim.env.obstacles):
                x1,y1,x2,y2 = self.sim.env.obstacles[idx]
                self._flash_rect.set_bounds(x1, y1, x2-x1, y2-y1)
                self._flash_rect.set_alpha(fa)
                self._flash_alpha = max(0.0, fa - 0.12)
            else:
                self._flash_rect.set_alpha(0); self._flash_alpha = 0.0
        elif hasattr(self, '_flash_rect'):
            self._flash_rect.set_alpha(0)

    def _update_lidar(self):
        if not self.sim.lidar_raw: return
        r = self.sim.robot
        raw_pts = np.array(self.sim.lidar_raw)
        raw_rel = raw_pts - [r.x, r.y]
        self._sc_raw.set_offsets(raw_rel)

        raw_ranges = self.sim.lidar_raw_ranges
        if len(raw_ranges) == 0:
            self._sc_clust.set_offsets(np.empty((0, 2)))
            return

        n = len(raw_ranges)
        max_r = self.sim.lidar.max_range
        cluster_ids = np.full(n, -1, dtype=int)
        cid = -1; prev_i = -1
        for i, d in enumerate(raw_ranges):
            if d >= max_r * 0.98:
                prev_i = -1
                continue
            if prev_i == -1:
                cid += 1
            else:
                gap = math.hypot(raw_pts[i, 0] - raw_pts[prev_i, 0],
                                 raw_pts[i, 1] - raw_pts[prev_i, 1])
                if gap > 0.5:
                    cid += 1
            cluster_ids[i] = cid
            prev_i = i

        mask = cluster_ids >= 0
        if not mask.any():
            self._sc_clust.set_offsets(np.empty((0, 2)))
            return

        clust_rel = raw_rel[mask]
        colors = [CLUSTER_PAL[c % len(CLUSTER_PAL)] for c in cluster_ids[mask]]
        self._sc_clust.set_offsets(clust_rel)
        self._sc_clust.set_facecolor(colors)

    def _update_loc(self):
        if len(self.sim.true_path)<2: return
        tp=np.array(self.sim.true_path)
        self._ln_loc_true.set_data(tp[:,0],tp[:,1])
        if len(self.sim.ekf.history)>1:
            eh=np.array(self.sim.ekf.history)
            self._ln_loc_ekf.set_data(eh[:,0],eh[:,1])
        if len(self.sim.dr.history)>1:
            dh=np.array(self.sim.dr.history)
            self._ln_loc_dr.set_data(dh[:,0],dh[:,1])

    def _update_err(self):
        t = self.sim.times
        if len(t) < 2: return
        ekf_arr = np.array(self.sim.errors_ekf)
        dr_arr  = np.array(self.sim.errors_dr)
        self._ln_err_ekf.set_data(t, ekf_arr)
        self._ln_err_dr.set_data(t, dr_arr)
                                              
        ax = self.ax_err
        ax.set_xlim(0, max(t[-1], 1.0))
        ymax = max(float(ekf_arr.max()), float(dr_arr.max()), 0.01) * 1.15
        ax.set_ylim(0, ymax)
        if len(t) > 5:
            re = math.sqrt(np.mean(ekf_arr**2))
            rd = math.sqrt(np.mean(dr_arr**2))
            self._hl_ekf.set_data([0, t[-1]], [re, re]); self._hl_ekf.set_visible(True)
            self._hl_dr.set_data([0, t[-1]], [rd, rd]);   self._hl_dr.set_visible(True)
            x0 = t[-1] * 0.02
            self._tx_ekf.set_position((x0, re + ymax * 0.02))
            self._tx_ekf.set_text(f"EKF RMSE={re:.3f}m"); self._tx_ekf.set_visible(True)
            self._tx_dr.set_position((x0, rd + ymax * 0.02))
            self._tx_dr.set_text(f"DR  RMSE={rd:.3f}m");  self._tx_dr.set_visible(True)

    def _update_metrics(self):
        sim = self.sim

        if sim.path_error:
            st_txt, st_col = "ROTA YOK!", C_ERR
        elif sim.done:
            st_txt, st_col = "HEDEFE ULASILDI", C_OK
        elif len(sim.true_path) > 1:
            st_txt, st_col = "CALISIYOR...", C_WARN
        else:
            st_txt, st_col = "HAZIR", C_MID

        if len(sim.true_path) < 2:
            dist_goal = math.hypot(sim.env.goal[0]-sim.env.start[0],
                                   sim.env.goal[1]-sim.env.start[1])
            pl_s, t_s, dg_s = "—", "—", f"{dist_goal:.1f}m"
            re_s, rd_s = "—", "—"
            ekf_col, dr_col = C_MID, C_MID
        else:
            pl = sum(math.hypot(sim.true_path[i][0]-sim.true_path[i-1][0],
                                sim.true_path[i][1]-sim.true_path[i-1][1])
                     for i in range(1, len(sim.true_path)))
            re = math.sqrt(np.mean(np.array(sim.errors_ekf)**2)) if sim.errors_ekf else 0
            rd = math.sqrt(np.mean(np.array(sim.errors_dr)**2))  if sim.errors_dr  else 0
            dist_goal = math.hypot(sim.robot.x-sim.env.goal[0],
                                   sim.robot.y-sim.env.goal[1])
            pl_s  = f"{pl:.1f}m"
            t_s   = f"{sim.t:.1f}s"
            dg_s  = f"{dist_goal:.1f}m"
            re_s  = f"{re:.3f}m"
            rd_s  = f"{rd:.3f}m"
            ekf_col = C_OK if re < 0.15 else (C_WARN if re < 0.40 else C_ERR)
            dr_col  = C_OK if rd < 0.30 else (C_WARN if rd < 0.80 else C_ERR)

        left = [
            ("─ METRİKLER ─", ACC,     True),
            (f"Yol : {pl_s}",  C_MID,  False),
            (f"Sure: {t_s}",   C_MID,  False),
            (f"Hdf : {dg_s}",  C_BRIGHT, False),
            (f"EKF : {re_s}",  ekf_col, False),
            (f"DR  : {rd_s}",  dr_col,  False),
            (st_txt,           st_col,  True),
        ]
                                                        
        right = [
            ("─ AYARLAR ─",              ACC,      True),
            (f"{sim.nav_algo[:9]}",       C_BRIGHT, False),
            (f"{sim.robot_type[:10]}",    C_BRIGHT, False),
            (f"Lok: {sim.loc_algo[:3]}", C_BRIGHT, False),
            ("",                          C_MID,    False),
        ]

        for t_obj, (txt, col, bold) in zip(self._met_texts[:7], left):
            t_obj.set_text(txt); t_obj.set_color(col)
            t_obj.set_fontweight("bold" if bold else "normal")
        for t_obj, (txt, col, bold) in zip(self._met_texts[7:], right):
            t_obj.set_text(txt); t_obj.set_color(col)
            t_obj.set_fontweight("bold" if bold else "normal")

    def _on_resize(self, event):
        self._bg = None

    def _on_draw(self, event):
        if getattr(self, '_refreshing_bg', False):
            return
        if self._bg is None:
            self._refresh_bg()
            if hasattr(self, '_animated_arts') and hasattr(self, '_ln_plan'):
                self._update_metrics()
                self._preview_plan()

    def _refresh_bg(self):
        self._refreshing_bg = True
        
        hidden_states = []
        if hasattr(self, '_animated_arts'):
            for _, art in self._animated_arts:
                if art is not None:
                    hidden_states.append((art, art.get_visible()))
                    art.set_visible(False)
                    
        self.fig.canvas.draw()
        try:
            self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        except Exception:
            self._bg = None
            
        for art, state in hidden_states:
            art.set_visible(state)
            
        self._refreshing_bg = False

    def _collect_animated_arts(self):
        m = self.ax_map
        arts = [(m, self._ln_plan), (m, self._ln_true),
                (m, self._ln_ekf),  (m, self._ln_dr),
                (m, self._lidar_lc),(m, self._ekf_ell),
                (m, self._done_banner),(m, self._fps_text),
                (m, self._flash_rect)]
        for a in self._robot_arts.values():
            arts.append((m, a))
        arts += [(self.ax_lidar, self._sc_raw),
                 (self.ax_lidar, self._sc_clust),
                 (self.ax_loc, self._ln_loc_true),
                 (self.ax_loc, self._ln_loc_ekf),
                 (self.ax_loc, self._ln_loc_dr),
                 (self.ax_err, self._ln_err_ekf),
                 (self.ax_err, self._ln_err_dr),
                 (self.ax_err, self._hl_ekf),
                 (self.ax_err, self._hl_dr),
                 (self.ax_err, self._tx_ekf),
                 (self.ax_err, self._tx_dr)]
        arts += [(self.ax_met, t) for t in self._met_texts]
        self._animated_arts = arts

    def _tick(self):
        try:
            _t=time.perf_counter()
            if hasattr(self,'_fps_t'):
                raw_fps=1.0/max(_t-self._fps_t,1e-6)
                self._fps=getattr(self,'_fps',raw_fps)*0.85+raw_fps*0.15
            self._fps_t=_t

            self._step_acc = getattr(self, '_step_acc', 0.0) + getattr(self, '_speed_factor', 0.5)
            steps = int(self._step_acc)
            self._step_acc -= steps
            for _ in range(steps):
                self.sim.step(dt=0.1)
            self._tick_n = getattr(self, '_tick_n', 0) + 1

            need_bg_refresh = False
            if getattr(self.sim, '_teleport_flash', 0) > 0:
                self.sim._teleport_flash = 0
                self._flash_alpha = 0.9                            
                self._draw_map()
                self._collect_animated_arts()
                need_bg_refresh = True
                                                                            
            pe = self.sim.path_error; dn = self.sim.done
            if pe != getattr(self,'_prev_pe',None) or dn != getattr(self,'_prev_dn',None):
                self._prev_pe = pe; self._prev_dn = dn
                if not need_bg_refresh:
                    self._draw_map(); self._collect_animated_arts()
                need_bg_refresh = True
            if need_bg_refresh:
                self._refresh_bg()

            self._update_map()
            if self._tick_n % 2 == 0:
                self._update_lidar()
                self._update_loc()
            if self._tick_n % 3 == 0:
                self._update_err()
            self._update_metrics()

            try:
                if self._bg is None:
                    self._refresh_bg()
                canvas = self.fig.canvas
                canvas.restore_region(self._bg)
                for ax, art in self._animated_arts:
                    ax.draw_artist(art)
                canvas.blit(self.fig.bbox)
            except Exception:
                                                                                         
                self._bg = None
                try:
                    self._collect_animated_arts()
                    self._refresh_bg()
                except Exception:
                    pass
                self.fig.canvas.draw_idle()

            if self.sim.done and not self.sim.path_error and self.timer is not None:
                self._update_err()
                self._update_metrics()
                                                                                              
                for _, _art in self._animated_arts:
                    _art.set_animated(False)
                self.fig.canvas.draw()
                self.timer.stop()
        except Exception:
                                                                        
            import traceback; traceback.print_exc()

    def _show_algo_info(self, algo_name):
        title, body = ALGO_INFO.get(algo_name, (algo_name, "Bilgi bulunamadı."))
        for i in plt.get_fignums():
            try:
                if plt.figure(i).canvas.manager.get_window_title().startswith("Algoritma:"):
                    plt.close(plt.figure(i))
            except Exception:
                pass

        popup = plt.figure(figsize=(6.4, 5.6), facecolor="#12121e")
        popup.canvas.manager.set_window_title(f"Algoritma: {algo_name}")
        popup.subplots_adjust(left=0, right=1, top=1, bottom=0)

        ax_title = popup.add_axes([0, 0.86, 1, 0.14], facecolor="#2a2a42")
        ax_title.axis("off")
        ax_title.text(0.04, 0.50, title,
                      color="#f0e8d8", fontsize=13, fontweight="bold",
                      va="center", transform=ax_title.transAxes)

        ax_body = popup.add_axes([0, 0.12, 1, 0.74], facecolor="#12121e")
        ax_body.axis("off")
        ax_body.text(0.04, 0.97, body,
                     color="#ddd5c5", fontsize=10,
                     va="top", transform=ax_body.transAxes,
                     linespacing=1.8, wrap=False)

        ax_btn = popup.add_axes([0.33, 0.02, 0.34, 0.08])
        btn = Button(ax_btn, "  Kapat  ", color="#c8622a", hovercolor="#e07030")
        btn.label.set_color("white")
        btn.label.set_fontsize(10)
        btn.label.set_fontweight("bold")
        btn.on_clicked(lambda _: plt.close(popup))
        popup._close_btn = btn                                     

        popup.canvas.draw()
        try:
            popup.canvas.manager.window.lift()
        except Exception:
            pass
        plt.pause(0.001)

    def _on_click(self, event):
                                                              
        if event.button == 3:
            try:
                nav_bbox = self.ax_nav.get_window_extent()
                in_nav   = nav_bbox.contains(event.x, event.y)
            except Exception:
                in_nav = False
            if in_nav:
                labels = list(PLANNERS.keys())
                n      = len(labels)
                nav_bb = self.ax_nav.get_window_extent()
                                                                                 
                rel    = (event.y - nav_bb.y0) / max(nav_bb.height, 1.0)
                idx    = max(0, min(n-1, int((1.0 - rel) * n)))
                self._show_algo_info(labels[idx])
                return

        if event.inaxes != self.ax_map:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        if event.button == 3:
            if self.sim.env.is_free(x, y, margin=0.1):
                self._stop_timer()
                current_theta = self.sim.robot.theta
                self.sim.env.start = (self.sim.robot.x, self.sim.robot.y)
                self.sim.env.goal = (x, y)
                self.sim.reset(initial_theta=current_theta)
                self._draw_map()
                self._init_lidar_artists(); self._init_loc_artists(); self._init_err_artists()
                self._full_redraw()
            return

        if event.button == 1:
            self._on_map_left_click(x, y)

    def _on_map_left_click(self, x, y):
        env = self.sim.env

        hit_idx = None
        for i, (x1, y1, x2, y2) in enumerate(env.obstacles):
            if x1 <= x <= x2 and y1 <= y <= y2:
                hit_idx = i
                break

        if hit_idx is not None:
            env.obstacles.pop(hit_idx)
            env.labels.pop(hit_idx)
        else:
                                                      
            ow, oh = 0.8, 0.5
            nx1 = round(x - ow / 2, 2);  ny1 = round(y - oh / 2, 2)
            nx2 = round(nx1 + ow, 2);     ny2 = round(ny1 + oh, 2)
                              
            nx1 = max(0.3, nx1);          ny1 = max(0.3, ny1)
            nx2 = min(env.width - 0.3, nx2); ny2 = min(env.height - 0.3, ny2)
            if nx2 - nx1 < 0.2 or ny2 - ny1 < 0.2:
                return
                                                    
            cx, cy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
            if (math.hypot(cx - env.start[0], cy - env.start[1]) < 1.0 or
                    math.hypot(cx - env.goal[0],  cy - env.goal[1])  < 1.0 or
                    math.hypot(cx - self.sim.robot.x, cy - self.sim.robot.y) < 1.0):
                return
            env.obstacles.append((nx1, ny1, nx2, ny2))
            env.labels.append(f"U{len(env.obstacles)}")

        env._build_obstacle_cache()

        sim = self.sim
        if self.timer is not None:
                                                                    
            if sim.nav_algo == "D* Lite" and getattr(sim, '_dstar', None) is not None:
                pp  = ROBOT_PLAN_PARAMS[sim.robot_type]
                sim._dstar._g = sim._dstar._backward_dijkstra(sim.env.goal)
                raw = sim._dstar._extract(
                    (sim.robot.x, sim.robot.y), sim._dstar._g, sim.env.goal)
                if len(raw) < 3:
                    raw = sim._dstar.plan((sim.robot.x, sim.robot.y), sim.env.goal)
                if len(raw) >= 3:
                    sim.plan_path  = _smooth_path(raw, sim.env, pp["margin"])
                    sim.path_idx   = 0
                    sim.path_error = False
                    sim.done       = False
                sim._replan_cooldown = 15
            sim._teleport_flash = 1                                    
        else:
                                                           
            sim.reset()
            self._draw_map()
            self._init_lidar_artists(); self._init_loc_artists(); self._init_err_artists()
            self._full_redraw()

    def _on_start(self,_):
        self._stop_timer()
                                                                          
        self._prev_pe = self.sim.path_error
        self._prev_dn = self.sim.done
        self._refresh_bg()
        self.timer=self.fig.canvas.new_timer(interval=16)                 
        self.timer.add_callback(self._tick)
        self.timer.start()

    def _on_speed(self,val):
        self._speed_factor=float(val)
        self._bg=None

    def _full_redraw(self):
        self._collect_animated_arts()
        self._update_metrics()
        self._refresh_bg()
        self._preview_plan()

    def _preview_plan(self):
        if self._bg is None:
            return
        if len(self.sim.plan_path) > 1:
            pp = np.array(self.sim.plan_path)
            self._ln_plan.set_data(pp[:,0], pp[:,1])
        else:
            self._ln_plan.set_data([], [])
        try:
            canvas = self.fig.canvas
            canvas.restore_region(self._bg)
            for ax, art in self._animated_arts:
                ax.draw_artist(art)
            canvas.blit(self.fig.bbox)
        except Exception:
            self.fig.canvas.draw_idle()

    def _on_reset(self,_):
        self._stop_timer(); self.sim.reset(); self._draw_map()
        self._init_lidar_artists(); self._init_loc_artists(); self._init_err_artists()
        self._full_redraw()

    def _on_nav(self,label):
        self._stop_timer(); self.sim.nav_algo=label
        self.sim.reset(); self._draw_map()
        self._init_lidar_artists(); self._init_loc_artists(); self._init_err_artists()
        self._full_redraw()

    def _on_robot(self,label):
        self._stop_timer(); self.sim.robot_type=label
        self.sim.reset(); self._draw_map()
        self._init_lidar_artists(); self._init_loc_artists(); self._init_err_artists()
        self._full_redraw()

    def _on_loc(self,label):
        self._stop_timer(); self.sim.loc_algo=label
        self.sim.reset(); self._draw_map()
        self._init_loc_artists(); self._init_err_artists()
        self._full_redraw()

    def _on_rand(self,_):
        self._stop_timer()
        self.sim.env.randomize()
        self.sim.reset(); self._draw_map()
        self._init_lidar_artists(); self._init_loc_artists(); self._init_err_artists()
        self._full_redraw()

    def _on_compare(self, _):
        self._stop_timer()
        self._comparison_progress("Hazirlanıyor...", 0, len(PLANNERS))
        results = self._run_comparison_data()
        self._show_comparison(results)

    def _comparison_progress(self, label, done, total):
        pct = f"{done}/{total}" if total else ""
        self.ax_map.set_title(
            f"Karsilastirma hesaplaniyor  {pct}  —  {label}",
            color=C_WARN, fontsize=9, pad=4, fontweight="bold")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _make_comparison_sim(self, algo):
        sim = Simulation.__new__(Simulation)
                                                        
        env = Environment.__new__(Environment)
        env.width     = self.sim.env.width
        env.height    = self.sim.env.height
        env.start     = self.sim.env.start
        env.goal      = self.sim.env.goal
        env.is_random = self.sim.env.is_random
        env.obstacles = list(self.sim.env.obstacles)
        env.labels    = list(self.sim.env.labels)
        env._build_obstacle_cache()                             
        sim.env     = env
        sim.lidar   = LiDAR(env)
        sim.imu     = IMU()
        sim.encoder = Encoder()
        sim.nav_algo   = algo
        sim.loc_algo   = "EKF"
        sim.robot_type = self.sim.robot_type
        sim.reset()
        return sim

    def _run_comparison_data(self):
        results = {}
        algo_list = list(PLANNERS.keys())
        total = len(algo_list)
        for idx, algo in enumerate(algo_list):
            self._comparison_progress(algo, idx, total)
            try:
                sim = self._make_comparison_sim(algo)
            except Exception:
                results[algo] = None; continue

            if sim.path_error:
                results[algo] = None; continue

            goal = sim.env.goal
            best_dist = math.hypot(sim.robot.x - goal[0], sim.robot.y - goal[1])
            stall = 0
            for step_i in range(2000):
                if sim.done: break
                sim.step(dt=0.1)
                d = math.hypot(sim.robot.x - goal[0], sim.robot.y - goal[1])
                if d < best_dist - 0.1:
                    best_dist = d; stall = 0
                else:
                    stall += 1
                if stall >= 400:
                    break
                if step_i % 250 == 249:
                    self._comparison_progress(algo, idx, total)

            reached = (not sim.path_error and
                       math.hypot(sim.robot.x-goal[0], sim.robot.y-goal[1]) < 1.5)
            pl = (sum(math.hypot(sim.true_path[i][0]-sim.true_path[i-1][0],
                                 sim.true_path[i][1]-sim.true_path[i-1][1])
                      for i in range(1, len(sim.true_path)))
                  if len(sim.true_path)>1 else 0)
            re = (math.sqrt(np.mean(np.array(sim.errors_ekf)**2))
                  if sim.errors_ekf else 0)
            rd = (math.sqrt(np.mean(np.array(sim.errors_dr)**2))
                  if sim.errors_dr else 0)
            results[algo] = {
                "reached":  reached,
                "time":     sim.t if reached else None,
                "path_len": pl     if reached else None,
                "rmse_ekf": re     if reached else None,
                "rmse_dr":  rd     if reached else None,
            }
        return results

    def _show_comparison(self, results):
        algos   = list(results.keys())
        n       = len(algos)
        colors  = [C_OK if (results[a] and results[a]["reached"]) else C_ERR
                   for a in algos]
        bar_kw  = dict(edgecolor="#222", linewidth=0.8, zorder=3)

        def val(a, key, fallback=0):
            r = results[a]
            if r is None or r.get(key) is None: return fallback
            return r[key]

        times    = [val(a,"time", 0)    for a in algos]
        paths    = [val(a,"path_len",0) for a in algos]
        rmse_ekf = [val(a,"rmse_ekf",0) for a in algos]
        rmse_dr  = [val(a,"rmse_dr", 0) for a in algos]

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=DARK)
        fig.suptitle(
            f"Algoritma Karsilastirmasi  —  Robot: {self.sim.robot_type}  "
            f"|  Ortam: {'Rastgele' if self.sim.env.is_random else 'Varsayilan'}",
            color=C_BRIGHT, fontsize=12, fontweight="bold", y=0.99)

        specs = [
            (axes[0,0], times,    "Hedefe Ulasma Suresi (s)",
             "Daha kisa = daha hizli", ACC),
            (axes[0,1], paths,    "Kat Edilen Yol Uzunlugu (m)",
             "Daha kisa = daha verimli", C_WARN),
            (axes[1,0], rmse_ekf, "RMSE EKF Lokalizasyon Hatasi (m)",
             "Daha kucuk = daha dogru EKF", C_EKF),
            (axes[1,1], rmse_dr,  "RMSE Dead Reckoning Hatasi (m)",
             "Daha kucuk = daha dogru DR", C_DR),
        ]

        x = np.arange(n)
        for ax, vals, ylabel, note, base_col in specs:
            ax.set_facecolor(MAP_BG)
            ax.tick_params(colors=C_MID, labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor(SPINE_C)
            ax.grid(True, alpha=0.6, color=GRID_C, axis="y", zorder=0)

            bars = ax.bar(x, vals, color=colors, **bar_kw, width=0.55)

            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005*max(vals,default=1),
                            f"{v:.2f}", ha="center", va="bottom",
                            color=C_BRIGHT, fontsize=7.5, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(algos, rotation=20, ha="right", fontsize=8.5, color=C_MID)
            ax.set_ylabel(ylabel, color=C_MID, fontsize=8)
            ax.set_title(note, color=base_col, fontsize=8, pad=4)
            ax.tick_params(axis="x", bottom=False)

        from matplotlib.patches import Patch
        legend_els = [Patch(fc=C_OK,  ec="#222", label="Hedefe ulasti"),
                      Patch(fc=C_ERR, ec="#222", label="Hedefe ulasamadi / rota yok")]
        fig.legend(handles=legend_els, loc="lower center", ncol=2,
                   facecolor=PANEL, labelcolor=C_BRIGHT, fontsize=9,
                   framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        try:
            plt.show(block=True)
        except Exception:
            pass

        self._draw_map()
        self._init_lidar_artists(); self._init_loc_artists(); self._init_err_artists()
        self._full_redraw()

    def _stop_timer(self):
        if self.timer is not None:
            self.timer.stop(); self.timer=None

    def run(self):
        plt.show()

if __name__ == "__main__":
    gui=GUI()
    gui.run()

import simpy
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. LIVE STATE TRACKER (Shared Memory)
# ==========================================
class LiveState:
    def __init__(self):
        self.current_time = "Day 1 - 10:00 AM"
        self.sim_minute = 0
        self.configs = ['Baseline', 'Optimized']
        
        self.staff = {
            'Baseline': {'Teller': 3, 'Helpdesk': 1, 'RM': 1},
            'Optimized': {'Teller': 7, 'Helpdesk': 3, 'RM': 2}
        }
        
        self.daily_opex = {
            'Baseline': (3*1500) + (1*2000) + (1*4000),   # ₹10,500
            'Optimized': (7*1500) + (3*2000) + (2*4000)   # ₹24,500
        }
        
        self.queues = {c: {'Teller': 0, 'Helpdesk': 0, 'RM': 0} for c in self.configs}
        self.active = {c: {'Teller': 0, 'Helpdesk': 0, 'RM': 0} for c in self.configs}
        
        self.total_wait_time = {c: {'Teller': 0.0, 'Helpdesk': 0.0, 'RM': 0.0} for c in self.configs}
        self.served_count = {c: {'Teller': 0.001, 'Helpdesk': 0.001, 'RM': 0.001} for c in self.configs}
        
        self.queue_samples = {c: {'Teller': 0, 'Helpdesk': 0, 'RM': 0} for c in self.configs}
        self.sample_count = 0.001
        
        self.lost_rev = {'Baseline': 0, 'Optimized': 0}
        self.earned_rev = {'Baseline': 0, 'Optimized': 0}

state = LiveState()
SERVICE_VALUES = {'Teller': 150, 'Helpdesk': 600, 'RM': 6500}

# ==========================================
# 2. SIMPY ENGINE
# ==========================================
def customer_lifecycle(env, srv_type, branch, config, tolerance, patience, srv_time):
    arrival_time = env.now
    target_staff = branch[srv_type]
    
    state.queues[config][srv_type] += 1
    
    prob_join = 1.0 / (1.0 + math.exp(0.8 * (state.queues[config][srv_type] - tolerance)))
    if np.random.uniform(0, 1) > prob_join:
        state.queues[config][srv_type] -= 1
        state.lost_rev[config] += SERVICE_VALUES[srv_type]
        return

    req = target_staff.request()
    with req as request:
        results = yield env.any_of([request, env.timeout(patience)])
        state.queues[config][srv_type] -= 1
        
        if request in results:
            wait_time = env.now - arrival_time
            state.total_wait_time[config][srv_type] += wait_time
            state.served_count[config][srv_type] += 1
            
            state.active[config][srv_type] += 1
            yield env.timeout(srv_time)
            state.active[config][srv_type] -= 1
            state.earned_rev[config] += SERVICE_VALUES[srv_type]
        else:
            state.lost_rev[config] += SERVICE_VALUES[srv_type]

def queue_monitor(env):
    while True:
        state.sim_minute = env.now
        for config in state.configs:
            for srv_type in ['Teller', 'Helpdesk', 'RM']:
                state.queue_samples[config][srv_type] += state.queues[config][srv_type]
        state.sample_count += 1
        yield env.timeout(1)

def arrival_generator(env, branches):
    while env.now < 480: 
        hour = int((env.now % 480) // 60) + 10
        minute = int(env.now % 60)
        state.current_time = f"Day 1 - {hour:02d}:{minute:02d} {'AM' if hour < 12 else 'PM'}"
        
        rate = 3.0 if 120 <= env.now < 240 else 1.2 
        yield env.timeout(np.random.exponential(1.0 / rate))
        
        srv_type = np.random.choice(['Teller', 'Helpdesk', 'RM'], p=[0.60, 0.25, 0.15])
        tolerance = np.random.uniform(3, 10)
        patience = np.random.weibull(2.0) * 15
        
        if srv_type == 'Teller': srv_time = np.random.gamma(3.0, 1.0)
        elif srv_type == 'Helpdesk': srv_time = np.random.gamma(5.0, 1.2)
        else: srv_time = np.random.lognormal(2.6, 0.5)
        
        env.process(customer_lifecycle(env, srv_type, branches['Baseline'], 'Baseline', tolerance, patience, srv_time))
        env.process(customer_lifecycle(env, srv_type, branches['Optimized'], 'Optimized', tolerance, patience, srv_time))
        
        time.sleep(0.04) 

def run_simulation():
    env = simpy.Environment()
    branches = {
        'Baseline': {srv: simpy.Resource(env, cap) for srv, cap in state.staff['Baseline'].items()},
        'Optimized': {srv: simpy.Resource(env, cap) for srv, cap in state.staff['Optimized'].items()}
    }
    env.process(queue_monitor(env))
    env.process(arrival_generator(env, branches))
    env.run()

# ==========================================
# 3. LIVE MATPLOTLIB DASHBOARD
# ==========================================
plt.style.use('dark_background')
# Increased figure size for better spacing
fig = plt.figure(figsize=(19, 10))

# Explicit spacing between rows and columns to prevent overlap
gs = fig.add_gridspec(2, 6, height_ratios=[1.3, 1], hspace=0.4, wspace=0.6)

ax_base_floor = fig.add_subplot(gs[0, 0:3])
ax_opt_floor = fig.add_subplot(gs[0, 3:6])

ax_wait = fig.add_subplot(gs[1, 0:2])
ax_queue = fig.add_subplot(gs[1, 2:4])
ax_fin = fig.add_subplot(gs[1, 4:6])

fig.suptitle("🏦 Live Operations Comparison (Baseline vs. Optimized)", fontsize=24, fontweight='bold', color='white')

services = ['Teller', 'Helpdesk', 'RM']
colors = ['#3498db', '#f1c40f', '#9b59b6']

def draw_floor_plan(ax, config, title):
    ax.clear()
    ax.axis('off')
    
    # Dynamic X-Axis to prevent long queues from overlapping text
    max_queue_len = max([state.queues[config][s] for s in services])
    ax.set_xlim(-2.5, max(12, max_queue_len * 0.5 + 3))
    ax.set_ylim(-0.5, 2.8)
    
    # Title (Top Center)
    ax.text(0.5, 1.05, title, color='white', fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
    
    time_ratio = state.sim_minute / 480.0 if state.sim_minute <= 480 else 1.0
    current_opex = state.daily_opex[config] * time_ratio
    current_net = state.earned_rev[config] - current_opex
    
    # Financials (Bottom Center)
    ax.text(0.5, -0.1, f"Net Profit: ₹{current_net/1000:,.1f}K  |  Walkout Loss: ₹{state.lost_rev[config]/1000:,.1f}K", 
            color='#e74c3c' if config == 'Baseline' else '#2ecc71', fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)
    
    for i, srv in enumerate(services):
        active = state.active[config][srv]
        total = state.staff[config][srv]
        
        # Staff Text
        ax.text(-0.2, i, f"{srv}\n{active}/{total} Staff", color='white', fontsize=12, fontweight='bold', va='center', ha='right')
        
        queue_len = state.queues[config][srv]
        if queue_len > 0:
            x_vals = np.arange(1, queue_len + 1) * 0.5
            y_vals = np.full(queue_len, i)
            # Scatter Dots
            ax.scatter(x_vals, y_vals, s=180, color=colors[i], edgecolors='white', zorder=3)
            # Waiting Text perfectly spaced
            ax.text(x_vals[-1] + 0.4, i, f"{queue_len} waiting", color='lightgray', va='center', fontsize=11)

def update_dashboard(frame):
    # 1. Update Floor Plans
    draw_floor_plan(ax_base_floor, 'Baseline', f"Baseline (3T, 1H, 1RM) | {state.current_time}")
    draw_floor_plan(ax_opt_floor, 'Optimized', f"Optimized (7T, 3H, 2RM) | {state.current_time}")
    
    x_srv = np.arange(len(services))
    width = 0.35
    
    # 2. Avg Wait Time Bar Chart
    ax_wait.clear()
    base_w = [state.total_wait_time['Baseline'][s] / state.served_count['Baseline'][s] for s in services]
    opt_w = [state.total_wait_time['Optimized'][s] / state.served_count['Optimized'][s] for s in services]
    
    bars1 = ax_wait.bar(x_srv - width/2, base_w, width, label='Baseline', color='#e74c3c')
    bars2 = ax_wait.bar(x_srv + width/2, opt_w, width, label='Optimized', color='#2ecc71')
    ax_wait.set_title("Avg Wait Time (Minutes)", color='white', fontweight='bold', pad=15)
    ax_wait.set_xticks(x_srv)
    ax_wait.set_xticklabels(services, color='white', fontsize=11)
    ax_wait.tick_params(colors='white')
    ax_wait.legend()
    # Add exact numbers on top of bars
    ax_wait.bar_label(bars1, fmt='%.1f', padding=3, color='white')
    ax_wait.bar_label(bars2, fmt='%.1f', padding=3, color='white')

    # 3. Avg Queue Length Bar Chart
    ax_queue.clear()
    base_q = [state.queue_samples['Baseline'][s] / state.sample_count for s in services]
    opt_q = [state.queue_samples['Optimized'][s] / state.sample_count for s in services]
    
    bars3 = ax_queue.bar(x_srv - width/2, base_q, width, label='Baseline', color='#e74c3c')
    bars4 = ax_queue.bar(x_srv + width/2, opt_q, width, label='Optimized', color='#2ecc71')
    ax_queue.set_title("Avg Queue Length (People)", color='white', fontweight='bold', pad=15)
    ax_queue.set_xticks(x_srv)
    ax_queue.set_xticklabels(services, color='white', fontsize=11)
    ax_queue.tick_params(colors='white')
    # Add exact numbers on top of bars
    ax_queue.bar_label(bars3, fmt='%.1f', padding=3, color='white')
    ax_queue.bar_label(bars4, fmt='%.1f', padding=3, color='white')

    # 4. Financial Bar Chart (Profit vs. Loss)
    ax_fin.clear()
    time_ratio = state.sim_minute / 480.0 if state.sim_minute <= 480 else 1.0
    
    net_base = state.earned_rev['Baseline'] - (state.daily_opex['Baseline'] * time_ratio)
    net_opt = state.earned_rev['Optimized'] - (state.daily_opex['Optimized'] * time_ratio)
    
    lost_base = state.lost_rev['Baseline']
    lost_opt = state.lost_rev['Optimized']
    
    fin_labels = ['Net Profit', 'Walkout Loss']
    b_fin = [net_base, lost_base]
    o_fin = [net_opt, lost_opt]
    x_fin = np.arange(len(fin_labels))
    
    bars5 = ax_fin.bar(x_fin - width/2, b_fin, width, label='Baseline', color='#e74c3c')
    bars6 = ax_fin.bar(x_fin + width/2, o_fin, width, label='Optimized', color='#2ecc71')
    ax_fin.set_title("Live Financials (INR)", color='white', fontweight='bold', pad=15)
    ax_fin.set_xticks(x_fin)
    ax_fin.set_xticklabels(fin_labels, color='white', fontsize=11)
    ax_fin.tick_params(colors='white')
    ax_fin.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"₹{val/1000:.0f}K"))
    
    # Custom format for large currency values on top of bars
    ax_fin.bar_label(bars5, labels=[f"₹{v/1000:,.0f}K" for v in b_fin], padding=3, color='white', fontsize=10)
    ax_fin.bar_label(bars6, labels=[f"₹{v/1000:,.0f}K" for v in o_fin], padding=3, color='white', fontsize=10)

sim_thread = threading.Thread(target=run_simulation, daemon=True)
sim_thread.start()

ani = FuncAnimation(fig, update_dashboard, interval=100, cache_frame_data=False)
plt.tight_layout() # Added tight layout to wrap it all together cleanly
plt.show()
import os
import math
import warnings
import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION & DIRECTORY SETUP
# ==========================================
# Folder path for saving plots
OUTPUT_DIR = "outputs2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Simulation Time Parameters
WEEKS_TO_SIMULATE = 4
DAYS_PER_WEEK = 5
MINUTES_PER_DAY = 480          # 10 AM to 6 PM (8-hour shift)
SIM_TIME = WEEKS_TO_SIMULATE * DAYS_PER_WEEK * MINUTES_PER_DAY
WARMUP = MINUTES_PER_DAY       # Discard the first day for steady-state
N_REPS = 15                   # Number of independent replications to run

# Baseline Configuration
BASE_TELLERS = 3
BASE_HD = 1
BASE_RM = 1

# Grid Search Parameters
TELLER_OPTIONS = [3, 4, 5, 6, 7]
HELPDESK_OPTIONS = [1, 2, 3]
RM_OPTIONS = [1, 2, 3]         

# Financial Data & Costs
SERVICE_VALUES = {'Teller': 150, 'Helpdesk': 600, 'RM': 6500}
STAFF_DAILY_COST = {'Teller': 1500, 'Helpdesk': 2000, 'RM': 4000}

# Global aesthetic settings for matplotlib/seaborn
plt.rcParams.update({
    'font.size': 12, 
    'axes.spines.top': False, 
    'axes.spines.right': False,
    'figure.dpi': 200
})
sns.set_theme(style="whitegrid", palette="muted")


# ==========================================
# 2. ANALYTICS ENGINE
# ==========================================
class AnalyticsEngine:
    """Handles tracking and logging of simulation metrics across replications."""
    def __init__(self, scenario_name, rep_id):
        self.scenario_name = scenario_name
        self.rep_id = rep_id
        self.served_data = []      
        self.abandoned_data = []   
        self.arrival_log = []
        self.queue_snapshots = []

    def record_arrival(self, time_minute, day_of_week):
        if time_minute > WARMUP:
            hour_of_day = int((time_minute % MINUTES_PER_DAY) // 60) + 10
            self.arrival_log.append({'Day': day_of_week, 'Hour': hour_of_day})

    def record_served(self, time_minute, service, wait_time, service_time):
        if time_minute > WARMUP:
            self.served_data.append({
                'Scenario': self.scenario_name, 'Rep': self.rep_id,
                'Service': service, 'Wait_Time': wait_time, 'Service_Time': service_time
            })

    def record_abandoned(self, time_minute, service, reason):
        if time_minute > WARMUP:
            self.abandoned_data.append({
                'Scenario': self.scenario_name, 'Rep': self.rep_id,
                'Service': service, 'Reason': reason
            })

    def record_queue_snapshot(self, minute_of_day, tellers_q, hd_q, rm_q):
        self.queue_snapshots.append({
            'Scenario': self.scenario_name, 'Rep': self.rep_id,
            'Minute_of_Day': minute_of_day,
            'Teller_Q': tellers_q, 'HD_Q': hd_q, 'RM_Q': rm_q,
            'Total_Queue': tellers_q + hd_q + rm_q
        })


# ==========================================
# 3. CORE SIMULATION CLASSES
# ==========================================
class Customer:
    """Represents a single bank customer with specific tolerances and needs."""
    def __init__(self, env):
        self.service_type = np.random.choice(['Teller', 'Helpdesk', 'RM'], p=[0.60, 0.25, 0.15])
        self.priority = 1 if np.random.uniform(0, 1) < 0.10 else 2
        self.tolerance_k = np.random.uniform(3, 10)  
        self.patience_limit = np.random.weibull(2.0) * 15  

class RetailBranch:
    """Manages the branch resources (staff) and their respective queues."""
    def __init__(self, env, n_tellers, n_hd, n_rms, discipline="pooled"):
        self.env = env
        self.discipline = discipline
        
        if discipline == "priority":
            self.tellers = simpy.PriorityResource(env, capacity=n_tellers)
            self.helpdesk = simpy.PriorityResource(env, capacity=n_hd)
            self.rms = simpy.PriorityResource(env, capacity=n_rms)
        else:
            self.tellers = simpy.Resource(env, capacity=n_tellers)
            self.helpdesk = simpy.Resource(env, capacity=n_hd)
            self.rms = simpy.Resource(env, capacity=n_rms)

    def get_service_time(self, service_type):
        if service_type == 'Teller': 
            return np.random.gamma(shape=3.0, scale=1.0)     
        elif service_type == 'Helpdesk': 
            return np.random.gamma(shape=5.0, scale=1.2) 
        else: 
            return np.random.lognormal(mean=2.6, sigma=0.5)                         


# ==========================================
# 4. DISCRETE EVENT PROCESSES
# ==========================================
def monitor_queues(env, branch, analytics):
    """Background process to snapshot queue lengths every 10 minutes."""
    yield env.timeout(WARMUP)
    while True:
        minute_of_day = env.now % MINUTES_PER_DAY
        analytics.record_queue_snapshot(
            minute_of_day, len(branch.tellers.queue), 
            len(branch.helpdesk.queue), len(branch.rms.queue)
        )
        yield env.timeout(10)

def customer_lifecycle(env, customer, branch, analytics):
    """Handles the full flow of a customer entering, waiting, and leaving the branch."""
    arrival_time = env.now
    
    if customer.service_type == 'Teller': target_staff = branch.tellers
    elif customer.service_type == 'Helpdesk': target_staff = branch.helpdesk
    else: target_staff = branch.rms

    q_length = len(target_staff.queue)
    prob_join = 1.0 / (1.0 + math.exp(0.8 * (q_length - customer.tolerance_k)))
    
    if np.random.uniform(0, 1) > prob_join:
        analytics.record_abandoned(arrival_time, customer.service_type, 'Balked_Crowd')
        return

    if branch.discipline == "priority":
        req = target_staff.request(priority=customer.priority)
    else:
        req = target_staff.request()
        
    with req as request:
        results = yield env.any_of([request, env.timeout(customer.patience_limit)])
        
        if request in results:
            wait_time = env.now - arrival_time
            service_time = branch.get_service_time(customer.service_type)
            
            if np.random.uniform(0, 1) < 0.05: 
                service_time += np.random.uniform(5, 15)
                
            yield env.timeout(service_time)
            analytics.record_served(env.now, customer.service_type, wait_time, service_time)
        else:
            analytics.record_abandoned(env.now, customer.service_type, 'Reneged_Patience')

def nhpp_arrival_generator(env, branch, analytics):
    """Non-Homogeneous Poisson Process for generating realistic foot traffic."""
    lambda_max_overall = 4.0 
    
    while True:
        u1 = np.random.uniform(0, 1)
        inter_arrival = -math.log(u1) / lambda_max_overall
        yield env.timeout(inter_arrival)
        
        current_minute = env.now
        day_of_week = int(current_minute // MINUTES_PER_DAY) % DAYS_PER_WEEK
        time_of_day = current_minute % MINUTES_PER_DAY
        
        if day_of_week == 0: day_multiplier = 1.2      
        elif day_of_week == 4: day_multiplier = 1.3    
        else: day_multiplier = 1.0                     
        
        if 0 <= time_of_day < 60: time_multiplier = 1.2      
        elif 60 <= time_of_day < 120: time_multiplier = 0.8  
        elif 120 <= time_of_day < 240: time_multiplier = 1.1 
        elif 240 <= time_of_day < 360: time_multiplier = 1.3 
        else: time_multiplier = 0.85
            
        lambda_t = (2.5 * time_multiplier) * day_multiplier
        prob_accept = lambda_t / lambda_max_overall
        
        if np.random.uniform(0, 1) < prob_accept:
            analytics.record_arrival(current_minute, day_of_week)
            customer = Customer(env)
            env.process(customer_lifecycle(env, customer, branch, analytics))


# ==========================================
# 5. EXECUTION WRAPPER (MULTIPLE REPS)
# ==========================================
def run_scenario(scenario_name, tellers, hd, rms, discipline="pooled", verbose=True):
    """Runs N_REPS for a specific staffing configuration and returns aggregated DataFrames."""
    if verbose:
        print(f" Simulating: {scenario_name} (Tellers: {tellers}, HD: {hd}, RM: {rms})")
        
    all_served, all_abandoned, all_queues, all_arrivals = [], [], [], []
    
    for rep in range(N_REPS):
        np.random.seed(42 + rep * 137) 
        env = simpy.Environment()
        analytics = AnalyticsEngine(scenario_name, rep)
        branch = RetailBranch(env, tellers, hd, rms, discipline)
        
        env.process(nhpp_arrival_generator(env, branch, analytics))
        env.process(monitor_queues(env, branch, analytics))
        env.run(until=SIM_TIME)
        
        all_served.extend(analytics.served_data)
        all_abandoned.extend(analytics.abandoned_data)
        all_queues.extend(analytics.queue_snapshots)
        all_arrivals.extend(analytics.arrival_log)
        
    return (
        pd.DataFrame(all_served), 
        pd.DataFrame(all_abandoned), 
        pd.DataFrame(all_queues), 
        pd.DataFrame(all_arrivals)
    )

def calculate_financials(df_served, df_abandoned, tellers, hd, rms):
    """Calculates average daily Revenue, OpEx, and Lost Revenue due to walkouts."""
    simulated_days = ((WEEKS_TO_SIMULATE * DAYS_PER_WEEK) - (WARMUP / MINUTES_PER_DAY)) * N_REPS
    
    revenue = 0
    if not df_served.empty:
        revenue = df_served.apply(lambda row: SERVICE_VALUES.get(row['Service'], 0), axis=1).sum()
        
    lost_revenue = 0
    if not df_abandoned.empty:
        lost_revenue = df_abandoned.apply(lambda row: SERVICE_VALUES.get(row['Service'], 0), axis=1).sum()
        
    opex = simulated_days * ((tellers * STAFF_DAILY_COST['Teller']) + 
                             (hd * STAFF_DAILY_COST['Helpdesk']) + 
                             (rms * STAFF_DAILY_COST['RM']))
    
    avg_daily_rev = revenue / simulated_days
    avg_daily_opex = opex / simulated_days
    avg_daily_lost = lost_revenue / simulated_days
    net_profit = avg_daily_rev - avg_daily_opex
    
    return net_profit, avg_daily_rev, avg_daily_opex, avg_daily_lost


# ==========================================
# 6. MAIN EXECUTION ROUTINE
# ==========================================
if __name__ == "__main__":
    print(f"==================================================")
    print(f" STARTING SIMULATION PIPELINE ({N_REPS} Reps/Scenario) ")
    print(f"==================================================\n")
    
    # --- STEP 1: RUN BASELINE ---
    print("Evaluating Baseline Operations")
    df_s_base, df_a_base, df_q_base, df_arr = run_scenario(
        "Baseline", tellers=BASE_TELLERS, hd=BASE_HD, rms=BASE_RM
    )

    # --- STEP 2: RUN GRID SEARCH TO FIND OPTIMAL ---
    print("\nExecuting Full 3D Grid Search Optimization")
    results = []
    
    total_combinations = len(TELLER_OPTIONS) * len(HELPDESK_OPTIONS) * len(RM_OPTIONS)
    count = 1

    for t in TELLER_OPTIONS:
        for h in HELPDESK_OPTIONS:
            for r in RM_OPTIONS:
                scenario_id = f"Grid_T{t}_H{h}_R{r}"
                
                # Dynamic progress print to terminal
                print(f" Testing configuration {count}/{total_combinations}: {t} Tellers, {h} HD, {r} RM", end='\r')
                
                df_s, df_a, _, _ = run_scenario(scenario_id, tellers=t, hd=h, rms=r, discipline="priority", verbose=False)
                net_profit, rev, opex, lost = calculate_financials(df_s, df_a, t, h, r)
                
                results.append({
                    'Tellers': t,
                    'Helpdesk': h,
                    'RM': r,
                    'Net_Profit': net_profit,
                    'Revenue': rev,
                    'OpEx': opex,
                    'Lost_Revenue_Walkouts': lost
                })
                count += 1
                
    results_df = pd.DataFrame(results)
    optimal_config = results_df.loc[results_df['Net_Profit'].idxmax()]
    
    opt_tellers = int(optimal_config['Tellers'])
    opt_hd = int(optimal_config['Helpdesk'])
    opt_rm = int(optimal_config['RM'])
    
    print(f"\nGrid Search Complete! Optimal identified as: {opt_tellers} Tellers, {opt_hd} Helpdesk, {opt_rm} RM.")

    # --- STEP 3: RUN OPTIMIZED SCENARIO ---
    print("\nSimulating the Optimized System Configuration")
    df_s_opt, df_a_opt, df_q_opt, _ = run_scenario(
        "Optimized", tellers=opt_tellers, hd=opt_hd, rms=opt_rm, discipline="priority"
    )

    # --- STEP 4: DATA AGGREGATION & PLOTTING ---
    print("\nGenerating Comparative Analytics & Plots...")
    
    df_served = pd.concat([df_s_base, df_s_opt])
    df_abandoned = pd.concat([df_a_base, df_a_opt])
    df_queues = pd.concat([df_q_base, df_q_opt])

    # PLOT 1: Traffic Heatmap
    plt.figure(figsize=(10, 6))
    heatmap_data = pd.crosstab(df_arr['Day'], df_arr['Hour'])
    heatmap_data.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    sns.heatmap(heatmap_data, cmap="YlOrRd", annot=True, fmt="d", cbar_kws={'label': 'Customer Volume'})
    plt.title("Traffic Heatmap Validating NHPP Input Model", fontweight='bold')
    plt.xlabel("Hour of Day (10 AM to 6 PM)")
    plt.ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_traffic_heatmap.png"))
    plt.close()

    # PLOT 2: Queue Backlog Line Chart
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_queues, x='Minute_of_Day', y='Total_Queue', hue='Scenario', 
                 errorbar='ci', linewidth=2.5, palette=['#e74c3c', '#2ecc71'])
    plt.title("Queue Backlog Collapse During Peak Stress (95% CI)", fontweight='bold')
    plt.xlabel("Minute of Day (0 = 10 AM, 480 = 6 PM)")
    plt.ylabel("Expected Customers in Queue (Lq)")
    plt.axvspan(180, 240, color='gray', alpha=0.15, label='Lunch Peak Dip')
    plt.axvspan(240, 360, color='orange', alpha=0.1, label='Post-Lunch Rush')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_queue_backlog.png"))
    plt.close()

    # PLOT 3: Wait Time Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_served, x='Scenario', y='Wait_Time', hue='Service',
                   split=False, inner="quart", palette='Set2')
    plt.title("Reduction in Variance and Extreme Wait Times", fontweight='bold')
    plt.ylabel("Wait Time (Minutes)")
    plt.xlabel("Staffing Configuration")
    plt.axhline(15, color='red', linestyle='--', alpha=0.5, label='15 Min SLA Limit')
    plt.legend(title='Service Type')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_wait_time_violin.png"))
    plt.close()

    # Calculate both baseline and optimized metrics for final prints and plotting
    b_profit, b_rev, b_opex, b_lost = calculate_financials(df_s_base, df_a_base, BASE_TELLERS, BASE_HD, BASE_RM)
    o_profit, o_rev, o_opex, o_lost = calculate_financials(df_s_opt, df_a_opt, opt_tellers, opt_hd, opt_rm)

    # PLOT 4: Financial Opportunity Optimization
    fin_df = pd.DataFrame({
        'Scenario': ['Baseline (Understaffed)', 'Optimized (Well-Staffed)', 
                     'Baseline (Understaffed)', 'Optimized (Well-Staffed)', 
                     'Baseline (Understaffed)', 'Optimized (Well-Staffed)'],
        'Metric': ['1. Daily Salary Costs', '1. Daily Salary Costs', 
                   '2. Money Lost to Walkouts', '2. Money Lost to Walkouts', 
                   '3. Actual Money Earned (Profit)', '3. Actual Money Earned (Profit)'],
        'Value': [b_opex, o_opex, b_lost, o_lost, b_profit, o_profit]
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left Plot: Salary costs
    salary_df = fin_df[fin_df['Metric'] == '1. Daily Salary Costs']
    sns.barplot(data=salary_df, x='Metric', y='Value', hue='Scenario', palette=['#e74c3c', '#2ecc71'], ax=axes[0])
    axes[0].set_title("The Cost (Salaries)", fontweight='bold')
    axes[0].set_ylabel("Total Cost (INR)")
    axes[0].set_xlabel("")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"₹{x/1000:.0f}K"))
    axes[0].get_legend().remove()

    # Right Plot: Lost Revenue vs Profit
    revenue_df = fin_df[fin_df['Metric'] != '1. Daily Salary Costs']
    sns.barplot(data=revenue_df, x='Metric', y='Value', hue='Scenario', palette=['#e74c3c', '#2ecc71'], ax=axes[1])
    axes[1].set_title("The Result (Walkouts vs Profit)", fontweight='bold')
    axes[1].set_ylabel("Total Value (INR)")
    axes[1].set_xlabel("")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"₹{x/1e6:.1f}M"))
    axes[1].legend(title="Staffing Configuration")

    plt.suptitle("Financial Impact — Spending a Little on Salaries Saves Millions in Walkouts", fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_financial_impact.png"))
    plt.close()

    # PLOT 5: Optimization Landscape Heatmap 
    optimal_rm_slice = results_df[results_df['RM'] == opt_rm]
    plt.figure(figsize=(8, 6))
    profit_matrix = optimal_rm_slice.pivot(index='Helpdesk', columns='Tellers', values='Net_Profit')
    sns.heatmap(profit_matrix, annot=True, fmt=".0f", cmap="viridis", cbar_kws={'label': 'Net Daily Profit (₹)'})
    plt.title(f"Grid Search Landscape (Sliced at Optimal RM = {opt_rm})", fontweight='bold')
    plt.xlabel("Number of Tellers")
    plt.ylabel("Number of Helpdesk Staff")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_optimization_landscape.png"))
    plt.close()

    # PLOT 6: Customer Abandonment
    simulated_days = ((WEEKS_TO_SIMULATE * DAYS_PER_WEEK) - (WARMUP / MINUTES_PER_DAY)) * N_REPS
    if not df_abandoned.empty:
        walkout_summary = df_abandoned.groupby(['Scenario', 'Service']).size().reset_index(name='Total_Walkouts')
        walkout_summary['Daily_Walkouts'] = walkout_summary['Total_Walkouts'] / simulated_days
        walkout_summary['Scenario'] = pd.Categorical(walkout_summary['Scenario'], categories=['Baseline', 'Optimized'], ordered=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=walkout_summary, x='Service', y='Daily_Walkouts', hue='Scenario', palette=['#e74c3c', '#2ecc71'])
        plt.title("Average Daily Customer Walkouts (Before vs After)", fontweight='bold', fontsize=14)
        plt.ylabel("Average Number of Customers Lost per Day")
        plt.xlabel("What Service Were They Waiting For?")
        
        ax = plt.gca()
        for p in ax.patches:
            height = p.get_height()
            if height > 0:  
                ax.annotate(f'{height:.1f}', (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.legend(title="Staffing Configuration")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "6_customer_walkouts.png"))
        plt.close()

    # --- STEP 5: FINAL TERMINAL OUTPUT ---
    print("\n==========================================================================")
    print(" FINAL SYSTEM OPTIMIZATION RESULTS (Daily Averages)")
    print("==========================================================================")
    print(f"{'Metric':<25} | {'Baseline (Understaffed)':<25} | {'Optimized (Well-Staffed)'}")
    print("-" * 74)
    print(f"{'Staffing (T, HD, RM)':<25} | {f'{BASE_TELLERS}, {BASE_HD}, {BASE_RM}':<25} | {f'{opt_tellers}, {opt_hd}, {opt_rm}'}")
    print(f"{'Daily Operating Cost':<25} | {f'₹{b_opex:,.2f}':<25} | ₹{o_opex:,.2f}")
    print(f"{'Lost Daily to Walkouts':<25} | {f'₹{b_lost:,.2f}':<25} | ₹{o_lost:,.2f}")
    print(f"{'Maximized Daily Profit':<25} | {f'₹{b_profit:,.2f}':<25} | ₹{o_profit:,.2f}")
    print("==========================================================================")
    print(f"Profit Improvement: +₹{(o_profit - b_profit):,.2f} per day")
    print(f"\nAll outputs successfully saved to '{OUTPUT_DIR}/' directory.")
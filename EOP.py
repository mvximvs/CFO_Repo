import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import math
from datetime import datetime, date, timedelta
from typing import List, Dict, Tuple, Optional
import calendar

DB_PATH = "employees_enhanced.db"

# ======================================================
# DB INIT & SCHEMA ENHANCEMENT - FIXED VERSION
# ======================================================

EXPECTED_COLUMNS = [
    "id", "name", "role_type", "seniority", "command_level", "role_impact",
    "salary", "tokens", "fte", "performance", "tvl", "company_stage", "hire_cohort",
    "hire_date", "departure_date", "dilution_factor", "final_equity_pct",
    "cliff_months", "vesting_years", "notes", "equity_grant_date"
]

def init_database():
    """Initialize database with all required tables"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create employees table
    c.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        role_type TEXT,
        seniority INTEGER,
        command_level INTEGER,
        role_impact INTEGER,
        salary REAL,
        tokens REAL,
        fte REAL,
        performance INTEGER,
        tvl REAL,
        company_stage TEXT,
        hire_cohort TEXT,
        hire_date TEXT,
        departure_date TEXT,
        dilution_factor REAL DEFAULT 1.0,
        final_equity_pct REAL,
        cliff_months INTEGER DEFAULT 12,
        vesting_years INTEGER DEFAULT 4,
        notes TEXT,
        equity_grant_date TEXT
    )
    """)
    
    # Create dilution_history table
    c.execute("""
    CREATE TABLE IF NOT EXISTS dilution_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        dilution_event TEXT NOT NULL,
        pre_dilution_factor REAL,
        post_dilution_factor REAL,
        description TEXT
    )
    """)
    
    # Create equity_transactions table
    c.execute("""
    CREATE TABLE IF NOT EXISTS equity_transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id INTEGER,
        date TEXT NOT NULL,
        transaction_type TEXT NOT NULL,
        equity_pct REAL,
        tokens REAL,
        cumulative_equity_pct REAL,
        cumulative_tokens REAL,
        notes TEXT,
        FOREIGN KEY (employee_id) REFERENCES employees (id)
    )
    """)
    
    # Create initial dilution event if table is empty
    c.execute("SELECT COUNT(*) FROM dilution_history")
    if c.fetchone()[0] == 0:
        c.execute("""
        INSERT INTO dilution_history (date, dilution_event, pre_dilution_factor, post_dilution_factor, description)
        VALUES (?, ?, ?, ?, ?)
        """, ("2023-01-01", "Initial", 1.0, 1.0, "Initial state - no dilution"))
    
    conn.commit()
    conn.close()

def db_fetch(query, params=()):
    """Execute a SELECT query and return results"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(query, params)
        rows = c.fetchall()
    except sqlite3.OperationalError as e:
        print(f"Database error: {e}")
        print(f"Query: {query}")
        rows = []
    finally:
        conn.close()
    return rows

def db_execute(query, params=()):
    """Execute an INSERT/UPDATE/DELETE query"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(query, params)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        print(f"Query: {query}")
    finally:
        conn.close()

def db_execute_many(query, params_list):
    """Execute multiple queries"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.executemany(query, params_list)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

# ======================================================
# DILUTION ENGINE
# ======================================================

class DilutionEngine:
    """Manages dilution events and their impact on equity allocations"""
    
    def __init__(self):
        self.dilution_history = self.load_dilution_history()
    
    def load_dilution_history(self):
        rows = db_fetch("SELECT * FROM dilution_history ORDER BY date")
        if rows:
            return pd.DataFrame(rows, columns=[
                "id", "date", "dilution_event", "pre_dilution_factor", 
                "post_dilution_factor", "description"
            ])
        return pd.DataFrame()
    
    def add_dilution_event(self, date_str: str, event_type: str, 
                          dilution_percent: float, description: str = ""):
        """Add a dilution event (fundraising, employee pool increase, etc.)"""
        # Get current dilution factor
        current_factor = self.get_current_dilution_factor()
        
        # Calculate new factor
        new_factor = current_factor * (1 - dilution_percent / 100)
        
        db_execute("""
        INSERT INTO dilution_history (date, dilution_event, pre_dilution_factor, 
                                    post_dilution_factor, description)
        VALUES (?, ?, ?, ?, ?)
        """, (date_str, event_type, current_factor, new_factor, description))
        
        # Update all employees' dilution_factor
        db_execute("UPDATE employees SET dilution_factor = ?", (new_factor,))
        
        return new_factor
    
    def get_current_dilution_factor(self):
        """Get the current dilution factor (1.0 = no dilution)"""
        rows = db_fetch("""
        SELECT post_dilution_factor FROM dilution_history 
        ORDER BY date DESC LIMIT 1
        """)
        if rows:
            return rows[0][0]
        return 1.0  # Default no dilution
    
    def get_dilution_factor_at_date(self, target_date: str):
        """Get dilution factor at a specific historical date"""
        rows = db_fetch("""
        SELECT post_dilution_factor FROM dilution_history 
        WHERE date <= ? ORDER BY date DESC LIMIT 1
        """, (target_date,))
        if rows:
            return rows[0][0]
        return 1.0
    
    def get_dilution_timeline(self):
        """Get timeline of dilution events for plotting"""
        return self.load_dilution_history()

# ======================================================
# ENHANCED EQUITY ENGINE
# ======================================================

MAX_T = 11
BASE_POOL = 0.003      # 0.30 %
VEST_POOL = 0.007      # 0.70 %
BASE_TVL_BONUS = 0.0002

STAGE_COEFF = {
    "Pre-seed": 1.00,
    "Seed": 0.75,
    "Series A": 0.45,
    "Series B": 0.20,
    "Growth": 0.10,
}

PERFORMANCE_MULTIPLIER = {
    1: 0.0,   # Underperform
    2: 0.25,  # Meets
    3: 0.75,  # Exceeds
    4: 1.20,  # Outstanding
}

def calculate_role_score(S: int, C: int, R: int) -> int:
    """Calculate T = S + C + R"""
    return S + C + R

def calculate_base_equity(T: int, base_pool: float = BASE_POOL) -> float:
    """Base equity grant (upfront)"""
    return base_pool * (T / MAX_T)

def calculate_vesting_equity(T: int, vest_pool: float = VEST_POOL) -> float:
    """Vesting equity grant"""
    return vest_pool * (T / MAX_T)

def calculate_performance_bonus(vest_eq: float, perf_score: int) -> float:
    """Performance bonus on vesting equity"""
    multiplier = PERFORMANCE_MULTIPLIER.get(perf_score, 0.25)
    return vest_eq * multiplier

def calculate_tvl_bonus(tvl: float, base_bonus: float = BASE_TVL_BONUS) -> float:
    """TVL-based bonus"""
    if tvl <= 0:
        return 0.0
    return base_bonus * math.log10(1 + tvl)

def calculate_final_equity_pct(
    T: int, 
    base_eq: float, 
    vest_eq: float,
    perf_b: float, 
    tvl_b: float, 
    fte: float, 
    stage: str,
    dilution_factor: float = 1.0,
    hire_date: Optional[str] = None
) -> float:
    """Calculate final equity percentage with dilution adjustment"""
    total = base_eq + vest_eq + perf_b + tvl_b
    coeff = STAGE_COEFF.get(stage, 1.0)
    
    # Apply dilution factor (lower factor = more dilution = higher equity needed)
    adjusted_total = total * coeff * fte * dilution_factor
    
    return adjusted_total

def generate_equity_waterfall(
    employee_id: int,
    total_supply: float = 10_000_000,
    current_date: Optional[date] = None
) -> pd.DataFrame:
    """Generate detailed equity waterfall for an employee"""
    if current_date is None:
        current_date = date.today()
    
    # Get employee data
    rows = db_fetch("""
        SELECT hire_date, cliff_months, vesting_years, final_equity_pct, 
               dilution_factor, equity_grant_date
        FROM employees WHERE id = ?
    """, (employee_id,))
    
    if not rows:
        return pd.DataFrame()
    
    hire_date_str, cliff_m, vest_years, equity_pct, dilution_factor, grant_date = rows[0]
    
    # Parse dates
    try:
        hire_date = datetime.strptime(hire_date_str, "%Y-%m-%d").date() if hire_date_str else current_date
        grant_date = datetime.strptime(grant_date, "%Y-%m-%d").date() if grant_date else hire_date
    except:
        hire_date = current_date
        grant_date = current_date
    
    # Ensure numeric values
    try:
        cliff_m = int(cliff_m) if cliff_m else 12
        vest_years = int(vest_years) if vest_years else 4
        equity_pct = float(equity_pct) if equity_pct else 0.0
    except:
        cliff_m = 12
        vest_years = 4
        equity_pct = 0.0
    
    # Calculate total tokens
    total_tokens = equity_pct * total_supply
    
    # Generate monthly schedule
    schedule = []
    cumulative_tokens = 0
    cumulative_pct = 0
    
    for month in range(0, vest_years * 12 + 1):
        current_month_date = grant_date + timedelta(days=30 * month)
        
        if month == 0:
            # Grant date
            schedule.append({
                "date": current_month_date,
                "month": month,
                "event": "Grant",
                "tokens_vested": 0,
                "tokens_cumulative": 0,
                "pct_vested": 0,
                "pct_cumulative": 0,
                "type": "grant"
            })
        elif month == cliff_m:
            # Cliff vesting
            cliff_tokens = total_tokens * (cliff_m / (vest_years * 12))
            cumulative_tokens = cliff_tokens
            cumulative_pct = equity_pct * (cliff_m / (vest_years * 12))
            
            schedule.append({
                "date": current_month_date,
                "month": month,
                "event": "Cliff",
                "tokens_vested": cliff_tokens,
                "tokens_cumulative": cumulative_tokens,
                "pct_vested": equity_pct * (cliff_m / (vest_years * 12)),
                "pct_cumulative": cumulative_pct,
                "type": "cliff"
            })
        elif month > cliff_m and month <= vest_years * 12:
            # Monthly vesting
            monthly_tokens = total_tokens / (vest_years * 12)
            cumulative_tokens = min(total_tokens, cumulative_tokens + monthly_tokens)
            cumulative_pct = min(equity_pct, cumulative_pct + (equity_pct / (vest_years * 12)))
            
            schedule.append({
                "date": current_month_date,
                "month": month,
                "event": f"Month {month}",
                "tokens_vested": monthly_tokens,
                "tokens_cumulative": cumulative_tokens,
                "pct_vested": equity_pct / (vest_years * 12),
                "pct_cumulative": cumulative_pct,
                "type": "vest"
            })
        elif month == vest_years * 12:
            # Final vest
            remaining = total_tokens - cumulative_tokens
            if remaining > 0:
                cumulative_tokens = total_tokens
                cumulative_pct = equity_pct
                
                schedule.append({
                    "date": current_month_date,
                    "month": month,
                    "event": "Final",
                    "tokens_vested": remaining,
                    "tokens_cumulative": cumulative_tokens,
                    "pct_vested": equity_pct - cumulative_pct + (equity_pct / (vest_years * 12)),
                    "pct_cumulative": cumulative_pct,
                    "type": "final"
                })
    
    return pd.DataFrame(schedule)

# ======================================================
# ENHANCED MOCK DATA
# ======================================================

def load_mock_data_if_empty():
    """Load mock data if the employees table is empty"""
    rows = db_fetch("SELECT COUNT(*) FROM employees")
    
    if not rows or rows[0][0] == 0:
        # Enhanced mock data with more realistic scenarios
        mock_employees = [
            # Founders (early, high equity)
            ("Yann LeCun", "Research", 3, 3, 5, 95000, 220000, 1.0, 3, 5000000, 
             "Pre-seed", "founder", "2023-01-15", None, 1.0, 0.022, 12, 4, 
             "Co-founder, Head of Research", "2023-01-15"),
            
            ("Marc Andreessen", "Engineering", 3, 3, 5, 92000, 210000, 1.0, 3, 4500000,
             "Pre-seed", "founder", "2023-01-15", None, 1.0, 0.021, 12, 4,
             "Co-founder, CTO", "2023-01-15"),
            
            ("Sarah Guo", "Product", 2, 2, 4, 88000, 175000, 1.0, 3, 3000000,
             "Pre-seed", "early", "2023-03-01", None, 1.0, 0.0175, 12, 4,
             "Head of Product", "2023-03-01"),
            
            # Early hires (Seed round)
            ("Ahmed Khan", "Risk Ops", 2, 1, 4, 82000, 140000, 1.0, 3, 2000000,
             "Seed", "early", "2023-06-01", None, 0.9, 0.014, 12, 4,
             "Risk Operations Lead", "2023-06-01"),
            
            ("Mei Chen", "Bizdev", 2, 2, 3, 78000, 125000, 1.0, 3, 1500000,
             "Seed", "early", "2023-06-15", None, 0.9, 0.0125, 12, 4,
             "Business Development", "2023-06-15"),
            
            # Recent hires (Series A)
            ("Lucas Silva", "Engineering", 1, 0, 3, 75000, 80000, 1.0, 2, 500000,
             "Series A", "recent", "2024-01-15", None, 0.7, 0.008, 12, 4,
             "Software Engineer", "2024-01-15"),
            
            ("Noah Williams", "Research", 1, 0, 3, 72000, 75000, 1.0, 2, 400000,
             "Series A", "recent", "2024-02-01", None, 0.7, 0.0075, 12, 4,
             "Quantitative Analyst", "2024-02-01"),
            
            ("Ines Rodriguez", "Product", 1, 0, 2, 68000, 50000, 1.0, 2, 200000,
             "Series A", "recent", "2024-02-15", None, 0.7, 0.005, 12, 4,
             "Product Manager", "2024-02-15"),
        ]

        for emp in mock_employees:
            db_execute("""
            INSERT INTO employees (
                name, role_type, seniority, command_level, role_impact,
                salary, tokens, fte, performance, tvl, company_stage, hire_cohort,
                hire_date, departure_date, dilution_factor, final_equity_pct,
                cliff_months, vesting_years, notes, equity_grant_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, emp)
        
        # Add dilution events
        dilution_engine = DilutionEngine()
        dilution_engine.add_dilution_event(
            "2023-06-01",
            "Seed Round",
            20.0,
            "Seed fundraising round - $2M at $10M valuation"
        )
        dilution_engine.add_dilution_event(
            "2024-01-01",
            "Series A",
            25.0,
            "Series A round - $5M at $20M valuation"
        )
        
        return True
    return False

# ======================================================
# VISUALIZATION HELPERS
# ======================================================

def create_equity_timeline_chart(df: pd.DataFrame) -> go.Figure:
    """Create a timeline chart of equity vesting for all employees"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    fig = go.Figure()
    
    # Get unique employees
    employees = df['name'].unique()[:10]  # Limit to 10 for clarity
    colors = px.colors.qualitative.Set3
    
    for idx, emp in enumerate(employees):
        emp_data = df[df['name'] == emp]
        
        if not emp_data.empty:
            # Add area for total grant
            fig.add_trace(go.Scatter(
                x=emp_data['date'],
                y=emp_data['pct_cumulative'] * 100,
                mode='lines',
                fill='tozeroy',
                name=emp,
                line=dict(color=colors[idx % len(colors)], width=2),
                hovertemplate=(
                    f"<b>{emp}</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Vested: %{y:.3f}%<br>"
                    "Cumulative: %{y:.3f}%<extra></extra>"
                )
            ))
    
    fig.update_layout(
        title="Equity Vesting Timeline",
        xaxis_title="Date",
        yaxis_title="Equity Vested (%)",
        hovermode="x unified",
        template="plotly_white",
        height=500
    )
    
    return fig

def create_dilution_waterfall_chart() -> go.Figure:
    """Create a waterfall chart showing dilution impact"""
    dilution_engine = DilutionEngine()
    dilution_df = dilution_engine.get_dilution_timeline()
    
    if dilution_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No dilution events recorded",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    fig = go.Figure()
    
    fig.add_trace(go.Waterfall(
        name="Dilution",
        orientation="v",
        measure=["absolute"] + ["relative"] * (len(dilution_df) - 1) + ["total"],
        x=dilution_df['date'].tolist() + ["Current"],
        y=dilution_df['post_dilution_factor'].tolist() + [dilution_df['post_dilution_factor'].iloc[-1]],
        textposition="outside",
        text=[f"{x:.1%}" for x in dilution_df['post_dilution_factor']] + 
             [f"{dilution_df['post_dilution_factor'].iloc[-1]:.1%}"],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="Dilution Waterfall Over Time",
        xaxis_title="Date",
        yaxis_title="Dilution Factor",
        yaxis_tickformat=".0%",
        template="plotly_white",
        showlegend=False
    )
    
    return fig

# ======================================================
# DATA LOADING FUNCTIONS
# ======================================================

def load_employees_data():
    """Load all employees data"""
    rows = db_fetch("SELECT * FROM employees ORDER BY id")
    if rows:
        df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['seniority', 'command_level', 'role_impact', 'salary', 
                          'tokens', 'fte', 'performance', 'tvl', 'dilution_factor',
                          'final_equity_pct', 'cliff_months', 'vesting_years']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    return pd.DataFrame()

def load_data():
    """Main data loading function - wrapper for compatibility"""
    return load_employees_data()

# ======================================================
# STREAMLIT UI - ENHANCED
# ======================================================

st.set_page_config(
    page_title="DeFine – Advanced Equity Planning",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database first
init_database()
load_mock_data_if_empty()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/blockchain-new-logo.png", width=80)
    st.title("DeFine HR Dashboard")
    
    st.markdown("---")
    st.markdown("### Company Metrics")
    
    # Load data for metrics
    df = load_data()
    
    if not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Employees", len(df))
            active_employees = len(df[df['departure_date'].isna()])
            st.metric("Active Employees", active_employees)
        with col2:
            total_equity = df['final_equity_pct'].sum() * 100
            st.metric("Total Equity", f"{total_equity:.2f}%")
            avg_salary = df['salary'].mean()
            st.metric("Avg Salary", f"${avg_salary:,.0f}")
        
        # Dilution factor display
        dilution_engine = DilutionEngine()
        current_dilution = dilution_engine.get_current_dilution_factor()
        st.progress(current_dilution, f"Dilution Factor: {current_dilution:.1%}")
    else:
        st.info("No employee data available")
    
    st.markdown("---")
    st.markdown("### Quick Actions")
    if st.button("🔄 Refresh All Data"):
        st.rerun()
    
    if st.button("🗑️ Reset Database"):
        if st.checkbox("Confirm reset (will delete all data)"):
            init_database()  # This will recreate empty tables
            st.success("Database reset complete")
            st.rerun()
    
    st.markdown("---")
    st.markdown("### Info")
    st.info("""
    This is an advanced equity planning tool for DeFine.
    Features include:
    - Dilution-aware equity calculations
    - Detailed vesting schedules
    - Scenario modeling
    - Compensation analysis
    """)

# Main content
st.title("🏛️ DeFine — Advanced Equity & Token Allocation Platform")

# Create tabs
tab_dashboard, tab_employees, tab_equity, tab_vesting, tab_dilution, tab_scatter, tab_docs = st.tabs(
    ["📊 Enhanced Dashboard", "👥 Employee Management", "🧮 Equity Calculator", 
     "📅 Vesting Schedules", "📉 Dilution Analysis", "📈 Compensation Analysis", "📘 Documentation"]
)

# ------------------------------------------------------
# ENHANCED DASHBOARD
# ------------------------------------------------------
with tab_dashboard:
    st.subheader("Executive Dashboard")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.info("No employees yet. Add your first team members.")
        if st.button("Load Sample Data"):
            load_mock_data_if_empty()
            st.rerun()
    else:
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_equity = df['final_equity_pct'].sum() * 100
            st.metric("Total Equity Pool", f"{total_equity:.2f}%")
        with col2:
            avg_salary = df['salary'].mean()
            st.metric("Average Salary", f"${avg_salary:,.0f}")
        with col3:
            total_tokens = df['tokens'].sum()
            st.metric("Total Tokens", f"{total_tokens:,.0f}")
        with col4:
            dilution_engine = DilutionEngine()
            dilution = dilution_engine.get_current_dilution_factor()
            st.metric("Dilution Factor", f"{dilution:.1%}")
        
        st.markdown("---")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            # Equity distribution by role
            if 'role_type' in df.columns and 'final_equity_pct' in df.columns:
                equity_by_role = df.groupby('role_type')['final_equity_pct'].sum() * 100
                if not equity_by_role.empty:
                    fig1 = px.pie(
                        values=equity_by_role.values,
                        names=equity_by_role.index,
                        title="Equity Distribution by Role",
                        hole=0.4
                    )
                    st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Salary distribution
            if 'role_type' in df.columns and 'salary' in df.columns:
                fig2 = px.box(
                    df, x='role_type', y='salary',
                    title="Salary Distribution by Role",
                    color='role_type'
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed table
        st.subheader("Employee Summary")
        display_cols = ['name', 'role_type', 'hire_date', 'salary', 'tokens',
                       'final_equity_pct', 'cliff_months', 'vesting_years', 'performance']
        
        # Filter columns that exist
        available_cols = [col for col in display_cols if col in df.columns]
        display_df = df[available_cols].copy()
        
        if 'final_equity_pct' in display_df.columns:
            display_df['final_equity_pct'] = display_df['final_equity_pct'] * 100
        
        display_df = display_df.rename(columns={
            'final_equity_pct': 'Equity %',
            'cliff_months': 'Cliff (mo)',
            'vesting_years': 'Vest (yr)'
        })
        
        format_dict = {}
        if 'salary' in display_df.columns:
            format_dict['salary'] = '${:,.0f}'
        if 'tokens' in display_df.columns:
            format_dict['tokens'] = '{:,.0f}'
        if 'Equity %' in display_df.columns:
            format_dict['Equity %'] = '{:.3f}%'
        
        st.dataframe(
            display_df.style.format(format_dict),
            use_container_width=True,
            height=400
        )

# ------------------------------------------------------
# ENHANCED EMPLOYEE MANAGEMENT
# ------------------------------------------------------
with tab_employees:
    st.subheader("Employee Database Management")
    
    df = load_data()
    
    # Toggle between add/edit mode
    mode = st.radio(
        "Operation Mode",
        ["Add New Employee", "Edit Existing Employee", "Bulk Operations"],
        horizontal=True
    )
    
    if mode == "Add New Employee":
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name")
            role = st.selectbox(
                "Role Type",
                ["Engineering", "Research", "Product", "Bizdev", "Risk Ops", "Operations", "Executive"]
            )
            hire_date = st.date_input("Hire Date", value=date.today())
            fte = st.slider("FTE", 0.1, 1.0, 1.0, 0.1)
            
            # Role scoring
            st.subheader("Role Scoring")
            seniority = st.select_slider(
                "Seniority Level",
                options=[0, 1, 2, 3],
                value=1,
                format_func=lambda x: ["Junior", "Mid", "Senior", "Principal"][x]
            )
            command = st.select_slider(
                "Command Level",
                options=[0, 1, 2, 3],
                value=0,
                format_func=lambda x: ["IC", "Lead", "Manager", "Executive"][x]
            )
            impact = st.slider("Role Impact", 1, 5, 3)
        
        with col2:
            # Compensation
            st.subheader("Compensation")
            salary = st.number_input("Base Salary (USD)", min_value=0, value=80000, step=5000)
            performance = st.selectbox(
                "Performance Rating",
                [1, 2, 3, 4],
                format_func=lambda x: ["Underperform", "Meets", "Exceeds", "Outstanding"][x-1]
            )
            tvl = st.number_input("TVL Contribution (USD)", min_value=0.0, value=0.0, step=10000.0)
            
            # Equity parameters
            st.subheader("Equity Parameters")
            stage = st.selectbox("Company Stage", list(STAGE_COEFF.keys()))
            cliff = st.number_input("Cliff (months)", 0, 24, 12)
            vest_years = st.number_input("Vesting Period (years)", 1, 10, 4)
            
            # Notes
            notes = st.text_area("Notes / Special Conditions")
        
        # Calculate equity preview
        if st.button("Preview Equity Calculation"):
            T = calculate_role_score(seniority, command, impact)
            base_eq = calculate_base_equity(T)
            vest_eq = calculate_vesting_equity(T)
            perf_bonus = calculate_performance_bonus(vest_eq, performance)
            tvl_bonus = calculate_tvl_bonus(tvl)
            
            dilution_engine = DilutionEngine()
            dilution_factor = dilution_engine.get_current_dilution_factor()
            
            final_pct = calculate_final_equity_pct(
                T, base_eq, vest_eq, perf_bonus, tvl_bonus,
                fte, stage, dilution_factor, str(hire_date)
            )
            
            st.info(f"Estimated Equity Grant: {final_pct*100:.4f}% of company")
        
        # Save button
        if st.button("Save Employee", type="primary"):
            # Calculate final equity
            T = calculate_role_score(seniority, command, impact)
            base_eq = calculate_base_equity(T)
            vest_eq = calculate_vesting_equity(T)
            perf_bonus = calculate_performance_bonus(vest_eq, performance)
            tvl_bonus = calculate_tvl_bonus(tvl)
            
            dilution_engine = DilutionEngine()
            dilution_factor = dilution_engine.get_current_dilution_factor()
            
            final_pct = calculate_final_equity_pct(
                T, base_eq, vest_eq, perf_bonus, tvl_bonus,
                fte, stage, dilution_factor, str(hire_date)
            )
            
            # Calculate tokens based on assumed total supply
            total_supply = 10_000_000  # Default assumption
            tokens = final_pct * total_supply
            
            # Determine hire cohort
            hire_cohort = "founder" if stage == "Pre-seed" and seniority >= 2 else "early"
            
            # Save to database
            db_execute("""
            INSERT INTO employees (
                name, role_type, seniority, command_level, role_impact,
                salary, tokens, fte, performance, tvl, company_stage, hire_cohort,
                hire_date, dilution_factor, final_equity_pct,
                cliff_months, vesting_years, notes, equity_grant_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name, role, seniority, command, impact,
                salary, tokens, fte, performance, tvl, stage, hire_cohort,
                str(hire_date), dilution_factor, final_pct,
                cliff, vest_years, notes, str(hire_date)
            ))
            
            st.success(f"Employee {name} added successfully!")
            st.rerun()
    
    elif mode == "Edit Existing Employee":
        if not df.empty:
            selected_name = st.selectbox("Select Employee", df["name"])
            emp = df[df["name"] == selected_name].iloc[0]
            
            # Similar form as above, pre-filled with emp data
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name", value=emp["name"])
                role = st.selectbox(
                    "Role Type",
                    ["Engineering", "Research", "Product", "Bizdev", "Risk Ops", "Operations", "Executive"],
                    index=["Engineering", "Research", "Product", "Bizdev", "Risk Ops", "Operations", "Executive"].index(
                        emp["role_type"] if emp["role_type"] in ["Engineering", "Research", "Product", "Bizdev", "Risk Ops", "Operations", "Executive"] else "Engineering"
                    )
                )
                
                hire_date_val = emp["hire_date"]
                if hire_date_val and isinstance(hire_date_val, str):
                    hire_date = st.date_input("Hire Date", value=datetime.strptime(hire_date_val, "%Y-%m-%d").date())
                else:
                    hire_date = st.date_input("Hire Date", value=date.today())
                
                fte = st.slider("FTE", 0.1, 1.0, float(emp.get("fte", 1.0)), 0.1)
                
                seniority = st.select_slider(
                    "Seniority Level",
                    options=[0, 1, 2, 3],
                    value=int(emp.get("seniority", 1)),
                    format_func=lambda x: ["Junior", "Mid", "Senior", "Principal"][x]
                )
                command = st.select_slider(
                    "Command Level",
                    options=[0, 1, 2, 3],
                    value=int(emp.get("command_level", 0)),
                    format_func=lambda x: ["IC", "Lead", "Manager", "Executive"][x]
                )
                impact = st.slider("Role Impact", 1, 5, int(emp.get("role_impact", 3)))
            
            with col2:
                salary = st.number_input("Base Salary (USD)", min_value=0, value=int(emp.get("salary", 80000)), step=5000)
                performance = st.selectbox(
                    "Performance Rating",
                    [1, 2, 3, 4],
                    index=int(emp.get("performance", 2))-1,
                    format_func=lambda x: ["Underperform", "Meets", "Exceeds", "Outstanding"][x-1]
                )
                tvl = st.number_input("TVL Contribution (USD)", min_value=0.0, value=float(emp.get("tvl", 0.0)), step=10000.0)
                
                stage = st.selectbox(
                    "Company Stage",
                    list(STAGE_COEFF.keys()),
                    index=list(STAGE_COEFF.keys()).index(emp.get("company_stage", "Seed"))
                    if emp.get("company_stage") in STAGE_COEFF else 0
                )
                cliff = st.number_input("Cliff (months)", 0, 24, int(emp.get("cliff_months", 12)))
                vest_years = st.number_input("Vesting Period (years)", 1, 10, int(emp.get("vesting_years", 4)))
                
                notes = st.text_area("Notes / Special Conditions", value=emp.get("notes", "") or "")
            
            col3, col4, col5 = st.columns(3)
            with col3:
                if st.button("Update Employee", type="primary"):
                    # Recalculate equity
                    T = calculate_role_score(seniority, command, impact)
                    base_eq = calculate_base_equity(T)
                    vest_eq = calculate_vesting_equity(T)
                    perf_bonus = calculate_performance_bonus(vest_eq, performance)
                    tvl_bonus = calculate_tvl_bonus(tvl)
                    
                    dilution_factor = float(emp.get("dilution_factor", 1.0))
                    
                    final_pct = calculate_final_equity_pct(
                        T, base_eq, vest_eq, perf_bonus, tvl_bonus,
                        fte, stage, dilution_factor, str(hire_date)
                    )
                    
                    total_supply = 10_000_000
                    tokens = final_pct * total_supply
                    
                    db_execute("""
                    UPDATE employees SET
                        name=?, role_type=?, seniority=?, command_level=?, role_impact=?,
                        salary=?, tokens=?, fte=?, performance=?, tvl=?, company_stage=?,
                        hire_date=?, dilution_factor=?, final_equity_pct=?,
                        cliff_months=?, vesting_years=?, notes=?, equity_grant_date=?
                    WHERE id=?
                    """, (
                        name, role, seniority, command, impact,
                        salary, tokens, fte, performance, tvl, stage,
                        str(hire_date), dilution_factor, final_pct,
                        cliff, vest_years, notes, str(hire_date),
                        int(emp["id"])
                    ))
                    st.success("Employee updated!")
                    st.rerun()
            
            with col4:
                if st.button("Mark as Departed"):
                    departure_date = date.today()
                    db_execute(
                        "UPDATE employees SET departure_date = ? WHERE id = ?",
                        (str(departure_date), int(emp["id"]))
                    )
                    st.warning(f"Employee {name} marked as departed")
                    st.rerun()
            
            with col5:
                if st.button("Delete Employee", type="secondary"):
                    if st.checkbox("Confirm deletion"):
                        db_execute("DELETE FROM employees WHERE id = ?", (int(emp["id"]),))
                        st.error(f"Employee {name} deleted")
                        st.rerun()
        else:
            st.info("No employees to edit.")
    
    # Display current employees
    st.markdown("---")
    st.subheader("Current Employees")
    
    if not df.empty:
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_active = st.checkbox("Show Active Only", value=True)
        with col2:
            role_filter = st.multiselect(
                "Filter by Role",
                df['role_type'].unique(),
                default=df['role_type'].unique()
            )
        with col3:
            sort_by = st.selectbox(
                "Sort By",
                ["name", "salary", "tokens", "hire_date", "final_equity_pct"],
                format_func=lambda x: x.replace("_", " ").title()
            )
        
        # Apply filters
        filtered_df = df.copy()
        if show_active:
            filtered_df = filtered_df[filtered_df['departure_date'].isna()]
        if role_filter:
            filtered_df = filtered_df[filtered_df['role_type'].isin(role_filter)]
        
        filtered_df = filtered_df.sort_values(sort_by, ascending=False)
        
        # Display
        display_cols = [
            'name', 'role_type', 'hire_date', 'salary', 'tokens',
            'final_equity_pct', 'fte', 'performance', 'company_stage'
        ]
        
        # Filter only existing columns
        display_cols = [col for col in display_cols if col in filtered_df.columns]
        
        st.dataframe(
            filtered_df[display_cols].style.format({
                'salary': '${:,.0f}',
                'tokens': '{:,.0f}',
                'final_equity_pct': '{:.4f}%'.format,
                'fte': '{:.1f}'.format
            }),
            use_container_width=True,
            height=400
        )
        
        # Export option
        if st.button("Export to CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="employees_export.csv",
                mime="text/csv"
            )

# ------------------------------------------------------
# EQUITY CALCULATOR TAB
# ------------------------------------------------------
with tab_equity:
    st.subheader("Advanced Equity Calculator")
    
    df = load_data()
    
    if df.empty:
        st.info("No employees available.")
    else:
        # Calculator with scenario modeling
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Employee Selection")
            emp_name = st.selectbox("Select Employee", df["name"])
            emp = df[df["name"] == emp_name].iloc[0]
            
            # Display current stats
            current_equity = float(emp.get('final_equity_pct', 0)) * 100
            st.info(f"""
            **Current Equity:** {current_equity:.4f}%
            **Tokens:** {float(emp.get('tokens', 0)):,.0f}
            **Cliff:** {int(emp.get('cliff_months', 12))} months
            **Vesting:** {int(emp.get('vesting_years', 4))} years
            """)
        
        with col2:
            st.subheader("Scenario Modeling")
            
            # Adjustable parameters
            new_performance = st.slider(
                "What-if Performance",
                1, 4, int(emp.get('performance', 2)),
                help="Simulate different performance ratings"
            )
            
            new_tvl = st.number_input(
                "What-if TVL (USD)",
                min_value=0.0,
                value=float(emp.get('tvl', 0.0)),
                step=100000.0
            )
            
            new_stage = st.selectbox(
                "What-if Company Stage",
                list(STAGE_COEFF.keys()),
                index=list(STAGE_COEFF.keys()).index(emp.get('company_stage', 'Seed'))
                if emp.get('company_stage') in STAGE_COEFF else 0
            )
            
            # Calculate new equity based on what-if
            S = int(emp.get('seniority', 1))
            C = int(emp.get('command_level', 0))
            R = int(emp.get('role_impact', 3))
            fte = float(emp.get('fte', 1.0))
            dilution_factor = float(emp.get('dilution_factor', 1.0))
            
            T = calculate_role_score(S, C, R)
            base_eq = calculate_base_equity(T)
            vest_eq = calculate_vesting_equity(T)
            perf_b = calculate_performance_bonus(vest_eq, new_performance)
            tvl_b = calculate_tvl_bonus(new_tvl)
            
            new_equity_pct = calculate_final_equity_pct(
                T, base_eq, vest_eq, perf_b, tvl_b,
                fte, new_stage, dilution_factor, emp.get('hire_date', str(date.today()))
            )
            
            current_equity = float(emp.get('final_equity_pct', 0))
            equity_change = ((new_equity_pct - current_equity) / current_equity) * 100 if current_equity > 0 else 100
            
            st.metric(
                "Projected Equity",
                f"{new_equity_pct*100:.4f}%",
                f"{equity_change:+.1f}%"
            )
        
        # Detailed breakdown
        st.markdown("---")
        st.subheader("Detailed Equity Breakdown")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Role Score (T)", T)
            st.metric("Base Equity", f"{base_eq*100:.4f}%")
        
        with col2:
            st.metric("Vesting Equity", f"{vest_eq*100:.4f}%")
            st.metric("Performance Bonus", f"{perf_b*100:.4f}%")
        
        with col3:
            st.metric("TVL Bonus", f"{tvl_b*100:.4f}%")
            st.metric("Stage Coefficient", STAGE_COEFF[new_stage])
        
        # Visualization
        breakdown_data = pd.DataFrame({
            'Component': ['Base Grant', 'Vesting', 'Performance Bonus', 'TVL Bonus'],
            'Percentage': [base_eq*100, vest_eq*100, perf_b*100, tvl_b*100],
            'Type': ['Base', 'Vesting', 'Bonus', 'Bonus']
        })
        
        fig = px.sunburst(
            breakdown_data,
            path=['Type', 'Component'],
            values='Percentage',
            title="Equity Component Breakdown"
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------
# VESTING SCHEDULES TAB
# ------------------------------------------------------
with tab_vesting:
    st.subheader("Vesting Schedule Visualization")
    
    df = load_data()
    
    if df.empty:
        st.info("No employees available.")
    else:
        # Multi-select for employees
        selected_employees = st.multiselect(
            "Select Employees to Display",
            df["name"].tolist(),
            default=df["name"].head(3).tolist() if len(df) >= 3 else df["name"].tolist()
        )
        
        # Parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            total_supply = st.number_input(
                "Total Token Supply",
                min_value=1_000_000.0,
                value=10_000_000.0,
                step=1_000_000.0
            )
        with col2:
            end_year = st.number_input(
                "Project Years Ahead",
                min_value=1,
                max_value=10,
                value=5
            )
        with col3:
            view_type = st.selectbox(
                "View Type",
                ["Cumulative", "Monthly Vest", "Remaining"]
            )
        
        # Generate schedules
        all_schedules = []
        
        for emp_name in selected_employees:
            emp = df[df["name"] == emp_name].iloc[0]
            schedule = generate_equity_waterfall(
                int(emp["id"]),
                total_supply,
                date.today() + timedelta(days=365*end_year)
            )
            
            if not schedule.empty:
                schedule["employee"] = emp_name
                all_schedules.append(schedule)
        
        if all_schedules:
            combined_df = pd.concat(all_schedules)
            
            # Create visualization based on view type
            if view_type == "Cumulative":
                fig = px.line(
                    combined_df,
                    x="date",
                    y="pct_cumulative",
                    color="employee",
                    title="Cumulative Equity Vesting",
                    labels={"pct_cumulative": "Cumulative Equity %", "date": "Date"}
                )
            elif view_type == "Monthly Vest":
                monthly_df = combined_df[combined_df["type"] == "vest"]
                if not monthly_df.empty:
                    fig = px.bar(
                        monthly_df,
                        x="date",
                        y="pct_vested",
                        color="employee",
                        title="Monthly Vesting Amounts",
                        labels={"pct_vested": "Monthly Vest %", "date": "Date"},
                        barmode="group"
                    )
                else:
                    fig = go.Figure()
                    fig.add_annotation(text="No monthly vesting data", x=0.5, y=0.5, showarrow=False)
            else:  # Remaining
                combined_df["remaining_pct"] = combined_df.groupby("employee")["pct_cumulative"].transform(
                    lambda x: x.iloc[-1] - x if len(x) > 0 else 0
                )
                fig = px.area(
                    combined_df,
                    x="date",
                    y="remaining_pct",
                    color="employee",
                    title="Remaining Equity to Vest",
                    labels={"remaining_pct": "Remaining Equity %", "date": "Date"}
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Download option
            csv = combined_df.to_csv(index=False)
            st.download_button(
                label="Download Vesting Schedule",
                data=csv,
                file_name="vesting_schedule.csv",
                mime="text/csv"
            )
        else:
            st.warning("No vesting schedules generated for selected employees.")

# ------------------------------------------------------
# DILUTION ANALYSIS TAB
# ------------------------------------------------------
with tab_dilution:
    st.subheader("Dilution Impact Analysis")
    
    dilution_engine = DilutionEngine()
    current_dilution = dilution_engine.get_current_dilution_factor()
    dilution_history = dilution_engine.get_dilution_timeline()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Current Dilution Factor",
            f"{current_dilution:.1%}",
            help="1.0 = no dilution, lower = more dilution"
        )
    
    with col2:
        if not dilution_history.empty:
            total_dilution = (1 - current_dilution) * 100
            st.metric("Total Dilution", f"{total_dilution:.1f}%")
    
    # Add new dilution event
    st.markdown("---")
    st.subheader("Add Dilution Event")
    
    with st.form("dilution_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            event_date = st.date_input("Event Date", value=date.today())
        with col2:
            event_type = st.selectbox(
                "Event Type",
                ["Fundraising", "Employee Pool", "Strategic Round", "Token Generation", "Other"]
            )
        with col3:
            dilution_pct = st.number_input(
                "Dilution Percentage",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                help="Percentage of new shares created"
            )
        
        description = st.text_input("Description")
        
        if st.form_submit_button("Add Dilution Event"):
            new_factor = dilution_engine.add_dilution_event(
                str(event_date),
                event_type,
                dilution_pct,
                description
            )
            st.success(f"Dilution event added. New factor: {new_factor:.1%}")
            st.rerun()
    
    # Dilution history chart
    st.markdown("---")
    st.subheader("Dilution History")
    
    if not dilution_history.empty:
        fig = create_dilution_waterfall_chart()
        st.plotly_chart(fig, use_container_width=True)
        
        # Show dilution events table
        st.dataframe(
            dilution_history.style.format({
                'pre_dilution_factor': '{:.1%}'.format,
                'post_dilution_factor': '{:.1%}'.format
            }),
            use_container_width=True
        )
    else:
        st.info("No dilution events recorded yet.")

# ------------------------------------------------------
# COMPENSATION ANALYSIS TAB
# ------------------------------------------------------
with tab_scatter:
    st.subheader("Advanced Compensation Analysis")
    
    df = load_data()
    
    if df.empty:
        st.info("No employees to display.")
    else:
        # Analysis controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox(
                "X-Axis",
                ["salary", "tokens", "final_equity_pct", "seniority", "tvl"],
                format_func=lambda x: x.replace("_", " ").title()
            )
        
        with col2:
            y_axis = st.selectbox(
                "Y-Axis",
                ["tokens", "salary", "final_equity_pct", "role_impact", "tvl"],
                format_func=lambda x: x.replace("_", " ").title()
            )
        
        with col3:
            size_var = st.selectbox(
                "Bubble Size",
                ["fte", "seniority", "command_level", "performance", "tvl"],
                format_func=lambda x: x.replace("_", " ").title()
            )
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color="role_type",
            size=size_var,
            hover_name="name",
            hover_data=["hire_date", "company_stage", "performance"],
            title=f"{y_axis.replace('_', ' ').title()} vs {x_axis.replace('_', ' ').title()}",
            trendline="ols" if len(df) > 1 else None,
            trendline_scope="overall"
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------
# DOCUMENTATION TAB
# ------------------------------------------------------
with tab_docs:
    st.subheader("Advanced Documentation & Methodology")
    
    # Create expandable sections
    with st.expander("📊 Equity Model Methodology", expanded=True):
        st.markdown("""
        ### Enhanced Equity Calculation Model
        
        Our equity model now includes **dilution-aware calculations**:
        
        ```
        Final Equity = (Base + Vesting + Performance_Bonus + TVL_Bonus) 
                      × FTE × Stage_Coefficient × Dilution_Factor
        ```
        
        **Dilution Factor:**
        - Starts at 1.0 (no dilution)
        - Decreases with each fundraising round
        - Applied to new hires to account for reduced ownership
        - Historical dilution is tracked for each employee
        """)
    
    with st.expander("📅 Vesting Schedule Engine"):
        st.markdown("""
        ### Advanced Vesting Schedule Features
        
        **Flexible Vesting Parameters:**
        - Custom cliff periods (0-24 months)
        - Variable vesting durations (1-10 years)
        - Monthly vesting calculations
        - Accelerated vesting scenarios
        """)
    
    with st.expander("📉 Dilution Management"):
        st.markdown("""
        ### Dilution Tracking System
        
        **Dilution Events:**
        1. **Fundraising Rounds** (Seed, Series A, B, etc.)
        2. **Employee Pool Increases**
        3. **Strategic Investment**
        4. **Token Generation Events**
        """)
    
    with st.expander("👥 Role Framework & Scoring"):
        st.markdown("""
        ### Comprehensive Role Scoring
        
        **Seniority Levels (S):**
        - 0: Junior (0-2 years)
        - 1: Mid-level (2-5 years)
        - 2: Senior (5-8 years)
        - 3: Principal (8+ years)
        
        **Command Levels (C):**
        - 0: Individual Contributor
        - 1: Lead (informal leadership)
        - 2: Manager (team leadership)
        - 3: Executive (department/company leadership)
        
        **Role Impact (R):**
        - 1: Support role
        - 2: Contributing role
        - 3: Important role
        - 4: Critical role
        - 5: Mission-critical role
        """)

# Add custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4e8cff;
    }
    .stButton button {
        width: 100%;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8cff;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

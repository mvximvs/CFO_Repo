# DeFine – Advanced Equity & Token Allocation Platform

This repository contains a Streamlit application for **employee equity planning**, **token allocation**, and **dilution-aware compensation analysis**.

The app is built around a SQLite database (`employees_enhanced.db`) and provides HR / founders / CFOs with a cockpit to:

- Design and compare equity grants
- Track dilution events over time
- Visualize vesting schedules per employee
- Analyse salary, equity and token compensation across the team

---

## Features

### 🔧 Data & Models

- **SQLite database** (`employees_enhanced.db`) auto-created on first run
- Employee table with:
  - Role type, seniority, command level, role impact
  - Salary, token allocation, FTE, performance, TVL contribution
  - Company stage, hire cohort, hire / departure dates
  - Cliff / vesting parameters, final equity %
- **Dilution engine**:
  - Tracks dilution events over time (fundraising, employee pool, etc.)
  - Computes and stores a global `dilution_factor`
  - Applies dilution factor into new hire equity grants
- **Equity engine**:
  - Role score `T = S + C + R`
  - Base grant, vesting grant, performance bonus, TVL bonus
  - Stage coefficient (Pre-seed, Seed, Series A, Series B, Growth)
  - Final equity:
    \[
    \text{Final Equity} = (Base + Vesting + Performance + TVL\_Bonus) \times FTE \times Stage\_Coeff \times Dilution\_Factor
    \]

### 🖥️ UI Overview (Tabs)

1. **📊 Enhanced Dashboard**
   - Global metrics: total equity, average salary, total tokens, current dilution factor
   - Equity distribution by role (pie chart)
   - Salary distribution by role (boxplot)
   - Employee summary table (salary, tokens, equity %, cliff, vesting, performance)

2. **👥 Employee Management**
   - Add new employees (with role scoring, compensation, equity params)
   - Edit existing employees (update fields & recompute equity)
   - Mark employees as departed
   - Delete employees
   - Filter & sort employees, export data to CSV

3. **🧮 Equity Calculator**
   - Select an employee and view current equity, tokens, cliff & vesting
   - Scenario modeling:
     - “What-if” performance
     - “What-if” TVL contribution
     - “What-if” company stage
   - Displays projected equity % and detailed component breakdown (sunburst chart)

4. **📅 Vesting Schedules**
   - Multi-employee vesting schedule visualization
   - Choose view:
     - Cumulative equity
     - Monthly vest amounts
     - Remaining equity to vest
   - Export full vesting schedule as CSV

5. **📉 Dilution Analysis**
   - Current dilution factor & total dilution since inception
   - Add dilution events (date, type, % and description)
   - Waterfall chart of dilution over time
   - Dilution events table

6. **📈 Compensation Analysis**
   - Bubble scatter plot of compensation variables:
     - X/Y axes among salary, tokens, equity %, seniority, TVL, role impact…
     - Bubble size by FTE / seniority / command level / performance / TVL
   - OLS regression trendline (requires `statsmodels`)

7. **📘 Documentation**
   - In-app methodology for:
     - Equity model
     - Vesting schedule logic
     - Dilution management
     - Role framework & scoring

---

## Installation

1. **Clone the repository** (or copy the file):

```bash
git clone <your-repo-url>
cd <your-repo-folder>

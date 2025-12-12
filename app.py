import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import json

# Set page config
st.set_page_config(page_title="Academic Performance Prediction", layout="wide")

# --- Constants & Config ---
ADMIN_CONFIG_FILE = 'admin_config.json'
DB_FILE = 'student_database.csv'

# --- Helper Functions ---

# Admin Config Management
def load_admin_config():
    if not os.path.exists(ADMIN_CONFIG_FILE):
        return {"username": "admin", "password": "Admin@2025"}
    with open(ADMIN_CONFIG_FILE, 'r') as f:
        return json.load(f)

def save_admin_config(username, password):
    config = load_admin_config()
    config['username'] = username
    config['password'] = password
    with open(ADMIN_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

def save_student_cred(student_name, password):
    config = load_admin_config()
    if 'student_users' not in config:
        config['student_users'] = {}
    config['student_users'][student_name] = password
    with open(ADMIN_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

# Model Loading
@st.cache_resource
def load_model_data():
    try:
        model = joblib.load('student_performance_model_v2.pkl')
        feature_names = joblib.load('feature_names.pkl')
        categorical_cols = joblib.load('categorical_cols.pkl')
        numerical_cols = joblib.load('numerical_cols.pkl')
        unique_values = joblib.load('categorical_unique_values.pkl')
        return model, feature_names, categorical_cols, numerical_cols, unique_values
    except FileNotFoundError:
        st.error("Model files not found. Please run 'model_training.py' first.")
        return None, None, None, None, None

model, feature_names, categorical_cols, numerical_cols, unique_values = load_model_data()

# Database Management
if not os.path.exists(DB_FILE):
    if feature_names:
        # Initial columns + Student Name, Predicted Score, Risk Level, Password
        initial_cols = ['Student_Name'] + feature_names + ['Predicted_Score', 'Risk_Level']
        df_init = pd.DataFrame(columns=initial_cols)
        df_init.to_csv(DB_FILE, index=False)

def load_data():
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE)
    return pd.DataFrame()

def save_data(data):
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        df = pd.concat([df, data], ignore_index=True)
        df.to_csv(DB_FILE, index=False)
    else:
        data.to_csv(DB_FILE, index=False)

# --- Session State Init ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_type' not in st.session_state:
    st.session_state['user_type'] = None
if 'user_name' not in st.session_state:
    st.session_state['user_name'] = None
if 'login_attempts' not in st.session_state:
    st.session_state['login_attempts'] = 0
if 'admin_view' not in st.session_state:
    st.session_state['admin_view'] = 'menu' # menu, predict, manage

# --- Styling (Vibrant Midnight Theme) ---
st.markdown("""
    <style>
    /* Import Google Font - Outfit */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    :root {
        --primary: #06B6D4;
        --primary-hover: #0891B2;
        --bg-glass: rgba(30, 41, 59, 0.7);
        --border-glass: rgba(148, 163, 184, 0.1);
        --shadow-glow: 0 0 20px rgba(6, 182, 212, 0.15);
    }

    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    /* --- Main Background with Gradient --- */
    .stApp {
        background: radial-gradient(circle at top right, #1e293b 0%, #0f172a 40%, #020617 100%);
    }

    /* --- Sidebar Styling --- */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid var(--border-glass);
    }
    
    /* --- Modern Glass Cards --- */
    .card, .stMetric, .action-card {
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border-glass);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        color: #F8FAFC;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    /* Hover Effects with Glow */
    .action-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: var(--shadow-glow);
        border-color: var(--primary);
    }

    /* Distinctive Gradient Accents */
    .card-predict::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 6px; height: 100%;
        background: linear-gradient(to bottom, #06B6D4, #3B82F6);
    }
    .card-manage::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 6px; height: 100%;
        background: linear-gradient(to bottom, #8B5CF6, #EC4899);
    }
    
    /* --- Inputs & Widgets --- */
    .stTextInput>div>div>input, .stSelectbox>div>div>button {
        background-color: rgba(30, 41, 59, 0.5);
        color: #F8FAFC;
        border-radius: 10px;
        border: 1px solid #475569;
        transition: all 0.2s;
    }
    .stTextInput>div>div>input:focus, .stSelectbox>div>div>button:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.2);
    }

    /* --- Buttons with Gradient Shine --- */
    .stButton>button {
        background: linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%);
        color: white;
        border: none;
        padding: 12px 32px;
        border-radius: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 8px 16px rgba(6, 182, 212, 0.4);
        transform: translateY(-2px);
        filter: brightness(1.1);
    }
    
    /* --- Metrics Text --- */
    div[data-testid="stMetricValue"] {
        background: -webkit-linear-gradient(120deg, #22D3EE, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* --- Headers --- */
    h1, h2, h3 {
        color: #F8FAFC;
        font-weight: 700;
        letter-spacing: -0.03em;
    }
    h1 {
        background: -webkit-linear-gradient(120deg, #F8FAFC, #94A3B8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)


# --- Authentication Logic ---

def login():
    st.title("🎓 Academic Performance System")
    
    tabs = st.tabs(["Admin Login", "Student Login"])
    
    # --- ADMIN LOGIN ---
    with tabs[0]:
        st.subheader("Admin Access")
        
        # 3-Strike Reset Option
        if st.session_state['login_attempts'] >= 3:
            st.warning("Too many failed attempts.")
            with st.expander("Reset Admin Password", expanded=True):
                with st.form("reset_password_form"):
                    st.write("Enter new credentials to reset access.")
                    new_user = st.text_input("Confirm Username")
                    new_pass = st.text_input("New Password", type="password")
                    reset_submit = st.form_submit_button("Reset & Update Password")
                    
                    if reset_submit:
                        config = load_admin_config()
                        if new_user == config['username']: # Simple check to ensure they know at least the username
                            save_admin_config(new_user, new_pass)
                            st.session_state['login_attempts'] = 0
                            st.success("Password updated! Please login with new password.")
                            st.rerun()
                        else:
                            st.error("Username does not match.")
        
        # Normal Login Form
        with st.form("admin_login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                config = load_admin_config()
                # Check with strip() to remove accidental spaces
                if username.strip() == config['username'] and password.strip() == config['password']:
                    st.session_state['logged_in'] = True
                    st.session_state['user_type'] = "Admin"
                    st.session_state['user_name'] = username
                    st.session_state['login_attempts'] = 0
                    st.rerun()
                else:
                    st.session_state['login_attempts'] += 1
                    st.error(f"Invalid Credentials. Attempts: {st.session_state['login_attempts']}/3")
                    if st.session_state['login_attempts'] >= 3:
                        st.rerun()

    # --- STUDENT LOGIN ---
    with tabs[1]:
        st.subheader("Student Portal")
        # Removed default password hint for security
        with st.form("student_login_form"):
            student_name = st.text_input("Full Name")
            student_pass = st.text_input("Password", type="password")
            submitted_student = st.form_submit_button("Login")
            
            if submitted_student:
                df = load_data()
                if not df.empty:
                    # Check Name and Password
                    config = load_admin_config()
                    student_creds = config.get('student_users', {})
                    stored_pass = student_creds.get(student_name)
                    
                    user_record = df[df['Student_Name'] == student_name]
                    
                    if stored_pass and stored_pass == student_pass and not user_record.empty:
                        st.session_state['logged_in'] = True
                        st.session_state['user_type'] = "Student"
                        st.session_state['user_name'] = student_name
                        st.rerun()
                    else:
                        st.error("Invalid Name or Password.")
                else:
                    st.error("No student records found.")

def logout():
    st.session_state['logged_in'] = False
    st.session_state['user_type'] = None
    st.session_state['user_name'] = None
    st.session_state['admin_view'] = 'menu'
    st.rerun()

# --- Admin Dashboard Views ---

def admin_dashboard():
    # Sidebar
    with st.sidebar:
        st.title(f"Admin Panel")
        st.write(f"User: **{st.session_state['user_name']}**")
        if st.button("🏠 Home / Menu"):
            st.session_state['admin_view'] = 'menu'
            st.rerun()
        st.divider()
        if st.button("Logout", type="primary"):
            logout()

    # Views
    if st.session_state['admin_view'] == 'menu':
        st.title("Admin Dashboard")
        
        # --- KPI Visuals (Replacing white space) ---
        df = load_data()
        if not df.empty:
            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                st.metric("Total Students", len(df), delta=None, border=True)
            with kpi2:
                avg_val = df['Predicted_Score'].mean()
                st.metric("Avg Class Score", f"{avg_val:.2f}", border=True)
            with kpi3:
                high_risk_count = len(df[df['Predicted_Score'] < 60])
                st.metric("High Risk Students", high_risk_count, delta_color="inverse", border=True)
            
            # --- New Admin Visual: Score Distribution ---
            st.markdown("### 📊 Class Performance Overview")
            fig_dist = px.histogram(
                df, 
                x='Predicted_Score', 
                nbins=20, 
                title="Exam Score Distribution",
                color_discrete_sequence=['#0EA5E9'] # Cyan
            )
            fig_dist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#F8FAFC'),
                bargap=0.1
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("No student data available yet.")

        st.divider()
        st.write("### Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='action-card card-predict'>
                <div class='card-icon'>📊</div>
                <div class='card-title'>Predict & Add</div>
                <div class='card-desc'>Enter new student data</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open Prediction Tool", key="btn_predict", use_container_width=True):
                st.session_state['admin_view'] = 'predict'
                st.rerun()
                
        with col2:
            st.markdown("""
            <div class='action-card card-manage'>
                <div class='card-icon'>📂</div>
                <div class='card-title'>Manage Records</div>
                <div class='card-desc'>View and edit databse</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open Database", key="btn_manage", use_container_width=True):
                st.session_state['admin_view'] = 'manage'
                st.rerun()
    
    elif st.session_state['admin_view'] == 'predict':
        render_prediction_form()
        
    elif st.session_state['admin_view'] == 'manage':
        render_management_view()

def render_prediction_form():
    st.header("Predict & Add Student")
    if st.button("← Back"):
        st.session_state['admin_view'] = 'menu'
        st.rerun()
        
    with st.form("data_entry_form"):
        st.subheader("New Student Details")
        
        c1, c2 = st.columns(2)
        with c1:
            student_name = st.text_input("Student Name")
        with c2:
            student_pass = st.text_input("Assign Password", value="123456")
        
        inputs = {}
        cols = st.columns(3)
        if feature_names:
            for i, col in enumerate(feature_names):
                with cols[i % 3]:
                    if col in categorical_cols:
                        options = unique_values.get(col, [])
                        inputs[col] = st.selectbox(f"{col}", options)
                    else:
                        # Sliders logic similar to before
                        if col == 'Attendance':
                            inputs[col] = st.slider(f"{col} (%)", 0, 100, 75)
                        elif col == 'Hours_Studied/Week':
                            inputs[col] = st.slider(f"{col}", 0, 40, 5)
                        elif col == 'Previous_Scores':
                            inputs[col] = st.slider(f"{col}", 0, 100, 70)
                        elif col == 'Sleep_Hours':
                            inputs[col] = st.slider(f"{col}", 0, 12, 7)
                        elif col == 'Tutoring_Sessions':
                            inputs[col] = st.number_input(f"{col}", 0, 10, 0)
                        elif col == 'Physical_Activity':
                            inputs[col] = st.number_input(f"{col} (Hrs/Week)", 0, 20, 2)
                        else:
                            inputs[col] = st.number_input(f"{col}", min_value=0.0)
        
        submitted = st.form_submit_button("Predict & Save Result")
        
        if submitted:
            if not student_name:
                st.error("Student Name is required.")
            else:
                # Prediction
                input_df = pd.DataFrame([inputs])
                prediction = model.predict(input_df)[0]
                
                risk_level = "Low Risk"
                if prediction < 60: risk_level = "High Risk"
                elif prediction < 75: risk_level = "Medium Risk"
                
                # Display
                st.info(f"Predicted Score: {prediction:.2f} | Risk: {risk_level}")
                
                # Save
                inputs['Student_Name'] = student_name
                inputs['Student_Name'] = student_name
                # Password saved to config, not CSV
                save_student_cred(student_name, student_pass)
                
                inputs['Predicted_Score'] = prediction
                inputs['Risk_Level'] = risk_level
                
                save_data(pd.DataFrame([inputs]))
                st.success(f"Student '{student_name}' saved successfully!")

def render_management_view():
    st.header("Student Records Management")
    if st.button("← Back"):
        st.session_state['admin_view'] = 'menu'
        st.rerun()
    
    # Split into Tabs
    man_tabs = st.tabs(["View Students", "Delete Students"])
    
    df_records = load_data()
    
    with man_tabs[0]:
        st.subheader("All Student Records")
        st.dataframe(df_records, use_container_width=True)
    
    with man_tabs[1]:
        st.subheader("Remove Records")
        if not df_records.empty:
            to_delete = st.selectbox("Select Student to Delete", df_records['Student_Name'].unique())
            if st.button("Delete Selected Student", type="primary"):
                df_new = df_records[df_records['Student_Name'] != to_delete]
                df_new.to_csv(DB_FILE, index=False)
                st.success("Deleted.")
                st.rerun()
        else:
            st.info("No records to delete.")

# --- Student Dashboard ---

def student_dashboard():
    with st.sidebar:
        st.title(f"Hi, {st.session_state['user_name']}")
        if st.button("Logout"): logout()
        
    st.title("My Performance Dashboard")
    
    df = load_data()
    # Logic to handle if checking against password here or just assuming session is valid
    # Just filter by name as we authenticated already
    user_rows = df[df['Student_Name'] == st.session_state['user_name']]
    
    if user_rows.empty:
        st.error("Record not found.")
        return
        
    student_data = user_rows.iloc[-1]
    
    # 1. Score Card
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Score", f"{student_data['Predicted_Score']:.2f}")
    with col2:
        risk = student_data.get('Risk_Level', 'N/A')
        color = "green" if "Low" in risk else "orange" if "Medium" in risk else "red"
        st.markdown(f"**Risk Status**")
        st.markdown(f"### :{color}[{risk}]")
    with col3:
        pass # Placeholder
        
    st.divider()
    
    # 2. Charts (Gauge & Radar)
    st.subheader("Performance Analysis")
    c1, c2 = st.columns(2)
    
    with c1:
        # --- NEW VISUAL: Gauge Chart ---
        avg_score = df['Predicted_Score'].mean()
        score = student_data['Predicted_Score']
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Predicted Score", 'font': {'size': 24, 'color': '#F8FAFC'}},
            delta = {'reference': avg_score, 'increasing': {'color': "#0EA5E9"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#F8FAFC"},
                'bar': {'color': "#06B6D4"},
                'bgcolor': "rgba(30, 41, 59, 0.7)",
                'borderwidth': 2,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 60], 'color': 'rgba(239, 68, 68, 0.3)'},
                    {'range': [60, 75], 'color': 'rgba(234, 179, 8, 0.3)'},
                    {'range': [75, 100], 'color': 'rgba(34, 197, 94, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': score
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "#F8FAFC", 'family': "Outfit"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        # --- Metrics Bar Chart (Original Restored) ---
        labels = ['Attendance', 'Previous_Scores', 'Hours_Studied/Week']
        # Let's verify columns exist
        valid_labels = [l for l in labels if l in student_data]
        if valid_labels:
            vals = [student_data[l] for l in valid_labels]
            fig2 = go.Figure(go.Bar(
                x=valid_labels, 
                y=vals, 
                marker_color='#06B6D4' # Cyan
            ))
            fig2.update_layout(
                title={'text': "My Key Metrics", 'font': {'color': '#F8FAFC'}},
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#F8FAFC", 'family': "Outfit"},
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.caption("Data based on your latest input.")

    # 3. What-If Simulator
    st.divider()
    with st.expander("🚀 Simulate Score Improvement", expanded=False):
        st.write("Modify your stats below to see how they impact your score!")
        
        sim_inputs = {}
        # We need a form or just interactive widgets? 
        # Interactive widgets are better for immediate feedback.
        
        sim_cols = st.columns(3)
        
        # We need to recreate inputs, pre-filled with student_data
        for i, col in enumerate(feature_names):
            with sim_cols[i % 3]:
                # Get current value for default
                current_val = student_data.get(col)
                
                if col in categorical_cols:
                    options = unique_values.get(col, [])
                    # Find index of current_val
                    try: 
                        idx = options.index(current_val)
                    except: 
                        idx = 0
                    sim_inputs[col] = st.selectbox(f"Simulated {col}", options, index=idx)
                    
                else:
                    # Numerical
                    # Heuristics for min/max/step based on training or typical usage
                    # We can reuse the logic from admin panel or just set reasonable defaults
                    min_v = 0.0
                    max_v = 100.0
                    step = 1.0
                    
                    if col == 'Attendance': max_v=100.0; step=1.0
                    elif col == 'Hours_Studied/Week': max_v=40.0; step=1.0
                    elif col == 'Previous_Scores': max_v=100.0; step=1.0
                    elif col == 'Sleep_Hours': max_v=12.0; step=1.0
                    elif col == 'Tutoring_Sessions': max_v=10.0; step=1.0
                    elif col == 'Physical_Activity': max_v=20.0; step=1.0
                    
                    # Ensure current_val is float/int
                    val = float(current_val) if pd.notnull(current_val) else 0.0
                    
                    sim_inputs[col] = st.slider(f"Sim {col}", min_value=min_v, max_value=max_v, value=val, step=step)

        # Predict Button for Simulation
        if st.button("Calculate Simulated Score"):
            sim_df = pd.DataFrame([sim_inputs])
            # Reorder columns to match feature_names
            sim_df = sim_df[feature_names]
            
            sim_pred = model.predict(sim_df)[0]
            
            # Recalculate original score using CURRENT model to avoid "Stale Data" issues
            # This ensures that if inputs are unchanged, delta is exactly 0
            original_inputs = student_data[feature_names].to_frame().T
            original_score_live = model.predict(original_inputs)[0]
            
            delta = sim_pred - original_score_live
            
            st.metric(
                label="New Predicted Score",
                value=f"{sim_pred:.2f}",
                delta=f"{delta:.2f} points"
            )
            
            # --- AI Suggestions (Benchmarking) ---
            st.divider()
            st.markdown("### 💡 Path to Excellence: AI Coaching")
            st.caption("We compared your simulated stats with top-performing students (85+ Score).")
            
            suggestions = []
            
            # 1. Benchmarking Logic
            # Filter for "Excellent" students
            top_performers = df[df['Predicted_Score'] >= 85]
            
            # If no top performers yet, use high static benchmarks
            if top_performers.empty:
                benchmark_hours = 30
                benchmark_attendance = 95
                benchmark_sleep = 8
            else:
                benchmark_hours = top_performers['Hours_Studied/Week'].mean()
                benchmark_attendance = top_performers['Attendance'].mean()
                benchmark_sleep = top_performers['Sleep_Hours'].mean()
            
            # 2. Analyze User vs Benchmark
            
            # Study Hours
            sim_hrs = sim_inputs.get('Hours_Studied/Week', 0)
            if sim_hrs < benchmark_hours:
                diff = benchmark_hours - sim_hrs
                suggestions.append(f"📚 **Study Habits**: Top students study about **{benchmark_hours:.1f} hours/week**. Try adding **{diff:.1f} hours** to your routine.")
            else:
                suggestions.append(f"✅ **Study Habits**: Your study time ({sim_hrs}h) is on par with top performers!")

            # Attendance
            sim_att = sim_inputs.get('Attendance', 0)
            if sim_att < benchmark_attendance:
                diff = benchmark_attendance - sim_att
                suggestions.append(f"🏫 **Attendance**: To be excellent, aim for **{benchmark_attendance:.0f}%** attendance (You are at {sim_att}%).")
            
            # Sleep
            sim_sleep = sim_inputs.get('Sleep_Hours', 0)
            if sim_sleep < 6:
                suggestions.append(f"😴 **Sleep**: Your sleep is too low. Top students average **{benchmark_sleep:.1f} hours**.")
            elif sim_sleep > 10:
                suggestions.append(f"😴 **Sleep**: Oversleeping might affect productivity. High achievers average **{benchmark_sleep:.1f} hours**.")
                
            # Previous Score (Mental Check)
            # If nothing changed
            if delta == 0:
                st.info("💡 **Tip**: Try adjusting **Study Hours** or **Attendance** to see the biggest impact.")
            
            # Display
            for s in suggestions:
                st.markdown(s)
            
            if delta > 0:
                st.balloons()
            elif delta < 0:
                st.warning("⚠️ These changes might lower your score compared to your current prediction.")

# --- Main Entry ---
if __name__ == "__main__":
    if not st.session_state['logged_in']:
        login()
    else:
        if st.session_state['user_type'] == "Admin":
            admin_dashboard()
        elif st.session_state['user_type'] == "Student":
            student_dashboard()

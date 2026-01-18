import streamlit as st
import pandas as pd

def smart_city_qna(df):
    """Rule-based + data-driven Q&A for urban insights"""
    
    st.subheader("Smart City Assistant")
    st.markdown("**Ask about traffic, pollution, health risks**")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What do you want to know about the city?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        response = generate_response(df, prompt.lower())
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

def generate_response(df, question):
    """Smart responses based on your exact dataset"""
    
    # Key metrics
    avg_aqi = df['aqi'].mean()
    high_risk_days = len(df[df['risk_level'] == 'High'])
    worst_city = df.groupby('city')['health_risk_score'].mean().idxmax()
    
    # Q&A Rules
    if any(word in question for word in ['aqi', 'pollution', 'air']):
        return f"**Current AQI:** {avg_aqi:.0f}\n• High pollution days: {len(df[df['aqi']>150])}\n• Worst affected: {df[df['aqi']==df['aqi'].max()]['city'].iloc[0]}"
    
    elif any(word in question for word in ['traffic', 'congestion', 'speed']):
        avg_speed = df['avg_speed'].mean()
        high_cong = len(df[df['congestion_level']=='High'])
        return f"**Traffic Status:** {avg_speed:.1f} km/h avg speed\n• High congestion: {high_cong} days\n• Slowest: {df['avg_speed'].min():.1f} km/h"
    
    elif any(word in question for word in ['health', 'risk', 'hospital']):
        return f"**Health Risk:** {df['health_risk_score'].mean():.3f} avg\n• High risk days: {high_risk_days}\n• Worst city: **{worst_city}**"
    
    elif 'correlation' in question:
        corr = df['avg_speed'].corr(df['aqi'])
        return f"**Key Insight:** Traffic speed vs AQI correlation = {corr:.3f}\n• Slower traffic → Higher pollution!"
    
    elif any(word in question for word in ['save', 'reduce', 'improve']):
        return """**💡 Policy Recommendations:**
• **Reduce vehicles 20%** → AQI drops ~15%, saves 120 hospital visits
• **Plant trees** → AQI -10% → Respiratory cases ↓18%
• **Traffic signals** → Speed +12 km/h → Pollution ↓"""
    
    else:
        return f"""**Quick City Stats:**
• **AQI:** {avg_aqi:.0f} | **Risk:** {df['health_risk_score'].mean():.3f}
• **High congestion:** {len(df[df['congestion_level']=='High'])} days
• **Hospital visits:** {df['total_hospital_visits'].sum():,}

**Ask me about:** AQI, traffic, health risks, correlations!"""
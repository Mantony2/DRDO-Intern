import streamlit as st
import time
st.title(" Rock 🪨, Paper 📝 and Scissors ✂️ Predictor!")
st.write("🎉 Welcome to the Rock, Paper and Scissors Predictor app! This app will eventually predict your next move and counter it.")

st.subheader("Enter Your Details")
age = st.slider("Enter your age:", 1, 100)
gender = st.selectbox("Select your gender:", ["👨 Male", "👩 Female"])
previous_move = st.selectbox("What was your previous move 🤔?", ["Rock 🪨", "Paper 📝", "Scissors ✂️"])
# if st.button("Predict My Next Move"):
#     st.write("Predicted Move: Coming soon")
#     st.write("My Move to defeat you: Coming soon")
# "Let's Play" Button
if st.button(" Let's Play 🎮"):
  moves = ["🪨 Rock", "📝 Paper", "✂️ Scissors"]

  placeholder = st.empty()

  for move in moves:
     placeholder.write(f"**{move}**")  
     time.sleep(1)

  placeholder.write("**Show! 🤯**")

st.markdown(
    """
    ---
    ### 🎮 How to Play:
    - Select your age and gender.
    - Choose your last move (Rock, Paper, or Scissors).
    - Click the "Let's Play" button to see the magic (once model is prepared)!
    """
)

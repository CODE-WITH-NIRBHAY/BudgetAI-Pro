import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import speech_recognition as sr
import pyttsx3
import re
import random
import os
from datetime import datetime
from warnings import warn

# ========================
# 🔈 Voice Engine Setup
# ========================
class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Slower speech
        
    def listen(self):
        with self.mic as source:
            print("\n🎤 Speak now (e.g., '500 rupees for pizza')...")
            audio = self.recognizer.listen(source, phrase_time_limit=5)
        try:
            return self.recognizer.recognize_google(audio).lower()
        except:
            return None
            
    def speak(self, text):
        print(f"🔊 AI: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

# ========================
# 🧠 Smarter AI Classifier
# ========================
class ExpenseClassifier:
    def __init__(self):
        self.patterns = {
            "Food": r"(zomato|swiggy|food|pizza|burger|coffee|lunch|dinner|groceries)",
            "Transport": r"(uber|ola|taxi|bus|train|metro|petrol|fuel)",
            "Rent": r"(rent|room|pg|hostel|deposit|lease)"
        }
        
    def classify(self, text):
        text = text.lower()
        for category, pattern in self.patterns.items():
            if re.search(pattern, text):
                return category
        return "Others"

# ========================
# 💎 Optimized BudgetAI Core
# ========================
class BudgetAI:
    def __init__(self, data_file="expenses.csv"):
        self.file = data_file
        self.voice = VoiceAssistant()
        self.classifier = ExpenseClassifier()
        self.df = self._load_data()
        
        # Pre-loaded tips database
        self.tips_db = {
            "Food": [
                "🍱 Meal prepping can save you ₹2000/month!",
                "☕ Making coffee at home saves ₹150/week!"
            ],
            "Transport": [
                "🚲 Cycling to college 2x/week = ₹800 saved!",
                "🚗 Carpool with 3 friends to split fuel costs!"
            ],
            "Rent": [
                "🏠 Negotiate rent during summer vacations!",
                "💡 Switch to LED bulbs to reduce electricity bills!"
            ]
        }
        
    def _load_data(self):
        """Load data with memory optimization"""
        try:
            return pd.read_csv(
                self.file, 
                parse_dates=['Date'],
                dtype={'Amount': 'float32', 'Category': 'category'}
            )
        except:
            return pd.DataFrame(columns=["Amount", "Category", "Date"])

    def _optimize_dataframe(self):
        """Reduce memory usage"""
        if not self.df.empty:
            self.df['Category'] = self.df['Category'].astype('category')
            self.df['Amount'] = pd.to_numeric(self.df['Amount'], downcast='float')

    def log_expense(self, amount, description, date=None):
        """Smart logging with auto-categorization"""
        category = self.classifier.classify(description)
        new_entry = {
            "Amount": float(amount),
            "Category": category,
            "Date": pd.to_datetime(date) if date else datetime.now()
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_entry])], 
                          ignore_index=True)
        self._optimize_dataframe()
        self.df.to_csv(self.file, index=False)
        
    def voice_log_expense(self):
        """Process voice input like '500 rupees for pizza'"""
        try:
            spoken = self.voice.listen()
            if not spoken:
                raise ValueError("Couldn't understand audio")
                
            # Extract amount and description
            amount = re.search(r"(\d+)\s*(rs|rupees|₹)?", spoken)
            desc = re.sub(r"\d+\s*(rs|rupees|₹)?\s*", "", spoken).strip()
            
            if not amount:
                raise ValueError("No amount detected")
                
            self.log_expense(amount.group(1), desc)
            self.voice.speak(f"Logged ₹{amount.group(1)} for {desc}")
            return True
            
        except Exception as e:
            self.voice.speak(f"Error: {str(e)}")
            return False

    def get_analysis(self):
        """Lightning-fast analysis with caching"""
        if self.df.empty:
            return None
            
        analysis = {
            "total": self.df['Amount'].sum(),
            "top_category": self.df['Category'].mode()[0],
            "monthly": self.df.groupby(
                self.df['Date'].dt.to_period('M'))['Amount'].sum().to_dict()
        }
        return analysis

    def get_ai_tip(self):
        """Context-aware tips"""
        if self.df.empty:
            return "💡 Start by logging your first expense!"
            
        top_cat = self.df['Category'].mode()[0]
        return random.choice(self.tips_db.get(top_cat, [
            "💰 Save 10% of every paycheck automatically!"
        ]))

    def predict_spending(self):
        """Enhanced prediction with trend analysis"""
        if len(self.df) < 7:
            return "Need at least 7 entries for accurate predictions"
            
        # Create time-based features
        self.df['Days'] = (self.df['Date'] - self.df['Date'].min()).dt.days
        model = LinearRegression()
        model.fit(self.df[['Days']], self.df['Amount'])
        
        next_day = self.df['Days'].max() + 1
        prediction = model.predict([[next_day]])[0]
        
        # Add trend analysis
        trend = "↑ Increasing" if model.coef_[0] > 0 else "↓ Decreasing"
        return f"Predicted: ₹{prediction:.2f} ({trend} trend)"

    def show_insights(self):
        """Interactive visual dashboard"""
        if self.df.empty:
            print("No data to visualize")
            return
            
        plt.figure(figsize=(15, 5))
        
        # Spending Trend
        plt.subplot(1, 3, 1)
        monthly = self.df.groupby(self.df['Date'].dt.to_period('M'))['Amount'].sum()
        monthly.plot(kind='bar', color='#4CAF50')
        plt.title("Monthly Spending Trend")
        plt.ylabel("Amount (₹)")
        
        # Category Distribution
        plt.subplot(1, 3, 2)
        self.df['Category'].value_counts().plot.pie(
            autopct='%1.1f%%', 
            colors=['#FF5722', '#2196F3', '#9C27B0', '#607D8B']
        )
        plt.title("Spending by Category")
        
        # Daily Heatmap
        plt.subplot(1, 3, 3)
        self.df['Weekday'] = self.df['Date'].dt.day_name()
        self.df['Hour'] = self.df['Date'].dt.hour
        pd.pivot_table(
            self.df, 
            values='Amount', 
            index='Weekday', 
            columns='Hour',
            aggfunc='sum',
            fill_value=0
        ).plot(kind='box', vert=False)
        plt.title("Spending Patterns")
        
        plt.tight_layout()
        plt.show()

# ========================
# 🎮 Enhanced CLI Interface
# ========================
def run_app():
    ai = BudgetAI()
    
    while True:
        print("\n" + "="*40)
        print("💰 AI BUDGET TRACKER PRO".center(40))
        print("="*40)
        print("1. Log Expense (Typing)")
        print("2. Log Expense (Voice)")
        print("3. View Analysis")
        print("4. Get AI Tip")
        print("5. Predict Spending")
        print("6. Show Insights")
        print("7. Exit")
        
        choice = input("\nChoose (1-7): ").strip()
        
        if choice == "1":
            try:
                amount = float(input("Amount (₹): "))
                desc = input("Description: ")
                ai.log_expense(amount, desc)
                print("✅ Expense logged!")
            except ValueError:
                print("❌ Invalid input!")
                
        elif choice == "2":
            print("\nSpeak naturally like: '300 rupees for dominos pizza'")
            if ai.voice_log_expense():
                print("✅ Voice expense logged!")
                
        elif choice == "3":
            analysis = ai.get_analysis()
            if analysis:
                print(f"\n📊 Total Spent: ₹{analysis['total']:.2f}")
                print(f"🏆 Top Category: {analysis['top_category']}")
                print("📅 Monthly Breakdown:")
                for month, amount in analysis['monthly'].items():
                    print(f"  - {month}: ₹{amount:.2f}")
            else:
                print("No data available")
                
        elif choice == "4":
            print("\n" + ai.get_ai_tip())
            
        elif choice == "5":
            print("\n" + ai.predict_spending())
            
        elif choice == "6":
            ai.show_insights()
            
        elif choice == "7":
            ai.voice.speak("Goodbye! Keep saving smart!")
            break
            
        else:
            print("Invalid choice")

if __name__ == "__main__":
    run_app()